"""Quantiv dashboard (Phase 6, optional): submit jobs, poll progress, fetch reports.

Run: `quantiv dashboard` (serves on 127.0.0.1:8020). Job state in SQLite.
Heavy work runs in a background thread calling the same run_pipeline the CLI uses.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  model TEXT, goal TEXT, method TEXT,
  status TEXT, created REAL, updated REAL,
  run_dir TEXT, error TEXT, log TEXT
);
"""


def _db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    return conn


def _result_summary(run_dir: str) -> str:
    """One-line verdict for the jobs table, read from report.json when present."""
    if not run_dir:
        return ""
    rep = Path(run_dir) / "report.json"
    if not rep.exists():
        return ""
    try:
        data = json.loads(rep.read_text(encoding="utf-8"))
        comp = data.get("comparison", {})
        gate = comp.get("gate_pass")
        inc = comp.get("ppl_increase")
        pct = f" {inc:+.1%}" if isinstance(inc, (int, float)) else ""
        if gate is True:
            return f"PASS{pct}"
        if gate is False:
            return f"FAIL{pct}"
        return "done"
    except Exception:
        return ""


def create_app(state_dir: str | Path = "runs/dashboard"):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    db_path = state / "jobs.db"

    app = FastAPI(title="Quantiv")

    def set_job(job_id: str, **fields) -> None:
        conn = _db(db_path)
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE jobs SET {cols}, updated = ? WHERE id = ?", (*fields.values(), time.time(), job_id))
        conn.commit()
        conn.close()

    def run_job(job_id: str, model: str, goal: str, method: str, max_samples: int, max_attempts: int) -> None:
        from quantiv.planner.pipeline import run_pipeline
        from quantiv.quantizers.base import resolve_device

        logs: list[str] = []

        def on_step(m: str) -> None:
            logs.append(m)
            set_job(job_id, log="\n".join(logs[-200:]))

        try:
            out = run_pipeline(
                model,
                goal=goal,
                method=method,
                device=resolve_device(),
                max_attempts=max_attempts,
                max_samples=max_samples,
                run_dir=state / job_id,
                on_step=on_step,
            )
            set_job(job_id, status="done", run_dir=out["run_dir"])
        except Exception as e:  # noqa: BLE001
            set_job(job_id, status="failed", error=f"{type(e).__name__}: {e}")

    @app.get("/api/backends")
    def backends() -> JSONResponse:
        from quantiv.quantizers import available_backends

        return JSONResponse(
            {k: {"available": v.available, "reason": v.reason, "version": v.version}
             for k, v in available_backends().items()}
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quantiv — auto-quantize LLMs</title>
<style>
:root{--bg:#f7f6f3;--panel:#ffffff;--border:#e5e2d9;--txt:#1c1b18;--dim:#6f6a5e;
--acc:#1f6feb;--acc-soft:#e8f0fe;--ok:#1a7f37;--ok-soft:#dcf2e3;--bad:#c93c37;--bad-soft:#fbe9e7;--warn:#9a6700;--warn-soft:#fef3d8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:32px 24px}
header{display:flex;align-items:center;gap:14px;margin-bottom:8px}
.logo{width:38px;height:38px;border-radius:10px;background:var(--txt);color:#fff;
display:flex;align-items:center;justify-content:center;font-weight:800;font-size:21px}
header h1{font-size:21px;margin:0;letter-spacing:-.01em}
header h1 span{color:var(--dim);font-weight:400;font-size:14px}
.tag{font-size:12px;color:var(--dim)}
#backends{margin-left:auto;text-align:right;font-size:12px;color:var(--dim)}
.sub{color:var(--dim);margin:0 0 24px;font-size:14px}
.steps{display:flex;align-items:center;gap:10px;margin:0 0 20px;color:var(--dim);font-size:13px}
.steps b{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--txt);
color:#fff;align-items:center;justify-content:center;font-size:12px;margin-right:6px;font-weight:700}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
padding:20px 22px;margin-bottom:16px;box-shadow:0 1px 2px #1c1b1808}
.card h2{font-size:12px;margin:0 0 14px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);font-weight:700}
label{display:block;font-size:12.5px;color:var(--dim);margin-bottom:5px;font-weight:600}
input,select{width:100%;background:#fff;border:1px solid var(--border);color:var(--txt);
border-radius:8px;padding:9px 12px;font-size:14px;font-family:inherit}
input:focus,select:focus{outline:2px solid var(--acc-soft);border-color:var(--acc)}
button{background:var(--acc);border:0;color:#fff;border-radius:8px;padding:10px 22px;
font-size:14px;font-weight:700;cursor:pointer;font-family:inherit}
button:hover{filter:brightness(1.08)}button:disabled{opacity:.55;cursor:wait}
.goals{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:4px 0 12px}
.goal{border:1px solid var(--border);border-radius:8px;padding:9px 11px;cursor:pointer;background:#fff}
.goal b{display:block;font-size:13px}.goal small{color:var(--dim);font-size:11.5px}
.goal.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset;background:var(--acc-soft)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.06em}
tr:last-child td{border-bottom:0}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:700}
.b-run{background:var(--warn-soft);color:var(--warn)}.b-done{background:var(--ok-soft);color:var(--ok)}
.b-fail{background:var(--bad-soft);color:var(--bad)}
a{color:var(--acc);cursor:pointer;text-decoration:none}a:hover{text-decoration:underline}
#detail h3{margin:4px 0 10px}.mono{white-space:pre-wrap;background:#f1efe9;border:1px solid var(--border);
border-radius:8px;padding:10px;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;
max-height:260px;overflow:auto;color:#3d3a33}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.kv div{background:#faf9f6;border:1px solid var(--border);border-radius:8px;padding:9px 13px}
.kv small{display:block;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.kv b{font-size:18px;font-variant-numeric:tabular-nums}
.pass{color:var(--ok);font-weight:700}.fail{color:var(--bad);font-weight:700}
.hint{color:var(--dim);font-size:12.5px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.dot.on{background:var(--ok)}.dot.off{background:#cfc9ba}
summary{cursor:pointer}
.grid{display:grid;gap:10px}
</style></head><body><div class="wrap">
<header><div class="logo">Q</div><div><h1>Quantiv <span>· v1.0</span></h1>
<div class="tag">agentic LLM quantization — measured, never fabricated</div></div>
<div id="backends" style="margin-left:auto;text-align:right">backends…</div></header>
<div class="steps"><div><b>1</b> Pick a model</div><div>→</div><div><b>2</b> Set the goal</div><div>→</div><div><b>3</b> Get verified artifact + report</div></div>
<div class="card"><h2>Step 1 · Model</h2>
<select id="modelpick"></select>
<div id="customwrap" style="display:none;margin-top:8px"><input id="model" value="HuggingFaceTB/SmolLM2-135M" placeholder="org/model-name or /local/path"></div>
<p class="hint" id="modelnote"></p></div>
<div class="card"><h2>Step 2 · Goal &amp; method</h2>
<div class="goals" id="goals"></div>
<div class="grid" style="grid-template-columns:1fr 1fr auto;margin-top:4px">
<div><label>Method</label><select id="method"><option value="auto">auto — planner picks ✓</option><option>gptq</option><option>awq</option><option>hqq</option><option>gguf</option><option>bnb</option></select></div>
<div><label>Eval samples</label><select id="samples"><option>8</option><option selected>16</option><option>32</option></select></div>
<div><label>&nbsp;</label><button id="go">▶ Quantize</button></div>
</div>
<details style="margin-top:10px"><summary class="hint">Advanced</summary>
<div style="margin-top:8px"><label>Max escalation attempts (1–5)</label>
<select id="attempts" style="max-width:120px"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></div>
</details></div>
<div class="card"><h2>Step 3 · Runs</h2><table id="jobs">
<tr><th>Run</th><th>Model</th><th>Goal</th><th>Status</th><th>Result</th><th></th></tr>
</table></div>
<div class="card" id="detail" style="display:none"><h2>Run detail</h2><div id="d"></div></div>
<div class="hint" style="text-align:center;margin:24px 0">Quantiv · <a href="https://github.com/ArshDhole/Quantiv">GitHub</a> · reports in <span class="mono" style="display:inline;padding:1px 6px">runs/</span></div>
</div>
<script>
const $=id=>document.getElementById(id);
const badge=s=>s==='done'?'<span class="badge b-done">done</span>':s==='failed'?'<span class="badge b-fail">failed</span>':'<span class="badge b-run">running</span>';
const esc=s=>String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
let watch=null,cur=null;
async function refreshJobs(){
 const jobs=await (await fetch('/jobs')).json();
 const t=$('jobs');
 const short=m=>{const p=m.split('/');return esc(p.length>1?p[1]:m);};
 t.innerHTML='<tr><th>Run</th><th>Model</th><th>Goal</th><th>Status</th><th>Result</th><th></th></tr>'+
  jobs.map(j=>`<tr><td class="hint">${j.id}</td><td>${short(j.model)}</td><td>${esc(j.goal)}</td><td>${badge(j.status)}</td><td class="hint">${esc(j.result||'')}</td><td><a onclick="show('${j.id}')">open →</a></td></tr>`).join('')
  ||'<tr><td colspan=6 class="hint">no runs yet — start one above</td></tr>';
}
async function show(id){
 cur=id;$('detail').style.display='block';
 clearInterval(watch);await detail(id);
 watch=setInterval(()=>detail(id),2000);
}
async function detail(id){
 const s=await (await fetch('/jobs/'+id)).json();
 let h=`<h3>${esc(s.model)} <small class="hint">${s.id} · ${s.goal} · ${s.method}</small> ${badge(s.status)}</h3>`;
 if(s.status==='failed')h+=`<p class="fail">${esc(s.error||'failed')}</p>`;
 if(s.status==='done'){
  const rep=await (await fetch('/jobs/'+id+'/report')).json();
  const q=rep.quantized||{},c=rep.comparison||{},base=rep.baseline||{};
  const gate=c.gate_pass===true?'<span class="pass">PASS</span>':c.gate_pass===false?'<span class="fail">FAIL</span>':'n/a';
  let shrink='—';
  if(base.disk_size_gb&&q.disk_size_gb&&q.disk_size_gb>0)shrink=(base.disk_size_gb/q.disk_size_gb).toFixed(1)+'× smaller';
  h+=`<div class="kv">
   <div><small>Perplexity Δ</small><b>${c.ppl_increase??'—'}</b></div>
   <div><small>Gate</small><b>${gate}</b></div>
   <div><small>Tokens/sec</small><b>${q.tokens_per_sec??'—'}</b></div>
   <div><small>Size</small><b>${shrink}</b></div>
   <div><small>Method</small><b>${esc((rep.quant||{}).method??'')} ${(rep.quant||{}).quant??''}</b></div>
  </div>
  <p><a href="/jobs/${id}/report" target="_blank">full report.json →</a></p>`;
  const at=(rep.attempts||[]).map(a=>`<tr><td>${a.n}</td><td>${esc(a.method)}</td><td>${esc(a.quant)}</td><td>${a.ppl}</td><td>${a.gate_pass?'PASS':'FAIL'}</td></tr>`).join('');
  if(at)h+='<table><tr><th>#</th><th>Method</th><th>Quant</th><th>PPL</th><th>Gate</th></tr>'+at+'</table>';
 }
 if(s.log)h+=`<h3>Log</h3><div class="mono">${esc(s.log)}</div>`;
 $('d').innerHTML=h;
 if(s.status==='done'||s.status==='failed'){clearInterval(watch);refreshJobs();}
}
$('go').onclick=async()=>{
 $('go').disabled=true;
 const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({model:$('model').value,goal:GOAL,method:$('method').value,
   max_samples:parseInt($('samples').value||'16'),max_attempts:parseInt($('attempts').value||'3')})});
 const j=await r.json();$('go').disabled=false;
 refreshJobs();show(j.id);
};
async function loadBackends(){
 try{
  const b=await (await fetch('/api/backends')).json();
  $('backends').innerHTML=Object.entries(b).map(([k,v])=>
   `<span class="dot ${v.available?'on':'off'}"></span>${k}`).join(' ');
 }catch(e){$('backends').textContent='';}
}
loadBackends();
const MODELS=[
 ["HuggingFaceTB/SmolLM2-135M","SmolLM2 135M · tiny, fast","llama"],
 ["HuggingFaceTB/SmolLM2-360M","SmolLM2 360M","llama"],
 ["HuggingFaceTB/SmolLM2-1.7B","SmolLM2 1.7B","llama"],
 ["Qwen/Qwen2.5-0.5B-Instruct","Qwen2.5 0.5B · chat","qwen2"],
 ["Qwen/Qwen2.5-1.5B-Instruct","Qwen2.5 1.5B · chat","qwen2"],
 ["Qwen/Qwen2.5-3B-Instruct","Qwen2.5 3B · chat","qwen2"],
 ["Qwen/Qwen2.5-7B-Instruct","Qwen2.5 7B · needs 8GB+","qwen2"],
 ["TinyLlama/TinyLlama-1.1B-Chat-v1.0","TinyLlama 1.1B · chat","llama"],
 ["meta-llama/Llama-3.2-1B-Instruct","Llama 3.2 1B · 🔒 gated","llama"],
 ["meta-llama/Llama-3.2-3B-Instruct","Llama 3.2 3B · 🔒 gated","llama"],
 ["mistralai/Mistral-7B-Instruct-v0.3","Mistral 7B · needs 16GB+","mistral"],
 ["google/gemma-2-2b-it","Gemma 2 2B · 🔒 gated","gemma"],
 ["microsoft/Phi-3-mini-4k-instruct","Phi-3 mini · MIT","phi"],
 ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B","DeepSeek-R1 distill 1.5B","qwen2"],
];
const GOALS=[
 ["balanced","Balanced","quality ≈ size"],
 ["min-size","Min size","smallest disk"],
 ["max-quality","Max quality","lowest PPL loss"],
 ["min-latency","Min latency","fastest tok/s"],
 ["cpu-efficient","CPU","no-GPU friendly"],
];
let GOAL="balanced";
const GROUPS=[["Small & fast",0,3],["Qwen chat",3,7],["Popular",7,14],["Custom",14,14]];
$('modelpick').innerHTML=GROUPS.map(g=>
 `<optgroup label="${g[0]}">`+MODELS.slice(g[1],g[2]).map(m=>`<option value="${m[0]}">${m[1]}</option>`).join('')+`</optgroup>`).join('')
 +`<option value="__custom">⌨ Custom HF id or local path…</option>`;
function syncModel(){
 const v=$('modelpick').value;
 $('customwrap').style.display=v==='__custom'?'block':'none';
 if(v!=='__custom')$('model').value=v;
 const m=MODELS.find(x=>x[0]===$('model').value);
 $('modelnote').textContent=m?`family ${m[2]} · ${m[1]}`:'Any public HF repo id or a local model directory.';
}
$('modelpick').onchange=syncModel;syncModel();
$('goals').innerHTML=GOALS.map(g=>`<div class="goal${g[0]===GOAL?' sel':''}" data-g="${g[0]}"><b>${g[1]}</b><small>${g[2]}</small></div>`).join('');
document.querySelectorAll('.goal').forEach(el=>el.onclick=()=>{
 GOAL=el.dataset.g;
 document.querySelectorAll('.goal').forEach(x=>x.classList.toggle('sel',x===el));
});
refreshJobs();setInterval(()=>{if(!cur)refreshJobs();},5000);
</script></body></html>"""

    @app.post("/jobs")
    def submit(payload: dict) -> JSONResponse:
        job_id = uuid.uuid4().hex[:12]
        conn = _db(db_path)
        conn.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                payload.get("model", ""),
                payload.get("goal", "balanced"),
                payload.get("method", "auto"),
                "running",
                time.time(),
                time.time(),
                "",
                "",
                "",
            ),
        )
        conn.commit()
        conn.close()
        max_samples = max(1, min(64, int(payload.get("max_samples", 16) or 16)))
        max_attempts = max(1, min(5, int(payload.get("max_attempts", 3) or 3)))
        threading.Thread(
            target=run_job,
            args=(
                job_id,
                payload.get("model", ""),
                payload.get("goal", "balanced"),
                payload.get("method", "auto"),
                max_samples,
                max_attempts,
            ),
            daemon=True,
        ).start()
        return JSONResponse({"id": job_id, "status": "running"})

    @app.get("/jobs")
    def list_jobs() -> JSONResponse:
        conn = _db(db_path)
        rows = conn.execute(
            "SELECT id,model,goal,method,status,created,updated,run_dir,error FROM jobs ORDER BY created DESC"
        ).fetchall()
        conn.close()
        keys = ("id", "model", "goal", "method", "status", "created", "updated", "run_dir", "error")
        out = []
        for r in rows:
            job = dict(zip(keys, r, strict=False))
            job["result"] = _result_summary(job.get("run_dir", ""))
            out.append(job)
        return JSONResponse(out)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        conn = _db(db_path)
        row = conn.execute(
            "SELECT id,model,goal,method,status,created,updated,run_dir,error,log FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        conn.close()
        if not row:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        keys = ("id", "model", "goal", "method", "status", "created", "updated", "run_dir", "error", "log")
        return JSONResponse(dict(zip(keys, row, strict=False)))

    @app.get("/jobs/{job_id}/report")
    def job_report(job_id: str) -> JSONResponse:
        conn = _db(db_path)
        row = conn.execute("SELECT run_dir,status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if not row:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        rep = Path(row[0]) / "report.json" if row[0] else None
        if not rep or not rep.exists():
            return JSONResponse({"status": row[1], "report": None})
        return JSONResponse(json.loads(rep.read_text(encoding="utf-8")))

    return app
