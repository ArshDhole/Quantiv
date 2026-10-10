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

    def run_job(job_id: str, model: str, goal: str, method: str) -> None:
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
                max_attempts=3,
                max_samples=16,
                run_dir=state / job_id,
                on_step=on_step,
            )
            set_job(job_id, status="done", run_dir=out["run_dir"])
        except Exception as e:  # noqa: BLE001
            set_job(job_id, status="failed", error=f"{type(e).__name__}: {e}")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quantiv — auto-quantize LLMs</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--txt:#e6edf3;--dim:#8b949e;
--acc:#2f81f7;--ok:#3fb950;--bad:#f85149;--warn:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;gap:12px;margin-bottom:20px}
header h1{font-size:22px;margin:0}header h1 span{color:var(--acc)}
.sub{color:var(--dim);margin:0 0 20px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:18px}
.card h2{font-size:15px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
.grid{display:grid;grid-template-columns:2fr 1fr 1fr auto;gap:10px}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:4px}
input,select{width:100%;background:#0d1117;border:1px solid var(--border);color:var(--txt);
border-radius:6px;padding:8px 10px;font-size:14px}
button{background:var(--acc);border:0;color:#fff;border-radius:6px;padding:9px 18px;
font-size:14px;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.15)}button:disabled{opacity:.5;cursor:wait}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.05em}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600}
.b-run{background:#1f6feb33;color:#79c0ff}.b-done{background:#3fb95033;color:#56d364}
.b-fail{background:#f8514933;color:#ff7b72}
a{color:#79c0ff;cursor:pointer}a:hover{text-decoration:underline}
#detail h3{margin:4px 0 10px}.mono{white-space:pre-wrap;background:#0d1117;border:1px solid var(--border);
border-radius:6px;padding:10px;font:12px ui-monospace,Consolas,monospace;max-height:260px;overflow:auto}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.kv div{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:8px 12px}
.kv small{display:block;color:var(--dim);font-size:11px;text-transform:uppercase}
.kv b{font-size:17px}
.pass{color:var(--ok);font-weight:700}.fail{color:var(--bad);font-weight:700}
.hint{color:var(--dim);font-size:12px}
</style></head><body><div class="wrap">
<header><h1>⚡ Quantiv <span>v1.0</span></h1></header>
<p class="sub">Give it a model + a goal — get back a verified quantized artifact with measured quality, speed &amp; memory numbers. No fabricated metrics, ever.</p>
<div class="card"><h2>New quantization job</h2>
<div class="grid">
<div><label>Model (HF id or local path)</label><input id="model" value="HuggingFaceTB/SmolLM2-135M"></div>
<div><label>Goal</label><select id="goal"><option>balanced</option><option>min-size</option><option>max-quality</option><option>min-latency</option><option>cpu-efficient</option></select></div>
<div><label>Method</label><select id="method"><option>auto</option><option>gptq</option><option>awq</option><option>hqq</option><option>gguf</option><option>bnb</option></select></div>
<div><label>&nbsp;</label><button id="go">Quantize</button></div>
</div><p class="hint">Runs the full pipeline: analyze → quantize → evaluate → quality gate → escalate if needed.</p></div>
<div class="card"><h2>Jobs</h2><table id="jobs">
<tr><th>Job</th><th>Model</th><th>Goal</th><th>Method</th><th>Status</th><th>Updated</th></tr>
</table></div>
<div class="card" id="detail" style="display:none"><h2>Job detail</h2><div id="d"></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
const badge=s=>s==='done'?'<span class="badge b-done">done</span>':s==='failed'?'<span class="badge b-fail">failed</span>':'<span class="badge b-run">running</span>';
const esc=s=>String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
let watch=null,cur=null;
async function refreshJobs(){
 const jobs=await (await fetch('/jobs')).json();
 const t=$('jobs');
 t.innerHTML='<tr><th>Job</th><th>Model</th><th>Goal</th><th>Method</th><th>Status</th><th>Updated</th></tr>'+
  jobs.map(j=>`<tr><td><a onclick="show('${j.id}')">${j.id}</a></td><td>${esc(j.model)}</td><td>${esc(j.goal)}</td><td>${esc(j.method)}</td><td>${badge(j.status)}</td><td>${new Date(j.updated*1000).toLocaleTimeString()}</td></tr>`).join('')
  ||'<tr><td colspan=6 class="hint">no jobs yet — submit one above</td></tr>';
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
  const q=rep.quantized||{},c=rep.comparison||{};
  const gate=c.gate_pass===true?'<span class="pass">PASS</span>':c.gate_pass===false?'<span class="fail">FAIL</span>':'n/a';
  h+=`<div class="kv">
   <div><small>Perplexity Δ</small><b>${c.ppl_increase??'—'}</b></div>
   <div><small>Gate</small><b>${gate}</b></div>
   <div><small>Tokens/sec</small><b>${q.tokens_per_sec??'—'}</b></div>
   <div><small>Disk GB</small><b>${q.disk_size_gb??'—'}</b></div>
   <div><small>Method</small><b>${esc((rep.quant||{}).method??'')} ${(rep.quant||{}).quant??''}</b></div>
  </div>`;
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
  body:JSON.stringify({model:$('model').value,goal:$('goal').value,method:$('method').value})});
 const j=await r.json();$('go').disabled=false;
 refreshJobs();show(j.id);
};
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
        threading.Thread(
            target=run_job,
            args=(job_id, payload.get("model", ""), payload.get("goal", "balanced"), payload.get("method", "auto")),
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
        return JSONResponse([dict(zip(keys, r, strict=False)) for r in rows])

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
