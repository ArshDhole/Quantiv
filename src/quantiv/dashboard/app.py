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

    @app.get("/api/plan")
    def plan_preview(model: str = "", goal: str = "balanced", target: str = "local") -> JSONResponse:
        """Ranked plan preview before committing to a run (config-only, no weights)."""
        from quantiv.analyzer import analyze_model
        from quantiv.hardware import get_target_profile, profile_hardware
        from quantiv.planner.rank import rank_candidates
        from quantiv.quantizers import available_backends

        try:
            tgt = get_target_profile(target)
        except KeyError:
            tgt = {"vram_gb": None}
        hw = profile_hardware()
        vram = tgt.get("vram_gb", hw.gpu_vram_gb)
        try:
            prof = analyze_model(model) if model else None
        except Exception:
            prof = None
        params_b = (prof.param_count / 1e9) if prof and prof.param_count else None
        avail = {k: v.available for k, v in available_backends().items()}
        cands = rank_candidates(goal=goal, params_b=params_b, target_vram_gb=vram, available=avail)
        return JSONResponse({
            "model": model,
            "params_b": params_b,
            "target_vram_gb": vram,
            "license": getattr(prof, "license", None),
            "license_flag": getattr(prof, "license_is_restrictive_or_gated", False),
            "candidates": [c.to_dict() for c in cands],
        })

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quantiv — auto-quantize LLMs</title>
<style>
:root{--bg:#0b0d11;--panel:#14171e;--panel2:#101319;--line:#262c37;--ink:#ece7d9;--dim:#8f8875;
--acc:#f0a832;--acc-ink:#171106;--ok:#7ee2a0;--ok-soft:#7ee2a022;--bad:#ff8f86;--bad-soft:#ff8f8622;
--warn:#f0a832;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 var(--mono);
background-image:linear-gradient(#ffffff05 1px,transparent 1px),linear-gradient(90deg,#ffffff05 1px,transparent 1px);
background-size:34px 34px}
.wrap{max-width:1060px;margin:0 auto;padding:22px 26px 44px}
.topbar{display:flex;align-items:center;gap:12px;padding:12px 0}
.mark{width:32px;height:32px;border-radius:8px;background:var(--acc);color:var(--acc-ink);
display:flex;align-items:center;justify-content:center;font-weight:900;font-size:17px}
.brand{font-weight:800;font-size:16px;letter-spacing:.02em}
.brand small{display:block;font-weight:400;font-size:10px;letter-spacing:.3em;color:var(--dim)}
.pills{margin-left:auto;display:flex;gap:8px}
.pill{border:1px solid var(--line);background:var(--panel);border-radius:20px;
padding:5px 14px;font-size:12px;color:var(--dim)}
.pill b{color:var(--ink)}.dotlive{color:var(--ok)}
.ledger{border:1px solid var(--line);border-radius:12px;background:var(--panel2);
padding:26px 28px;margin:26px 0 8px;position:relative;overflow:hidden}
.ledger h1{font-size:clamp(30px,4.6vw,50px);line-height:1.04;margin:6px 0 12px;
letter-spacing:-.02em;font-weight:800}
.ledger h1 em{font-style:normal;color:var(--acc)}
.eyebrow{font-size:11.5px;letter-spacing:.32em;color:var(--dim)}
.lede{color:#c9c2b0;max-width:60em;margin:0 0 18px;font-size:15px}
.receipt{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:center;
border:1px dashed #ffffff2e;border-radius:10px;padding:14px 18px;margin:16px 0;font-size:13px}
.receipt .cell small{display:block;color:var(--dim);font-size:10.5px;letter-spacing:.14em}
.receipt .cell b{font-size:19px;font-variant-numeric:tabular-nums}
.receipt .arrow{color:var(--acc);font-size:20px}
.receipt .stamp{border:2px solid var(--ok);color:var(--ok);border-radius:6px;padding:6px 14px;
font-weight:800;letter-spacing:.14em;transform:rotate(-4deg);font-size:13px;white-space:nowrap}
#bits{width:100%;height:120px;display:block;margin-top:6px}
.bitcap{display:flex;justify-content:space-between;color:var(--dim);font-size:11.5px;letter-spacing:.12em}
.bench{display:grid;grid-template-columns:220px 1fr;gap:0;margin-top:22px;border:1px solid var(--line);
border-radius:14px;overflow:hidden;background:var(--panel)}
.rail{background:var(--panel2);padding:22px 0;border-right:1px solid var(--line)}
.rail div{padding:11px 22px;color:var(--dim);font-size:13px;cursor:pointer;border-left:3px solid transparent}
.rail div.on{color:var(--ink);border-left-color:var(--acc);background:#ffffff06}
.rail div b{color:var(--acc);margin-right:8px}
.stage{padding:24px 26px;min-height:340px}
.stage h3{margin:0 0 4px;font-size:21px;letter-spacing:-.01em}
.stage .k{font-size:11px;letter-spacing:.26em;color:var(--dim);margin-bottom:6px}
.stage p.d{color:var(--dim);font-size:13px;margin:0 0 16px;max-width:52em}
label{display:block;font-size:11.5px;color:var(--dim);margin-bottom:5px;font-weight:600;letter-spacing:.08em}
input,select{width:100%;background:#0a0c10;border:1px solid var(--line);color:var(--ink);
border-radius:8px;padding:9px 12px;font-size:14px;font-family:inherit}
input:focus,select:focus{outline:1px solid var(--acc);border-color:var(--acc)}
button{background:var(--acc);border:0;color:var(--acc-ink);border-radius:8px;padding:11px 20px;
font-size:14px;font-weight:800;cursor:pointer;font-family:inherit;width:100%}
button:hover{filter:brightness(1.1)}button:disabled{opacity:.55;cursor:wait}
.goals{display:grid;gap:8px;margin:4px 0 12px}
.goal{border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;background:#0a0c10}
.goal b{display:block;font-size:13.5px}.goal small{color:var(--dim);font-size:11.5px}
.goal.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset;background:#f0a83214}
.planprev{margin-top:14px;font-size:12.5px}
.planprev table{font-size:12.5px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;text-transform:uppercase;font-size:10.5px;letter-spacing:.08em}
tr:last-child td{border-bottom:0}
.badge{display:inline-block;padding:2px 10px;border-radius:5px;font-size:12px;font-weight:700}
.b-run{background:#f0a83226;color:var(--warn)}.b-done{background:#7ee2a026;color:var(--ok)}
.b-fail{background:#ff8f8626;color:var(--bad)}
a{color:var(--acc);cursor:pointer;text-decoration:none}a:hover{text-decoration:underline}
#detail h3{margin:4px 0 10px}.mono{white-space:pre-wrap;background:#0a0c10;border:1px solid var(--line);
border-radius:8px;padding:10px;font-size:12px;max-height:240px;overflow:auto;color:#c9c2b0}
.cert{border:1px dashed #ffffff30;border-radius:10px;padding:18px 20px;margin:14px 0;position:relative;background:#0a0c10}
.cert .stamp{position:absolute;top:14px;right:16px}
.cert h4{margin:0 0 10px;font-size:13px;letter-spacing:.2em;color:var(--dim)}
.certgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.certgrid small{display:block;color:var(--dim);font-size:10.5px;letter-spacing:.1em}
.certgrid b{font-size:17px;font-variant-numeric:tabular-nums}
.pass{color:var(--ok);font-weight:700}.fail{color:var(--bad);font-weight:700}
.hint{color:var(--dim);font-size:12.5px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.dot.on{background:var(--ok)}.dot.off{background:#4a4438}
summary{cursor:pointer}
.grid{display:grid;gap:10px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin-bottom:16px}
.panel h2{font-size:11.5px;margin:0 0 14px;text-transform:uppercase;letter-spacing:.14em;color:var(--dim);font-weight:700}
footer{text-align:center;color:var(--dim);font-size:12px;margin:26px 0 10px}
@media(max-width:860px){.bench{grid-template-columns:1fr}.rail{display:flex;overflow:auto;border-right:0;border-bottom:1px solid var(--line)}.receipt{grid-template-columns:1fr;text-align:center}.receipt .arrow{transform:rotate(90deg)}}
</style></head><body><div class="wrap">
<div class="topbar"><div class="mark">Q</div><div class="brand">Quantiv<small>PRECISION LAB · v1.0</small></div>
<div class="pills"><span class="pill">planner: auto</span><span class="pill"><span class="dotlive">●</span>&nbsp;<b id="nb">…</b>&nbsp;backends live</span></div></div>
<div class="ledger">
<div class="eyebrow">EVERY BIT, ACCOUNTED FOR</div>
<h1>Smaller models.<br><em>Receipts included.</em></h1>
<p class="lede">Quantiv packs an open-source LLM into fewer bits per weight — then proves, with measured perplexity, speed and memory numbers, that nothing important was lost. Each run ships a certificate. Nothing is ever fabricated.</p>
<div class="receipt">
<div class="cell"><small>IN · fp16</small><b>272 MB</b></div><div class="arrow">→</div>
<div class="cell"><small>OUT · 8-bit</small><b>173 MB</b></div><div class="arrow">→</div>
<div class="cell"><small>smolLM2-135M · measured</small><b>Δppl −0.1%</b></div>
<div class="stamp">GATE&nbsp;PASS</div>
</div>
<canvas id="bits"></canvas>
<div class="bitcap"><span>FP16 · 16 BITS/WEIGHT</span><span>PACKING…</span><span>INT4 · 4 BITS/WEIGHT</span></div>
</div>
<div class="bench">
<div class="rail" id="rail">
<div data-s="0" class="on"><b>01</b>INTAKE</div>
<div data-s="1"><b>02</b>CALIBRATE</div>
<div data-s="2"><b>03</b>VERIFY &amp; FIRE</div>
<div data-s="3"><b>04</b>LEDGER</div>
</div>
<div class="stage" id="stage"></div>
</div>
<div class="panel" id="detail" style="display:none"><h2>Certificate of quantization</h2><div id="d"></div></div>
<footer>Quantiv · <a href="https://github.com/ArshDhole/Quantiv">GitHub</a> · every number measured</footer>
</div>
<script>
const $=id=>document.getElementById(id);
const badge=s=>s==='done'?'<span class="badge b-done">done</span>':s==='failed'?'<span class="badge b-fail">failed</span>':'<span class="badge b-run">running</span>';
const esc=s=>String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
let watch=null,cur=null,STAGE=0;
const STAGE_META=[
 ["01 · INTAKE","What are we weighing?","Pick a model to put on the bench."],
 ["02 · CALIBRATE","What does good look like?","Set the goal — the planner prices every strategy before you spend a GPU-minute."],
 ["03 · VERIFY & FIRE","Run it, then check the receipt.","Quantize, measure both versions, gate the result. Escalates automatically on a miss."],
 ["04 · LEDGER","Every run, on record.","Finished and running jobs with their verdicts. Nothing is ever fabricated."],
];
function stageHTML(i){
 if(i===0)return `<div class="k">01 — INTAKE</div><h3>What are we weighing?</h3>
  <p class="d">Pick a model to put on the bench. Anything public works — or point at a local directory.</p>
  <label>MODEL</label><select id="modelpick"></select>
  <div id="customwrap" style="display:none;margin-top:8px"><input id="model" value="HuggingFaceTB/SmolLM2-135M" placeholder="org/model-name or /local/path"></div>
  <p class="hint" id="modelnote"></p>`;
 if(i===1)return `<div class="k">02 — CALIBRATE</div><h3>What does good look like?</h3>
  <p class="d">Set the goal. The bench prices every strategy <i>before</i> spending a GPU-minute.</p>
  <div class="goals" id="goals"></div><div id="planprev" class="planprev hint">pick a model to price strategies…</div>`;
 if(i===2)return `<div class="k">03 — VERIFY &amp; FIRE</div><h3>Run it, check the receipt.</h3>
  <p class="d">Quantize → measure both versions → gate. A miss escalates automatically (bounded).</p>
  <div><label>ENGINE</label><select id="method"><option value="auto">Auto — planner picks</option><option>gptq</option><option>awq</option><option>hqq</option><option>gguf</option><option>bnb</option></select></div>
  <div class="grid" style="grid-template-columns:1fr 1fr;margin-top:8px">
  <div><label>EVAL SAMPLES</label><select id="samples"><option>8</option><option selected>16</option><option>32</option></select></div>
  <div><label>MAX ATTEMPTS</label><select id="attempts"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></div></div>
  <div style="margin-top:12px"><button id="go">▶ Quantize</button></div>`;
 return `<div class="k">04 — LEDGER</div><h3>Every run, on record.</h3>
  <p class="d">Finished and running jobs with verdicts. Click a row for its certificate.</p>
  <table id="jobs"><tr><th>Run</th><th>Model</th><th>Goal</th><th>Status</th><th>Result</th><th></th></tr></table>
  <div id="d" style="margin-top:14px"></div>`;
}
function setStage(i){
 STAGE=i;
 document.querySelectorAll('#rail div').forEach((el,k)=>el.classList.toggle('on',k===i));
 $('stage').innerHTML=stageHTML(i);
 if(i===0)initIntake();
 if(i===1)initCalibrate();
 if(i===2)initFire();
 if(i===3){refreshJobs();}
}
document.querySelectorAll('#rail div').forEach(el=>el.onclick=()=>setStage(parseInt(el.dataset.s)));
async function refreshJobs(){
 const t=$('jobs');
 if(!t)return;
 const jobs=await (await fetch('/jobs')).json();
 const short=m=>{const p=m.split('/');return esc(p.length>1?p[1]:m);};
 t.innerHTML='<tr><th>Run</th><th>Model</th><th>Goal</th><th>Status</th><th>Result</th><th></th></tr>'+
  jobs.map(j=>`<tr><td class="hint">${j.id}</td><td>${short(j.model)}</td><td>${esc(j.goal)}</td><td>${badge(j.status)}</td><td class="hint">${esc(j.result||'')}</td><td><a onclick="show('${j.id}')">open →</a></td></tr>`).join('')
  ||'<tr><td colspan=6 class="hint">ledger empty — fire a run from 03</td></tr>';
}
async function show(id){
 cur=id;
 if(STAGE!==3)setStage(3);
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
  const pass=c.gate_pass===true;
  const stamp=pass?'<span class="stamp" style="border-color:var(--ok);color:var(--ok)">GATE&nbsp;PASS</span>':'<span class="stamp" style="border-color:var(--bad);color:var(--bad)">GATE&nbsp;FAIL</span>';
  let shrink='—';
  if(base.disk_size_gb&&q.disk_size_gb&&q.disk_size_gb>0)shrink=(base.disk_size_gb/q.disk_size_gb).toFixed(1)+'× smaller';
  h+=`<div class="cert">${stamp}<h4>CERTIFICATE OF QUANTIZATION · QUANTIV v1.0</h4>
   <div class="certgrid">
   <div><small>METHOD</small><b>${esc((rep.quant||{}).method??'')} ${(rep.quant||{}).quant??''}</b></div>
   <div><small>PPL CHANGE</small><b>${c.ppl_increase??'—'}</b></div>
   <div><small>SIZE</small><b>${shrink}</b></div>
   <div><small>SPEED</small><b>${q.tokens_per_sec??'—'} tok/s</b></div>
   <div><small>TEXTS</small><b>${esc(q.text_source??'').split('(')[0]}</b></div>
   <div><small>DEVICE</small><b>${esc(q.device??rep.quant?.device??'')}</b></div>
   </div>
   <p class="hint">measured, not claimed · <a href="/jobs/${id}/report" target="_blank">full report.json →</a></p></div>`;
  const at=(rep.attempts||[]).map(a=>`<tr><td>${a.n}</td><td>${esc(a.method)}</td><td>${esc(a.quant)}</td><td>${a.ppl}</td><td>${a.gate_pass?'PASS':'FAIL'}</td></tr>`).join('');
  if(at)h+='<table><tr><th>#</th><th>Method</th><th>Quant</th><th>PPL</th><th>Gate</th></tr>'+at+'</table>';
 }
 if(s.log)h+=`<h3>Log</h3><div class="mono">${esc(s.log)}</div>`;
 const d=$('d');if(d)d.innerHTML=h;
 if(s.status==='done'||s.status==='failed'){clearInterval(watch);refreshJobs();}
}
function fireSubmit(){
 const go=$('go');if(!go)return;
 go.onclick=async()=>{
  go.disabled=true;
  const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({model:(typeof MODEL!=='undefined'&&MODEL)||'HuggingFaceTB/SmolLM2-135M',goal:GOAL,method:$('method').value,
    max_samples:parseInt($('samples').value||'16'),max_attempts:parseInt($('attempts').value||'3')})});
  const j=await r.json();go.disabled=false;
  show(j.id);
 };
}
function initIntake(){
 $('modelpick').innerHTML=GROUPS.map(g=>
  `<optgroup label="${g[0]}">`+MODELS.slice(g[1],g[2]).map(m=>`<option value="${m[0]}">${m[1]}</option>`).join('')+`</optgroup>`).join('')
  +`<option value="__custom">⌨ Custom HF id or local path…</option>`;
 if(typeof MODEL!=='undefined'&&MODEL&&MODELS.some(m=>m[0]===MODEL))$('modelpick').value=MODEL;
 $('modelpick').onchange=syncModel;syncModel();
}
function initCalibrate(){
 $('goals').innerHTML=GOALS.map(g=>`<div class="goal${g[0]===GOAL?' sel':''}" data-g="${g[0]}"><b>${g[1]}</b><small>${g[2]}</small></div>`).join('');
 document.querySelectorAll('.goal').forEach(el=>el.onclick=()=>{GOAL=el.dataset.g;
  document.querySelectorAll('.goal').forEach(x=>x.classList.toggle('sel',x===el));price();});
 price();
}
let priceT=null;
async function price(){
 const box=$('planprev');if(!box)return;
 const mv=(typeof MODEL!=='undefined'&&MODEL)||'';
 if(!mv){box.textContent='pick a model to price strategies…';return;}
 box.textContent='pricing…';
 clearTimeout(priceT);
 priceT=setTimeout(async()=>{
  try{
   const p=await (await fetch('/api/plan?model='+encodeURIComponent(mv)+'&goal='+GOAL)).json();
   const rows=(p.candidates||[]).slice(0,4).map(c=>
    `<tr><td>${esc(c.method)}</td><td>${esc(c.quant)}</td><td>${c.score}</td><td>${c.fits_target?'✓':'⚠'}</td></tr>`).join('');
   box.innerHTML=`priced for ${esc(mv)}${p.params_b?` · ${p.params_b.toFixed(2)}B params`:''}${p.license?` · ${esc(p.license)}`:''}<table><tr><th>Engine</th><th>Quant</th><th>Score</th><th>Fit</th></tr>${rows}</table>`;
  }catch(e){box.textContent='pricing unavailable offline';}
 },400);
}
function initFire(){fireSubmit();}
function paintBits(){
 const cv=$('bits');if(!cv)return;
 const ctx=cv.getContext('2d');
 const W=cv.width=cv.offsetWidth*2,H=cv.height=240;
 let t=0;
 function frame(){
  t+=0.016;ctx.clearRect(0,0,W,H);
  const n=16,bw=W/(n+2);
  for(let i=0;i<n;i++){
   const ph=(t*0.7+i*0.35)%4;
   const k=ph<2?1:(3-ph); // 1 → hold → merge
   const gx=(i%4)*2+Math.floor(i/4)%2, target=Math.floor(i/4);
   const x=(gx+(target-gx)*Math.max(0,k-1))*(bw)+bw;
   const s=bw*0.62*(1-0.35*Math.max(0,k-1));
   const a=0.25+0.75*Math.min(1,2-k+1);
   ctx.fillStyle=i%2?`rgba(240,168,50,${a})`:`rgba(236,231,217,${a*0.8})`;
   const y=H/2-s/2+Math.sin(t*2+i)*4;
   ctx.fillRect(x,y,s,s*0.7);
  }
  requestAnimationFrame(frame);
 }
 frame();
}
async function loadBackends(){
 try{
  const b=await (await fetch('/api/backends')).json();
  const n=Object.values(b).filter(v=>v.available).length;
  $('nb').textContent=n+'/'+Object.keys(b).length;
 }catch(e){$('nb').textContent='…';}
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
 MODEL=$('model').value;
 const m=MODELS.find(x=>x[0]===MODEL);
 $('modelnote').textContent=m?`family ${m[2]} · ${m[1]}`:'Any public HF repo id or a local model directory.';
 price();
}
let MODEL='HuggingFaceTB/SmolLM2-135M';
setStage(0);paintBits();
setInterval(()=>{if(STAGE===3&&!cur)refreshJobs();},5000);
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
