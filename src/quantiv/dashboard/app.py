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
        return """<html><head><title>Quantiv</title></head><body>
<h1>Quantiv</h1>
<form id="f">Model: <input id="model" size="50" value="HuggingFaceTB/SmolLM2-135M">
Goal: <select id="goal"><option>balanced</option><option>min-size</option><option>max-quality</option></select>
Method: <select id="method"><option>auto</option><option>hqq</option><option>gguf</option>
<option>gptq</option><option>bnb</option></select>
<button>Quantize</button></form>
<pre id="out"></pre>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();
 const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({model:document.getElementById('model').value,goal:document.getElementById('goal').value,method:document.getElementById('method').value})});
 const j=await r.json();out.textContent='job '+j.id+' ...';
 const t=setInterval(async()=>{
  const s=await (await fetch('/jobs/'+j.id)).json();
  out.textContent=JSON.stringify(s,null,1)+'\\n'+(s.log||'');
  if(s.status==='done'||s.status==='failed'){clearInterval(t);
   if(s.status==='done'){const rep=await(await fetch('/jobs/'+j.id+'/report')).json();
    out.textContent+='\\n'+JSON.stringify(rep.comparison,null,1);}}},2000);};
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
            "SELECT id,model,goal,method,status,created,updated,run_dir,error "
            "FROM jobs ORDER BY created DESC"
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
