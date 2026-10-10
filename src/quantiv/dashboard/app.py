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
    from fastapi.responses import JSONResponse

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

    def run_job(
        job_id: str, model: str, goal: str, method: str,
        max_samples: int, max_attempts: int, bits: int,
    ) -> None:
        from quantiv.planner.pipeline import run_pipeline
        from quantiv.quantizers.base import resolve_device

        logs: list[str] = []

        def on_step(m: str) -> None:
            logs.append(m)
            set_job(job_id, log="\n".join(logs[-200:]))

        set_job(job_id, log=f"job accepted · {model} · warming up engine…")

        try:
            out = run_pipeline(
                model,
                goal=goal,
                method=method,
                bits=bits,
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
            {
                k: {"available": v.available, "reason": v.reason, "version": v.version}
                for k, v in available_backends().items()
            }
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
        return JSONResponse(
            {
                "model": model,
                "params_b": params_b,
                "target_vram_gb": vram,
                "license": getattr(prof, "license", None),
                "license_flag": getattr(prof, "license_is_restrictive_or_gated", False),
                "candidates": [c.to_dict() for c in cands],
            }
        )

    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    def index():
        from fastapi.responses import FileResponse

        react = Path(__file__).parent.parent.parent.parent / "frontend" / "dist" / "index.html"
        legacy = Path(__file__).parent / "static" / "index.html"
        return FileResponse(react if react.exists() else legacy)

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
        bits = int(payload.get("bits", 4) or 4)
        bits = bits if bits in (2, 3, 4, 5, 6, 8) else 4
        threading.Thread(
            target=run_job,
            args=(
                job_id,
                payload.get("model", ""),
                payload.get("goal", "balanced"),
                payload.get("method", "auto"),
                max_samples,
                max_attempts,
                bits,
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

    @app.get("/jobs/{job_id}/download")
    def job_download(job_id: str):
        from fastapi.responses import FileResponse

        conn = _db(db_path)
        row = conn.execute("SELECT run_dir,status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if not row or not row[0]:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        if row[1] != "done":
            return JSONResponse({"error": f"job is {row[1]}, nothing to download yet"}, status_code=409)
        try:
            archive = build_artifact_zip(Path(row[0]))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)
        return FileResponse(archive, filename=archive.name, media_type="application/zip")

    return app


def build_artifact_zip(run_dir: str | Path) -> Path:
    """Bundle the final quantized artifact + reports into a cached zip.

    Uses report.json's attempt_dir (falls back to legacy quantized/ layout).
    Raises FileNotFoundError/RuntimeError with a clear message when incomplete.
    """
    import zipfile

    run_dir = Path(run_dir)
    rep_file = run_dir / "report.json"
    if not rep_file.exists():
        raise FileNotFoundError(f"no report.json in {run_dir}")
    rep = json.loads(rep_file.read_text(encoding="utf-8"))
    artifact = rep.get("quant", {}).get("attempt_dir") or rep.get("quantized", {}).get("model")
    artifact_dir = Path(artifact) if artifact else None
    if artifact_dir is None or not artifact_dir.exists():
        legacy = run_dir / "quantized"
        artifact_dir = legacy if legacy.exists() else None
    if artifact_dir is None:
        raise FileNotFoundError("quantized artifact directory is missing")
    files = [p for p in artifact_dir.iterdir() if p.is_file()]
    if not files:
        raise RuntimeError("quantized artifact directory is empty")
    archive = run_dir / f"{run_dir.name}-artifact.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"quantized/{p.name}")
        for extra in ("report.json", "report.md", "MODEL_CARD.md", "quantiv_manifest.json"):
            ep = run_dir / extra
            if ep.exists():
                z.write(ep, extra)
    return archive
