"""Phase 6 tests: model card, license gate, compare/package CLI, dashboard API."""

import json
import sqlite3
import time

from typer.testing import CliRunner

from quantiv.cli.main import app
from quantiv.dashboard.app import _progress, _stall_min, check_stalls
from quantiv.packaging import build_model_card, check_publish_allowed

runner = CliRunner()


def test_model_card_honest():
    card = build_model_card(
        "org/model",
        {"method": "hqq", "quant": "4bit", "device": "cpu"},
        {"ppl_increase": 0.13, "gate_pass": False},
        "apache-2.0",
        "abc123",
    )
    assert "0.13" in card and "FAIL" in card and "apache-2.0" in card


def test_license_gate():
    assert check_publish_allowed("apache-2.0", False, False).allowed is True
    assert check_publish_allowed("Meta Llama 3.1 Community License", False, False).allowed is False
    assert check_publish_allowed("Meta Llama 3.1 Community License", False, True).allowed is True
    assert check_publish_allowed("apache-2.0", True, True).allowed is False  # gated never


def _fake_run(d, method="hqq", ppl_inc=0.13, gate=False):
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(
        json.dumps(
            {
                "baseline": {"model": "org/model"},
                "quantized": {"tokens_per_sec": 10.0, "disk_size_gb": 0.5},
                "quant": {"method": method, "quant": "4bit"},
                "comparison": {"ppl_increase": ppl_inc, "gate_pass": gate},
            }
        )
    )
    (d / "quantiv_manifest.json").write_text(json.dumps({"model": "org/model", "x": 1}))


def test_compare_and_package(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _fake_run(a, "hqq")
    _fake_run(b, "gguf", ppl_inc=0.2, gate=True)
    r = runner.invoke(app, ["compare", str(a), str(b)])
    assert r.exit_code == 0 and "hqq" in r.output and "gguf" in r.output


def test_dashboard_submit_and_status(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantiv.dashboard import create_app

    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(tmp_path / "dash"))
    r = client.post("/jobs", json={"model": "x", "goal": "balanced", "method": "hqq"})
    assert r.status_code == 200
    job_id = r.json()["id"]
    s = client.get(f"/jobs/{job_id}").json()
    assert s["status"] in ("running", "done", "failed")
    assert isinstance(client.get("/jobs").json(), list)
    assert client.get("/jobs/nope/download").status_code == 404


def test_build_artifact_zip(tmp_path):
    import zipfile

    from quantiv.dashboard.app import build_artifact_zip

    run = tmp_path / "run1"
    qdir = run / "attempt1-hqq"
    qdir.mkdir(parents=True)
    (qdir / "qmodel.pt").write_bytes(b"fake-weights")
    (qdir / "config.json").write_text("{}")
    (run / "report.json").write_text(
        json.dumps(
            {
                "quant": {"method": "hqq", "quant": "4bit", "attempt_dir": str(qdir)},
                "comparison": {"gate_pass": True},
            }
        )
    )
    (run / "report.md").write_text("# report")
    archive = build_artifact_zip(run)
    names = zipfile.ZipFile(archive).namelist()
    assert "quantized/qmodel.pt" in names
    assert "report.json" in names and "report.md" in names


def test_download_endpoint_serves_zip(tmp_path, monkeypatch):
    import sqlite3
    import time

    from fastapi.testclient import TestClient

    from quantiv.dashboard import create_app

    monkeypatch.chdir(tmp_path)
    dash = tmp_path / "dash"
    client = TestClient(create_app(dash))
    run = dash / "job1"
    qdir = run / "attempt1-hqq"
    qdir.mkdir(parents=True)
    (qdir / "model.gguf").write_bytes(b"fake")
    (run / "report.json").write_text(
        json.dumps(
            {
                "quant": {"method": "gguf", "attempt_dir": str(qdir)},
            }
        )
    )
    client.get("/jobs")  # ensure jobs table exists before raw insert
    conn = sqlite3.connect(dash / "jobs.db")
    conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("job1", "m", "balanced", "gguf", "done", time.time(), time.time(), str(run), "", ""),
    )
    conn.commit()
    conn.close()
    r = client.get("/jobs/job1/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert len(r.content) > 0


def _seed_running(db_path, job_id="stalled1", age_min=60):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, model TEXT, goal TEXT, method TEXT,"
        " status TEXT, created REAL, updated REAL, run_dir TEXT, error TEXT, log TEXT)"
    )
    now = time.time()
    conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (job_id, "m", "balanced", "hqq", "running", now - age_min * 60, now - age_min * 60, "", "", "old log"),
    )
    conn.commit()
    conn.close()


def test_startup_cleanup_fails_interrupted(tmp_path):
    from quantiv.dashboard import create_app

    db = tmp_path / "dash" / "jobs.db"
    db.parent.mkdir(parents=True)
    _seed_running(db)
    create_app(tmp_path / "dash")  # startup cleanup runs on creation
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT status,error FROM jobs WHERE id='stalled1'").fetchone()
    conn.close()
    assert row[0] == "failed" and "restarted" in row[1]


def test_check_stalls_flags_silent_job(tmp_path):
    db = tmp_path / "jobs.db"
    _seed_running(db, age_min=60)
    _progress["stalled1"] = time.time() - 3600
    _stall_min["stalled1"] = 30
    try:
        flagged = check_stalls(db, default_stall_min=30)
        assert flagged == ["stalled1"]
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT status,error FROM jobs WHERE id='stalled1'").fetchone()
        conn.close()
        assert row[0] == "failed" and "no progress" in row[1]
    finally:
        _progress.pop("stalled1", None)
        _stall_min.pop("stalled1", None)


def test_check_stalls_spares_fresh_job(tmp_path):
    db = tmp_path / "jobs.db"
    _seed_running(db, job_id="fresh", age_min=2)
    _progress["fresh"] = time.time()
    try:
        assert check_stalls(db, default_stall_min=30) == []
    finally:
        _progress.pop("fresh", None)
