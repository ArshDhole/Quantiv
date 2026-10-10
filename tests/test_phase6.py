"""Phase 6 tests: model card, license gate, compare/package CLI, dashboard API."""

import json

from typer.testing import CliRunner

from quantiv.cli.main import app
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
