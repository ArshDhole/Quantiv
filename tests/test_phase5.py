"""Phase 5 tests: tool guardrails + echo-agent end-to-end on toy weights."""

from quantiv.agent import TOOLS, EchoProvider, call_tool, run_agent


def test_tool_registry_closed():
    assert set(TOOLS) == {"analyze", "profile_hardware", "plan", "quantize", "evaluate", "report"}
    r = call_tool("exec_shell", {"cmd": "rm -rf /"})
    assert r.ok is False and "unknown tool" in r.error


def test_tool_arg_validation():
    r = call_tool("quantize", {"method": "hqq"})  # missing model/output_dir
    assert r.ok is False and "missing required args" in r.error


def test_plan_tool_ranks():
    r = call_tool("plan", {"goal": "balanced", "params_b": 0.5, "target_vram_gb": 8})
    assert r.ok and len(r.data["candidates"]) >= 3
    assert r.data["candidates"][0]["score"] >= r.data["candidates"][-1]["score"]


def test_echo_agent_e2e_toy(tiny_hf_model, tmp_path):
    """Full agent loop on toy weights: 7 tool steps ending in a real report."""
    res = run_agent(
        str(tiny_hf_model),
        goal="balanced",
        run_dir=str(tmp_path / "agent"),
        provider=EchoProvider(),
        max_steps=10,
        time_budget_s=1200,
    )
    assert res.success, res.error
    assert [s["tool"] for s in res.steps] == [
        "analyze",
        "profile_hardware",
        "plan",
        "quantize",
        "evaluate",
        "evaluate",
        "report",
    ]
    assert all(s["ok"] for s in res.steps)
    assert res.report_md and (tmp_path / "agent" / "report.md").exists()


def test_agent_step_budget():
    res = run_agent(
        "does-not-exist-model-xyz", goal="balanced", run_dir="runs/never", provider=EchoProvider(), max_steps=2
    )
    assert res.success is False
    assert "budget" in res.error
