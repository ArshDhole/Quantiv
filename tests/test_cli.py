"""CLI + planner + config tests."""

from typer.testing import CliRunner

from quantiv.cli.main import app
from quantiv.planner import recommend_plans
from quantiv.utils.config import RunConfig

runner = CliRunner()


def test_doctor_exits_zero():
    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_analyze_tiny_dir(tiny_model_dir):
    r = runner.invoke(app, ["analyze", str(tiny_model_dir)])
    assert r.exit_code == 0
    assert "LlamaForCausalLM" in r.output


def test_plan_lists_candidates(tiny_model_dir):
    r = runner.invoke(app, ["plan", str(tiny_model_dir), "--target", "cpu-only", "--goal", "balanced"])
    assert r.exit_code == 0
    assert "Q4_K_M" in r.output


def test_run_unknown_method_fails():
    r = runner.invoke(app, ["run", "some-model", "--method", "bogus-backend"])
    assert r.exit_code != 0


def test_run_agent_openai_compat_needs_args():
    r = runner.invoke(app, ["run", "some-model", "--agent", "--agent-provider", "openai-compat"])
    assert r.exit_code == 2


def test_planner_goals():
    for goal in ("min-size", "balanced", "max-quality", "min-latency", "cpu-efficient"):
        cands = recommend_plans(1.1, 8, goal)
        assert len(cands) == 3
        assert all("method" in c for c in cands)


def test_run_config_defaults():
    c = RunConfig(model="x")
    assert c.goal == "balanced"
    assert c.no_agent is True
