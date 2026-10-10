"""Orchestrator: bounded tool-calling loop with guardrails.

Guardrails (master prompt 4.8): bounded iterations, wall-time budget, typed
tool set only (no shell), no invented numbers (tools measure), failures are
recorded and stop the run loudly after the retry budget.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from quantiv.agent.providers import EchoProvider, Provider
from quantiv.agent.tools import TOOLS, call_tool


@dataclass
class AgentResult:
    success: bool
    steps: list[dict] = field(default_factory=list)
    report_md: str = ""
    report_json: str = ""
    error: str = ""


def run_agent(
    model: str,
    goal: str = "balanced",
    run_dir: str = "runs/agent",
    provider: Provider | None = None,
    max_steps: int = 10,
    time_budget_s: int = 3600,
    on_step=None,
) -> AgentResult:
    """Run the orchestration loop. Deterministic with EchoProvider."""
    provider = provider or EchoProvider()
    history: list[dict] = [{"model": model, "goal": goal, "run_dir": run_dir}]
    tools = sorted(TOOLS)
    t0 = time.time()
    failures = 0

    for _ in range(max_steps):
        if time.time() - t0 > time_budget_s:
            return AgentResult(False, history[1:], error="time budget exceeded")
        decision = provider.decide(goal, history, tools)
        if on_step:
            on_step(f"agent: {decision.tool} ({decision.reasoning})")
        result = call_tool(decision.tool, decision.args)
        entry = {
            "tool": decision.tool,
            "args": decision.args,
            "reasoning": decision.reasoning,
            "ok": result.ok,
            "data": result.data,
            "error": result.error,
        }
        history.append(entry)
        if not result.ok:
            failures += 1
            if failures >= 3:
                return AgentResult(False, history[1:], error=f"3 tool failures, last: {result.error}")
            continue
        if decision.tool == "report" and result.ok:
            return AgentResult(
                True,
                history[1:],
                report_md=result.data.get("report_md", ""),
                report_json=result.data.get("report_json", ""),
            )
    return AgentResult(False, history[1:], error=f"step budget exceeded ({max_steps})")


def agent_result_to_dict(res: AgentResult) -> dict:
    return asdict(res)
