"""LLM provider adapters. The orchestrator is model-agnostic.

- EchoProvider: deterministic offline planner (follows ranked candidates).
  Default. Needs no API key, fully tested.
- OpenAICompatProvider: talks to any OpenAI-compatible chat-completions
  endpoint (hosted APIs, local llama.cpp/vLLM servers). Optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ProviderDecision:
    tool: str
    args: dict
    reasoning: str = ""


class Provider:
    name = "base"

    def decide(self, goal: str, history: list[dict], tools: list[str]) -> ProviderDecision:
        raise NotImplementedError


class EchoProvider(Provider):
    """Offline deterministic policy: analyze -> hardware -> plan -> quantize best -> evaluate both -> report."""

    name = "echo"

    def decide(self, goal: str, history: list[dict], tools: list[str]) -> ProviderDecision:
        seed = history[0]  # {"model":..., "run_dir":...}, never a tool call
        steps = [h for h in history[1:] if h.get("ok")]
        data = {h["tool"]: h.get("data", {}) for h in steps}
        evals = [h.get("data", {}) for h in steps if h["tool"] == "evaluate"]
        n = len(steps)
        if n == 0:
            return ProviderDecision("analyze", {"model": seed.get("model", "")}, "profile the model first")
        if n == 1:
            return ProviderDecision("profile_hardware", {"target": "local"}, "know the machine")
        if n == 2:
            prof = data.get("analyze", {})
            params_b = (prof.get("param_count") or 0) / 1e9 or None
            return ProviderDecision("plan", {"goal": goal, "params_b": params_b}, "rank candidates")
        if n == 3:
            cands = data.get("plan", {}).get("candidates", [])
            best = cands[0] if cands else {"method": "hqq", "quant": "", "bits": 4}
            return ProviderDecision(
                "quantize",
                {
                    "model": data["analyze"]["source"],
                    "method": best["method"],
                    "output_dir": f"{seed.get('run_dir', 'runs/agent')}/quantized",
                    "bits": best.get("bits", 4),
                    "quant": best.get("quant", ""),
                    "goal": goal,
                },
                f"execute top candidate {best.get('method')}",
            )
        if n == 4:
            return ProviderDecision(
                "evaluate", {"model_ref": data["analyze"]["source"], "max_samples": 8}, "measure baseline"
            )
        if n == 5:
            qdir = data.get("quantize", {}).get("output_dir", "")
            return ProviderDecision("evaluate", {"model_ref": qdir, "max_samples": 8}, "measure quantized")
        return ProviderDecision(
            "report",
            {
                "run_dir": seed.get("run_dir", "runs/agent"),
                "baseline": evals[0] if evals else {},
                "quantized": evals[1] if len(evals) > 1 else {},
                "quant_info": {
                    "method": data.get("quantize", {}).get("method", ""),
                    "quant": data.get("quantize", {}).get("quant", ""),
                },
                "gates": {"max_ppl_increase": 0.05},
            },
            "write the comparison report",
        )


class OpenAICompatProvider(Provider):
    """Minimal OpenAI-compatible chat provider (no extra deps; stdlib http)."""

    name = "openai-compat"

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def decide(self, goal: str, history: list[dict], tools: list[str]) -> ProviderDecision:
        import urllib.request

        tool_list = ",".join(tools)
        system = (
            "You orchestrate model quantization. Reply with JSON only, shaped like "
            '{"tool": "<one of ' + tool_list + '>", "args": {...}, "reasoning": "..."}. '
            "Workflow: analyze, profile_hardware, plan, quantize, "
            "evaluate (twice: baseline then quantized), report. "
            f"Overall goal: {goal}."
        )
        messages = [{"role": "system", "content": system}]
        for h in history[-10:]:
            messages.append({"role": "user", "content": json.dumps(h)[:2000]})
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.load(r)
            content = payload["choices"][0]["message"]["content"]
            start, end = content.find("{"), content.rfind("}") + 1
            dec = json.loads(content[start:end])
            tool = dec.get("tool", "")
            if tool not in tools:
                raise ValueError(f"provider chose unknown tool '{tool}'")
            return ProviderDecision(tool, dec.get("args", {}), dec.get("reasoning", ""))
        except Exception as e:
            # Provider failure degrades to the deterministic policy (guardrail:
            # the run continues offline instead of dying on LLM errors).
            fallback = EchoProvider().decide(goal, history, tools)
            fallback.reasoning = f"provider error ({e}); echo fallback"
            return fallback
