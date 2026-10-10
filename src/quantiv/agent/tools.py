"""Typed agent tools: the ONLY actions the orchestrator may take.

Each tool is a validated Python callable over the deterministic engine —
no shell, no network beyond model/dataset fetch, no invented numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    name: str
    description: str
    args: dict  # name -> description (validated at call time)


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)
    error: str = ""


def _require(args: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in args]
    if missing:
        raise ValueError(f"missing required args: {missing}")


def tool_analyze(args: dict) -> ToolResult:
    _require(args, "model")
    from quantiv.analyzer import analyze_model

    return ToolResult(ok=True, data=analyze_model(args["model"]).to_dict())


def tool_profile_hardware(args: dict) -> ToolResult:
    from quantiv.hardware import get_target_profile, profile_hardware

    hw = profile_hardware().to_dict()
    target = args.get("target", "local")
    try:
        tgt = get_target_profile(target)
    except KeyError as e:
        return ToolResult(ok=False, error=str(e))
    return ToolResult(ok=True, data={"hardware": hw, "target": tgt})


def tool_plan(args: dict) -> ToolResult:
    _require(args, "goal")
    from quantiv.planner.rank import rank_candidates
    from quantiv.quantizers import available_backends

    avail = {k: v.available for k, v in available_backends().items()}
    cands = rank_candidates(
        goal=args.get("goal", "balanced"),
        params_b=args.get("params_b"),
        target_vram_gb=args.get("target_vram_gb"),
        available=avail,
    )
    return ToolResult(ok=True, data={"candidates": [c.to_dict() for c in cands]})


def tool_quantize(args: dict) -> ToolResult:
    _require(args, "model", "method", "output_dir")
    from quantiv.quantizers import run_quantization
    from quantiv.quantizers.base import QuantizeRequest

    result = run_quantization(
        QuantizeRequest(
            model=args["model"],
            method=args["method"],
            bits=int(args.get("bits", 4)),
            group_size=int(args.get("group_size", 64)),
            output_dir=args["output_dir"],
            device=args.get("device", "auto"),
            seed=int(args.get("seed", 42)),
            mixed=args.get("mixed"),
            quant=args.get("quant", ""),
        ),
        goal=args.get("goal", "balanced"),
    )
    return ToolResult(
        ok=True,
        data={
            "output_dir": result.output_dir,
            "method": result.method,
            "quant": result.quant,
            "files": result.files,
            "elapsed_s": result.elapsed_s,
        },
    )


def tool_evaluate(args: dict) -> ToolResult:
    _require(args, "model_ref")
    from quantiv.evaluation import evaluate_gguf_full, evaluate_hf_model

    ref = args["model_ref"]
    if ref.endswith(".gguf"):
        rep = evaluate_gguf_full(ref, max_samples=int(args.get("max_samples", 16)))
    else:
        rep = evaluate_hf_model(ref, device=args.get("device", "cpu"), max_samples=int(args.get("max_samples", 16)))
    return ToolResult(ok=True, data=rep.to_dict())


def tool_report(args: dict) -> ToolResult:
    _require(args, "run_dir", "baseline", "quantized", "quant_info")
    from quantiv.evaluation.metrics import EvalReport
    from quantiv.packaging.report import write_report

    base = EvalReport(**args["baseline"])
    quant = EvalReport(**args["quantized"])
    jp, md = write_report(args["run_dir"], base, quant, args["quant_info"], args.get("gates", {}), args.get("attempts"))
    return ToolResult(ok=True, data={"report_json": str(jp), "report_md": str(md)})


TOOLS: dict[str, tuple[ToolSpec, object]] = {
    "analyze": (
        ToolSpec("analyze", "Profile a model (arch, size, license, memory).", {"model": "HF id or path"}),
        tool_analyze,
    ),
    "profile_hardware": (
        ToolSpec("profile_hardware", "Profile build machine + target.", {"target": "device profile name"}),
        tool_profile_hardware,
    ),
    "plan": (
        ToolSpec(
            "plan",
            "Rank quantization candidates.",
            {
                "goal": "min-size|balanced|max-quality|min-latency|cpu-efficient",
                "params_b": "billions of params (optional)",
                "target_vram_gb": "target VRAM (optional)",
            },
        ),
        tool_plan,
    ),
    "quantize": (
        ToolSpec(
            "quantize",
            "Run a quantization backend.",
            {
                "model": "HF id or path",
                "method": "backend",
                "output_dir": "dir",
                "bits": "4",
                "group_size": "64",
                "device": "auto",
                "seed": "42",
            },
        ),
        tool_quantize,
    ),
    "evaluate": (
        ToolSpec(
            "evaluate",
            "Measure ppl/speed/memory of a model.",
            {"model_ref": "HF id or path", "device": "cpu", "max_samples": "16"},
        ),
        tool_evaluate,
    ),
    "report": (
        ToolSpec(
            "report",
            "Write comparison report + manifest data.",
            {"run_dir": "dir", "baseline": "eval dict", "quantized": "eval dict", "quant_info": "dict"},
        ),
        tool_report,
    ),
}


def call_tool(name: str, args: dict) -> ToolResult:
    """Dispatch with schema validation. Unknown tools are refused (guardrail)."""
    if name not in TOOLS:
        return ToolResult(ok=False, error=f"unknown tool '{name}'. Allowed: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        return ToolResult(ok=False, error="args must be a dict")
    try:
        return TOOLS[name][1](args)
    except Exception as e:  # tools fail loudly with context, never silently
        return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
