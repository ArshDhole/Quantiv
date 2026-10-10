"""Phase 4 pipeline: analyze -> quantize -> evaluate -> gate -> escalate (bounded).

Escalation ladder: requested plan -> higher bits -> next-ranked method ->
mixed precision (sensitivity-driven). Every attempt is recorded; the final
report shows the full ladder, not just the winner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from quantiv.evaluation import EvalReport, evaluate_gguf_full, evaluate_hf_model
from quantiv.planner.mixed import assign_mixed_precision
from quantiv.planner.rank import rank_candidates
from quantiv.planner.sensitivity import layer_sensitivity
from quantiv.quantizers import available_backends, run_quantization
from quantiv.quantizers.base import QuantizeRequest


@dataclass
class Attempt:
    n: int
    method: str
    quant: str
    bits: int
    mixed: bool
    ppl: float | None
    ppl_increase: float | None
    gate_pass: bool | None
    elapsed_s: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_ladder(method: str, bits: int, goal: str, max_attempts: int = 3) -> list[dict]:
    """Ordered escalation plans. Deterministic; unavailable backends skipped."""
    avail = {k: v.available for k, v in available_backends().items()}
    ranked = [c for c in rank_candidates(goal=goal, available=avail)]
    ladder: list[dict] = []
    start_method = method
    if method == "auto":
        start_method = ranked[0].method if ranked else "hqq"
    ladder.append({"method": start_method, "bits": bits, "mixed": False})
    if bits < 8:
        ladder.append({"method": start_method, "bits": 8, "mixed": False})
    for cand in ranked:
        if cand.method != start_method and avail.get(cand.method):
            ladder.append({"method": cand.method, "bits": cand.bits, "mixed": False})
            break
    if avail.get("hqq"):
        ladder.append({"method": "hqq", "bits": 4, "mixed": True})
    # Dedupe preserving order; mixed precision stays the last resort within cap.
    seen, unique, mixed_step = set(), [], None
    for step in ladder:
        key = (step["method"], step["bits"], step["mixed"])
        if key not in seen:
            seen.add(key)
            if step["mixed"]:
                mixed_step = step
            else:
                unique.append(step)
    cap = max(1, max_attempts)
    if mixed_step is not None and cap > 1:
        unique = unique[: cap - 1] + [mixed_step]
    return unique[:cap]


def _evaluate_quantized(result_method: str, quant_dir: Path, device: str, max_samples: int) -> EvalReport:
    if result_method == "gguf":
        ggufs = sorted(quant_dir.glob("*.gguf"))
        if ggufs:
            return evaluate_gguf_full(ggufs[0], max_samples=max_samples)
    return evaluate_hf_model(str(quant_dir), device=device, max_samples=max_samples)


def run_pipeline(
    model: str,
    goal: str = "balanced",
    method: str = "auto",
    bits: int = 4,
    device: str = "cpu",
    max_ppl_increase: float = 0.05,
    max_attempts: int = 3,
    max_samples: int = 16,
    group_size: int = 64,
    run_dir: str | Path = "runs/pipeline",
    on_step=None,
) -> dict:
    """Execute the full ladder. Returns dict with baseline, final, attempts, ladder."""
    from quantiv.packaging.manifest import build_manifest, write_manifest
    from quantiv.packaging.report import write_report

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ladder = build_ladder(method, bits, goal, max_attempts)
    if on_step:
        plan = ", ".join(f"{s['method']}/{s['bits']}b{' +mixed' if s['mixed'] else ''}" for s in ladder)
        on_step(f"plan: {plan} (max {len(ladder)} attempts)")
        on_step("analyzing + loading baseline model…")
    baseline = evaluate_hf_model(model, device=device, max_samples=max_samples)
    if on_step:
        on_step(f"baseline ppl={baseline.perplexity}")

    attempts: list[Attempt] = []
    final_quant: EvalReport | None = None
    final_info: dict = {}
    for i, step in enumerate(ladder, 1):
        qdir = run_dir / f"attempt{i}-{step['method']}"
        mixed_map = None
        note = ""
        if on_step:
            on_step(f"attempt {i}/{len(ladder)}: quantizing {step['method']} {step['bits']}b…")
        if step["mixed"]:
            if on_step:
                on_step("gate missed: running sensitivity for mixed precision")
            sens = layer_sensitivity(model, device=device)
            mixed_map = assign_mixed_precision(sens)
            note = f"sensitive blocks kept at 8-bit: {sum(1 for b in mixed_map.values() if b == 8)}"
        result = run_quantization(
            QuantizeRequest(
                model=model,
                method=step["method"],
                bits=step["bits"],
                group_size=group_size,
                output_dir=str(qdir),
                device=device,
                seed=42,
                mixed=mixed_map,
                quant="",  # backends resolve "" from bits (e.g. GGUF k-quants)
            ),
            goal=goal,
        )
        quant = _evaluate_quantized(result.method, qdir, device, max_samples)
        inc = None
        gate = None
        if baseline.perplexity and quant.perplexity:
            inc = round((quant.perplexity - baseline.perplexity) / baseline.perplexity, 4)
            gate = inc <= max_ppl_increase
        attempts.append(
            Attempt(
                i,
                result.method,
                result.quant,
                step["bits"],
                step["mixed"],
                quant.perplexity,
                inc,
                gate,
                result.elapsed_s,
                note,
            )
        )
        final_quant, final_info = (
            quant,
            {
                "method": result.method,
                "quant": result.quant,
                "elapsed_s": result.elapsed_s,
                "files": result.files,
                "device": device,
                "attempt_dir": str(qdir),
            },
        )
        if on_step:
            on_step(f"attempt {i}: {result.method} {result.quant} ppl={quant.perplexity} gate={gate}")
        if gate:
            break

    assert final_quant is not None
    gates = {"max_ppl_increase": max_ppl_increase}
    json_path, md_path = write_report(
        run_dir,
        baseline,
        final_quant,
        final_info,
        gates,
        attempts=[a.to_dict() for a in attempts],
    )
    manifest = build_manifest(
        model,
        {
            "run_dir": str(run_dir),
            "goal": goal,
            "device": device,
            "attempts": [a.to_dict() for a in attempts],
        },
    )
    write_manifest(run_dir, manifest)
    return {
        "run_dir": str(run_dir),
        "baseline": baseline.to_dict(),
        "quantized": final_quant.to_dict(),
        "quant_info": final_info,
        "attempts": [a.to_dict() for a in attempts],
        "report_md": str(md_path),
        "report_json": str(json_path),
    }
