"""Benchmark report writer: side-by-side baseline vs quantized (Markdown + JSON)."""

from __future__ import annotations

import json
from pathlib import Path

from quantiv.evaluation.metrics import EvalReport


def _row(label: str, base: object, quant: object) -> str:
    return f"| {label} | {base} | {quant} |"


def write_report(
    run_dir: str | Path,
    baseline: EvalReport,
    quantized: EvalReport,
    quant_info: dict,
    quality_gates: dict,
    attempts: list[dict] | None = None,
) -> tuple[Path, Path]:
    """Write report.json + report.md. Returns (json_path, md_path)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ppl_inc = None
    if baseline.perplexity and quantized.perplexity:
        ppl_inc = round((quantized.perplexity - baseline.perplexity) / baseline.perplexity, 4)
    gate_max = quality_gates.get("max_ppl_increase", 0.05)
    gate_pass = (ppl_inc <= gate_max) if ppl_inc is not None else None

    payload = {
        "baseline": baseline.to_dict(),
        "quantized": quantized.to_dict(),
        "quant": quant_info,
        "comparison": {
            "ppl_increase": ppl_inc,
            "gate_max_ppl_increase": gate_max,
            "gate_pass": gate_pass,
        },
        "attempts": attempts or [],
        "note": "Every number above was measured in this run; see text_source/device fields for method.",
    }
    json_path = run_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Quantiv benchmark report",
        "",
        f"- Baseline: `{baseline.model}` (device `{baseline.device}`, text `{baseline.text_source}`)",
        f"- Quantized: `{quantized.model}` via **{quant_info.get('method')} {quant_info.get('quant')}** "
        f"in {quant_info.get('elapsed_s', '?')}s",
        "",
        "| Metric | Baseline | Quantized |",
        "|---|---|---|",
        _row("Perplexity (lower better)", baseline.perplexity, quantized.perplexity),
        _row("PPL increase", "-", ppl_inc),
        _row(
            f"Quality gate (max +{gate_max})", "-", "PASS" if gate_pass else ("FAIL" if gate_pass is False else "n/a")
        ),
        _row("Tokens/sec (decode)", baseline.tokens_per_sec, quantized.tokens_per_sec),
        _row("Time to first token (s)", baseline.time_to_first_token_s, quantized.time_to_first_token_s),
        _row("Peak memory (GB)", baseline.peak_memory_gb, quantized.peak_memory_gb),
        _row("Disk size (GB)", baseline.disk_size_gb, quantized.disk_size_gb),
        _row("Sanity", baseline.sanity, quantized.sanity),
        "",
        "## Warnings",
        "",
    ]
    for w in baseline.warnings:
        md.append(f"- baseline: {w}")
    for w in quantized.warnings:
        md.append(f"- quantized: {w}")
    if not baseline.warnings and not quantized.warnings:
        md.append("- none")
    if attempts:
        md += ["", "## Attempts (escalation ladder)", ""]
        md.append("| # | Method | Quant | PPL | Gate | Note |")
        md.append("|---|---|---|---|---|---|")
        for a in attempts:
            gate = "PASS" if a.get("gate_pass") else ("FAIL" if a.get("gate_pass") is False else "n/a")
            md.append(
                f"| {a.get('n')} | {a.get('method')} | {a.get('quant')} | "
                f"{a.get('ppl')} | {gate} | {a.get('note', '')} |"
            )
    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path
