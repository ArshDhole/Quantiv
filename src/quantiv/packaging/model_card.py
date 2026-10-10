"""Model card generator: honest, measurement-backed cards for quantized artifacts."""

from __future__ import annotations

from pathlib import Path


def build_model_card(
    model: str,
    quant_info: dict,
    comparison: dict,
    source_license: str | None,
    manifest_hash: str = "",
) -> str:
    """Render MODEL_CARD.md. Reports measured numbers; never invents any."""
    ppl = comparison.get("ppl_increase")
    gate = comparison.get("gate_pass")
    lines = [
        f"# {model} — quantized ({quant_info.get('method')} {quant_info.get('quant', '')})",
        "",
        "Quantized with [Quantiv](https://github.com/ArshDhole/Quantiv). "
        "All metrics below were measured; see `report.json` for method details.",
        "",
        "## Quantization",
        "",
        f"- Method: `{quant_info.get('method')}` {quant_info.get('quant', '')}",
        f"- Device: `{quant_info.get('device', 'unknown')}`",
        f"- Manifest: `{manifest_hash}`",
        "",
        "## Measured results",
        "",
        f"- Perplexity increase vs baseline: **{ppl}**",
        f"- Quality gate: **{'PASS' if gate else ('FAIL' if gate is False else 'n/a')}**",
        "",
        "## Source license",
        "",
        f"This artifact derives from `{model}`, licensed `{source_license or 'unknown'}`. "
        "The source license travels with this artifact — check it before redistributing.",
        "",
        "## Limitations",
        "",
        "- Quality numbers are specific to the reported eval texts and device.",
        "- Per-GGUF perplexity uses the GGUF-embedded tokenizer (approximate delta).",
        "",
    ]
    return "\n".join(lines)


def write_model_card(run_dir: str | Path, card: str) -> Path:
    p = Path(run_dir) / "MODEL_CARD.md"
    p.write_text(card, encoding="utf-8")
    return p
