"""Candidate ranking: score quantization plans by fit, goal, and availability.

Pure functions (no torch) — deterministic and unit-tested. The LLM agent in
Phase 5 reasons over these ranked candidates; the rules engine alone (default)
picks rank[0].
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    method: str
    quant: str
    bits: int
    score: float = 0.0
    fits_target: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "quant": self.quant,
            "bits": self.bits,
            "score": round(self.score, 3),
            "fits_target": self.fits_target,
            "reasons": self.reasons,
        }


# (method, quant, bits, size_factor vs fp16, quality_rank 1..5 higher=better)
_CATALOG: dict[str, list[tuple[str, str, int, float, int]]] = {
    "min-size": [
        ("gguf", "Q2_K", 2, 0.16, 1),
        ("gguf", "Q4_K_M", 4, 0.30, 3),
        ("hqq", "4bit", 4, 0.30, 3),
        ("torchao", "int4", 4, 0.30, 3),
    ],
    "balanced": [
        ("gptq", "4bit-g128", 4, 0.30, 4),
        ("awq", "W4A16", 4, 0.30, 4),
        ("gguf", "Q4_K_M", 4, 0.30, 3),
        ("hqq", "4bit", 4, 0.30, 3),
        ("bnb", "nf4", 4, 0.32, 3),
    ],
    "max-quality": [
        ("awq", "W4A16", 4, 0.30, 4),
        ("gptq", "4bit-g128", 4, 0.30, 4),
        ("bnb", "int8", 8, 0.55, 5),
        ("gguf", "Q8_0", 8, 0.55, 5),
        ("hqq", "8bit", 8, 0.55, 4),
    ],
    "min-latency": [
        ("gguf", "Q4_K_M", 4, 0.30, 3),
        ("gguf", "Q5_K_M", 5, 0.36, 4),
        ("awq", "W4A16", 4, 0.30, 4),
        ("gptq", "4bit-g128", 4, 0.30, 4),
    ],
    "cpu-efficient": [
        ("gguf", "Q4_K_M", 4, 0.30, 3),
        ("gguf", "Q5_K_M", 5, 0.36, 4),
        ("hqq", "4bit", 4, 0.30, 3),
    ],
}

_GOAL_WEIGHTS = {
    "min-size": {"size": 0.6, "quality": 0.2, "speed": 0.2},
    "balanced": {"size": 0.35, "quality": 0.4, "speed": 0.25},
    "max-quality": {"size": 0.15, "quality": 0.7, "speed": 0.15},
    "min-latency": {"size": 0.25, "quality": 0.25, "speed": 0.5},
    "cpu-efficient": {"size": 0.4, "quality": 0.3, "speed": 0.3},
}


def rank_candidates(
    goal: str = "balanced",
    params_b: float | None = None,
    target_vram_gb: float | None = None,
    available: dict[str, bool] | None = None,
) -> list[Candidate]:
    """Score and rank plans. Unavailable backends sink to the bottom (kept, flagged)."""
    goal = goal if goal in _CATALOG else "balanced"
    weights = _GOAL_WEIGHTS[goal]
    out: list[Candidate] = []
    for method, quant, bits, size_factor, quality in _CATALOG[goal]:
        reasons: list[str] = []
        size_score = max(0.0, 1.0 - size_factor)
        quality_score = quality / 5.0
        speed_score = 0.9 if method == "gguf" else (0.7 if method in ("awq", "gptq") else 0.5)
        score = weights["size"] * size_score + weights["quality"] * quality_score + weights["speed"] * speed_score
        fits = True
        if params_b and target_vram_gb:
            need_gb = params_b * size_factor + 0.5
            fits = need_gb <= target_vram_gb
            if fits:
                reasons.append(f"est. {need_gb:.1f}GB <= target {target_vram_gb}GB")
                score += 0.1
            else:
                reasons.append(f"est. {need_gb:.1f}GB > target {target_vram_gb}GB — risky")
                score -= 0.3
        is_avail = (available or {}).get(method, True)
        if not is_avail:
            reasons.append(f"backend '{method}' unavailable on this machine")
            score -= 1.0
        reasons.insert(0, f"goal={goal}: size {size_score:.2f}, quality {quality_score:.2f}, speed {speed_score:.2f}")
        out.append(Candidate(method, quant, bits, round(score, 3), fits, reasons))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
