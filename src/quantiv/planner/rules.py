"""Rules-engine planner stub (full ranking lands in Phase 4).

Phase 1: deterministic, dependency-free recommendations so `quantiv plan`
works without torch/LLMs.
"""

from __future__ import annotations


def recommend_plans(model_params_b: float | None, vram_gb: float | None, goal: str) -> list[dict]:
    """Return ranked candidate plans. Pure function — easy to test."""
    goal = (goal or "balanced").lower()
    if goal == "min-size":
        cands = [
            {
                "method": "gguf",
                "quant": "Q2_K",
                "bits": 2,
                "reason": "smallest CPU/offline artifact",
            },
            {
                "method": "gguf",
                "quant": "Q4_K_M",
                "bits": 4,
                "reason": "fallback if Q2_K fails quality gate",
            },
            {"method": "torchao", "quant": "int4", "bits": 4, "reason": "PyTorch-native fallback"},
        ]
    elif goal == "max-quality":
        cands = [
            {"method": "bnb", "quant": "int8", "bits": 8, "reason": "highest fidelity, larger"},
            {"method": "gguf", "quant": "Q8_0", "bits": 8, "reason": "near-lossless CPU path"},
            {"method": "hqq", "quant": "4bit", "bits": 4, "reason": "calibration-free fallback"},
        ]
    elif goal in ("min-latency", "cpu-efficient"):
        cands = [
            {
                "method": "gguf",
                "quant": "Q4_K_M",
                "bits": 4,
                "reason": "best CPU latency/size tradeoff",
            },
            {
                "method": "gguf",
                "quant": "Q5_K_M",
                "bits": 5,
                "reason": "slightly larger, faster convergence",
            },
            {"method": "torchao", "quant": "int8", "bits": 8, "reason": "torch.compile-friendly"},
        ]
    else:  # balanced
        cands = [
            {"method": "gguf", "quant": "Q4_K_M", "bits": 4, "reason": "default balanced choice"},
            {
                "method": "hqq",
                "quant": "4bit",
                "bits": 4,
                "reason": "calibration-free, GPU-friendly",
            },
            {"method": "bnb", "quant": "nf4", "bits": 4, "reason": "fast baseline"},
        ]
    # Tiny-VRAM nudge: prefer GGUF first (already first in most lists).
    if vram_gb is not None and vram_gb <= 8:
        pass
    _ = model_params_b  # used in Phase 4 for fine ranking
    return cands
