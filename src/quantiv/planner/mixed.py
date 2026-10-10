"""Mixed-precision assignment from measured sensitivity.

Top-fraction most sensitive blocks keep high bits; the rest take low bits.
Executed block-by-block (true per-block mixed precision, not just per-type).
"""

from __future__ import annotations


def assign_mixed_precision(
    sensitivity: dict[str, float],
    high_bits: int = 8,
    low_bits: int = 4,
    top_fraction: float = 0.25,
) -> dict[str, int]:
    """Map block_name -> bits. Deterministic; most sensitive go high."""
    if not sensitivity:
        return {}
    names = sorted(sensitivity, key=lambda k: sensitivity[k], reverse=True)
    n_high = max(1, round(len(names) * top_fraction))
    return {n: (high_bits if i < n_high else low_bits) for i, n in enumerate(names)}
