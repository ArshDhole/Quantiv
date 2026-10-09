"""Calibration data manager stub (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationSet:
    name: str = "default-general-text"
    seed: int = 42
    dataset_hash: str = "pending-phase-3"
