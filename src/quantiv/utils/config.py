"""Pydantic + YAML config system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Goal = Literal["min-size", "balanced", "max-quality", "min-latency", "cpu-efficient"]
Method = Literal["auto", "gguf", "gptq", "awq", "hqq", "bnb", "torchao"]


class QualityGates(BaseModel):
    max_ppl_increase: float = Field(default=0.05, ge=0, le=10)
    max_quality_drop: float = Field(default=0.02, ge=0, le=1)
    max_attempts: int = Field(default=3, ge=1, le=10)


class TargetDevice(BaseModel):
    name: str = "local"
    vram_gb: float | None = None
    ram_gb: float | None = None
    kind: Literal["cuda", "cpu", "mps", "rocm", "auto"] = "auto"


class RunConfig(BaseModel):
    model: str
    target: str = "local"
    goal: Goal = "balanced"
    method: Method = "auto"
    max_ppl_increase: float = 0.05
    no_agent: bool = True
    seed: int = 42
    output_dir: str = "runs"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_default_config() -> dict:
    here = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    if here.exists():
        return load_yaml(here)
    return {}
