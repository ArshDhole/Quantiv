"""Unified Quantizer plugin interface (Phase 2: hqq/bnb/gguf live; gptq/awq/torchao in Phase 3+)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QuantizeRequest:
    model: str
    method: str = "auto"  # auto|gguf|gptq|awq|hqq|bnb|torchao
    bits: int = 4
    group_size: int = 64
    quant: str = ""  # backend-specific label, e.g. Q4_K_M / nf4 / int8
    output_dir: str = "runs/out"
    seed: int = 42
    device: str = "auto"  # auto|cuda|cpu


@dataclass
class QuantizeResult:
    output_dir: str
    method: str
    quant: str
    success: bool
    message: str = ""
    files: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class BackendStatus:
    name: str
    available: bool
    reason: str = ""
    version: str = ""


class Quantizer(ABC):
    """All backends implement this. Import-safe: __init__ must not import heavy deps."""

    name: str = "base"

    @classmethod
    def probe(cls) -> BackendStatus:
        """Check availability without importing heavy deps at module scope."""
        return BackendStatus(name=cls.name, available=False, reason="not implemented")

    @abstractmethod
    def prepare(self, request: QuantizeRequest) -> None: ...

    @abstractmethod
    def calibrate(self, request: QuantizeRequest) -> None: ...

    @abstractmethod
    def quantize(self, request: QuantizeRequest) -> QuantizeResult: ...

    def validate_loadable(self, path: str | Path) -> bool:
        return Path(path).exists()


def resolve_device(requested: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. Never crashes if torch is missing."""
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
