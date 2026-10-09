"""Unified Quantizer plugin interface (implemented in Phase 2+)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuantizeRequest:
    model: str
    method: str = "auto"
    bits: int = 4
    group_size: int = 128
    output_dir: str = "runs/out"
    seed: int = 42


@dataclass
class QuantizeResult:
    output_dir: str
    method: str
    success: bool
    message: str = ""


class Quantizer(ABC):
    """All backends (gguf/gptq/awq/hqq/bnb/torchao) implement this."""

    name: str = "base"

    @abstractmethod
    def prepare(self, request: QuantizeRequest) -> None: ...

    @abstractmethod
    def calibrate(self, request: QuantizeRequest) -> None: ...

    @abstractmethod
    def quantize(self, request: QuantizeRequest) -> QuantizeResult: ...

    @abstractmethod
    def save(self, result: QuantizeResult, path: str | Path) -> Path: ...

    def validate_loadable(self, path: str | Path) -> bool:
        return Path(path).exists()
