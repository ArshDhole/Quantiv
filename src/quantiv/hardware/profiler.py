"""Hardware profiler: detect build-machine resources + named target profiles."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareProfile:
    os: str
    cpu: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_features: list[str]
    ram_gb: float
    disk_free_gb: float
    gpu_name: str | None = None
    gpu_vendor: str | None = None
    gpu_vram_gb: float | None = None
    gpu_compute_capability: str | None = None
    cuda_version: str | None = None
    torch_cuda_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_features() -> list[str]:
    feats: list[str] = []
    try:
        import cpuinfo  # type: ignore # optional

        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        for cand in ("avx2", "avx512f", "avx512_vnni", "neon", "fma", "bmi2"):
            if cand in flags:
                feats.append(cand)
    except Exception:
        pass
    # Fallback on Windows x86_64: assume SSE4.2 at minimum; try AVX2 via wmic is unreliable.
    arch = platform.machine().lower()
    if arch in ("amd64", "x86_64") and not feats:
        feats.append("x86_64 (feature detail unavailable without py-cpuinfo)")
    return feats


def _nvidia_smi() -> dict | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        return {
            "name": parts[0],
            "vram_mb": float(parts[1]),
            "compute_cap": parts[2],
            "driver": parts[3] if len(parts) > 3 else None,
        }
    except Exception:
        return None


def profile_hardware() -> HardwareProfile:
    """Detect current machine's CPU/GPU/RAM/disk. Never crashes — degrades gracefully."""
    info = _nvidia_smi()
    torch_cuda = False
    cuda_ver: str | None = None
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        if torch_cuda:
            try:
                cuda_ver = torch.version.cuda
            except Exception:
                cuda_ver = None
    except ImportError:
        pass

    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(".")
    uname = platform.uname()
    return HardwareProfile(
        os=f"{uname.system} {uname.release} ({uname.machine})",
        cpu=uname.processor or uname.machine or "unknown",
        cpu_cores_physical=psutil.cpu_count(logical=False) or 0,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 0,
        cpu_features=_cpu_features(),
        ram_gb=round(vm.total / (1024**3), 2),
        disk_free_gb=round(disk.free / (1024**3), 2),
        gpu_name=info["name"] if info else None,
        gpu_vendor="NVIDIA" if info else None,
        gpu_vram_gb=round(info["vram_mb"] / 1024, 2) if info else None,
        gpu_compute_capability=info["compute_cap"] if info else None,
        cuda_version=cuda_ver or (info["driver"] if info else None),
        torch_cuda_available=torch_cuda,
    )
