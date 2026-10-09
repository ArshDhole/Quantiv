"""GGUF backend: HF -> GGUF-F16 (pinned convert script) -> k-quant (llama.cpp bindings).

No compiler needed at runtime: conversion uses the pure-Python converter from the
pinned llama.cpp tag, quantization uses the already-built `llama_cpp` package.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir

from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult

LLAMACPP_TAG = "v0.6.0"  # verified 2026-10-09: latest release; converter at repo root + conversion/ pkg

SUPPORTED_QUANTS = (
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_0",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16",
    "F32",
)


class GGUQuantizer(Quantizer):
    name = "gguf"

    @classmethod
    def probe(cls) -> BackendStatus:
        try:
            import gguf  # noqa: F401
            import llama_cpp  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            return BackendStatus(name=cls.name, available=False, reason=f"missing dependency: {e.name}")
        return BackendStatus(name=cls.name, available=True, version=LLAMACPP_TAG)

    def prepare(self, request: QuantizeRequest) -> None:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

    def calibrate(self, request: QuantizeRequest) -> None:
        # Plain k-quants need no calibration. (imatrix path is a Phase 4+ extension.)
        _ = request

    def _convert_toolkit(self) -> Path:
        """Fetch the pinned converter (script + conversion/ package) into the cache.

        Since llama.cpp refactored the converter into a package, the lone script
        is not enough. File list comes from the pinned tag's git tree (deterministic).
        """
        import json

        cache = Path(user_cache_dir("quantiv")) / "llamacpp" / LLAMACPP_TAG
        cache.mkdir(parents=True, exist_ok=True)
        script = cache / "convert_hf_to_gguf.py"
        pkg = cache / "conversion"
        gguf_pkg = cache / "gguf-py" / "gguf" / "__init__.py"
        if script.exists() and gguf_pkg.exists() and pkg.is_dir() and any(pkg.iterdir()):
            return script
        tree_url = f"https://api.github.com/repos/ggml-org/llama.cpp/git/trees/{LLAMACPP_TAG}?recursive=1"
        with urllib.request.urlopen(tree_url, timeout=120) as r:
            tree = json.load(r)["tree"]
        wanted = [
            "convert_hf_to_gguf.py",
            *(
                x["path"]
                for x in tree
                if (x["path"].startswith("conversion/") or x["path"].startswith("gguf-py/gguf/"))
                and x["type"] == "blob"
                and x["path"].endswith(".py")
            ),
        ]
        for rel in wanted:
            dest = cache / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMACPP_TAG}/{rel}"
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
                f.write(r.read())
        return script

    def quantize(self, request: QuantizeRequest) -> QuantizeResult:
        import llama_cpp

        t0 = time.time()
        self.prepare(request)
        quant = (request.quant or "Q4_K_M").upper()
        if quant not in SUPPORTED_QUANTS:
            raise ValueError(f"Unsupported GGUF quant '{quant}'. Supported: {SUPPORTED_QUANTS}")
        out = Path(request.output_dir)
        f16 = out / "model-f16.gguf"
        final = out / f"model-{quant}.gguf"

        # 0. The converter only accepts local dirs: snapshot HF ids first.
        src = request.model
        if not Path(src).exists():
            from huggingface_hub import snapshot_download

            src = snapshot_download(src)

        # 1. HF -> GGUF F16 (repo-pinned gguf-py first on sys.path)
        import os

        env = dict(os.environ)
        toolkit = Path(user_cache_dir("quantiv")) / "llamacpp" / LLAMACPP_TAG
        env["PYTHONPATH"] = str(toolkit / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, str(self._convert_toolkit()), src, "--outfile", str(f16), "--outtype", "f16"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        if proc.returncode != 0 or not f16.exists():
            raise RuntimeError(f"GGUF conversion failed:\n{proc.stderr[-3000:]}")

        # 2. F16 -> k-quant (or keep F16)
        if quant in ("F16", "F32"):
            f16.rename(final)
        else:
            import os

            params = llama_cpp.llama_model_quantize_default_params()
            params.nthread = max(1, (os.cpu_count() or 8) - 2)
            params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{quant}")
            params.allow_requantize = False
            ret = llama_cpp.llama_model_quantize(str(f16).encode(), str(final).encode(), params)
            if ret != 0 or not final.exists():
                raise RuntimeError(f"llama_model_quantize failed with code {ret}")
            f16.unlink(missing_ok=True)

        files = sorted(p.name for p in out.iterdir())
        return QuantizeResult(
            output_dir=str(out),
            method=self.name,
            quant=quant,
            success=True,
            message=f"GGUF {quant} via llama.cpp {LLAMACPP_TAG}",
            files=files,
            elapsed_s=round(time.time() - t0, 2),
        )
