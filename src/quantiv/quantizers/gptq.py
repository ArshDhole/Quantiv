"""GPTQ backend via the maintained `gptqmodel` fork (AutoGPTQ is deprecated).

Calibration-based 4-bit (also 3/8-bit). Requires CUDA. Supports resumable
checkpoints for long jobs (gptqmodel>=7.4 native support).
"""

from __future__ import annotations

import time
from pathlib import Path

from quantiv.calibration import get_calibration_set
from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult


class GPTQQuantizer(Quantizer):
    name = "gptq"

    @classmethod
    def probe(cls) -> BackendStatus:
        try:
            import gptqmodel  # noqa: F401
            import torch
            import transformers  # noqa: F401
        except ImportError as e:
            return BackendStatus(name=cls.name, available=False, reason=f"missing dependency: {e.name}")
        if not torch.cuda.is_available():
            return BackendStatus(name=cls.name, available=False, reason="requires CUDA GPU (none detected)")
        return BackendStatus(name=cls.name, available=True, version=getattr(gptqmodel, "__version__", ""))

    def prepare(self, request: QuantizeRequest) -> None:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

    def calibrate(self, request: QuantizeRequest) -> None:
        # Calibration is fetched inside quantize() so the dataset hash lands in one place.
        _ = request

    def quantize(self, request: QuantizeRequest) -> QuantizeResult:
        import os

        from gptqmodel import GPTQModel, QuantizeConfig
        from transformers import AutoTokenizer

        t0 = time.time()
        self.prepare(request)
        bits = request.bits or 4
        out = Path(request.output_dir)

        quant_config = QuantizeConfig(bits=bits, group_size=request.group_size or 128)
        try:
            model = GPTQModel.load(request.model, quantize_config=quant_config)
        except Exception as e:
            raise RuntimeError(
                f"GPTQ load failed for '{request.model}': {e}. "
                "The architecture may be unsupported by gptqmodel — "
                "try --method hqq (generic) or --method gguf."
            ) from e

        cal = get_calibration_set(num_samples=128, seed=request.seed)
        # Resumable checkpoints: gptqmodel's directory-fsync is POSIX-only
        # (os.O_DIRECTORY missing on Windows) — enable where supported.
        checkpoint = None
        ckpt_note = "checkpoints: off (Windows)"
        if os.name != "nt":
            from gptqmodel.looper.checkpoint_store import CheckpointConfig

            checkpoint = CheckpointConfig(path=str(out / "checkpoints"), resume="auto")
            ckpt_note = "checkpoints: on"
        model.quantize(list(cal.texts), batch_size=1, checkpoint=checkpoint)
        model.save_quantized(str(out))
        try:
            AutoTokenizer.from_pretrained(request.model, trust_remote_code=False).save_pretrained(str(out))
        except Exception as e:  # tokenizer must not fail the run
            (out / "tokenizer_warning.txt").write_text(str(e), encoding="utf-8")
        files = sorted(p.name for p in out.iterdir())
        return QuantizeResult(
            output_dir=str(out),
            method=self.name,
            quant=f"{bits}bit-g{request.group_size or 128}-cal:{cal.dataset_hash}",
            success=True,
            message=f"GPTQ {bits}-bit on CUDA (calibration: {cal.source} x{cal.num_samples}; {ckpt_note})",
            files=files,
            elapsed_s=round(time.time() - t0, 2),
        )
