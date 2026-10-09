"""HQQ backend: calibration-free 4-bit (also 8/3/2-bit) via the `hqq` package."""

from __future__ import annotations

import time
from pathlib import Path

from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult, resolve_device


class HQQQuantizer(Quantizer):
    name = "hqq"

    @classmethod
    def probe(cls) -> BackendStatus:
        try:
            import hqq  # noqa: F401
            import torch  # noqa: F401

            return BackendStatus(name=cls.name, available=True, version=getattr(hqq, "__version__", ""))
        except ImportError as e:
            return BackendStatus(name=cls.name, available=False, reason=f"missing dependency: {e.name}")

    def prepare(self, request: QuantizeRequest) -> None:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

    def calibrate(self, request: QuantizeRequest) -> None:
        # HQQ is calibration-free (uses weight statistics only). Nothing to do.
        _ = request

    def quantize(self, request: QuantizeRequest) -> QuantizeResult:
        import torch
        from hqq.core.quantize import BaseQuantizeConfig
        from hqq.engine.hf import HQQModelForCausalLM
        from transformers import AutoTokenizer

        t0 = time.time()
        self.prepare(request)
        device = resolve_device(request.device)
        bits = request.bits or 4
        quant_config = BaseQuantizeConfig(
            nbits=bits,
            group_size=request.group_size or 64,
            quant_zero=True,
            quant_scale=False,
        )
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = HQQModelForCausalLM.from_pretrained(request.model, quant_config=quant_config, torch_dtype=torch_dtype)
        if device == "cpu":
            model.to("cpu")
        out = Path(request.output_dir)
        model.save_quantized_(str(out))
        try:
            tok = AutoTokenizer.from_pretrained(request.model, trust_remote_code=False)
            tok.save_pretrained(str(out))
        except Exception as e:  # tokenizer must not fail the run
            (out / "tokenizer_warning.txt").write_text(str(e), encoding="utf-8")
        files = sorted(p.name for p in out.iterdir())
        return QuantizeResult(
            output_dir=str(out),
            method=self.name,
            quant=f"{bits}bit-g{request.group_size or 64}",
            success=True,
            message=f"HQQ {bits}-bit quantized on {device}",
            files=files,
            elapsed_s=round(time.time() - t0, 2),
        )
