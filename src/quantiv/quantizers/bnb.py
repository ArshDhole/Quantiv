"""bitsandbytes backend: 8-bit LLM.int8() and 4-bit NF4 via transformers integration."""

from __future__ import annotations

import time
from pathlib import Path

from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult


class BNBQuantizer(Quantizer):
    name = "bnb"

    @classmethod
    def probe(cls) -> BackendStatus:
        try:
            import bitsandbytes  # noqa: F401
            import torch
            import transformers  # noqa: F401
        except ImportError as e:
            return BackendStatus(name=cls.name, available=False, reason=f"missing dependency: {e.name}")
        if not torch.cuda.is_available():
            return BackendStatus(name=cls.name, available=False, reason="requires CUDA GPU (none detected)")
        return BackendStatus(name=cls.name, available=True, version=getattr(bitsandbytes, "__version__", ""))

    def prepare(self, request: QuantizeRequest) -> None:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

    def calibrate(self, request: QuantizeRequest) -> None:
        # bnb is calibration-free. Nothing to do.
        _ = request

    def quantize(self, request: QuantizeRequest) -> QuantizeResult:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        t0 = time.time()
        self.prepare(request)
        quant_label = (request.quant or "").lower()
        if request.bits == 8 or quant_label == "int8":
            qconf = BitsAndBytesConfig(load_in_8bit=True)
            label = "int8"
        else:
            qtype = "nf4" if quant_label in ("", "nf4") else "fp4"
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=qtype,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            label = qtype
        model = AutoModelForCausalLM.from_pretrained(request.model, quantization_config=qconf, device_map="auto")
        out = Path(request.output_dir)
        model.save_pretrained(str(out))
        try:
            AutoTokenizer.from_pretrained(request.model, trust_remote_code=False).save_pretrained(str(out))
        except Exception as e:  # tokenizer must not fail the run
            (out / "tokenizer_warning.txt").write_text(str(e), encoding="utf-8")
        files = sorted(p.name for p in out.iterdir())
        return QuantizeResult(
            output_dir=str(out),
            method=self.name,
            quant=label,
            success=True,
            message=f"bitsandbytes {label} loaded + saved",
            files=files,
            elapsed_s=round(time.time() - t0, 2),
        )
