"""AWQ backend via `llmcompressor` (vLLM project; AutoAWQ is deprecated).

Activation-aware 4-bit (W4A16). Requires CUDA. Calibration handled inside
oneshot() from the named dataset; count + seed recorded for reproducibility.
"""

from __future__ import annotations

import time
from pathlib import Path

from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult

CALIBRATION_SAMPLES = 128


class AWQQuantizer(Quantizer):
    name = "awq"

    @classmethod
    def probe(cls) -> BackendStatus:
        try:
            import llmcompressor  # noqa: F401
            import torch
            import transformers  # noqa: F401
        except ImportError as e:
            return BackendStatus(name=cls.name, available=False, reason=f"missing dependency: {e.name}")
        if not torch.cuda.is_available():
            return BackendStatus(name=cls.name, available=False, reason="requires CUDA GPU (none detected)")
        return BackendStatus(name=cls.name, available=True, version=getattr(llmcompressor, "__version__", ""))

    def prepare(self, request: QuantizeRequest) -> None:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

    def calibrate(self, request: QuantizeRequest) -> None:
        # Calibration runs inside oneshot(); counts recorded in the result message.
        _ = request

    def quantize(self, request: QuantizeRequest) -> QuantizeResult:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier

        t0 = time.time()
        self.prepare(request)
        out = Path(request.output_dir)

        recipe = AWQModifier(ignore=["lm_head"], scheme="W4A16")
        try:
            oneshot(
                model=request.model,
                recipe=recipe,
                dataset="wikitext",
                num_calibration_samples=CALIBRATION_SAMPLES,
                output_dir=str(out),
            )
        except Exception as e:
            raise RuntimeError(
                f"AWQ oneshot failed for '{request.model}': {e}. "
                "The architecture may be unsupported by llmcompressor — "
                "try --method gptq, --method hqq (generic) or --method gguf."
            ) from e
        files = sorted(p.name for p in out.iterdir())
        return QuantizeResult(
            output_dir=str(out),
            method=self.name,
            quant=f"W4A16-cal:wikitext-x{CALIBRATION_SAMPLES}",
            success=True,
            message=f"AWQ W4A16 on CUDA (calibration: wikitext x{CALIBRATION_SAMPLES})",
            files=files,
            elapsed_s=round(time.time() - t0, 2),
        )
