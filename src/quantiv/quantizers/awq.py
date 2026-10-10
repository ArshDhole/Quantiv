"""AWQ backend via `gptqmodel` METHOD.AWQ (activation-aware 4-bit, W4A16-G128).

Note: `llmcompressor.oneshot` was tried first but its calibration tracer is
incompatible with the installed transformers 5.x forwards (feeds float
activations as input_ids — systemic, reproduced on SmolLM2 and Qwen2.5).
gptqmodel's AWQ path is the documented alternative and is verified working.
Requires CUDA.
"""

from __future__ import annotations

import time
from pathlib import Path

from quantiv.calibration import get_calibration_set
from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult

CALIBRATION_SAMPLES = 128


def _valid_group_size(model_id: str, requested: int) -> int:
    """Largest group size <= requested dividing all linear dims (config only)."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        inter = int(getattr(cfg, "intermediate_size", 0) or 0)
        heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        kv_heads = int(getattr(cfg, "num_key_value_heads", heads) or heads)
        head_dim = hidden // heads if hidden and heads else 0
        dims = [d for d in (hidden, inter, kv_heads * head_dim if head_dim else 0) if d > 0]
    except Exception:
        dims = []
    for gs in (requested, 128, 64, 32, 16, 8):
        if gs <= 0:
            continue
        if not dims or all(d % gs == 0 for d in dims):
            return gs
    raise ValueError(f"No valid AWQ group size for dims {dims} (requested {requested}). Try --group-size 8.")


class AWQQuantizer(Quantizer):
    name = "awq"

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
        ver = getattr(gptqmodel, "__version__", "")
        return BackendStatus(name=cls.name, available=True, version=f"gptqmodel-{ver}")

    def prepare(self, request: QuantizeRequest) -> None:
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

    def calibrate(self, request: QuantizeRequest) -> None:
        # Calibration is fetched inside quantize() so the dataset hash lands in one place.
        _ = request

    def quantize(self, request: QuantizeRequest) -> QuantizeResult:
        import os

        from gptqmodel import GPTQModel, QuantizeConfig
        from gptqmodel.quantization.config import METHOD
        from transformers import AutoTokenizer

        t0 = time.time()
        self.prepare(request)
        out = Path(request.output_dir)
        gs = _valid_group_size(request.model, request.group_size or 128)

        quant_config = QuantizeConfig(bits=4, group_size=gs, method=METHOD.AWQ)
        try:
            model = GPTQModel.load(request.model, quantize_config=quant_config)
        except Exception as e:
            raise RuntimeError(
                f"AWQ load failed for '{request.model}': {e}. "
                "The architecture may be unsupported by gptqmodel — "
                "try --method gptq, --method hqq (generic) or --method gguf."
            ) from e

        cal = get_calibration_set(num_samples=CALIBRATION_SAMPLES, seed=request.seed)
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
            quant=f"W4A16-g{gs}-cal:{cal.dataset_hash}",
            success=True,
            message=(
                f"AWQ W4A16-g{gs} on CUDA via gptqmodel (calibration: {cal.source} x{cal.num_samples}; {ckpt_note})"
            ),
            files=files,
            elapsed_s=round(time.time() - t0, 2),
        )
