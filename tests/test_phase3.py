"""Phase 3 tests: calibration determinism + GPTQ/AWQ backends (GPU when available)."""

import pytest
import torch

from quantiv.calibration import CalibrationSet, get_calibration_set
from quantiv.quantizers import available_backends

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def test_calibration_fallback_deterministic():
    # Offline fallback path: same inputs -> same hash; different size -> different hash.
    a = get_calibration_set(name="nonexistent-source", num_samples=16, seed=7)
    b = get_calibration_set(name="nonexistent-source", num_samples=16, seed=7)
    c = get_calibration_set(name="nonexistent-source", num_samples=8, seed=7)
    assert a.dataset_hash and a.dataset_hash == b.dataset_hash
    assert a.dataset_hash != c.dataset_hash
    assert a.num_samples == 16
    assert "builtin-fallback" in a.source
    assert isinstance(CalibrationSet().to_dict()["dataset_hash"], str)


def test_calibration_wikitext_cached():
    cal = get_calibration_set(num_samples=8, seed=42)
    assert cal.num_samples == 8
    assert cal.dataset_hash
    assert len(cal.texts) == 8


@needs_cuda
def test_gptq_probe_available():
    assert available_backends()["gptq"].available


@needs_cuda
def test_awq_probe_available():
    assert available_backends()["awq"].available


@needs_cuda
def test_gptq_e2e_tiny(tiny_hf_model, tmp_path):
    """GPTQ on toy weights (CUDA): quantize with calibration, reload, measure."""
    from quantiv.evaluation import evaluate_hf_model
    from quantiv.quantizers import run_quantization
    from quantiv.quantizers.base import QuantizeRequest

    out = tmp_path / "gptq"
    result = run_quantization(
        QuantizeRequest(
            model=str(tiny_hf_model),
            method="gptq",
            bits=4,
            group_size=32,
            output_dir=str(out),
            device="cuda",
            seed=42,
        )
    )
    assert result.success
    assert any(p.suffix == ".safetensors" for p in out.iterdir())
    rep = evaluate_hf_model(str(out), device="cuda", max_samples=2)
    assert rep.perplexity is not None and rep.perplexity > 0
