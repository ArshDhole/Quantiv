"""Reproducibility: same inputs -> same hashes/configs (within tolerance)."""

from quantiv.analyzer import analyze_model
from quantiv.calibration import get_calibration_set
from quantiv.packaging.manifest import build_manifest, manifest_hash


def test_calibration_reproducible():
    a = get_calibration_set(num_samples=16, seed=123)
    b = get_calibration_set(num_samples=16, seed=123)
    assert a.dataset_hash == b.dataset_hash
    assert a.texts == b.texts


def test_analyzer_deterministic(tiny_model_dir):
    p1 = analyze_model(str(tiny_model_dir)).to_dict()
    p2 = analyze_model(str(tiny_model_dir)).to_dict()
    assert p1 == p2


def test_manifest_records_versions():
    m = build_manifest("org/model", {"seed": 42})
    assert m["quantiv_version"]
    assert m["created_utc"]
    h1, h2 = manifest_hash(m), manifest_hash(dict(m))
    assert h1 == h2  # hashing is stable for identical manifests
