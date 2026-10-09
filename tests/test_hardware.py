"""Hardware profiler tests (must pass on CPU-only CI too)."""

from quantiv.hardware import get_target_profile, list_targets, profile_hardware


def test_profile_never_crashes():
    hw = profile_hardware()
    assert hw.ram_gb > 0
    assert hw.cpu_cores_logical >= 1
    d = hw.to_dict()
    assert "gpu_name" in d


def test_targets_include_expected():
    targets = list_targets()
    for t in ("rtx3060-12gb", "cpu-only", "local"):
        assert t in targets


def test_get_target_profile():
    p = get_target_profile("rtx3060-12gb")
    assert p["vram_gb"] == 12
