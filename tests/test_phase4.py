"""Phase 4 tests: ranking, ladder, mixed assignment, sensitivity, pipeline."""

from quantiv.planner import (
    assign_mixed_precision,
    build_ladder,
    layer_sensitivity,
    rank_candidates,
    run_pipeline,
)


def test_rank_deterministic_and_scored():
    a = rank_candidates(goal="balanced", params_b=0.5, target_vram_gb=8)
    b = rank_candidates(goal="balanced", params_b=0.5, target_vram_gb=8)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]
    scores = [c.score for c in a]
    assert scores == sorted(scores, reverse=True)
    assert all(c.fits_target for c in a)  # 0.5B fits 8GB at any listed size


def test_rank_flags_misfit_and_unavailable():
    cands = rank_candidates(
        goal="min-size",
        params_b=70,
        target_vram_gb=8,
        available={"gguf": False, "hqq": True, "bnb": False, "gptq": False, "awq": False},
    )
    assert any(not c.fits_target for c in cands)  # 70B cannot fit 8GB
    assert cands[-1].method == "gguf"  # unavailable sinks (kept, flagged)
    assert any("unavailable" in r for r in cands[-1].reasons)


def test_build_ladder_shapes():
    ladder = build_ladder("hqq", 4, "balanced", max_attempts=3)
    assert ladder[0] == {"method": "hqq", "bits": 4, "mixed": False}
    assert len(ladder) <= 3
    assert ladder[-1]["mixed"] is True  # mixed precision is the last resort
    keys = [(s["method"], s["bits"], s["mixed"]) for s in ladder]
    assert len(keys) == len(set(keys))  # deduped


def test_assign_mixed_precision():
    sens = {f"model.layers.{i}": float(i) for i in range(8)}
    mixed = assign_mixed_precision(sens, high_bits=8, low_bits=4, top_fraction=0.25)
    assert sum(1 for b in mixed.values() if b == 8) == 2
    assert mixed["model.layers.7"] == 8  # most sensitive goes high
    assert mixed["model.layers.0"] == 4
    assert assign_mixed_precision({}) == {}


def test_sensitivity_smoke(tiny_hf_model):
    sens = layer_sensitivity(str(tiny_hf_model), device="cpu", num_texts=2, seq_len=64)
    assert len(sens) == 2  # toy has 2 blocks
    assert all(isinstance(v, float) for v in sens.values())


def test_pipeline_single_attempt_pass(tiny_hf_model, tmp_path):
    """Loose gate -> attempt 1 passes, attempts recorded in report."""
    out = run_pipeline(
        str(tiny_hf_model),
        goal="balanced",
        method="hqq",
        bits=4,
        device="cpu",
        max_ppl_increase=50.0,
        max_attempts=1,
        max_samples=2,
        group_size=8,
        run_dir=tmp_path / "pipe",
    )
    assert len(out["attempts"]) == 1
    assert out["attempts"][0]["gate_pass"] is True
    assert (tmp_path / "pipe" / "report.md").exists()
    assert (tmp_path / "pipe" / "report.json").exists()


def test_pipeline_emits_lifecycle_events(tiny_hf_model, tmp_path):
    """The dashboard polls these — first event must arrive before slow work."""
    from quantiv.planner import run_pipeline

    seen: list[str] = []
    run_pipeline(
        str(tiny_hf_model),
        goal="balanced",
        method="hqq",
        bits=4,
        device="cpu",
        max_ppl_increase=50.0,
        max_attempts=1,
        max_samples=2,
        group_size=8,
        run_dir=tmp_path / "pipe2",
        on_step=seen.append,
    )
    assert seen and seen[0].startswith("plan:")
    assert any("attempt 1/1" in m for m in seen)
