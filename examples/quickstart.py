"""Quantiv quickstart: analyze + hardware + ranked plan. CLI does the rest.

Full end-to-end:
    quantiv run <model> --goal balanced          # rules pipeline with retry ladder
    quantiv run <model> --agent                  # autonomous agent loop
    quantiv dashboard                            # web UI
"""
from quantiv.analyzer import analyze_model
from quantiv.hardware import get_target_profile, profile_hardware
from quantiv.planner import rank_candidates

hw = profile_hardware()
print("BUILD MACHINE:", hw.to_dict())
print("TARGET rtx3060-12gb:", get_target_profile("rtx3060-12gb"))

# Local dir example: python examples/quickstart.py ./tests-fixture
import sys
model = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    profile = analyze_model(model)
    print("MODEL:", profile.to_dict())
    params_b = (profile.param_count / 1e9) if profile.param_count else None
    for c in rank_candidates(goal="balanced", params_b=params_b, target_vram_gb=12):
        print("PLAN:", c.to_dict())
except Exception as e:
    print(f"analyze failed (offline?): {e}")
