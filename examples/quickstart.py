"""Phase 1 quickstart (analyzer + hardware + planner only — no torch needed)."""
from quantiv.analyzer import analyze_model
from quantiv.hardware import get_target_profile, profile_hardware
from quantiv.planner import recommend_plans

hw = profile_hardware()
print("BUILD MACHINE:", hw.to_dict())
print("TARGET rtx3060-12gb:", get_target_profile("rtx3060-12gb"))

# Local dir example: python examples/quickstart.py ./tests-fixture
import sys
model = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    profile = analyze_model(model)
    print("MODEL:", profile.to_dict())
    print("PLANS:", recommend_plans(
        (profile.param_count / 1e9) if profile.param_count else None, 12, "balanced"))
except Exception as e:
    print(f"analyze failed (offline?): {e}")
