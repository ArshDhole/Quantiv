"""Named target-device profiles (may differ from build machine)."""

from __future__ import annotations

from pathlib import Path

from quantiv.utils.config import load_yaml

_DEVICES_CACHE: dict | None = None


def load_device_profiles(config_path: str | Path | None = None) -> dict:
    global _DEVICES_CACHE
    if config_path is None:
        here = Path(__file__).resolve().parents[3] / "configs" / "devices.yaml"
        config_path = here
    data = load_yaml(config_path)
    _DEVICES_CACHE = data.get("targets", {})
    return _DEVICES_CACHE


def get_target_profile(name: str) -> dict:
    profiles = load_device_profiles()
    if name in profiles:
        return {"name": name, **profiles[name]}
    if name == "local":
        return {"name": "local", "description": "Build machine itself"}
    raise KeyError(f"Unknown target '{name}'. Available: {sorted(profiles)} + ['local']")


def list_targets() -> list[str]:
    return sorted(load_device_profiles()) + ["local"]
