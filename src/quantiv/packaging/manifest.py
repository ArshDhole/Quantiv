"""Manifest builder stub (Phase 6)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from quantiv import __version__


def build_manifest(model: str, extra: dict | None = None) -> dict:
    return {
        "quantiv_version": __version__,
        "model": model,
        "created_utc": datetime.now(UTC).isoformat(),
        "tool_versions": {"python": None},
        "extra": extra or {},
    }


def manifest_hash(manifest: dict) -> str:
    blob = json.dumps(manifest, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_manifest(run_dir: str | Path, manifest: dict) -> Path:
    p = Path(run_dir) / "quantiv_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p
