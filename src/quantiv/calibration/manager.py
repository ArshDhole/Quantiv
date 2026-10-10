"""Calibration data manager: reproducible held-in text sets for GPTQ/AWQ.

Every set is content-hashed and cached, so runs are reproducible and the
manifest can record exactly what the quantization observed.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir

from quantiv.evaluation.metrics import FALLBACK_TEXTS

WIKITEXT_SPECS = (
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext", "wikitext-2-raw-v1"),
)


@dataclass
class CalibrationSet:
    name: str = "wikitext"
    seed: int = 42
    num_samples: int = 128
    seq_len: int = 2048
    dataset_hash: str = ""
    source: str = ""
    texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("texts")  # manifest records the hash, not the corpus
        return d


def _hash_texts(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def _cache_path(name: str, num_samples: int, seed: int) -> Path:
    cache = Path(user_cache_dir("quantiv")) / "calibration"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{name}-{num_samples}-seed{seed}.json"


def get_calibration_set(
    name: str = "wikitext",
    num_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> CalibrationSet:
    """Load (or build + cache) a deterministic calibration set.

    Selection is logged and reproducible: fixed seed, content hash recorded.
    Falls back to built-in generic English when the Hub is unreachable.
    """
    cache_file = _cache_path(name, num_samples, seed)
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return CalibrationSet(**data)
        except Exception:
            pass  # corrupt cache -> rebuild below

    texts: list[str] = []
    source = ""
    if name == "wikitext":
        try:
            from datasets import load_dataset

            for spec in WIKITEXT_SPECS:
                try:
                    ds = load_dataset(*spec, split="train")
                    pool = [t.strip() for t in ds["text"] if len(t.strip()) > 100]
                    break
                except Exception:
                    pool = []
            if pool:
                rng = random.Random(seed)
                texts = rng.sample(pool, min(num_samples, len(pool)))
                source = f"{spec[0]}/{spec[1]}/train"
        except Exception:
            texts = []
    if not texts:
        # Documented offline fallback (also used for unit tests).
        repeats = (num_samples // len(FALLBACK_TEXTS)) + 1
        texts = (FALLBACK_TEXTS * repeats)[:num_samples]
        source = f"builtin-fallback (requested: {name})"

    cal = CalibrationSet(
        name=name,
        seed=seed,
        num_samples=len(texts),
        seq_len=seq_len,
        dataset_hash=_hash_texts(texts),
        source=source,
        texts=texts,
    )
    try:
        cache_file.write_text(json.dumps(asdict(cal)), encoding="utf-8")
    except Exception:
        pass
    return cal
