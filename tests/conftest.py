"""Shared fixtures: build a tiny fake HF-style model dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tiny-model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "vocab_size": 1000,
                "max_position_embeddings": 512,
                "torch_dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "hi"}), encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    return d
