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


@pytest.fixture(scope="module")
def tiny_hf_model(tmp_path_factory):
    """Random-weight toy Llama + locally-trained BPE tokenizer. No network needed."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from quantiv.evaluation.metrics import FALLBACK_TEXTS

    d = tmp_path_factory.mktemp("tiny-hf")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        FALLBACK_TEXTS * 20,
        BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=200),
    )
    tok.save(str(d / "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(d / "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tok.save_pretrained(str(d))
    cfg = LlamaConfig(
        vocab_size=hf_tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(str(d))
    return d
