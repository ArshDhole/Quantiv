"""Phase 2 tests: registry, eval metrics, report, HQQ e2e — all offline on CPU."""

import json

import pytest
import torch

from quantiv.evaluation import EvalReport, evaluate_hf_model
from quantiv.packaging.report import write_report
from quantiv.quantizers import auto_select, available_backends, get_backend, run_quantization
from quantiv.quantizers.base import QuantizeRequest


@pytest.fixture(scope="module")
def tiny_hf_model(tmp_path_factory):
    """Random-weight toy Llama + locally-trained BPE tokenizer. No network needed."""
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


def test_registry_probes_all_backends():
    statuses = available_backends()
    assert set(statuses) == {"hqq", "bnb", "gguf"}
    assert statuses["hqq"].available  # torch+hqq installed
    assert statuses["gguf"].available  # llama_cpp+gguf+torch installed


def test_auto_select_returns_available():
    name = auto_select(QuantizeRequest(model="x"), goal="balanced")
    assert available_backends()[name].available


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_backend("nope")


def test_evaluate_tiny_model(tiny_hf_model):
    rep = evaluate_hf_model(str(tiny_hf_model), device="cpu", max_samples=2)
    assert rep.perplexity is not None and rep.perplexity > 0
    assert rep.tokens_per_sec is not None and rep.tokens_per_sec > 0
    assert rep.sanity.get("loads") is True
    assert rep.sanity.get("generates") is True
    assert rep.sanity.get("no_nan_inf") is True


def test_write_report_gate_logic(tmp_path):
    base = EvalReport(model="b", device="cpu", perplexity=10.0)
    good = EvalReport(model="q", device="cpu", perplexity=10.2)  # +2%
    bad = EvalReport(model="q", device="cpu", perplexity=12.0)  # +20%
    _, md = write_report(tmp_path / "g", base, good, {"method": "hqq"}, {"max_ppl_increase": 0.05})
    assert "PASS" in md.read_text()
    _, md2 = write_report(tmp_path / "b", base, bad, {"method": "hqq"}, {"max_ppl_increase": 0.05})
    assert "FAIL" in md2.read_text()
    payload = json.loads((tmp_path / "g" / "report.json").read_text())
    assert payload["comparison"]["ppl_increase"] == pytest.approx(0.02)


def test_hqq_e2e_tiny(tiny_hf_model, tmp_path):
    """Full pipeline on toy weights: quantize -> evaluate both -> report with real numbers."""
    out = tmp_path / "q"
    result = run_quantization(
        QuantizeRequest(model=str(tiny_hf_model), method="hqq", group_size=8, output_dir=str(out), device="cpu")
    )
    assert result.success
    assert "qmodel.pt" in result.files  # HQQ format: torch pickle + config.json
    assert "config.json" in result.files
    base = evaluate_hf_model(str(tiny_hf_model), device="cpu", max_samples=2)
    quant = evaluate_hf_model(str(out), device="cpu", max_samples=2)
    assert base.perplexity and quant.perplexity
    run_dir = tmp_path / "run"
    jp, md = write_report(
        run_dir,
        base,
        quant,
        {"method": "hqq", "quant": result.quant, "elapsed_s": result.elapsed_s, "files": result.files, "device": "cpu"},
        {"max_ppl_increase": 5.0},
    )
    assert jp.exists() and md.exists()
