"""Analyzer unit tests (offline, tiny model)."""

from quantiv.analyzer import analyze_model
from quantiv.analyzer.model_analyzer import _license_flag


def test_analyze_local_dir(tiny_model_dir):
    p = analyze_model(str(tiny_model_dir))
    assert p.architecture == "LlamaForCausalLM"
    assert p.model_type == "llama"
    assert p.hidden_size == 64
    assert p.num_layers == 2
    assert p.chat_template is True
    assert p.memory_estimates_gb  # estimate path fills this
    assert p.param_count_source in ("estimate", "safetensors-header", "unknown")


def test_analyze_missing_dir_warns(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    p = analyze_model(str(d))
    assert any("config.json" in w for w in p.warnings)


def test_analyze_unsupported_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    p = analyze_model(str(f))
    assert any("Unsupported" in w for w in p.warnings)


def test_license_flag_permissive():
    # Regression: arch tags like "llama" must not flag apache-2.0.
    assert _license_flag("apache-2.0", ["pytorch", "llama", "text-generation"]) is False
    assert _license_flag("mit", ["llama"]) is False
    assert _license_flag(None, ["pytorch"]) is False


def test_license_flag_restrictive():
    assert _license_flag("Meta Llama 3.1 Community License", []) is True
    assert _license_flag("gemma terms of use", []) is True
    assert _license_flag("cc-by-nc-4.0", []) is True
    assert _license_flag("apache-2.0", ["license:restricted"]) is True
