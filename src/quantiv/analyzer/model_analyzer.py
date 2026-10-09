"""Model intake & analyzer.

Accepts: HF repo ID, local directory, or .safetensors/.gguf file.
Strategy: parse local config.json when available; otherwise query HF Hub API
(config only — no weight download). Estimates memory per bit-width.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelProfile:
    source: str
    source_type: str  # "hf-hub" | "local-dir" | "local-file"
    architecture: str | None = None
    model_type: str | None = None
    param_count: int | None = None
    param_count_source: str = "unknown"  # config | safetensors-header | estimate | unknown
    dtype: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    context_length: int | None = None
    is_moe: bool = False
    num_experts: int | None = None
    tokenizer: str | None = None
    chat_template: bool = False
    license: str | None = None
    license_is_restrictive_or_gated: bool = False
    gated: bool = False
    sha: str | None = None
    memory_estimates_gb: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


RESTRICTIVE_LICENSE_HINTS = (
    "llama",
    "community license",
    "gated",
    "accept",
    "meta",
    "gemma terms",
    "restricted",
    "non-commercial",
    "cc-by-nc",
)

DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "int8": 1,
    "int4": 0.5,
    "nf4": 0.5,
    "float4": 0.5,
}


def _looks_like_hf_id(s: str) -> bool:
    # Local paths exist on disk or end with known suffixes; else treat "org/name" as HF id.
    p = Path(s)
    if p.exists():
        return False
    if s.endswith((".safetensors", ".gguf", ".bin")):
        return False
    return "/" in s.strip() and " " not in s.strip()


def _memory_estimates(param_count: int | None, overhead_gb: float = 0.5) -> dict:
    if not param_count:
        return {}
    out: dict[str, float] = {}
    for label, bpp in (("fp32", 4), ("fp16_bf16", 2), ("int8", 1), ("int4", 0.5), ("2bit", 0.25)):
        out[label] = round(param_count * bpp / 1e9 + overhead_gb, 2)
    return out


def _license_flag(license_name: str | None, tags: list[str] | None = None) -> bool:
    """Flag restrictive/gated licenses.

    Only the license string itself + license-related tags (``license:*``) are
    checked. Generic Hub tags (arch names like ``llama``, region tags, etc.)
    must NOT trigger the flag — that caused apache-2.0 false positives.
    """
    lic = (license_name or "").lower()
    if any(h in lic for h in RESTRICTIVE_LICENSE_HINTS):
        return True
    lic_tags = [t.lower() for t in (tags or []) if t.lower().startswith(("license:", "gated"))]
    blob = " ".join(lic_tags)
    return any(h in blob for h in RESTRICTIVE_LICENSE_HINTS) or "gated" in blob


def _enrich_from_config_dict(cfg: dict, profile: ModelProfile) -> ModelProfile:
    profile.architecture = cfg.get("architectures", [None])[0] if cfg.get("architectures") else profile.architecture
    profile.model_type = cfg.get("model_type", profile.model_type)
    profile.hidden_size = cfg.get("hidden_size", profile.hidden_size)
    profile.num_layers = cfg.get("num_hidden_layers", cfg.get("n_layer", profile.num_layers))
    profile.vocab_size = cfg.get("vocab_size", profile.vocab_size)
    profile.context_length = cfg.get("max_position_embeddings", profile.context_length)
    profile.dtype = cfg.get("torch_dtype", profile.dtype)
    # MoE detection
    if any(k in cfg for k in ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts")):
        profile.is_moe = True
        profile.num_experts = cfg.get("num_local_experts", cfg.get("num_experts", cfg.get("n_routed_experts")))
    # Chat template
    if cfg.get("chat_template"):
        profile.chat_template = True
    # Param estimate from hidden_size/layers if safetensors count missing
    if profile.param_count is None and profile.hidden_size and profile.num_layers and profile.vocab_size:
        h, L, V = profile.hidden_size, profile.num_layers, profile.vocab_size
        # ~12*h^2 per layer (attn+mlp) + vocab*h embeddings — rough but documented.
        est = 12 * h * h * L + V * h
        profile.param_count = int(est)
        profile.param_count_source = "estimate"
        profile.warnings.append("param_count is a rough config-based estimate, not exact.")
    if profile.param_count:
        profile.memory_estimates_gb = _memory_estimates(profile.param_count)
    return profile


def analyze_local_dir(path: str | Path) -> ModelProfile:
    p = Path(path)
    profile = ModelProfile(source=str(p), source_type="local-dir")
    cfg_path = p / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            profile = _enrich_from_config_dict(cfg, profile)
        except Exception as e:
            profile.warnings.append(f"Failed to parse config.json: {e}")
    else:
        profile.warnings.append("No config.json found in directory.")
    # Tokenizer
    if (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists():
        profile.tokenizer = "present (local)"
        try:
            tc = p / "tokenizer_config.json"
            if tc.exists() and "chat_template" in tc.read_text(encoding="utf-8"):
                profile.chat_template = True
        except Exception:
            pass
    else:
        profile.warnings.append("No tokenizer files found.")
    # License
    for lic_file in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (p / lic_file).exists():
            profile.license = f"see {lic_file} (local)"
            break
    # Safetensors header param count (no full load)
    try:
        from safetensors import safe_open  # type: ignore

        st_files = sorted(p.glob("*.safetensors"))
        if st_files:
            total = 0
            for f in st_files:
                with safe_open(str(f), framework="pt") as h:  # type: ignore[attr-defined]
                    for k in h.keys():
                        t = h.get_slice(k)
                        n = 1
                        for d in t.get().shape:
                            n *= d
                        total += n
            if total:
                profile.param_count = total
                profile.param_count_source = "safetensors-header"
                profile.memory_estimates_gb = _memory_estimates(total)
    except ImportError:
        profile.warnings.append("safetensors not installed; skipping exact param count.")
    except Exception as e:
        profile.warnings.append(f"safetensors header read failed: {e}")
    # GGUF single-file inside dir? handled by analyze_local_file.
    return profile


def analyze_local_file(path: str | Path) -> ModelProfile:
    p = Path(path)
    profile = ModelProfile(source=str(p), source_type="local-file")
    if p.suffix == ".gguf":
        profile.dtype = "gguf (quantized)"
        profile.warnings.append("GGUF input: already quantized; re-quantization analysis is limited.")
        try:
            profile.memory_estimates_gb = {"file_size_gb": round(p.stat().st_size / 1e9, 2)}
        except Exception:
            pass
    elif p.suffix == ".safetensors":
        try:
            from safetensors import safe_open  # type: ignore

            total = 0
            with safe_open(str(p), framework="pt") as h:  # type: ignore[attr-defined]
                for k in h.keys():
                    t = h.get_slice(k)
                    n = 1
                    for d in t.get().shape:
                        n *= d
                    total += n
            if total:
                profile.param_count = total
                profile.param_count_source = "safetensors-header"
                profile.memory_estimates_gb = _memory_estimates(total)
        except ImportError:
            profile.warnings.append("safetensors not installed; cannot count params.")
        except Exception as e:
            profile.warnings.append(f"safetensors read failed: {e}")
    else:
        profile.warnings.append(f"Unsupported file type '{p.suffix}'. Expected .safetensors/.gguf or a directory.")
    return profile


def analyze_hf_hub(repo_id: str) -> ModelProfile:
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore

    profile = ModelProfile(source=repo_id, source_type="hf-hub")
    api = HfApi()
    try:
        info = api.model_info(repo_id)
    except HfHubHTTPError as e:
        profile.warnings.append(f"HF Hub lookup failed: {e}")
        return profile
    except Exception as e:
        profile.warnings.append(f"HF Hub lookup failed (offline?): {e}")
        return profile
    profile.sha = getattr(info, "sha", None)
    profile.gated = bool(getattr(info, "gated", False))
    profile.license = (
        getattr(info, "license", None) or getattr(info, "cardData", {}).get("license")
        if getattr(info, "cardData", None)
        else getattr(info, "license", None)
    )
    tags = list(getattr(info, "tags", []) or [])
    if _license_flag(profile.license, tags) or profile.gated:
        profile.license_is_restrictive_or_gated = True
        profile.warnings.append(
            f"License '{profile.license}' looks restrictive/gated — redistribution requires review. Never auto-upload."
        )
    profile.tokenizer = "present (hub)" if "tokenizers" in " ".join(tags).lower() or True else None
    # Try config.json only (no weights)
    try:
        cfg_path = hf_hub_download(repo_id, "config.json")
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        profile = _enrich_from_config_dict(cfg, profile)
    except Exception as e:
        profile.warnings.append(f"Could not fetch config.json from Hub: {e}")
    # Try tokenizer_config for chat template
    try:
        tok_path = hf_hub_download(repo_id, "tokenizer_config.json")
        if "chat_template" in Path(tok_path).read_text(encoding="utf-8"):
            profile.chat_template = True
    except Exception:
        pass
    # siblings for safetensors presence hint
    try:
        siblings = [getattr(s, "rfilename", "") for s in (info.siblings or [])]
        if any(s.endswith(".gguf") for s in siblings):
            profile.warnings.append("Hub repo contains GGUF files.")
    except Exception:
        pass
    return profile


def analyze_model(source: str) -> ModelProfile:
    """Main entry: dispatch to local-dir / local-file / HF Hub."""
    p = Path(source)
    if p.exists():
        if p.is_dir():
            return analyze_local_dir(p)
        return analyze_local_file(p)
    if source.endswith((".gguf", ".safetensors")):
        return analyze_local_file(source)
    if _looks_like_hf_id(source):
        return analyze_hf_hub(source)
    # Fallback: try Hub anyway (gives a clear error in warnings)
    return analyze_hf_hub(source)
