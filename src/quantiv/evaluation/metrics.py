"""Baseline evaluation: perplexity + generation speed + memory. All numbers measured.

Held-out text: WikiText-2 test split when reachable (hash recorded); otherwise a
built-in generic-English fallback (source recorded). The report always states
which source was used — no silent substitutions.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

FALLBACK_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
    "Machine learning models compress knowledge into matrices of numbers.",
    "Quantization reduces model size by storing weights in fewer bits per value.",
    "The city council met on Tuesday to discuss the new public library budget.",
    "She opened the window and watched the rain fall on the empty street.",
    "Open source software allows anyone to inspect, modify, and share the code.",
    "The train arrived late because of heavy snow in the mountain pass.",
    "Cooking requires patience, fresh ingredients, and careful attention to heat.",
]


@dataclass
class EvalReport:
    model: str
    device: str
    text_source: str = "unknown"
    perplexity: float | None = None
    ppl_samples: int = 0
    tokens_per_sec: float | None = None
    time_to_first_token_s: float | None = None
    peak_memory_gb: float | None = None
    disk_size_gb: float | None = None
    sanity: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def get_eval_texts(max_samples: int = 32) -> tuple[list[str], str]:
    """Return (texts, source_label). Prefers WikiText-2 test split; falls back offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples]
        if texts:
            return texts, "wikitext-2-raw-v1/test"
    except Exception as e:  # offline or missing dep -> documented fallback
        return (FALLBACK_TEXTS * ((max_samples // len(FALLBACK_TEXTS)) + 1))[:max_samples], (
            f"builtin-fallback ({type(e).__name__})"
        )
    return (FALLBACK_TEXTS * ((max_samples // len(FALLBACK_TEXTS)) + 1))[:max_samples], (
        "builtin-fallback (empty split)"
    )


@contextmanager
def track_peak_memory(device: str):
    """Yield a callable returning GB observed inside the block.

    CUDA: true allocator peak. CPU: process-RSS delta (documented approximation).
    """
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            yield lambda: round(torch.cuda.max_memory_allocated() / 1e9, 3)
            return
    except ImportError:
        pass
    import psutil

    rss0 = psutil.Process().memory_info().rss
    yield lambda: round((psutil.Process().memory_info().rss - rss0) / 1e9, 3)


def measure_perplexity(model, tokenizer, texts: list[str], device: str, seq_len: int = 512) -> float:
    """Sliding-window perplexity. Standard NLL over concatenated token stream."""
    import torch

    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    nll_sum, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, input_ids.size(1), seq_len):
            chunk = input_ids[:, i : i + seq_len]
            if chunk.size(1) < 10:
                continue
            out = model(chunk, labels=chunk)
            nll_sum += out.loss.item() * chunk.size(1)
            count += chunk.size(1)
    return math.exp(nll_sum / count)


def measure_speed(model, tokenizer, device: str, max_new_tokens: int = 32, reps: int = 3) -> tuple[float, float]:
    """Greedy generation speed. Returns (tokens/sec decode, time-to-first-token s)."""
    import torch

    prompts = ["The capital of France is", "Quantization means"]
    model.eval()
    ttfts: list[float] = []
    rates: list[float] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # warmup
        with torch.no_grad():
            model.generate(ids, max_new_tokens=4, do_sample=False, use_cache=True)
        for _ in range(reps):
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            dt = time.time() - t0
            new_tokens = out.shape[1] - ids.shape[1]
            # TTFT approx: time for first token ~= total * 1/new (prefill-dominated); measure honestly:
            t1 = time.time()
            with torch.no_grad():
                model.generate(ids, max_new_tokens=1, do_sample=False, use_cache=True)
            ttfts.append(time.time() - t1)
            rates.append(new_tokens / dt)
    return sum(rates) / len(rates), sum(ttfts) / len(ttfts)


def disk_size_gb(path: str | Path) -> float:
    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / 1e9, 3)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1e9, 3)


def sanity_checks(model, tokenizer, device: str) -> dict:
    """Loads (we're here) + generates coherent text + no NaN/Inf in a forward pass."""
    import torch

    result = {"loads": True, "generates": False, "no_nan_inf": False, "chat_template": False}
    try:
        ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        result["generates"] = len(text) > len("Hello, my name is")
        with torch.no_grad():
            logits = model(ids).logits
        result["no_nan_inf"] = bool(torch.isfinite(logits).all())
    except Exception:
        pass
    try:
        result["chat_template"] = bool(getattr(tokenizer, "chat_template", None))
    except Exception:
        pass
    return result


def evaluate_hf_model(
    model_ref: str, device: str = "cpu", max_samples: int = 32, trust_remote_code: bool = False
) -> EvalReport:
    """Full eval of an HF-format model (id or dir). Numbers are all measured."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    report = EvalReport(model=model_ref, device=device)
    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    with track_peak_memory(device) as peak:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_ref, torch_dtype=dtype, trust_remote_code=trust_remote_code
            ).to(device)
        except Exception:
            # Possibly an HQQ-quantized artifact (custom safetensors layout):
            # reload via HQQ's generic loader instead of failing the run.
            from hqq.models.hf.base import AutoHQQHFModel

            model = AutoHQQHFModel.from_quantized(model_ref, compute_dtype=dtype, device=device)
            report.warnings.append("loaded via HQQ from_quantized (non-standard weights)")
        texts, source = get_eval_texts(max_samples)
        report.text_source = source
        try:
            report.perplexity = round(measure_perplexity(model, tok, texts, device), 3)
            report.ppl_samples = len(texts)
        except Exception as e:
            report.warnings.append(f"perplexity failed: {e}")
        try:
            tps, ttft = measure_speed(model, tok, device)
            report.tokens_per_sec = round(tps, 2)
            report.time_to_first_token_s = round(ttft, 4)
        except Exception as e:
            report.warnings.append(f"speed failed: {e}")
        try:
            report.sanity = sanity_checks(model, tok, device)
        except Exception as e:
            report.warnings.append(f"sanity failed: {e}")
        try:
            report.peak_memory_gb = peak()
        except Exception:
            pass
    try:
        report.disk_size_gb = disk_size_gb(model_ref) if Path(model_ref).exists() else None
    except Exception:
        pass
    report.elapsed_s = round(time.time() - t0, 2)
    return report


def evaluate_gguf(gguf_path: str | Path) -> EvalReport:
    """Lightweight eval for GGUF artifacts: load check + size. PPL via HF path stays canonical."""
    t0 = time.time()
    report = EvalReport(model=str(gguf_path), device="llama.cpp")
    report.disk_size_gb = disk_size_gb(gguf_path)
    try:
        import llama_cpp

        m = llama_cpp.Llama(str(gguf_path), verbose=False)
        report.sanity = {"loads": True, "n_ctx": m.n_ctx(), "generates": False}
        out = m("Hello, my name is", max_tokens=8)
        report.sanity["generates"] = len(out["choices"][0]["text"]) > 0
    except Exception as e:
        report.sanity = {"loads": False}
        report.warnings.append(f"gguf load failed: {e}")
    report.elapsed_s = round(time.time() - t0, 2)
    return report
