"""Evaluation package: measured perplexity / speed / memory."""

from quantiv.evaluation.metrics import (
    EvalReport,
    disk_size_gb,
    evaluate_gguf,
    evaluate_gguf_full,
    evaluate_hf_model,
    get_eval_texts,
)

__all__ = [
    "EvalReport",
    "disk_size_gb",
    "evaluate_gguf",
    "evaluate_gguf_full",
    "evaluate_hf_model",
    "get_eval_texts",
]
