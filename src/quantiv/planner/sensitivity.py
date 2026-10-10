"""Per-layer sensitivity: how much does quantizing each block hurt?

Method: baseline NLL on a small calibration slice, then per block replace its
Linears with in-memory HQQ-quantized versions, re-measure NLL, restore.
The delta ranks blocks for mixed-precision assignment. All measured.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def find_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Locate decoder blocks: model.layers / model.model.layers, else heuristic."""
    for path in ("model.layers", "layers", "transformer.h", "gpt_neox.layers"):
        node: nn.Module | None = model
        for attr in path.split("."):
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and len(list(node)) > 1:
            return [(f"{path}.{i}", m) for i, m in enumerate(node)]
    # Fallback: largest ModuleList of identical-type children.
    best: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(list(module)) > 1:
            kids = [(f"{name}.{i}", m) for i, m in enumerate(module)]
            if len(kids) > len(best):
                best = kids
    return best


def _block_nll(model, input_ids: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return model(input_ids, labels=input_ids).loss.item()


def layer_sensitivity(
    model_id: str,
    device: str = "cpu",
    num_texts: int = 4,
    seq_len: int = 256,
    bits: int = 4,
    group_size: int = 64,
) -> dict[str, float]:
    """Return {block_name: NLL increase when that block alone is quantized}."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quantiv.calibration import get_calibration_set

    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype).to(device)

    cal = get_calibration_set(num_samples=max(num_texts, 4), seed=42)
    enc = tok("\n\n".join(cal.texts[:num_texts]), return_tensors="pt")
    input_ids = enc.input_ids[:, :seq_len].to(device)

    base_nll = _block_nll(model, input_ids)
    hidden = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    gs = min(group_size, hidden) if hidden else group_size
    gs = max(8, gs - (gs % 8))  # multiple of 8, floor 8 (tiny-model safe)
    cfg = BaseQuantizeConfig(nbits=bits, group_size=gs, quant_zero=True, quant_scale=False)
    deltas: dict[str, float] = {}
    for bname, block in find_blocks(model):
        saved: dict[str, nn.Module] = {}
        for lname, linear in list(block.named_modules()):
            if type(linear) is nn.Linear:
                # deepcopy: HQQLinear strips the source module's weight on wrap.
                saved[lname] = copy.deepcopy(linear)
                parent = block
                *path, leaf = lname.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(
                    parent,
                    leaf,
                    HQQLinear(linear, cfg, compute_dtype=compute_dtype, device=device),
                )
        try:
            deltas[bname] = round(_block_nll(model, input_ids) - base_nll, 5)
        finally:
            for lname, linear in saved.items():
                parent = block
                *path, leaf = lname.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, leaf, linear)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(sorted(deltas.items(), key=lambda kv: kv[1], reverse=True))
