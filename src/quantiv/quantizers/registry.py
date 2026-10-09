"""Backend registry + auto-selection (dependency-free; backends self-probe)."""

from __future__ import annotations

from quantiv.quantizers.base import BackendStatus, Quantizer, QuantizeRequest, QuantizeResult


def _backend_classes() -> dict[str, type[Quantizer]]:
    # Local imports so `quantiv.quantizers` stays import-safe without torch.
    from quantiv.quantizers.bnb import BNBQuantizer
    from quantiv.quantizers.gguf import GGUQuantizer
    from quantiv.quantizers.hqq import HQQQuantizer

    return {c.name: c for c in (HQQQuantizer, BNBQuantizer, GGUQuantizer)}


def available_backends() -> dict[str, BackendStatus]:
    """Probe every backend. Never raises — failures become unavailable statuses."""
    out: dict[str, BackendStatus] = {}
    for name, cls in _backend_classes().items():
        try:
            out[name] = cls.probe()
        except Exception as e:  # noqa: BLE001
            out[name] = BackendStatus(name=name, available=False, reason=f"probe failed: {e}")
    return out


def get_backend(name: str) -> Quantizer:
    classes = _backend_classes()
    if name not in classes:
        raise KeyError(f"Unknown backend '{name}'. Available: {sorted(classes)}")
    inst = classes[name]()
    status = inst.probe()
    if not status.available:
        raise RuntimeError(f"Backend '{name}' unavailable: {status.reason}")
    return inst


def auto_select(request: QuantizeRequest, goal: str = "balanced") -> str:
    """Pick the first available backend for the goal. Deterministic; tested."""
    avail = available_backends()
    goal_order = {
        "min-size": ["gguf", "hqq", "bnb"],
        "max-quality": ["bnb", "gguf", "hqq"],
        "cpu-efficient": ["gguf", "hqq", "bnb"],
        "min-latency": ["gguf", "hqq", "bnb"],
        "balanced": ["hqq", "gguf", "bnb"],
    }
    order = goal_order.get(goal, goal_order["balanced"])
    for name in order:
        if avail.get(name) and avail[name].available:
            return name
    raise RuntimeError("No quantization backend available: " + "; ".join(f"{k} ({v.reason})" for k, v in avail.items()))


def run_quantization(request: QuantizeRequest, goal: str = "balanced") -> QuantizeResult:
    method = request.method
    if method == "auto":
        method = auto_select(request, goal=goal)
        request.method = method
    return get_backend(method).quantize(request)
