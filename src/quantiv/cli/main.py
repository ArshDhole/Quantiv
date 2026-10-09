"""Quantiv CLI (Typer). Phase 1: analyze + doctor fully work; others validate + stub."""

from __future__ import annotations

import json
import sys
from datetime import UTC
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from quantiv import __version__
from quantiv.analyzer import analyze_model
from quantiv.hardware import get_target_profile, list_targets, profile_hardware
from quantiv.planner import recommend_plans
from quantiv.utils.config import RunConfig
from quantiv.utils.logging import get_logger, set_verbose

app = typer.Typer(
    name="quantiv",
    help="Auto-quantize open-source LLMs with verified reports.",
    no_args_is_help=True,
)
console = Console()
log = get_logger()


def _print_profile(profile) -> None:
    d = profile.to_dict()
    table = Table(title=f"Model: {d['source']}", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")
    for k in (
        "source_type",
        "architecture",
        "model_type",
        "param_count",
        "param_count_source",
        "dtype",
        "hidden_size",
        "num_layers",
        "vocab_size",
        "context_length",
        "is_moe",
        "num_experts",
        "tokenizer",
        "chat_template",
        "license",
        "license_is_restrictive_or_gated",
        "gated",
        "sha",
    ):
        table.add_row(k, str(d.get(k)))
    table.add_row("memory_estimates_gb", json.dumps(d.get("memory_estimates_gb", {})))
    if d.get("warnings"):
        table.add_row("warnings", "\n".join(d["warnings"]))
    console.print(table)


@app.command()
def analyze(
    model: str = typer.Argument(..., help="HF repo ID, local dir, or .safetensors/.gguf file"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write JSON report to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Inspect a model: arch, size, dtype, license, tokenizer, memory estimates."""
    set_verbose(verbose)
    log.info("Analyzing %s", model)
    profile = analyze_model(model)
    _print_profile(profile)
    if profile.license_is_restrictive_or_gated:
        console.print("[bold yellow]License warning:[/bold yellow] restrictive/gated — review before redistributing.")
    if output:
        Path(output).write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green]")


@app.command()
def plan(
    model: str = typer.Argument(...),
    target: str = typer.Option("local", "--target", help=f"Device profile ({'/'.join(list_targets())})"),
    goal: str = typer.Option("balanced", "--goal", help="min-size|balanced|max-quality|min-latency|cpu-efficient"),
    max_ppl_increase: float = typer.Option(0.05, "--max-ppl-increase"),
    output: str | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Show ranked quantization candidates (rules engine; LLM agent lands in Phase 5)."""
    cfg = RunConfig(model=model, target=target, goal=goal, max_ppl_increase=max_ppl_increase)  # type: ignore[arg-type]
    try:
        tgt = get_target_profile(cfg.target)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from e
    hw = profile_hardware()
    vram = tgt.get("vram_gb", hw.gpu_vram_gb)
    profile = analyze_model(model)
    params_b = (profile.param_count / 1e9) if profile.param_count else None
    cands = recommend_plans(params_b, vram, cfg.goal)
    table = Table(title=f"Plan for {model} -> {target} ({goal})")
    table.add_column("#")
    table.add_column("Method")
    table.add_column("Quant")
    table.add_column("Reason")
    for i, c in enumerate(cands, 1):
        table.add_row(str(i), c["method"], c["quant"], c["reason"])
    console.print(table)
    console.print(f"Quality gate: max_ppl_increase={cfg.max_ppl_increase} | target vram={vram}GB | params={params_b}B")
    if output:
        Path(output).write_text(
            json.dumps({"model": model, "target": tgt, "candidates": cands}, indent=2),
            encoding="utf-8",
        )


@app.command()
def run(
    model: str = typer.Argument(...),
    target: str = typer.Option("local", "--target"),
    goal: str = typer.Option("balanced", "--goal"),
    method: str = typer.Option("auto", "--method", help="auto|gguf|gptq|awq|hqq|bnb|torchao"),
    max_ppl_increase: float = typer.Option(0.05, "--max-ppl-increase"),
    max_samples: int = typer.Option(16, "--max-samples", help="Held-out samples for perplexity"),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o"),
    no_agent: bool = typer.Option(True, "--no-agent/--agent", help="Rules-only (default) vs LLM agent (Phase 5)"),
) -> None:
    """End-to-end: analyze -> quantize -> evaluate baseline + quantized -> report."""
    from datetime import datetime

    from quantiv.evaluation import evaluate_gguf, evaluate_hf_model
    from quantiv.packaging.manifest import build_manifest, write_manifest
    from quantiv.packaging.report import write_report
    from quantiv.quantizers import run_quantization
    from quantiv.quantizers.base import QuantizeRequest, resolve_device

    cfg = RunConfig(
        model=model,
        target=target,
        goal=goal,
        method=method,
        max_ppl_increase=max_ppl_increase,
        no_agent=no_agent,
    )  # type: ignore[arg-type]
    if not no_agent:
        console.print("[red]Agent mode (--agent) is not implemented until Phase 5. Re-run with --no-agent.[/red]")
        raise typer.Exit(3)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{stamp}-{cfg.method}-{cfg.goal}"
    quant_dir = run_dir / "quantized"
    device = resolve_device()
    console.print(f"Quantizing [bold]{model}[/bold] method={cfg.method} goal={goal} device={device}")

    result = run_quantization(
        QuantizeRequest(
            model=model,
            method=cfg.method,
            output_dir=str(quant_dir),
            device=device,
            group_size=group_size,
            quant="Q4_K_M" if cfg.method in ("auto", "gguf") else "",
        ),
        goal=goal,
    )
    if not result.success:
        console.print(f"[red]Quantization failed: {result.message}[/red]")
        raise typer.Exit(4)
    console.print(
        f"[green]Quantized:[/green] {result.method} {result.quant} in {result.elapsed_s}s -> {result.output_dir}"
    )

    console.print("Evaluating baseline (original model)...")
    baseline_report = evaluate_hf_model(model, device=device, max_samples=max_samples)
    console.print(f"Baseline ppl={baseline_report.perplexity} tok/s={baseline_report.tokens_per_sec}")
    console.print("Evaluating quantized model...")
    if result.method == "gguf":
        ggufs = sorted(quant_dir.glob("*.gguf"))
        quant_report = evaluate_gguf(ggufs[0]) if ggufs else baseline_report
    else:
        quant_report = evaluate_hf_model(result.output_dir, device=device, max_samples=max_samples)
    console.print(f"Quantized ppl={quant_report.perplexity} tok/s={quant_report.tokens_per_sec}")

    gates = {"max_ppl_increase": max_ppl_increase}
    json_path, md_path = write_report(
        run_dir,
        baseline_report,
        quant_report,
        {
            "method": result.method,
            "quant": result.quant,
            "elapsed_s": result.elapsed_s,
            "files": result.files,
            "device": device,
        },
        gates,
    )
    manifest = build_manifest(
        model, {"run_dir": str(run_dir), "method": result.method, "quant": result.quant, "device": device, "goal": goal}
    )
    write_manifest(run_dir, manifest)
    console.print(f"[green]Done.[/green] Report: {md_path} | JSON: {json_path}")


@app.command()
def eval(
    quantized_path: str = typer.Argument(...),
    baseline: str = typer.Option(..., "--baseline", help="Original model (HF id or path)"),
    max_samples: int = typer.Option(16, "--max-samples"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o"),
) -> None:
    """Evaluate quantized vs baseline and write a comparison report."""
    from datetime import datetime

    from quantiv.evaluation import evaluate_gguf, evaluate_hf_model
    from quantiv.packaging.report import write_report
    from quantiv.quantizers.base import resolve_device

    device = resolve_device()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{stamp}-eval"
    console.print(f"Evaluating baseline {baseline} ...")
    base = evaluate_hf_model(baseline, device=device, max_samples=max_samples)
    console.print(f"Evaluating {quantized_path} ...")
    qp = Path(quantized_path)
    if qp.is_file() and qp.suffix == ".gguf":
        quant = evaluate_gguf(qp)
    elif qp.is_dir() and list(qp.glob("*.gguf")):
        quant = evaluate_gguf(sorted(qp.glob("*.gguf"))[0])
    else:
        quant = evaluate_hf_model(quantized_path, device=device, max_samples=max_samples)
    json_path, md_path = write_report(
        run_dir, base, quant, {"method": "external", "quant": qp.name, "device": device}, {}
    )
    console.print(f"[green]Done.[/green] Report: {md_path} | JSON: {json_path}")


@app.command()
def compare(run_a: str = typer.Argument(...), run_b: str = typer.Argument(...)) -> None:
    """Compare two runs (Phase 2+)."""
    console.print(f"[yellow]Phase 1 stub:[/yellow] would compare {run_a} vs {run_b}.")


@app.command()
def package(run_dir: str = typer.Argument(...), push_to_hub: bool = typer.Option(False, "--push-to-hub")) -> None:
    """Package run dir → manifest + model card (Phase 6)."""
    if push_to_hub:
        console.print("[red]Refusing --push-to-hub in Phase 1: license check + manifest required (Phase 6).[/red]")
        raise typer.Exit(3)
    console.print(f"[yellow]Phase 1 stub:[/yellow] would package {run_dir}.")


@app.command()
def doctor(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Check environment, drivers, dependencies."""
    set_verbose(verbose)
    hw = profile_hardware()
    table = Table(title=f"Quantiv doctor (v{__version__})")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("OS", hw.os)
    table.add_row("CPU", f"{hw.cpu} ({hw.cpu_cores_physical}P/{hw.cpu_cores_logical}L) {hw.cpu_features}")
    table.add_row("RAM", f"{hw.ram_gb} GB")
    table.add_row("Disk free", f"{hw.disk_free_gb} GB")
    table.add_row("GPU", str(hw.gpu_name or "none detected"))
    table.add_row("GPU VRAM", str(hw.gpu_vram_gb or "-"))
    table.add_row("Compute cap", str(hw.gpu_compute_capability or "-"))
    table.add_row("torch.cuda", str(hw.torch_cuda_available))
    # Dependency presence (import-only, no install)
    for mod in (
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "datasets",
        "gptqmodel",
        "bitsandbytes",
        "torchao",
        "lm_eval",
    ):
        try:
            __import__(mod)
            table.add_row(mod, "[green]installed[/green]")
        except ImportError:
            table.add_row(mod, "[dim]missing (ok in Phase 1 unless needed)[/dim]")
    console.print(table)
    console.print(f"Targets: {list_targets()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
