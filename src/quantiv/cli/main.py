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
    try:
        from quantiv.quantizers import available_backends

        avail = {k: v.available for k, v in available_backends().items()}
    except Exception:
        avail = {}
    from quantiv.planner.rank import rank_candidates

    ranked = rank_candidates(goal=cfg.goal, params_b=params_b, target_vram_gb=vram, available=avail)
    table = Table(title=f"Plan for {model} -> {target} ({goal})")
    table.add_column("#")
    table.add_column("Method")
    table.add_column("Quant")
    table.add_column("Score")
    table.add_column("Fits")
    table.add_column("Reason")
    for i, c in enumerate(ranked, 1):
        table.add_row(str(i), c.method, c.quant, str(c.score), str(c.fits_target), "; ".join(c.reasons[:2]))
    console.print(table)
    console.print(f"Quality gate: max_ppl_increase={cfg.max_ppl_increase} | target vram={vram}GB | params={params_b}B")
    if output:
        Path(output).write_text(
            json.dumps({"model": model, "target": tgt, "candidates": [c.to_dict() for c in ranked]}, indent=2),
            encoding="utf-8",
        )


@app.command()
def run(
    model: str = typer.Argument(...),
    target: str = typer.Option("local", "--target"),
    goal: str = typer.Option("balanced", "--goal"),
    method: str = typer.Option("auto", "--method", help="auto|gguf|gptq|awq|hqq|bnb|torchao"),
    bits: int = typer.Option(4, "--bits", help="Starting bit-width"),
    max_ppl_increase: float = typer.Option(0.05, "--max-ppl-increase"),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Bounded escalation attempts"),
    max_samples: int = typer.Option(16, "--max-samples", help="Held-out samples for perplexity"),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o"),
    no_agent: bool = typer.Option(True, "--no-agent/--agent", help="Rules-only (default) vs LLM agent (Phase 5)"),
    agent_provider: str = typer.Option("echo", "--agent-provider", help="echo|openai-compat"),
    agent_model: str = typer.Option("", "--agent-model", help="Model for openai-compat provider"),
    agent_base_url: str = typer.Option("", "--agent-base-url", help="Base URL for openai-compat provider"),
) -> None:
    """End-to-end: analyze -> quantize -> evaluate -> gate -> escalate (bounded)."""
    from datetime import datetime

    from quantiv.planner.pipeline import run_pipeline
    from quantiv.quantizers.base import resolve_device

    cfg = RunConfig(
        model=model,
        target=target,
        goal=goal,
        method=method,
        max_ppl_increase=max_ppl_increase,
        no_agent=no_agent,
    )  # type: ignore[arg-type]
    if not no_agent:
        from quantiv.agent import EchoProvider, OpenAICompatProvider, run_agent

        if agent_provider == "openai-compat":
            if not agent_model or not agent_base_url:
                console.print("[red]--agent-provider openai-compat needs --agent-model and --agent-base-url.[/red]")
                raise typer.Exit(2)
            import os

            provider = OpenAICompatProvider(
                model=agent_model, base_url=agent_base_url, api_key=os.environ.get("QUANTIV_API_KEY", "")
            )
        else:
            provider = EchoProvider()
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        agent_dir = Path(output_dir) / f"{stamp}-agent-{cfg.goal}"
        res = run_agent(
            model, goal=goal, run_dir=str(agent_dir), provider=provider, on_step=lambda m: console.print(f"  {m}")
        )
        if not res.success:
            console.print(f"[red]Agent run failed: {res.error}[/red]")
            raise typer.Exit(4)
        console.print(f"[green]Agent done.[/green] Report: {res.report_md}")
        return

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{stamp}-{cfg.method}-{cfg.goal}"
    device = resolve_device()
    console.print(
        f"Quantizing [bold]{model}[/bold] method={cfg.method} bits={bits} goal={goal} "
        f"device={device} max_attempts={max_attempts}"
    )
    try:
        out = run_pipeline(
            model,
            goal=goal,
            method=cfg.method,
            bits=bits,
            device=device,
            max_ppl_increase=max_ppl_increase,
            max_attempts=max_attempts,
            max_samples=max_samples,
            group_size=group_size,
            run_dir=run_dir,
            on_step=lambda m: console.print(f"  {m}"),
        )
    except Exception as e:
        console.print(f"[red]Run failed: {e}[/red]")
        raise typer.Exit(4) from e
    final_gate = out["attempts"][-1]["gate_pass"] if out["attempts"] else None
    console.print(
        f"[green]Done.[/green] {out['quant_info']['method']} {out['quant_info']['quant']} "
        f"gate={final_gate} attempts={len(out['attempts'])} Report: {out['report_md']}"
    )


@app.command()
def eval(
    quantized_path: str = typer.Argument(...),
    baseline: str = typer.Option(..., "--baseline", help="Original model (HF id or path)"),
    max_samples: int = typer.Option(16, "--max-samples"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o"),
) -> None:
    """Evaluate quantized vs baseline and write a comparison report."""
    from datetime import datetime

    from quantiv.evaluation import evaluate_gguf_full, evaluate_hf_model
    from quantiv.packaging.report import write_report
    from quantiv.quantizers.base import resolve_device

    device = resolve_device()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{stamp}-eval"
    console.print(f"Evaluating baseline {baseline} ...")
    base = evaluate_hf_model(baseline, device=device, max_samples=max_samples)
    console.print(f"Evaluating {quantized_path} ...")
    qp = Path(quantized_path)
    qmethod = "external"
    if qp.is_file() and qp.suffix == ".gguf":
        quant = evaluate_gguf_full(qp, max_samples=max_samples)
        qmethod = "gguf"
    elif qp.is_dir() and list(qp.glob("*.gguf")):
        quant = evaluate_gguf_full(sorted(qp.glob("*.gguf"))[0], max_samples=max_samples)
        qmethod = "gguf"
    else:
        quant = evaluate_hf_model(quantized_path, device=device, max_samples=max_samples)
    json_path, md_path = write_report(run_dir, base, quant, {"method": qmethod, "quant": qp.name, "device": device}, {})
    console.print(f"[green]Done.[/green] Report: {md_path} | JSON: {json_path}")


@app.command()
def compare(run_a: str = typer.Argument(...), run_b: str = typer.Argument(...)) -> None:
    """Side-by-side table of two runs' report.json files (all numbers measured)."""
    import json as _json

    def load(run: str) -> dict:
        p = Path(run) / "report.json"
        if not p.exists():
            console.print(f"[red]No report.json in {run}[/red]")
            raise typer.Exit(2)
        return _json.loads(p.read_text(encoding="utf-8"))

    a, b = load(run_a), load(run_b)
    table = Table(title=f"Compare: {Path(run_a).name} vs {Path(run_b).name}")
    table.add_column("Metric")
    table.add_column(Path(run_a).name)
    table.add_column(Path(run_b).name)
    for label, key in (
        ("Method", ("quant", "method")),
        ("Quant", ("quant", "quant")),
        ("PPL increase", ("comparison", "ppl_increase")),
        ("Gate", ("comparison", "gate_pass")),
    ):
        node_a, node_b = a, b
        for k in key:
            node_a, node_b = node_a.get(k), node_b.get(k)
        table.add_row(label, str(node_a), str(node_b))
    for side, rep in (("A", a), ("B", b)):
        q = rep.get("quantized", {})
        table.add_row(f"{side} tok/s", str(q.get("tokens_per_sec")), "")
        table.add_row(f"{side} disk GB", str(q.get("disk_size_gb")), "")
    console.print(table)


@app.command()
def package(
    run_dir: str = typer.Argument(...),
    push_to_hub: bool = typer.Option(False, "--push-to-hub"),
    repo_id: str = typer.Option("", "--repo-id", help="Hub repo for --push-to-hub"),
    i_accept_license: bool = typer.Option(False, "--i-accept-license", help="Confirm publishing rights"),
) -> None:
    """Validate run dir, write MODEL_CARD.md, optionally upload (license-gated)."""
    import json as _json

    from quantiv.packaging import (
        build_model_card,
        check_publish_allowed,
        manifest_hash,
        write_model_card,
    )

    rd = Path(run_dir)
    rep_p, man_p = rd / "report.json", rd / "quantiv_manifest.json"
    if not rep_p.exists() or not man_p.exists():
        console.print(f"[red]{run_dir} is not a complete run (need report.json + manifest).[/red]")
        raise typer.Exit(2)
    rep = _json.loads(rep_p.read_text(encoding="utf-8"))
    man = _json.loads(man_p.read_text(encoding="utf-8"))
    model = man.get("model", rep.get("baseline", {}).get("model", "unknown"))
    try:
        from quantiv.analyzer import analyze_model

        prof = analyze_model(model)
        lic, gated = prof.license, prof.gated
    except Exception:
        lic, gated = None, False
    card = build_model_card(model, rep.get("quant", {}), rep.get("comparison", {}), lic, manifest_hash(man))
    card_p = write_model_card(rd, card)
    console.print(f"[green]Wrote {card_p}[/green]")
    if not push_to_hub:
        return
    verdict = check_publish_allowed(lic, gated, i_accept_license)
    console.print(f"License gate: {verdict.reason}")
    if not verdict.allowed:
        raise typer.Exit(3)
    if not repo_id:
        console.print("[red]--push-to-hub needs --repo-id.[/red]")
        raise typer.Exit(2)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(rd))
        console.print(f"[green]Uploaded to {repo_id}[/green]")
    except Exception as e:
        console.print(f"[red]Upload failed: {e}[/red]")
        raise typer.Exit(4) from e


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8020, "--port"),
) -> None:
    """Serve the optional web dashboard (submit jobs, poll progress, fetch reports)."""
    import uvicorn

    from quantiv.dashboard import create_app

    console.print(f"Serving Quantiv dashboard on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


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
