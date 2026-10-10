# Quantiv — auto-quantize open-source LLMs, with receipts

> **v1.2.** Give it a model + a goal — get back a verified quantized artifact with **measured** quality, speed, and memory numbers. No fabricated metrics, ever.

Quantiv takes a Hugging Face model (or local weights) and a goal like *"run on my 8GB GPU"* or *"smallest size under 5% loss"*, then: profiles the model → ranks strategies → quantizes → evaluates both versions → checks a quality gate → **escalates automatically on a miss** (higher bits → next backend → mixed precision) → ships weights + report + model card.

An LLM-driven planner orchestrates deterministic tooling — the agent decides, the tools do the math.

## Measured results (real runs, WikiText-2)

| Model | Backend | PPL change | Size | Gate |
|---|---|---|---|---|
| SmolLM2-135M | HQQ 8-bit (escalated) | **−0.1%** | 272 → 173 MB | PASS |
| SmolLM2-135M | GPTQ 4-bit + calibration | +15.8% | 272 → 118 MB | FAIL* |
| SmolLM2-135M | GGUF Q4_K_M | +38.6% | 272 → 105 MB | FAIL* |
| Qwen2.5-0.5B | HQQ 4-bit | +13.0% | 1.0 GB → 0.78 GB | FAIL* |
| Qwen2.5-0.5B | GGUF Q4_K_M | +24.4% | 1.0 GB → 0.40 GB | FAIL* |
| DeepSeek-R1-Distill-1.5B | HQQ 8-bit (escalated) | **+0.2%** | 3.56 → 2.34 GB | PASS |

\* Honest misses at a strict 5% gate — the ladder exists precisely for this. Full matrix: [`docs/supported-models.md`](docs/supported-models.md).

## Features

- **Precision control** — pick 2 / 3 / 4 / 5 / 6 / 8-bit per run (CLI `--bits`, dashboard dropdown). GGUF maps widths to k-quants (`Q2_K`…`Q8_0`); a gate miss still escalates automatically.
- **5 backends** behind one interface — HQQ (incl. per-block mixed precision), GGUF, GPTQ (calibrated, checkpointed), AWQ (experimental), bitsandbytes. Each self-probes; unavailable ones are skipped with reasons, never crashes.
- **Measured eval** — perplexity (WikiText-2), decode tok/s, TTFT, peak memory, disk size, sanity checks. GGUF artifacts get native perplexity via embedded-tokenizer scoring.
- **Retry ladder + quality gates** — bounded attempts, every attempt recorded in `report.md`/`report.json`.
- **Agent mode** — `quantiv run --agent` runs an autonomous tool-calling loop (offline echo provider by default, OpenAI-compatible endpoint optional). `--no-agent` rules path always works with no API key.
- **Dashboard** (React + FastAPI) — `quantiv dashboard` → model picker (14 curated models), live plan preview with fit checks, progress tracking with stall watchdog, quality-vs-size chart, before→after certificates, one-click **artifact download (.zip)**.
- **Packaging** — manifest (versions, hashes, seeds, calibration hash), model cards, license-gated opt-in Hub upload. Gated/restrictive models are never published silently.
- **Reproducibility** — hashed calibration sets, recorded seeds, deterministic planner; reproducibility tests in CI.

## Quickstart

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e ".[dev,core-ml,hqq,calib,web]"   # CPU starter stack (wheels only)

quantiv doctor                                     # env + dependency report
quantiv analyze Qwen/Qwen2.5-0.5B-Instruct         # arch, license, memory estimates
quantiv plan Qwen/Qwen2.5-0.5B-Instruct --target rtx3060-12gb --goal balanced
quantiv run HuggingFaceTB/SmolLM2-135M --bits 4 --goal balanced --max-attempts 3
quantiv dashboard                                  # web UI on http://127.0.0.1:8020
```

Full CUDA stack (GPTQ/bnb + GPU eval): install `torch` from the cu130 index, then
`uv pip install -e ".[gptq,awq,bnb]"`. See [`docs/supported-models.md`](docs/supported-models.md) for verified versions.

## CLI reference

| Command | What it does |
|---|---|
| `quantiv analyze <model>` | Profile a model (Hub id or path); JSON with `--output` |
| `quantiv plan <model> --target --goal` | Scored, ranked strategies with VRAM-fit check |
| `quantiv run <model> [--method auto] [--bits 4] [--goal balanced] [--max-attempts 3] [--agent]` | End-to-end pipeline with retry ladder |
| `quantiv eval <quantized> --baseline <model>` | Standalone comparison report (HF or GGUF) |
| `quantiv compare <run_a> <run_b>` | Side-by-side table of two runs |
| `quantiv package <run_dir> [--push-to-hub --repo-id …]` | Validate, write model card, license-gated upload |
| `quantiv dashboard [--port 8020]` | Web UI: jobs, progress, chart, certificates, downloads |
| `quantiv doctor` | Environment, drivers, dependency report |

## Verified backend choices (checked Oct 2026)

| Backend | Verdict |
|---|---|
| GPTQ | **`gptqmodel`** (ModelCloud). `AutoGPTQ` is unmaintained/deprecated upstream. |
| AWQ | **Experimental.** `AutoAWQ` deprecated; `llmcompressor` 0.14's tracer is incompatible with transformers 5.x forwards; gptqmodel AWQ kernels need a CUDA-toolkit JIT build. |
| torchao | Active (0.18.0). Not yet wrapped — planned backend. |
| bitsandbytes / HQQ | Fast baselines (8-bit/NF4, calibration-free). bnb needs CUDA. |
| CUDA torch | cu130 wheels (`2.14.x+cu130`) for Blackwell GPUs; CPU fallback works everywhere. |

## Repo layout

```
src/quantiv/
  cli/          analyze/plan/run/eval/compare/package/doctor/dashboard
  agent/        typed tools, echo + OpenAI-compatible providers, bounded orchestrator
  analyzer/     model intake, license flags, memory estimates
  hardware/     profiler + named target device profiles
  planner/      ranking, sensitivity, mixed precision, retry pipeline
  quantizers/   hqq / gguf / gptq / awq / bnb (+ registry with self-probing)
  calibration/  hashed, cached, seeded calibration sets
  evaluation/   perplexity, speed, memory, sanity (HF + GGUF-native)
  packaging/    manifest, reports, model cards, license gate
  dashboard/    FastAPI app (jobs, watchdog, plan API, downloads)
frontend/       React + Vite console (served by the dashboard)
configs/        devices.yaml + default.yaml (quality gates)
tests/          50 tests, CPU-green, GPU-gated skips
docker/         CPU + CUDA images
docs/           support matrix · examples/quickstart.py
```

## Hard rules (enforced)

1. No fabricated metrics — every number comes from a real run.
2. Reproducibility — seeds, hashes, versions, source revision in the manifest.
3. License safety — never publish gated/restrictive models without explicit opt-in.
4. Fail loudly with suggested fixes; bounded retries; honest limitations (see matrix).

## License

Apache-2.0. Quantized artifacts retain their source model's license — check `quantiv analyze` output before redistributing.
