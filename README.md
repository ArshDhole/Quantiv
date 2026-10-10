# Quantiv — auto-quantize open-source LLMs

> **Status: v1.0 — all phases complete.** See `docs/supported-models.md` for measured results and `CHANGELOG.md` for history.

Quantiv takes a model (HF repo ID or local path) + a goal ("run on 8GB GPU", "smallest size <2% loss") and produces a **verified, quantized, ready-to-run artifact** with real measured quality/speed/memory numbers. An LLM-driven planner orchestrates deterministic quantization tooling — the agent decides, the tools do the math, and **no metric is ever fabricated**.

## What works (v1.0)

- `quantiv analyze` — arch, params, dtype, license, tokenizer, memory estimates (Hub or local)
- `quantiv plan` — scored, ranked candidates with VRAM-fit check
- `quantiv run` — end-to-end with **bounded retry ladder** (higher bits → next method → mixed precision) and gate verdict
- `quantiv run --agent` — autonomous tool-calling loop (offline echo provider default; OpenAI-compatible optional)
- `quantiv eval` / `compare` — measured side-by-side reports (HF + GGUF artifacts)
- `quantiv package` — MODEL_CARD.md + manifest, license-gated opt-in Hub upload
- `quantiv dashboard` — web UI: submit jobs, watch progress, fetch reports
- Backends: **HQQ** (incl. per-block mixed precision), **GGUF** (Q2_K..Q8_0), **GPTQ** (calibrated, checkpointed), **AWQ** (experimental), **bitsandbytes** (8-bit/NF4, CUDA)
- `--no-agent` rules path always works with no API key

## Quickstart

```powershell
# Python 3.11 required (3.14 wheels don't exist yet for torch)
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"

# or minimal (no torch yet — analyzer/hardware/CLI only):
uv pip install -e .

quantiv doctor
quantiv analyze TinyLlama/TinyLlama-1.1B-Chat-v1.0
quantiv analyze ./path/to/model --output report.json
quantiv plan TinyLlama/TinyLlama-1.1B-Chat-v1.0 --target rtx3060-12gb --goal balanced
```

## Verified backend choices (checked Oct 2026)

| Backend | Verdict |
|---|---|
| GPTQ | **Use `gptqmodel` (ModelCloud/GPTQModel)**. `AutoGPTQ` is unmaintained/deprecated upstream. |
| AWQ | **`AutoAWQ` is officially deprecated** (PyPI notice, last tested torch 2.6 / transformers 4.51.3). Use `llm-compressor` (vLLM project) or GPTQModel AWQ kernels. |
| torchao | Active (`0.18.0`, Aug 2026). PyTorch-native int4/int8/float8. |
| lm-eval | Package `lm-eval`, base install no longer bundles torch/transformers — install `lm_eval[hf]`. CLI is now `lm-eval run / ls / validate`. |
| bitsandbytes / HQQ | Still the fast Phase-2 baselines (8-bit/NF4, calibration-free). |

## Repo layout

```
src/quantiv/
  cli/          Typer CLI (analyze/plan/run/eval/compare/package/doctor)
  agent/        orchestrator stub (Phase 5)
  analyzer/     model intake + architecture detection (Phase 1 done)
  hardware/     profiler + target device profiles (Phase 1 done)
  planner/      rules-engine stub (Phase 4)
  quantizers/   Quantizer plugin interface (Phase 2+)
  calibration/  calibration-data manager stub (Phase 3)
  evaluation/   eval-suite stub (Phase 2+)
  packaging/    manifest / model-card stub (Phase 6)
  utils/        logging + Pydantic/YAML config
configs/        devices.yaml + default.yaml (quality gates)
tests/          unit tests (tiny models, no GPU required)
examples/       quickstart.py
docker/         CPU + CUDA images (Phase 7)
docs/
```

## Hard rules (enforced)

1. No fabricated metrics — every number comes from a real run.
2. Reproducibility — seeds, hashes, versions, source revision in manifest.
3. License safety — never auto-upload gated/restrictive models.
4. Fail loudly with suggested fixes; testability; resource awareness; honest limitations.

## License

Apache-2.0. Quantized artifacts retain their source model's license — check `quantiv analyze` output before redistributing.
