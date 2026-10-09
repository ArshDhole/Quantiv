# Quantiv — auto-quantize open-source LLMs

> **Status: Phase 2 (first backends + baseline eval) done.** See `Quantiv_Master_Prompt.md` for the full project vision.

Quantiv takes a model (HF repo ID or local path) + a goal ("run on 8GB GPU", "smallest size <2% loss") and produces a **verified, quantized, ready-to-run artifact** with real measured quality/speed/memory numbers. An LLM-driven planner orchestrates deterministic quantization tooling — the agent decides, the tools do the math, and **no metric is ever fabricated**.

## What works today (v0.2)

- `quantiv doctor` — environment / driver / dependency checks
- `quantiv analyze <model>` — arch, params, dtype, license, tokenizer, memory estimates (local dir or HF Hub, no full weight download)
- `quantiv run <model> --method hqq|gguf|bnb|auto` — end-to-end: quantize, evaluate baseline + quantized (perplexity on WikiText-2, tokens/sec, peak memory, disk size, sanity), write `report.md`/`report.json` + manifest with quality-gate verdict
- `quantiv eval <quantized> --baseline <model>` — standalone comparison report (HF or GGUF artifacts)
- `quantiv plan` — ranked candidates via rules engine; `compare`/`package` still stubs (Phases 4/6)
- Backends: **HQQ** (4-bit, CPU+CUDA), **GGUF** (Q2_K..Q8_0 via pinned llama.cpp v0.6.0 toolkit + built bindings), **bitsandbytes** (8-bit/NF4, CUDA-only — correctly reports unavailable on CPU boxes)
- `--no-agent` rules-engine path is the default (agent layer lands in Phase 5)

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
4. See `Quantiv_Master_Prompt.md` §9 for the full list.

## License

Apache-2.0. Quantized artifacts retain their source model's license — check `quantiv analyze` output before redistributing.
