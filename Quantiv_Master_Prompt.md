# MASTER PROJECT PROMPT: Quantiv

> Paste this entire document into the AI coding agent that will build the project.

---

## 0. Your Role

You are a senior ML systems engineer and technical lead. You will design and build **Quantiv**, an open-source, agentic system that automatically quantizes open-source AI models (Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, etc.) and delivers an optimized, verified, ready-to-run artifact.

Work autonomously, in phases. After each phase: run tests, summarize what was done, list open issues, then proceed. Ask me a question only when a decision is truly blocked and cannot be resolved with a sensible default. State any default you pick.

---

## 1. Project Vision

A user gives Quantiv a model (a Hugging Face repo ID or local path) and a goal, for example:

- "Run this on an 8 GB GPU."
- "Make this run on CPU with the lowest latency."
- "Smallest possible size with under 2% quality loss."

Quantiv then, with minimal human input:

1. Inspects the model (architecture, size, dtype, license, tokenizer, special layers).
2. Decides on the best quantization strategy for the target hardware and goal.
3. Runs the quantization using the appropriate backend.
4. Evaluates the result against the original (quality, speed, memory).
5. Iterates if the result misses the target (for example, switches to mixed precision or a different method).
6. Produces the final artifact, a model card, and a benchmark report.

**The core idea: an LLM-driven planner/agent orchestrates deterministic quantization tooling.** The agent makes decisions; the tools do the numeric work. The agent must never fabricate benchmark numbers. All metrics must come from real evaluation runs.

---

## 2. Goals and Non-Goals

### Goals
- Support multiple quantization methods behind a single unified interface.
- Automatically select a method and bit-width from model + hardware + goal.
- Quantitatively verify quality loss, not just "it runs."
- Be reproducible: every run emits a config, logs, and a manifest with versions and hashes.
- Be usable via CLI first, then a Python API, then optionally a web dashboard.
- Run on modest hardware where possible, including a single consumer GPU or CPU-only with a layer-by-layer (streaming) approach for large models.

### Non-Goals (for v1)
- Training or fine-tuning new models from scratch.
- Inventing a new quantization algorithm (wrap and orchestrate existing ones; research extensions come later).
- Serving infrastructure at scale (a simple local test server is fine).

---

## 3. Supported Backends (via a plugin interface)

Implement each as a plugin behind a common `Quantizer` interface. Before implementing each, **check the current upstream docs and versions** and pin them. Do not rely on memory for APIs.

| Family | Typical use | Notes |
|---|---|---|
| GGUF (llama.cpp k-quants / imatrix) | CPU, Apple Silicon, mixed CPU/GPU | Convert HF -> GGUF, then quantize (Q2_K to Q8_0, IQ-quants with importance matrix) |
| GPTQ | GPU inference, 3/4/8-bit | Needs calibration data |
| AWQ | GPU inference, 4-bit | Activation-aware, needs calibration data |
| bitsandbytes | Quick 8-bit / NF4 | Good for fast baselines and QLoRA workflows |
| HQQ | Calibration-free quantization | Fast, no data needed |
| torchao | PyTorch-native int4/int8/float8 | Good for PyTorch deployment |
| ExLlama-style (EXL2/EXL3) | High-speed GPU inference | Optional, stretch goal |
| FP8 / INT8 (SmoothQuant-style) | Datacenter GPUs | Optional, stretch goal |

Backends can be added later without touching core code.

---

## 4. System Architecture

```
                 +--------------------------------------+
   User/CLI ---> |           Orchestrator Agent         |
                 |  (LLM planner + tool-calling loop)   |
                 +----------------+---------------------+
                                  |
        +-----------+-------------+-------------+--------------+
        |           |             |             |              |
   Model Intake  Hardware     Strategy      Quantization   Evaluation
   & Analyzer    Profiler     Planner       Engine         Suite
        |           |             |             |              |
   HF Hub/local  GPU/CPU/RAM  rules +      backend plugins  perplexity,
   arch detect   detection    heuristics   (GGUF, GPTQ,     lm-eval tasks,
   license check              + LLM        AWQ, HQQ, ...)   speed, memory
                                  |             |              |
                                  +------+------+--------------+
                                         |
                                 Packaging & Reporting
                                 (model card, manifest, HF upload)
```

### Components

**4.1 Model Intake & Analyzer**
- Accept HF repo ID, local directory, or safetensors/GGUF file.
- Detect architecture family, parameter count, dtype, layer structure, MoE vs dense, vocab size, context length, tokenizer, chat template.
- Read the **license** and flag restrictive or gated licenses (for example Llama community license terms) before redistribution. Never auto-upload a model whose license forbids it.
- Estimate memory needs for each candidate bit-width before running.

**4.2 Hardware Profiler**
- Detect GPU (vendor, VRAM, compute capability), CPU (cores, AVX2/AVX512/NEON), RAM, disk.
- Allow the user to specify a *target* device profile that differs from the build machine (for example "build on A100, target RTX 3060 12GB").

**4.3 Strategy Planner**
- Input: model profile + hardware profile + user goal + constraints.
- Output: ranked list of candidate plans (method, bits, group size, calibration set, mixed-precision map).
- Hybrid approach: a deterministic rules engine produces candidates; the LLM agent reasons over them, explains tradeoffs, and picks. The agent may propose changes, but they must pass schema validation.

**4.4 Quantization Engine**
- Unified `Quantizer` interface: `prepare()`, `calibrate()`, `quantize()`, `save()`, `validate_loadable()`.
- Layer-wise / streaming execution so models larger than VRAM can be processed.
- Support **mixed precision**: per-layer sensitivity analysis (for example measure loss change per layer when quantized) and assign higher bits to sensitive layers (embeddings, lm_head, first/last layers, attention output projections, etc.).
- Checkpointing and resume for long jobs.

**4.5 Calibration Data Manager**
- Default calibration sets (for example a general-text corpus slice plus code and multilingual slices as appropriate); user-supplied data allowed.
- Calibration set selection must be logged and reproducible (seed + dataset hash).

**4.6 Evaluation Suite**
- **Quality:** perplexity on a held-out set (for example WikiText-style), plus a configurable subset of `lm-evaluation-harness` tasks, plus optional KL divergence of output distributions vs the original.
- **Performance:** tokens/sec (prefill and decode), time-to-first-token, peak VRAM/RAM, model size on disk.
- **Sanity tests:** model loads, generates coherent text on a fixed prompt set, chat template works, no NaNs/Infs.
- Output a side-by-side comparison against the original and against other candidate quantizations.
- Define **quality gates** (for example max allowed perplexity increase). If a gate fails, the agent must iterate (higher bits, different method, mixed precision) up to a bounded number of attempts.

**4.7 Packaging & Reporting**
- Output directory with: quantized weights, tokenizer, config, `quantiv_manifest.json` (tool versions, commit hashes, seeds, calibration hash, source model revision, hardware used), benchmark report (Markdown + JSON), and an auto-generated **model card** that honestly reports measured results and the source model's license.
- Optional upload to Hugging Face Hub (opt-in only, with the license check passing).

**4.8 Orchestrator Agent**
- Tool-calling loop with a strict, typed tool set (analyze, profile_hardware, plan, quantize, evaluate, compare, package).
- Guardrails: bounded iterations, cost/time budget, no arbitrary shell execution outside the defined tools, no invented numbers.
- Must be **model-agnostic**: configurable LLM provider (a hosted API or a local model) via a simple adapter, so the planner can itself run offline.
- Also ship a **non-LLM mode** (`--no-agent`) that uses only the rules engine, so the tool works without any API key.

---

## 5. Interfaces

### CLI (primary, build first)
```
quantiv analyze  <model>
quantiv plan     <model> --target "rtx3060-12gb" --goal "min-size" --max-ppl-increase 0.05
quantiv run      <model> --target ... --goal ... [--method auto|gguf|gptq|awq|hqq|bnb]
quantiv eval     <quantized_path> --baseline <model>
quantiv compare  <run_a> <run_b>
quantiv package  <run_dir> [--push-to-hub]
quantiv doctor              # checks environment, drivers, dependencies
```

### Python API
```python
from quantiv import Quantiv
qf = Quantiv()
result = qf.run("meta-llama/Llama-3.1-8B-Instruct", target="rtx3060-12gb", goal="balanced")
print(result.report)
```

### Web Dashboard (phase 6, optional)
- Submit jobs, watch progress and logs, compare runs in tables and charts, download artifacts.

---

## 6. Tech Stack (defaults; justify any change)

- **Language:** Python 3.11+
- **Core libs:** PyTorch, Hugging Face `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `datasets`
- **Quantization libs:** llama.cpp (via subprocess/bindings), AutoGPTQ/GPTQModel, AutoAWQ, bitsandbytes, HQQ, torchao (verify which are currently maintained and pick the maintained forks)
- **Eval:** `lm-evaluation-harness`
- **CLI:** Typer or Click; **config:** Pydantic + YAML
- **Job state:** SQLite (local) to start
- **Web (optional):** FastAPI + a lightweight frontend
- **Packaging/CI:** `uv` or `poetry`, `pytest`, `ruff`, `mypy`, GitHub Actions, Docker images (CPU and CUDA variants)

Pin all versions. Check upstream compatibility matrices (CUDA, PyTorch, backend versions) before locking.

---

## 7. Repository Structure

```
quantiv/
  README.md
  pyproject.toml
  src/quantiv/
    cli/
    agent/            # orchestrator, tools, prompts, guardrails
    analyzer/         # model intake and architecture detection
    hardware/         # profiler and target device profiles
    planner/          # rules engine + candidate ranking
    quantizers/       # base.py + gguf.py, gptq.py, awq.py, hqq.py, bnb.py, torchao.py
    calibration/
    evaluation/       # perplexity, lm_eval wrapper, speed, memory, sanity
    packaging/        # model card, manifest, hub upload
    utils/
  tests/              # unit + integration (tiny models)
  examples/
  docs/
  docker/
  configs/            # default device profiles, quality gates
```

---

## 8. Phased Roadmap

Complete each phase with working code, tests, and docs before moving on.

**Phase 1: Foundation**
Repo scaffold, CLI skeleton, config system, logging, `doctor` command, model analyzer, hardware profiler. Tests using a tiny model.

**Phase 2: First backends and baseline eval**
Implement bitsandbytes (8-bit/NF4), HQQ, and GGUF backends. Implement perplexity + speed + memory evaluation. End-to-end `run` on a small model (for example a ~0.5B-1B parameter model) with a real report.

**Phase 3: Calibration-based methods**
Add GPTQ and AWQ with the calibration data manager. Streaming/layer-wise execution and checkpoint/resume.

**Phase 4: Planner and mixed precision**
Rules engine, candidate ranking, per-layer sensitivity analysis, mixed-precision assignment, quality gates with bounded retry loop.

**Phase 5: Agent layer**
LLM-driven orchestrator with typed tools, guardrails, provider adapter, and `--no-agent` fallback. Evaluate the agent: does it beat the rules-only planner on a benchmark set of models/goals?

**Phase 6: Packaging, model cards, dashboard**
Manifest, model card generator, license checks, opt-in Hub upload, optional web UI.

**Phase 7: Hardening and release**
Docker images, CI, docs site, example gallery, reproducibility test (same inputs produce same outputs within tolerance), v0.1.0 release.

**Stretch:** EXL2/EXL3, FP8, MoE-aware quantization, vision-language models, speculative-decoding-friendly drafts, QAT hooks, automatic regression benchmarks across new model releases.

---

## 9. Hard Requirements and Rules

1. **No fabricated metrics.** Every number in a report comes from an actual run, and the report records how it was measured.
2. **Reproducibility.** Seeds, dataset hashes, tool versions, and source model revision are recorded in the manifest.
3. **License safety.** Detect and surface model licenses. Never publish without an explicit opt-in and a passing license check.
4. **Verify before use.** Check current docs for every external library's API before coding against it. Do not assume APIs from memory.
5. **Fail loudly and helpfully.** Clear errors for unsupported architectures, insufficient memory, or driver mismatches, with suggested fixes.
6. **Safe execution.** The agent can only call defined tools. No arbitrary shell commands, no network calls outside model/dataset download and optional Hub upload.
7. **Testability.** Every module has unit tests; integration tests use tiny models so CI stays fast. Mark GPU-only tests so they skip gracefully.
8. **Resource awareness.** Estimate memory before running, and offload or stream when needed instead of crashing.
9. **Clean code.** Type hints, docstrings, linting, small modules, no dead code.
10. **Honest limitations.** Document which architectures and methods are supported, tested, or experimental.

---

## 10. Definition of Done (v0.1.0)

- `quantiv run <hf-model> --target <device> --goal <goal>` produces a quantized model plus a report with real, measured quality/speed/memory numbers versus the original.
- At least 4 backends working (GGUF, bitsandbytes, HQQ, plus GPTQ or AWQ) on at least 3 model families (for example Llama, Mistral, Qwen).
- Planner picks a reasonable method automatically, and quality gates trigger retries when violated.
- Agent mode and `--no-agent` mode both work.
- CI green, Docker images build, docs and examples published.
- A reproducibility test passes.

---

## 11. How to Start

1. Restate your understanding of the project in a few sentences and list any assumptions.
2. Propose the final repo layout and dependency list (after verifying current library status).
3. Begin **Phase 1** immediately. Deliver code, tests, and a short summary, then continue to Phase 2 without waiting unless blocked.

Be pragmatic: get a thin end-to-end slice working early, then deepen it.
