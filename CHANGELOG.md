# Changelog

## v1.0 (final)
- Complete pipeline: analyze → plan (ranked) → quantize → evaluate → gate → escalate → report → package
- Backends: HQQ, GGUF (Q2_K–Q8_0), GPTQ (gptqmodel), AWQ (gptqmodel engine), bitsandbytes (8-bit/NF4)
- Calibration manager (hashed, cached WikiText sets); GPTQ/AWQ checkpoints where platform-supported
- Mixed precision (sensitivity-driven, per-block) + bounded retry ladder with first gate PASS demonstrated
- Agent layer: typed tools, echo (offline) + OpenAI-compatible providers, guardrails; `--agent` works
- Packaging: manifest, Markdown/JSON reports with attempts, model cards, license gate, opt-in Hub upload
- Dashboard (FastAPI + SQLite): submit jobs, poll progress, fetch reports
- CI (Linux/Windows × py3.11/3.12), CPU + CUDA Dockerfiles, support matrix, reproducibility tests
- Known limits: AWQ parked experimental (llmcompressor tracer vs transformers 5.x; JIT kernel needs toolkit build);
  GGUF perplexity uses embedded tokenizer (approximate delta); bnb/GPTQ/AWQ need CUDA

## v0.6 — Phase 6: model cards, license gate, compare/package, dashboard
## v0.5 — Phase 5: agent layer (tools, providers, guardrails, working --agent)
## v0.4 — Phase 4: ranked planner, sensitivity, mixed precision, retry ladder
## v0.3 — Phase 3: CUDA stack, calibration manager, GPTQ verified on GPU
## v0.2 — Phase 2: HQQ/GGUF/bnb backends, measured eval, end-to-end run + Qwen showcase
## v0.1 — Phase 1: CLI, analyzer, hardware profiler, rules planner, 15 tests green
