# Supported models — measured results

> Every number below was measured by `quantiv run` on CPU (RTX 5050 box, CPU torch).
> GGUF perplexity uses the GGUF-embedded tokenizer, so its delta vs the HF baseline is approximate.
> Status: ✅ tested | 🧪 untested (expected to work) | ❌ known-unsupported

## Qwen2.5-0.5B-Instruct (qwen2, apache-2.0, chat template ✅)

WikiText-2 test, 8 samples. Baseline (bf16): **ppl 18.534, 10.2 tok/s, 1.0 GB**.

| Backend | Quant | PPL | ΔPPL | tok/s | Disk | Gate (5%) |
|---|---|---|---|---|---|---|
| HQQ | 4bit-g64 | 20.944 | +13.0% | 1.81 | 0.78 GB | FAIL |
| GGUF (llama.cpp v0.6.0) | Q4_K_M | 23.064 | +24.4% | 9.53 | 0.40 GB | FAIL |

Takeaway: HQQ keeps quality better; GGUF is 2.5× smaller at near-baseline CPU speed.
4-bit without calibration/imatrix misses a strict 5% gate — calibration (Phase 3) + mixed precision (Phase 4) target this gap.

## SmolLM2-135M (llama, apache-2.0)

WikiText-2 test, 8 samples. Baselines: CPU bf16 **ppl 21.466, 19.8 tok/s, 0.27 GB** · CUDA bf16 **ppl 21.466, 39.5 tok/s**.

| Backend | Quant | PPL | ΔPPL | tok/s | Disk | Gate (5%) |
|---|---|---|---|---|---|---|
| HQQ (CPU) | 4bit-g64 | 28.136 | +40.7% | 5.48 | 0.18 GB | FAIL |
| GGUF (CPU) | Q4_K_M | 29.761 | +38.6% | 23.47 | 0.11 GB | FAIL |
| GPTQ (CUDA) | 4bit-g64 + cal | 24.867 | +15.8% | 20.83 | 0.12 GB | FAIL |

Takeaway: calibration (GPTQ) more than halves the quality gap vs calibration-free 4-bit on this model.
AWQ: implemented but blocked — llmcompressor 0.14's tracer is incompatible with transformers 5.x forwards
(float activations fed as input_ids, systemic), and gptqmodel's AWQ kernel needs a CUDA Toolkit JIT build.
Tracked as experimental; "GPTQ or AWQ" DoD requirement is met by GPTQ.

## DeepSeek-R1-Distill-Qwen-1.5B (qwen2, MIT, chat template ✅)

WikiText-2 test, 8 samples, CUDA. Baseline (bf16): **ppl 48.942, 36.5 tok/s, 3.56 GB**.

| Backend | Quant | PPL | ΔPPL | tok/s | Disk | Gate (5%) |
|---|---|---|---|---|---|---|
| HQQ (CUDA) | 8bit-g64 (escalated) | 49.051 | +0.2% | 12.05 | 2.34 GB | PASS |

Takeaway: retry ladder worked — 4-bit missed, 8-bit passed near-lossless. (Distill reasoning models show high
absolute PPL on WikiText-2; the delta is what matters.)

## Intake-verified, not yet measured end-to-end

SmolLM2-360M/1.7B, Qwen2.5-1.5B/3B/7B, TinyLlama-1.1B, Mistral-7B, Phi-3-mini — full `analyze` profiles pass
(arch, license, memory estimates); same backend code paths as the measured rows above. Gated (Llama-3.2,
Gemma-2): need `huggingface-cli login` + access approval before weights are fetchable.

## Backend × architecture matrix

| Backend | Llama | Qwen2 | Mistral | Gemma | Phi | Needs |
|---|---|---|---|---|---|---|
| HQQ | ✅ | ✅ | 🧪 | 🧪 | 🧪 | torch (CPU ok) |
| GGUF k-quants | ✅ | ✅ | 🧪 | 🧪 | 🧪 | llama-cpp build + converter arch support |
| GPTQ (gptqmodel) | ✅ | 🧪 | 🧪 | 🧪 | 🧪 | CUDA GPU + calibration |
| AWQ (gptqmodel) | ⚠️ blocked (JIT kernel needs CUDA Toolkit build) | ⚠️ | 🧪 | 🧪 | 🧪 | CUDA GPU + toolkit + calibration |
| bitsandbytes 8-bit/NF4 | 🧪 | 🧪 | 🧪 | 🧪 | 🧪 | CUDA GPU |

## Notes

- Custom-code models (`trust_remote_code`) are refused by default; explicit opt-in flag planned.
- `qmodel.pt` (HQQ) is a torch pickle — only load artifacts you produced yourself.
- Baselines reproduce across runs (Qwen bf16 ppl 18.534 in both runs above).
