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

WikiText-2 test, 8 samples. Baseline (bf16): **ppl 21.466, 19.8 tok/s, 0.27 GB**.

| Backend | Quant | PPL | ΔPPL | tok/s | Disk | Gate (5%) |
|---|---|---|---|---|---|---|
| HQQ | 4bit-g64 | 28.136 | +40.7% | 5.48 | 0.18 GB | FAIL |
| GGUF (llama.cpp v0.6.0) | Q4_K_M | 29.761 | +38.6% | 23.47 | 0.11 GB | FAIL |

## Backend × architecture matrix

| Backend | Llama | Qwen2 | Mistral | Gemma | Phi | Needs |
|---|---|---|---|---|---|---|
| HQQ | ✅ | ✅ | 🧪 | 🧪 | 🧪 | torch (CPU ok) |
| GGUF k-quants | ✅ | ✅ | 🧪 | 🧪 | 🧪 | llama-cpp build + converter arch support |
| bitsandbytes 8-bit/NF4 | 🧪 | 🧪 | 🧪 | 🧪 | 🧪 | CUDA GPU |
| GPTQ | Phase 3 | Phase 3 | Phase 3 | Phase 3 | Phase 3 | CUDA + calibration |
| AWQ | Phase 3 | Phase 3 | Phase 3 | Phase 3 | Phase 3 | CUDA + calibration |

## Notes

- Custom-code models (`trust_remote_code`) are refused by default; explicit opt-in flag planned.
- `qmodel.pt` (HQQ) is a torch pickle — only load artifacts you produced yourself.
- Baselines reproduce across runs (Qwen bf16 ppl 18.534 in both runs above).
