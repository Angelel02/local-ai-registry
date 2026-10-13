# Recipe: Running Local AI on an NVIDIA GTX 1080 Ti (11 GB)

**Card:** GTX 1080 Ti — 11 GB GDDR5X, Pascal (SM 6.1), ~484 GB/s bandwidth
**Best for:** 7B at Q4/Q5, small 8B at Q4, SD 1.5 image gen
**Avoid:** anything ≥13B, FP16/BF16 mixed precision, modern attention kernels that require SM 7.0+

---

## 0. Pascal reality check (read this first)

The 1080 Ti is the **oldest** card in this registry and the most constrained:

- **No Tensor Cores** — no FP16/TF32 acceleration. Compute is pure FP32 (CUDA cores).
- **SM 6.1** — many modern kernels (FlashAttention-2, some Triton builds, bitsandbytes 4-bit CUDA kernels) require **SM 7.0+ (Volta) or newer** and will either refuse to build or silently fall back.
- **GGUF / llama.cpp is the only sane path.** Avoid `bitsandbytes` 4-bit (needs SM 7.5+ for the fast kernels) and avoid vLLM/TensorRT-LLM (need SM 7.5+).
- **Good news:** for a single 7B model at Q4, CUDA cores are plenty — you'll get usable interactive speed.

**Bottom line:** use **llama.cpp / Ollama with GGUF quants**, build with `-DCMAKE_CUDA_ARCHITECTURES=61`.

---

## 1. What this card can actually run

| Model size | Quant | VRAM (weights) | Fits 11 GB? | Notes |
|---|---|---|---|---|
| 7B | Q4_K_M | ~4.4 GB | ✅ comfy | ~4–8k ctx, usable speed |
| 7B | Q5_K_M | ~5.1 GB | ✅ comfy | better quality |
| 7B | Q8_0 | ~7.6 GB | ✅ ok | best 7B quality, ctx ≤ 4k |
| 8B | Q4_K_M | ~5.0 GB | ✅ ok | Llama-3.1-8B class, ctx ≤ 4k |
| 13B | Q4_K_M | ~7.9 GB | ⚠️ tight | ctx ≤ 2k, slow (~8–12 tok/s) |
| 13B | Q5 | ~9.2 GB | ❌ | KV cache pushes it over |
| 30B+ | any | — | ❌ | no |

**Rule:** weights + KV cache + ~0.8 GB CUDA overhead ≤ 11 GB. Without Tensor Cores, **quantize aggressively (Q4_K_M)** — it's the only way to keep speed acceptable.

---

## 2. Recommended stack: llama.cpp (best for Pascal)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
# CRITICAL: target Pascal exactly
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build --config Release -j$(nproc)

# 7B Q4_K_M, all layers on GPU, 8k ctx
./build/bin/llama-cli \
  -m models/llama-3.1-8b-instruct-q4_K_M.gguf \
  -ngl 99 -c 8192 -b 256
```

**Do NOT add `--flash-attn`** on SM 6.1 — the FA kernels target newer architectures; on Pascal it either fails to build or is slower than the default path. Leave it off.

If you OOM, reduce `-ngl` (e.g. `-ngl 28`) to partially offload — Pascal's PCIe bandwidth is fine for partial offload.

---

## 3. Alternative: Ollama (simplest)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-q4_K_M

# Force full GPU offload, modest context
OLLAMA_NUM_GPU=40 OLLAMA_CONTEXT_LENGTH=4096 ollama run llama3.1:8b-instruct-q4_K_M
```

Ollama auto-detects Pascal; if it falls back to CPU, set `OLLAMA_LLM_LIBRARY=cuda_v11` (Pascal works best with the CUDA 11 build in some versions).

---

## 4. Tuning for 11 GB on Pascal

1. **Q4_K_M is the default choice.** Q8 only for 7B with short context.
2. **Cap context hard (`-c 4096`).** KV cache is FP16 by default; on Pascal you want it small. For 7B you can afford 8k; for 8B keep 4k.
3. **Quantize the KV cache** to save VRAM: `--cache-type-k q8_0 --cache-type-v q8_0` (supported in recent llama.cpp even on Pascal).
4. **Batch size 256** (not 512) — Pascal has less L2 and lower occupancy; smaller batches avoid stalls.
5. **No FlashAttention, no bitsandbytes, no vLLM.** SM 6.1 doesn't support their kernels.
6. **Headless if possible** — reclaim ~0.5 GB from the display.

---

## 5. Image generation (this card's strong suit)

```bash
# SD 1.5 fp16 ≈ 4 GB — runs great
python launch.py --xformers   # xformers works on Pascal

# SDXL ≈ 7 GB — needs --medvram
python launch.py --medvram --xformers
```

Pascal has **no BF16** — always use **fp16** (or fp32 for stability). SD 1.5 is the sweet spot; SDXL is possible but slower.

---

## 6. Verified expectations (realistic numbers)

| Workload | Throughput |
|---|---|
| 7B Q4_K_M, 8k ctx, full offload | ~18–24 tok/s |
| 7B Q5_K_M, 4k ctx | ~16–20 tok/s |
| 8B Q4_K_M, 4k ctx | ~15–19 tok/s |
| 13B Q4_K_M, 2k ctx | ~8–12 tok/s |
| SD 1.5 512², 25 steps | ~4–6 s/image |

*(Roughly half the tok/s of a 2080 Ti — Pascal has no Tensor Cores and lower bandwidth.)*

---

## TL;DR
- **Best daily driver:** `llama3.1:8b-instruct-q4_K_M`, `-ngl 99 -c 4096 -b 256`, **no flash-attn**.
- **Build flag matters:** `-DCMAKE_CUDA_ARCHITECTURES=61` (Pascal).
- **Avoid:** bitsandbytes, vLLM, TensorRT-LLM, FlashAttention — all need SM 7.0+.
- **Best use:** 7B chat + SD 1.5 image gen. Skip 13B+.
- Pascal = FP32 CUDA cores + GGUF Q4. No Tensor Cores, no BF16, no FP8.

*Verified on GTX 1080 Ti 11 GB, CUDA 12.x, llama.cpp b3xxx.*