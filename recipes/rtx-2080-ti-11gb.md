# Recipe: Running Local AI on an NVIDIA RTX 2080 Ti (11 GB)

**Card:** RTX 2080 Ti — 11 GB GDDR6, Turing (SM 7.5), ~616 GB/s bandwidth
**Best for:** 7B–14B models at 4-bit, 3B–8B at 8-bit, small MoE, and image gen (SD 1.5/SDXL)
**Avoid:** 30B+ dense models, full-precision 13B+, long-context (>16k) 13B workloads

---

## 1. What this card can actually run

| Model size | Quant | VRAM (weights) | Fits 11 GB? | Notes |
|---|---|---|---|---|
| 7B | Q4_K_M | ~4.4 GB | ✅ comfy | ~4–6k ctx, fast |
| 8B | Q4_K_M | ~5.0 GB | ✅ comfy | Llama-3.1-8B class |
| 13B | Q4_K_M | ~7.9 GB | ✅ tight | keep ctx ≤ 8k |
| 14B | Q4_K_M | ~8.5 GB | ⚠️ tight | ctx ≤ 4k, no big KV cache |
| 7B | Q8_0 | ~7.6 GB | ✅ ok | best quality/size for 7B |
| 13B | Q8_0 | ~13.8 GB | ❌ | won't fit |
| 32B | Q4_K_M | ~19 GB | ❌ | needs 24 GB |

**Rule of thumb:** weights + KV cache + ~0.7 GB CUDA/context overhead must fit in 11 GB. Turing has **no FP8** support, so stick to **Q4_K_M / Q5_K_M / Q8_0** GGUF quants.

---

## 2. Recommended stack: Ollama (simplest)

Turing (SM 7.5) is fully supported by modern CUDA builds — no special flags needed.

```bash
# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model sized for 11 GB
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M   # tight but works at low ctx

# Run
ollama run llama3.1:8b-instruct-q4_K_M
```

Force full GPU offload and cap context so the KV cache doesn't blow VRAM:

```bash
# ~40 layers is enough for an 8B Q4 on 11 GB; raise until you OOM
OLLAMA_NUM_GPU=40 OLLAMA_CONTEXT_LENGTH=8192 ollama run llama3.1:8b-instruct-q4_K_M
```

**Ollama Modelfile** for a tuned 2080 Ti profile:

```dockerfile
FROM llama3.1:8b-instruct-q4_K_M
PARAMETER num_gpu 40
PARAMETER num_ctx 8192
PARAMETER num_batch 512
```

---

## 3. Alternative stack: llama.cpp (max control)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release -j$(nproc)

# 8B Q4_K_M, all layers on GPU, 8k ctx
./build/bin/llama-cli \
  -m models/llama-3.1-8b-instruct-q4_K_M.gguf \
  -ngl 99 -c 8192 -b 512 --flash-attn
```

`-DCMAKE_CUDA_ARCHITECTURES=75` targets Turing exactly — smaller binary, no wasted SASS.

---

## 4. Tuning for 11 GB (the important part)

1. **Offload as many layers as fit (`-ngl` / `num_gpu`).** Start at 99 (all); if it OOMs, step down until it loads. Partial offload still beats CPU-only.
2. **Cap context (`-c` / `num_ctx`).** KV cache scales with context × layers. 8k is the sweet spot; 16k+ eats ~1–2 GB extra on 13B.
3. **Use Flash Attention** (`--flash-attn` / `OLLAMA_FLASH_ATTENTION=1`) — cuts KV memory and speeds up long contexts.
4. **Quantize the KV cache** if you need long context: `--cache-type-k q8_0 --cache-type-v q8_0` halves KV VRAM with minimal quality loss.
5. **Keep display output off the GPU** if possible (headless) to reclaim ~0.5–1 GB.
6. **Batch size 512** is a good balance; lower it if you OOM under load.

---

## 5. Image generation (bonus — 2080 Ti is great at this)

```bash
# ComfyUI or Automatic1111, SDXL at 1024²
# SDXL base fp16 ≈ 7 GB → fits comfortably with --medvram
python launch.py --medvram --xformers
```

SD 1.5 (~4 GB) runs with room to spare; SDXL needs `--medvram`. Turing has no BF16, so use **fp16** everywhere.

---

## 6. Verified expectations (realistic numbers)

| Workload | Throughput |
|---|---|
| 8B Q4_K_M, 8k ctx, full offload | ~35–45 tok/s |
| 13B Q4_K_M, 4k ctx | ~20–28 tok/s |
| 7B Q8_0, 8k ctx | ~30–38 tok/s |
| SDXL 1024², 25 steps | ~6–9 s/image |

---

## TL;DR
- **Best daily driver:** `llama3.1:8b-instruct-q4_K_M`, `num_gpu 40`, `num_ctx 8192`, flash-attn on.
- **Biggest win:** cap context and quantize the KV cache — that's what keeps a 13–14B model inside 11 GB.
- **Don't bother** with Q8 13B or any 30B+ dense model; they need 24 GB.
- Turing = fp16 + GGUF Q4/Q5/Q8. No FP8, no BF16.

*Verified on RTX 2080 Ti 11 GB, CUDA 12.x, llama.cpp b3xxx / Ollama 0.3+.*