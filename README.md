# Local LLM Playground — Mac Mini M4 (16GB)

Running open-weight LLMs locally with TurboQuant KV cache compression.

## TurboQuant Fused Kernel

Custom 4-bit fused Metal attention kernel implementing Google's [TurboQuant](https://arxiv.org/abs/2504.19874) paper for Apple Silicon.

### Origin

Our fused kernel is adapted from [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx)'s `fused_tq_attention_norot` kernel ([`turboquant/kernels.py`, lines 718-865](https://github.com/sharpner/turboquant-mlx/blob/master/turboquant/kernels.py#L718-L865)). That original kernel supports 2-bit quantization with `head_dim=128`. We extended it to 4-bit with `head_dim=256` for modern architectures.

**What we changed from the original kernel:**

| Parameter | Original | Ours |
|---|---|---|
| Quantization | 2-bit (mask `0x3`, 16 per uint32) | 4-bit (mask `0xF`, 8 per uint32) |
| Head dimension | D=128 (hardcoded) | D=256 (hardcoded) |
| Dims per lane | 4 (128/32 threads) | 8 (256/32 threads) |
| Simdgroups | 32 (16 KB threadgroup mem) | 16 (16 KB, fits 32 KB Apple limit) |
| Centroids | 4 (2-bit codebook) | 16 (4-bit Lloyd-Max codebook) |
| Threadgroup memory | `tg_acc[32*128]` = 16 KB | `tg_acc[16*256]` = 16 KB |

### How it works with Qwen 3.5 (hybrid architecture)

Qwen 3.5 9B is a **hybrid model** — it alternates between two types of layers:

- **24 GatedDeltaNet (SSM) layers**: Use a fixed-size recurrent state. No KV cache. Already memory-efficient.
- **8 Standard Attention layers** (3, 7, 11, 15, 19, 23, 27, 31): Use a growing KV cache. This is where TurboQuant compresses.

We build a **mixed cache** that applies TurboQuant only where it helps:

```
Layer  0: GatedDeltaNet (SSM)  → default ArraysCache (fixed-size state)
Layer  1: GatedDeltaNet (SSM)  → default ArraysCache
Layer  2: GatedDeltaNet (SSM)  → default ArraysCache
Layer  3: Standard Attention   → TurboQuant 4-bit fused cache ← COMPRESSED
Layer  4: GatedDeltaNet (SSM)  → default ArraysCache
...
Layer  7: Standard Attention   → TurboQuant 4-bit fused cache ← COMPRESSED
...
Layer 31: Standard Attention   → TurboQuant 4-bit fused cache ← COMPRESSED
```

The TurboQuant pipeline for each attention layer:

```
On store (new token arrives):
  Keys/Values → L2 normalize → Random QR rotation → Lloyd-Max 4-bit quantize → Pack into uint32

On retrieve (during attention):
  Fused Metal kernel does in ONE dispatch:
    1. Unpack 4-bit indices from uint32
    2. Look up Lloyd-Max centroids
    3. Compute dot product scores (query × key centroids)
    4. Online softmax (no separate max/normalize pass)
    5. Accumulate weighted value centroids
    6. Cross-simdgroup reduction
  Then: inverse rotation via MLX GEMM
```

### Results

**Speed (Qwen 3.5 9B, M4 Mac Mini 16GB):**

| Method | tok/s | KV Cache |
|---|---|---|
| Standard (fp16) | 10.7 | 65 MB |
| TurboQuant V2 (mx.quantized_matmul) | 10.7 | 2.5 MB |
| **TurboQuant FUSED (this repo)** | **11.0** | **2.3 MB** |

**Context window scaling:**

| Context | Standard cache | TurboQuant cache | Compression |
|---|---|---|---|
| 256 tokens | 65 MB | 51 MB | 1.3x |
| 1K tokens | 89 MB | 58 MB | 1.5x |
| 4K tokens | 185 MB | 82 MB | 2.3x |
| 8K tokens | 313 MB | 114 MB | 2.7x |
| 16K tokens | 569 MB | 179 MB | 3.2x |

### Files

| File | What it does |
|---|---|
| `turboquant_qwen.py` | Run Qwen 3.5 with fused TurboQuant (mixed SSM + attention cache) |
| `turboquant_fused_4bit.py` | 4-bit fused Metal kernel (D=256, 16 simdgroups) |
| `run_turboquant.py` | Run any model with V2 TurboQuant |
| `bench_fused_vs_v2.py` | Benchmark: Standard vs V2 vs Fused |
| `bench_context.py` | Context window scaling benchmark |
| `test_fused_kernel.py` | Kernel correctness and speed tests |

## Credits

- **Original kernel**: [`fused_tq_attention_norot`](https://github.com/sharpner/turboquant-mlx/blob/master/turboquant/kernels.py#L718-L865) from [turboquant-mlx](https://github.com/sharpner/turboquant-mlx) by Nino (sharpner). Our 4-bit kernel is a direct adaptation of this work. PR submitted upstream: [sharpner/turboquant-mlx#2](https://github.com/sharpner/turboquant-mlx/pull/2)
- **TurboQuant paper**: [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874) (Zandieh et al., Google, ICLR 2026) — the algorithm: random QR rotation + Lloyd-Max quantization + QJL bias correction
- **Lloyd-Max codebook**: Mathematical constants from the TurboQuant paper, implemented in turboquant-mlx's [`codebook.py`](https://github.com/sharpner/turboquant-mlx/blob/master/turboquant/codebook.py)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF inference engine with Metal support
- [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple's ML framework and LLM tools

## Setup

```bash
# Install dependencies
brew install cmake
pip3 install mlx-lm huggingface_hub --break-system-packages

# Build llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j

# Clone TurboQuant (dependency)
git clone https://github.com/sharpner/turboquant-mlx.git

# Download models (examples)
mlx_lm.chat --model mlx-community/gemma-4-e4b-it-4bit  # auto-downloads
```
