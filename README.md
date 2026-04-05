# Local LLM Playground — Mac Mini M4 (16GB)

Running open-weight LLMs locally with TurboQuant KV cache compression.

## Models

| Model | Format | Size | Speed | Command |
|---|---|---|---|---|
| **Qwen 3.5 9B** | MLX 4-bit | 5 GB | 11 tok/s | `python3 turboquant_qwen.py` |
| **Gemma 4 E4B** | MLX 4-bit | 4.9 GB | 34 tok/s | `mlx_lm.chat --model models/gemma-4-e4b-it-4bit` |
| **Gemma 4 E2B** | MLX 4-bit | 3.3 GB | 64 tok/s | `mlx_lm.chat --model models/gemma-4-e2b-it-4bit` |
| **Gemma 3 12B** | GGUF Q4_K_M | 6.8 GB | 13 tok/s | `./llama.cpp/build/bin/llama-cli -m models/google_gemma-3-12b-it-Q4_K_M.gguf -c 2048 -fa on --conversation` |

## TurboQuant Fused Kernel

Custom 4-bit fused Metal attention kernel implementing Google's [TurboQuant](https://arxiv.org/abs/2504.19874) paper for Apple Silicon.

**Results on Qwen 3.5 9B (M4 Mac Mini 16GB):**
- 29x KV cache compression (65 MB -> 2.3 MB)
- Zero speed penalty (11.0 tok/s vs 10.7 baseline)
- Correct outputs (max_diff < 0.000001 vs reference)

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

- [turboquant-mlx](https://github.com/sharpner/turboquant-mlx) by Nino (sharpner) — MLX implementation of TurboQuant
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Zandieh et al., Google, ICLR 2026)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF inference engine
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference on MLX

## Setup

```bash
# Install dependencies
brew install cmake
pip3 install mlx-lm huggingface_hub --break-system-packages

# Build llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j

# Clone TurboQuant
git clone https://github.com/sharpner/turboquant-mlx.git

# Download models (examples)
mlx_lm.chat --model mlx-community/gemma-4-e4b-it-4bit  # auto-downloads
```
