"""Benchmark: Context window scaling — Standard vs TurboQuant FUSED.

Measures memory and speed at increasing context lengths to show
where TurboQuant's 29x compression actually matters.
"""

import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache

sys.path.insert(0, "turboquant-mlx")
from turboquant_qwen import TurboQuantKVCache4Bit, apply_patch, make_mixed_cache

MODEL = "models/qwen3.5-9b-mlx-4bit"
CONTEXT_LENGTHS = [256, 1024, 4096, 8192, 16384]


def fill_context(model, tokenizer, cache, n_tokens):
    """Fill the cache with n_tokens by feeding dummy input."""
    # Generate a long prompt by repeating text
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * (n_tokens // 8 + 1)
    tokens = tokenizer.encode(text)[:n_tokens]
    input_ids = mx.array([tokens])

    # Process in chunks to avoid OOM during prefill
    chunk_size = 512
    for i in range(0, len(tokens), chunk_size):
        chunk = input_ids[:, i:i + chunk_size]
        model(chunk, cache=cache)
        mx.synchronize()


def measure_generation(model, tokenizer, cache, n_gen=32):
    """Generate n_gen tokens and measure speed."""
    # Use a simple continuation token
    last_token = mx.array([[1]])
    tokens = []
    start = time.perf_counter()
    for _ in range(n_gen):
        logits = model(last_token, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        last_token = tok.reshape(1, 1)
        tokens.append(tok.item())
        mx.synchronize()
    elapsed = time.perf_counter() - start
    return n_gen, elapsed


def get_cache_bytes(cache):
    total = 0
    for c in cache:
        if hasattr(c, 'nbytes'):
            total += c.nbytes
    return total


print(f"Loading: {MODEL}")
model, tokenizer = mlx_lm.load(MODEL)
print(f"Loaded: {len(model.layers)} layers\n")

print(f"{'Context':>8} | {'Standard':>20} | {'TurboQuant FUSED':>20} | {'Compression':>12}")
print(f"{'':>8} | {'tok/s':>8} {'cache MB':>10} | {'tok/s':>8} {'cache MB':>10} | {'ratio':>12}")
print(f"{'-'*8}-+-{'-'*20}-+-{'-'*20}-+-{'-'*12}")

for ctx_len in CONTEXT_LENGTHS:
    # --- Standard ---
    try:
        std_cache = make_prompt_cache(model)
        fill_context(model, tokenizer, std_cache, ctx_len)
        std_n, std_t = measure_generation(model, tokenizer, std_cache)
        std_bytes = get_cache_bytes(std_cache)
        std_tps = std_n / std_t
        std_mb = std_bytes / (1024 * 1024)
        # Free memory
        del std_cache
        mx.synchronize()
    except Exception as e:
        std_tps = 0
        std_mb = 0
        std_err = str(e)[:30]

    # --- TurboQuant FUSED ---
    try:
        apply_patch()
        tq_cache, _ = make_mixed_cache(model)
        fill_context(model, tokenizer, tq_cache, ctx_len)
        tq_n, tq_t = measure_generation(model, tokenizer, tq_cache)
        tq_bytes = get_cache_bytes(tq_cache)
        tq_tps = tq_n / tq_t
        tq_mb = tq_bytes / (1024 * 1024)
        del tq_cache
        mx.synchronize()
    except Exception as e:
        tq_tps = 0
        tq_mb = 0

    ratio = f"{std_mb / tq_mb:.1f}x" if tq_mb > 0 else "N/A"

    print(f"{ctx_len:>8} | {std_tps:>8.1f} {std_mb:>9.1f}M | {tq_tps:>8.1f} {tq_mb:>9.1f}M | {ratio:>12}")

print("\nDone!")
