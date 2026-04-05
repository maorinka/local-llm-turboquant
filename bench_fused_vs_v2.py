"""Benchmark: Fused 4-bit vs V2 4-bit vs Standard on Qwen 3.5 9B"""

import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache

MODEL = "models/qwen3.5-9b-mlx-4bit"
PROMPT = "Write a Python function to check if a number is prime."
MAX_TOKENS = 256


def run_prompt(model, tokenizer, prompt, cache, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(formatted))
    tokens = []
    start = time.perf_counter()
    for token, _ in generate_step(
        prompt=input_ids, model=model, max_tokens=max_tokens, prompt_cache=cache,
    ):
        tok = token.item() if hasattr(token, "item") else int(token)
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)
    elapsed = time.perf_counter() - start
    return len(tokens), elapsed


print(f"Loading: {MODEL}")
model, tokenizer = mlx_lm.load(MODEL)
print(f"Loaded: {len(model.layers)} layers\n")

# --- 1. Standard (no TurboQuant) ---
print("1. Standard (fp16 KV cache)...")
import mlx_lm.models.base as _base
_base_sdpa = _base.scaled_dot_product_attention  # save original

std_cache = make_prompt_cache(model)
n, t = run_prompt(model, tokenizer, PROMPT, std_cache, MAX_TOKENS)
std_bytes = sum(c.nbytes for c in std_cache if hasattr(c, 'nbytes'))
print(f"   {n} tokens, {t:.1f}s, {n/t:.1f} tok/s, cache: {std_bytes:,}B\n")

# --- 2. V2 (rotation + mx.quantized_matmul) ---
print("2. TurboQuant V2 (rotation + mx.quantized_matmul)...")
sys.path.insert(0, "turboquant-mlx")
from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch
tq_patch.apply()

def make_v2_cache(model):
    default = make_prompt_cache(model)
    mixed = []
    for i, c in enumerate(default):
        if isinstance(c, KVCache):
            mixed.append(TurboQuantKVCacheV2(
                head_dim=model.layers[i].self_attn.head_dim,
                bits=4, group_size=64, use_qjl=False, seed=42 + i,
            ))
        else:
            mixed.append(c)
    return mixed

v2_cache = make_v2_cache(model)
n2, t2 = run_prompt(model, tokenizer, PROMPT, v2_cache, MAX_TOKENS)
v2_tq = [c for c in v2_cache if isinstance(c, TurboQuantKVCacheV2)]
v2_bytes = sum(c.nbytes for c in v2_tq)
print(f"   {n2} tokens, {t2:.1f}s, {n2/t2:.1f} tok/s, KV cache: {v2_bytes:,}B\n")

# --- 3. Fused 4-bit (Lloyd-Max + fused Metal kernel) ---
print("3. TurboQuant FUSED 4-bit (Lloyd-Max + fused Metal kernel)...")
tq_patch.revert()  # remove V2 patch

from turboquant_qwen import TurboQuantKVCache4Bit, apply_patch, make_mixed_cache
apply_patch()

fused_cache, _ = make_mixed_cache(model)
n3, t3 = run_prompt(model, tokenizer, PROMPT, fused_cache, MAX_TOKENS)
fused_tq = [c for c in fused_cache if isinstance(c, TurboQuantKVCache4Bit)]
fused_bytes = sum(c.nbytes for c in fused_tq)
print(f"   {n3} tokens, {t3:.1f}s, {n3/t3:.1f} tok/s, KV cache: {fused_bytes:,}B\n")

# --- Summary ---
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Method':<35} {'tok/s':>8} {'KV Cache':>12}")
print(f"{'-'*35} {'-'*8} {'-'*12}")
print(f"{'Standard (fp16)':<35} {n/t:>8.1f} {std_bytes:>12,}B")
print(f"{'V2 (rot + mx.quantized_matmul)':<35} {n2/t2:>8.1f} {v2_bytes:>12,}B")
print(f"{'FUSED (rot + Lloyd-Max + Metal)':<35} {n3/t3:>8.1f} {fused_bytes:>12,}B")
