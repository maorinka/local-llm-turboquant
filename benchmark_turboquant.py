"""Benchmark: TurboQuant vs Standard cache on Qwen 3.5 9B"""

import sys
import time
import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache

sys.path.insert(0, "turboquant-mlx")
from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch
tq_patch.apply()

MODEL = "models/qwen3.5-9b-mlx-4bit"

PROMPTS = [
    "What is 2+2?",
    "Write a Python function to check if a number is prime.",
    "Explain gravity in 3 sentences.",
]


def make_mixed_tq_cache(model, bits=4):
    default_cache = make_prompt_cache(model)
    mixed = []
    for i, c in enumerate(default_cache):
        if isinstance(c, KVCache):
            mixed.append(TurboQuantKVCacheV2(
                head_dim=model.layers[i].self_attn.head_dim,
                bits=bits, group_size=64, use_qjl=False, seed=42 + i,
            ))
        else:
            mixed.append(c)
    return mixed


def run_prompt(model, tokenizer, prompt, cache, max_tokens=512):
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
    text = tokenizer.decode(tokens)
    return text, len(tokens), elapsed


def main():
    print(f"Loading: {MODEL}")
    model, tokenizer = mlx_lm.load(MODEL)
    print(f"Loaded: {len(model.layers)} layers\n")

    for prompt in PROMPTS:
        print(f"{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")

        # Standard cache
        std_cache = make_prompt_cache(model)
        std_text, std_n, std_t = run_prompt(model, tokenizer, prompt, std_cache)
        std_bytes = sum(c.nbytes for c in std_cache if hasattr(c, 'nbytes'))

        # TurboQuant cache
        tq_cache = make_mixed_tq_cache(model, bits=4)
        tq_text, tq_n, tq_t = run_prompt(model, tokenizer, prompt, tq_cache)
        tq_tq_caches = [c for c in tq_cache if isinstance(c, TurboQuantKVCacheV2)]
        tq_bytes = sum(c.nbytes for c in tq_tq_caches)
        tq_fp16 = sum(c.nbytes_equivalent_fp16 for c in tq_tq_caches)

        print(f"\n--- Standard ---")
        print(f"  {std_n} tokens, {std_t:.1f}s, {std_n/std_t:.1f} tok/s")
        print(f"  Cache: {std_bytes:,} bytes")
        print(f"  Output: {std_text[:200]}")

        print(f"\n--- TurboQuant 4-bit ---")
        print(f"  {tq_n} tokens, {tq_t:.1f}s, {tq_n/tq_t:.1f} tok/s")
        print(f"  KV Cache: {tq_bytes:,} bytes ({tq_fp16/tq_bytes:.1f}x compression)")
        print(f"  Output: {tq_text[:200]}")
        print()


if __name__ == "__main__":
    main()
