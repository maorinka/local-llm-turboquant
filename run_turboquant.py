"""Run Qwen 3.5 9B with TurboQuant KV-cache compression.

Builds a mixed cache: TurboQuant for attention layers, default for SSM layers.
"""

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

DEFAULT_MODEL = "models/qwen3.5-9b-mlx-4bit"
BITS = 4


def make_mixed_cache(model, bits=BITS, group_size=64):
    """Create mixed cache: TurboQuant for attention layers, default for SSM."""
    default_cache = make_prompt_cache(model)
    mixed = []
    tq_count = 0
    for i, c in enumerate(default_cache):
        if isinstance(c, KVCache):
            # Run a dummy forward to get head_dim, or detect from model
            # KVCache layers have self_attn with head_dim
            attn = model.layers[i].self_attn
            head_dim = attn.head_dim
            mixed.append(TurboQuantKVCacheV2(
                head_dim=head_dim, bits=bits, group_size=group_size,
                use_qjl=False, seed=42 + i,
            ))
            tq_count += 1
        else:
            # SSM layer — keep default ArraysCache
            mixed.append(c)
    print(f"  {tq_count} attention layers → TurboQuant {bits}-bit")
    print(f"  {len(mixed) - tq_count} SSM layers → default cache")
    return mixed


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens=4096):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(formatted))

    tokens = []
    start = time.perf_counter()

    for token, logprobs in generate_step(
        prompt=input_ids,
        model=model,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        tok = token.item() if hasattr(token, "item") else int(token)
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)
        sys.stdout.write(tokenizer.decode([tok]))
        sys.stdout.flush()

    elapsed = time.perf_counter() - start
    return tokens, elapsed


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else BITS

    print(f"Loading: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)
    n_layers = len(model.layers)
    print(f"Loaded: {n_layers} layers, TurboQuant {bits}-bit")

    cache = make_mixed_cache(model, bits=bits)
    print()

    while True:
        try:
            prompt = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if prompt.strip().lower() in ("q", "quit", "exit"):
            break
        if not prompt.strip():
            continue

        # Reset cache for each turn (stateless chat)
        cache = make_mixed_cache(model, bits=bits)
        tokens, elapsed = generate_with_cache(model, tokenizer, prompt, cache)

        # Stats from TurboQuant layers only
        tq_caches = [c for c in cache if isinstance(c, TurboQuantKVCacheV2)]
        tq_bytes = sum(c.nbytes for c in tq_caches)
        tq_fp16 = sum(c.nbytes_equivalent_fp16 for c in tq_caches)
        ratio = tq_fp16 / tq_bytes if tq_bytes > 0 else 0

        print(f"\n\n[{len(tokens)} tokens, {elapsed:.1f}s, {len(tokens)/elapsed:.1f} tok/s, "
              f"KV cache: {tq_bytes:,}B ({ratio:.1f}x compression)]\n")


if __name__ == "__main__":
    main()
