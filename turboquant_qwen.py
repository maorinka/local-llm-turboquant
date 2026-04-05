"""TurboQuant fused pipeline for Qwen 3.5 9B.

4-bit Lloyd-Max KV cache with fused Metal attention kernel.
Mixed cache: TurboQuant for attention layers, default for SSM layers.
"""

import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache
import mlx_lm.models.base as _base

sys.path.insert(0, "turboquant-mlx")
from turboquant.codebook import get_codebook
from turboquant.rotation import generate_rotation_matrix
from turboquant.kernels import turboquant_encode
from turboquant_fused_4bit import pack_4bit_indices, fused_tq_attention_4bit


# ============================================================
# 4-bit TurboQuant KV Cache (Lloyd-Max codebook)
# ============================================================

class TurboQuantKVCache4Bit:
    """4-bit TurboQuant KV cache with fused Metal attention kernel.

    D=256 only. No QJL (not needed at 4-bit).
    Uses Lloyd-Max codebook for quantization.
    """

    def __init__(self, head_dim: int = 256, seed: int = 42):
        if head_dim != 256:
            raise ValueError(f"Only head_dim=256 supported, got {head_dim}")
        self.head_dim = head_dim
        self.offset = 0

        self.centroids, self.boundaries = get_codebook(4, head_dim)
        self.rotation_matrix = generate_rotation_matrix(head_dim, seed=seed)

        self.key_packed = None
        self.key_norms = None
        self.value_packed = None
        self.value_norms = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize new KV pairs, store packed.

        Args:
            keys: (B, n_kv_heads, T_new, D)
            values: (B, n_kv_heads, T_new, D)

        Returns:
            (keys, values): passthrough for compatibility
        """
        k_indices, k_norms, _ = turboquant_encode(
            keys, self.rotation_matrix, self.boundaries
        )
        v_indices, v_norms, _ = turboquant_encode(
            values, self.rotation_matrix, self.boundaries
        )

        k_packed = pack_4bit_indices(k_indices)
        v_packed = pack_4bit_indices(v_indices)

        if self.key_packed is None:
            self.key_packed = k_packed
            self.key_norms = k_norms
            self.value_packed = v_packed
            self.value_norms = v_norms
        else:
            self.key_packed = mx.concatenate([self.key_packed, k_packed], axis=2)
            self.key_norms = mx.concatenate([self.key_norms, k_norms], axis=2)
            self.value_packed = mx.concatenate([self.value_packed, v_packed], axis=2)
            self.value_norms = mx.concatenate([self.value_norms, v_norms], axis=2)

        self.offset += keys.shape[2]

        # Force eval: Metal barrier between concat and fused kernel
        mx.synchronize()

        return keys, values

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            from mlx_lm.models.base import create_causal_mask
            return create_causal_mask(N, offset=self.offset - N, window_size=window_size)
        return "causal"

    @property
    def state(self):
        if self.key_packed is None:
            return []
        return [self.key_packed, self.key_norms, self.value_packed, self.value_norms]

    @state.setter
    def state(self, v):
        if not v:
            return
        self.key_packed, self.key_norms, self.value_packed, self.value_norms = v

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return False

    def empty(self):
        return self.key_packed is None

    @property
    def nbytes(self):
        if self.key_packed is None:
            return 0
        return (self.key_packed.nbytes + self.key_norms.nbytes +
                self.value_packed.nbytes + self.value_norms.nbytes)

    @property
    def nbytes_equivalent_fp16(self):
        if self.key_packed is None:
            return 0
        B, n_kv_heads, T, _ = self.key_packed.shape
        return B * n_kv_heads * T * self.head_dim * 2 * 2


# ============================================================
# Fused SDPA dispatch patch
# ============================================================

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


def _fused_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    """Dispatch to fused 4-bit kernel for TurboQuantKVCache4Bit."""
    if isinstance(cache, TurboQuantKVCache4Bit) and cache.key_packed is not None:
        B, n_q_heads, T_q, D = queries.shape
        T_kv = cache.offset

        if T_q == 1 and B == 1:
            # Fused path: rotate query → fused kernel → rotate back
            q_flat = queries.reshape(n_q_heads, D)
            q_rot = (q_flat * scale) @ cache.rotation_matrix.T

            out_rot = fused_tq_attention_4bit(
                q_rot,
                cache.key_packed.squeeze(0),
                cache.centroids,
                cache.key_norms.squeeze(0),
                cache.value_packed.squeeze(0),
                cache.value_norms.squeeze(0),
                n_q_heads=n_q_heads,
                D=D,
            )

            output = out_rot @ cache.rotation_matrix
            return output.reshape(B, n_q_heads, T_q, D)

        # Fallback for prefill (T_q > 1): use standard attention with dequantized KV
        return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)

    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def apply_patch():
    global _patched
    if _patched:
        return
    _base.scaled_dot_product_attention = _fused_sdpa
    _patched = True


# ============================================================
# Mixed cache builder
# ============================================================

def make_mixed_cache(model):
    """TurboQuant for attention layers, default for SSM layers."""
    default_cache = make_prompt_cache(model)
    mixed = []
    tq_count = 0
    for i, c in enumerate(default_cache):
        if isinstance(c, KVCache):
            head_dim = model.layers[i].self_attn.head_dim
            mixed.append(TurboQuantKVCache4Bit(head_dim=head_dim, seed=42 + i))
            tq_count += 1
        else:
            mixed.append(c)
    return mixed, tq_count


# ============================================================
# Main
# ============================================================

DEFAULT_MODEL = "models/qwen3.5-9b-mlx-4bit"


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens=4096):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(formatted))

    tokens = []
    start = time.perf_counter()

    for token, _ in generate_step(
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

    apply_patch()

    print(f"Loading: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)
    print(f"Loaded: {len(model.layers)} layers")

    cache, tq_count = make_mixed_cache(model)
    print(f"  {tq_count} attention layers → TurboQuant 4-bit FUSED")
    print(f"  {len(cache) - tq_count} SSM layers → default cache\n")

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

        cache, _ = make_mixed_cache(model)
        tokens, elapsed = generate_with_cache(model, tokenizer, prompt, cache)

        tq_caches = [c for c in cache if isinstance(c, TurboQuantKVCache4Bit)]
        tq_bytes = sum(c.nbytes for c in tq_caches)
        tq_fp16 = sum(c.nbytes_equivalent_fp16 for c in tq_caches)
        ratio = tq_fp16 / tq_bytes if tq_bytes > 0 else 0

        print(f"\n\n[{len(tokens)} tokens, {elapsed:.1f}s, {len(tokens)/elapsed:.1f} tok/s, "
              f"KV cache: {tq_bytes:,}B ({ratio:.1f}x compression)]\n")


if __name__ == "__main__":
    main()
