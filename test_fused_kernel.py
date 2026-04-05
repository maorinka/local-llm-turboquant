"""Test the fused 4-bit TurboQuant kernel for correctness and speed."""

import sys
import time
import mlx.core as mx

sys.path.insert(0, "turboquant-mlx")
from turboquant.codebook import get_codebook
from turboquant.rotation import generate_rotation_matrix
from turboquant_fused_4bit import pack_4bit_indices, unpack_4bit_indices, fused_tq_attention_4bit

D = 256
N_Q_HEADS = 8
N_KV_HEADS = 4
BITS = 4
T_KV_VALUES = [64, 256, 1024]

# Get Lloyd-Max 4-bit codebook
centroids, boundaries = get_codebook(BITS, D)
rotation_matrix = generate_rotation_matrix(D, seed=42)


def make_test_data(T_kv):
    """Create test data: random keys/values, quantize them."""
    keys = mx.random.normal((N_KV_HEADS, T_kv, D))
    values = mx.random.normal((N_KV_HEADS, T_kv, D))

    k_norms = mx.linalg.norm(keys, axis=-1)
    v_norms = mx.linalg.norm(values, axis=-1)
    k_safe = mx.where(k_norms[..., None] < 1e-8, mx.ones_like(keys), keys)
    v_safe = mx.where(v_norms[..., None] < 1e-8, mx.ones_like(values), values)
    k_normalized = k_safe / mx.linalg.norm(k_safe, axis=-1, keepdims=True)
    v_normalized = v_safe / mx.linalg.norm(v_safe, axis=-1, keepdims=True)

    k_rotated = k_normalized @ rotation_matrix.T
    v_rotated = v_normalized @ rotation_matrix.T

    k_indices = mx.zeros(k_rotated.shape, dtype=mx.uint8)
    v_indices = mx.zeros(v_rotated.shape, dtype=mx.uint8)
    for b in range(len(boundaries)):
        k_indices = k_indices + (k_rotated > boundaries[b]).astype(mx.uint8)
        v_indices = v_indices + (v_rotated > boundaries[b]).astype(mx.uint8)

    k_packed = pack_4bit_indices(k_indices)
    v_packed = pack_4bit_indices(v_indices)

    return k_packed, k_norms, v_packed, v_norms


def reference_attention(q_rot, k_packed, centroids_arr, k_norms, v_packed, v_norms):
    """Pure MLX reference implementation for comparison."""
    n_repeats = N_Q_HEADS // N_KV_HEADS

    k_indices = unpack_4bit_indices(k_packed, D)
    k_centroids = centroids_arr[k_indices.astype(mx.uint32)]
    v_indices = unpack_4bit_indices(v_packed, D)
    v_centroids = centroids_arr[v_indices.astype(mx.uint32)]

    q_grouped = q_rot.reshape(N_KV_HEADS, n_repeats, D)
    scores = q_grouped @ k_centroids.transpose(0, 2, 1)
    scores = scores * k_norms[:, None, :]

    weights = mx.softmax(scores, axis=-1)

    v_normed = v_centroids * v_norms[:, :, None]
    output = weights @ v_normed

    return output.reshape(N_Q_HEADS, D)


print("=" * 60)
print("Testing fused 4-bit TurboQuant kernel (D=256)")
print("=" * 60)

# Test 1: Pack/unpack roundtrip
print("\n--- Test 1: 4-bit pack/unpack roundtrip ---")
test_indices = mx.random.randint(0, 16, (4, 32, D)).astype(mx.uint8)
packed = pack_4bit_indices(test_indices)
unpacked = unpack_4bit_indices(packed, D)
match = mx.array_equal(test_indices.astype(mx.uint32), unpacked)
mx.synchronize()
print(f"  Pack/unpack match: {match.item()}")

# Test 2: Kernel correctness
print("\n--- Test 2: Kernel correctness ---")
for T_kv in T_KV_VALUES:
    k_packed, k_norms, v_packed, v_norms = make_test_data(T_kv)
    q_rot = mx.random.normal((N_Q_HEADS, D)) * (1.0 / (D ** 0.5))
    mx.synchronize()

    fused_out = fused_tq_attention_4bit(
        q_rot, k_packed, centroids, k_norms, v_packed, v_norms,
        n_q_heads=N_Q_HEADS, D=D,
    )
    ref_out = reference_attention(q_rot, k_packed, centroids, k_norms, v_packed, v_norms)
    mx.synchronize()

    diff = mx.abs(fused_out - ref_out).max().item()
    print(f"  T_kv={T_kv:4d}: max_diff={diff:.6f} {'OK' if diff < 0.01 else 'FAIL'}")

# Test 3: Speed benchmark
print("\n--- Test 3: Speed benchmark ---")
for T_kv in T_KV_VALUES:
    k_packed, k_norms, v_packed, v_norms = make_test_data(T_kv)
    q_rot = mx.random.normal((N_Q_HEADS, D)) * (1.0 / (D ** 0.5))
    mx.synchronize()

    # Warmup
    for _ in range(3):
        out = fused_tq_attention_4bit(
            q_rot, k_packed, centroids, k_norms, v_packed, v_norms,
            n_q_heads=N_Q_HEADS, D=D,
        )
        mx.synchronize()

    # Timed
    N = 100
    start = time.perf_counter()
    for _ in range(N):
        out = fused_tq_attention_4bit(
            q_rot, k_packed, centroids, k_norms, v_packed, v_norms,
            n_q_heads=N_Q_HEADS, D=D,
        )
        mx.synchronize()
    elapsed = time.perf_counter() - start
    us_per_call = elapsed / N * 1e6
    print(f"  T_kv={T_kv:4d}: {us_per_call:.0f} us/call ({N} iterations)")

print("\nDone!")
