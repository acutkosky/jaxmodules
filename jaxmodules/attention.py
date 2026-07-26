import jax
from jax import numpy as jnp
import math
from typing import Callable, Union, Optional, Dict, Any, NamedTuple, Tuple
from jaxtyping import Array, Float, UInt
import equinox as eqx
from einops import rearrange, repeat
from jaxmodules.vectorize import array_from_coords, multi_vmap, multi_vmap_transposed_in_axes, nested_fori_loop, fancy_vmap#, einsum
from jaxmodules.block_mask import BlockMask
from functools import partial
from einops import einsum

def use_custom_einsum():
    global einsum
    from jaxmodules.vectorize import einsum as use_custom_einsum
    einsum = use_custom_einsum


def threshold_kernel(threshold: Optional[float] = None):
    """
    Create a kernel function that applies a threshold to the dot product before exponentiating.

    Args:
        threshold: Optional threshold to subtract from the dot product before exponentiating.
            If None, no threshold is applied.

    Returns:
        A kernel function that takes query and key vectors and returns their attention score
    """

    def kernel_fn(q, k):
        dotp = jnp.dot(q, k)
        if threshold is not None:
            dotp = dotp - threshold / jnp.sqrt(q.shape[0])
        return jnp.exp(dotp)

    return kernel_fn


def default_kernel(q, k):
    """
    Default attention kernel that computes exp(q^T k / sqrt(d)).

    Args:
        q: Query vector
        k: Key vector

    Returns:
        Attention score between q and k
    """
    return jnp.exp(jnp.dot(q, k) / jnp.sqrt(k.shape[-1]))


def _unmasked(b, h, q, k):
    del b, h, q, k
    return True


def _causal_mask(b, h, q, k):
    del b, h
    return q >= k


def _materialize_mask(mask_fn, B, Hq, Hkv, q_idx, k_idx):
    """Materialize one mask tile in grouped-query layout."""
    if mask_fn is _unmasked:
        return True
    if mask_fn is _causal_mask:
        return q_idx[None, :, None, None, None] >= k_idx[None, None, None, None, :]

    MQA_factor = Hq // Hkv
    mask = fancy_vmap(
        mask_fn,
        "mask[b, q, h, k] = mask_fn(B[b], Hq[h], q_idx[q], k_idx[k])"
    )(jnp.arange(B), jnp.arange(Hq), q_idx, k_idx)
    return rearrange(
        mask,
        "B Lq (Hkv MQA) Lk -> B Lq Hkv MQA Lk",
        Hkv=Hkv,
        MQA=MQA_factor,
    )


def _attn_kq_block_fn(
    max_score,  # [B, Lq, Hq]
    normalizer,  # [B, Lq, Hq]
    numerator,  # [B, Lq, Hq, dv]
    q_idx, # [Lq]
    k_idx, # [Lk]
    q_block, # [B, Lq, Hq, dq]
    k_block, # [B, Lk, Hkv, dk]
    v_block, # [B, Lk, Hkv, dv]
    mask_fn,
    kernel_fn,
):
    B, Lq, Hq, dq = q_block.shape
    _, Lk, Hkv, _ = k_block.shape
    _, _, _, dv = v_block.shape
    MQA_factor = Hq // Hkv
    accumulator_dtype = jnp.result_type(
        q_block.dtype,
        k_block.dtype,
        v_block.dtype,
        jnp.float32,
    )
    q_block = rearrange(q_block, "B Lq (Hkv MQA) dq -> B Lq Hkv MQA dq", Hkv=Hkv)

    mask = _materialize_mask(mask_fn, B, Hq, Hkv, q_idx, k_idx)

    max_score = rearrange(
        max_score,
        "B Lq (Hkv MQA) -> B Lq Hkv MQA 1",
        Hkv=Hkv,
    )
    normalizer = rearrange(normalizer, "B Lq (Hkv MQA)-> B Lq Hkv MQA 1", Hkv=Hkv)
    numerator = rearrange(
        numerator,
        "B Lq (Hkv MQA) dv -> B Lq Hkv MQA dv",
        Hkv=Hkv,
    )

    if kernel_fn is default_kernel:
        # Compute score tiles directly so XLA can lower the contraction as a
        # batched matrix multiplication. Accumulate low-precision inputs in
        # FP32, matching the usual scaled-dot-product attention policy.
        scores = jnp.einsum(
            "bqhmd,bkhd->bqhmk",
            q_block,
            k_block,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=accumulator_dtype,
        )
        scores = scores / jnp.sqrt(jnp.asarray(dq, dtype=accumulator_dtype))
        scores = jnp.where(mask, scores, -jnp.inf)

        local_max = jnp.max(scores, axis=-1, keepdims=True)
        new_max_score = jnp.maximum(max_score, local_max)
        # A fully masked prefix has max=-inf. Shift it by zero so that
        # (-inf)-(-inf) cannot introduce NaNs.
        safe_new_max = jnp.where(
            jnp.isfinite(new_max_score),
            new_max_score,
            jnp.zeros_like(new_max_score),
        )
        previous_scale = jnp.where(
            jnp.isfinite(max_score),
            jnp.exp(max_score - safe_new_max),
            jnp.zeros_like(max_score),
        )
        probabilities = jnp.exp(scores - safe_new_max)
        local_normalizer = jnp.sum(probabilities, axis=-1, keepdims=True)
        local_numerator = jnp.einsum(
            "bqhmk,bkhe->bqhme",
            probabilities,
            v_block,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=accumulator_dtype,
        )

        new_normalizer = previous_scale * normalizer + local_normalizer
        numerator = previous_scale * numerator + local_numerator
        max_score = new_max_score
    else:
        # Compatibility path for arbitrary positive kernels. Such kernels
        # expose weights rather than logits, so max-shifting is not generally
        # available without changing their API.
        scores = fancy_vmap(
            kernel_fn,
            (
                "scores[b, q, h, m, k] = "
                "kernel_fn(q_block[b, q, h, m, :], K[b, k, h, :])"
            ),
        )(q_block, k_block)
        scores = jnp.asarray(scores, dtype=accumulator_dtype)
        scores = jnp.where(mask, scores, jnp.zeros_like(scores))

        local_normalizer = jnp.sum(scores, axis=-1, keepdims=True)
        local_numerator = einsum(
            scores,
            v_block,
            "B Lq Hkv MQA Lk, B Lk Hkv d -> B Lq Hkv MQA d",
        )
        new_normalizer = normalizer + local_normalizer
        numerator = numerator + local_numerator

    max_score = rearrange(max_score, "B Lq Hkv MQA 1 -> B Lq (Hkv MQA)")
    normalizer = rearrange(new_normalizer, "B Lq Hkv MQA 1 -> B Lq (Hkv MQA)")
    numerator = rearrange(numerator, "B Lq Hkv MQA dv -> B Lq (Hkv MQA) dv")

    return max_score, normalizer, numerator

def _make_attention_kq_scanner(
    mask_fn,
    kernel_fn,
    q_idx,
    q_block,
):
    def scan_fn(
        carry,
        blocks,
    ):
        max_score, normalizer, numerator = carry
        k_idx, k_block, v_block = blocks
        max_score, normalizer, numerator = _attn_kq_block_fn(
            max_score,
            normalizer,
            numerator,
            q_idx,
            k_idx,
            q_block,
            k_block,
            v_block,
            mask_fn,
            kernel_fn)
        return (max_score, normalizer, numerator), None
    return scan_fn


def _attn_block_fn(
    block_idx,
    q_idx,
    q_block,
    K,
    V,
    kv_block_size,
    mask_fn,
    kernel_fn,
    window_left,
    window_num_blocks,
    is_causal=False,
    use_causal_block_skipping=True,
):
    """Compute attention for a single Q block against all K/V blocks.

    Args:
        use_causal_block_skipping: If True and is_causal=True, skip K/V blocks
            that are entirely masked out. Set to False during backward pass
            since fori_loop with dynamic bounds doesn't support reverse-mode AD.
    """
    B, Lq, Hq, dq = q_block.shape
    _, Lk, Hkv, _ = K.shape
    _, _, _, dv = V.shape

    k_block = rearrange(
        K,
        "B (blocks block_size) Hkv d -> blocks B block_size Hkv d",
        block_size=kv_block_size,
    )
    v_block = rearrange(
        V,
        "B (blocks block_size) Hkv d -> blocks B block_size Hkv d",
        block_size=kv_block_size,
    )
    k_indices = rearrange(
        jnp.arange(Lk),
        "(blocks block_size) -> blocks block_size",
        block_size=kv_block_size,
    )

    num_blocks = k_indices.shape[0]

    # Apply windowing if specified
    if window_left is not None:
        idx_start = jnp.maximum(
            0,
            jnp.floor_divide(q_idx[0] - window_left, kv_block_size),
        )
        k_indices = jax.lax.dynamic_slice_in_dim(
            k_indices,
            idx_start,
            window_num_blocks,
            axis=0,
        )
        k_block = jax.lax.dynamic_slice_in_dim(
            k_block,
            idx_start,
            window_num_blocks,
            axis=0,
        )
        v_block = jax.lax.dynamic_slice_in_dim(
            v_block,
            idx_start,
            window_num_blocks,
            axis=0,
        )
        num_blocks = window_num_blocks

    accumulator_dtype = jnp.result_type(
        q_block.dtype,
        K.dtype,
        V.dtype,
        jnp.float32,
    )
    init_carry = (
        jnp.full((B, Lq, Hq), -jnp.inf, dtype=accumulator_dtype),
        jnp.zeros((B, Lq, Hq), dtype=accumulator_dtype),
        jnp.zeros((B, Lq, Hq, dv), dtype=accumulator_dtype),
    )

    if is_causal and use_causal_block_skipping and window_left is None:
        # Causal optimization: only process blocks containing keys no later
        # than the final query in this tile.
        # Use fori_loop with dynamic upper bound instead of scan over all blocks
        # Note: This optimization only works for forward pass; backward pass
        # must use use_causal_block_skipping=False since fori_loop with dynamic
        # bounds doesn't support reverse-mode differentiation.
        def body_fn(i, carry):
            max_score, normalizer, numerator = carry
            max_score, normalizer, numerator = _attn_kq_block_fn(
                max_score,
                normalizer,
                numerator,
                q_idx,
                k_indices[i],
                q_block,
                k_block[i],
                v_block[i],
                mask_fn,
                kernel_fn,
            )
            return (max_score, normalizer, numerator)

        # Cross-attention may contain more query blocks than key/value blocks.
        # Clamp the bound instead of relying on JAX's clipped out-of-bounds
        # indexing, which would process the final K/V block repeatedly.
        upper_bound = jnp.minimum(
            jnp.floor_divide(q_idx[-1], kv_block_size) + 1,
            num_blocks,
        )
        max_score, normalizer, numerator = jax.lax.fori_loop(
            0, upper_bound, body_fn, init_carry
        )
    else:
        # Standard scan over all K/V blocks
        kq_scanner = _make_attention_kq_scanner(mask_fn, kernel_fn, q_idx, q_block)
        (max_score, normalizer, numerator), _ = jax.lax.scan(
            kq_scanner,
            init=init_carry,
            xs=(k_indices, k_block, v_block),
        )

    safe_normalizer = jnp.where(
        normalizer > 0,
        normalizer,
        jnp.ones_like(normalizer),
    )
    output = jnp.where(
        normalizer[..., None] > 0,
        numerator / safe_normalizer[..., None],
        jnp.zeros_like(numerator),
    )
    log_normalizer = jnp.where(
        normalizer > 0,
        max_score + jnp.log(safe_normalizer),
        jnp.zeros_like(normalizer),
    )
    return output, log_normalizer


def _window_block_config(
    window_size,
    query_block_size,
    kv_block_size,
    num_kv_blocks,
):
    """Return a fixed-size K/V slice covering every query tile's window."""
    if window_size is None:
        return None, None

    left, right = window_size
    period = kv_block_size // math.gcd(query_block_size, kv_block_size)
    span = left + query_block_size + right
    window_num_blocks = max(
        (
            ((query_index * query_block_size - left) % kv_block_size)
            + span
            + kv_block_size
            - 1
        )
        // kv_block_size
        for query_index in range(period)
    )
    return left, min(window_num_blocks, num_kv_blocks)


def _masked_attention_via_map_impl(
    Q: Array,
    K: Array,
    V: Array,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[Tuple[int,int,int], Array], Array]] = None,
    block_size=None,
    kv_block_size=None,
    window_size=None,
    is_causal=False,
) -> Array:

    B, L, Hk, d = K.shape
    B, N, Hq, dq = Q.shape
    B, Lv, Hv, dv = V.shape

    if block_size is None:
        block_size = N
    if kv_block_size is None:
        kv_block_size = block_size

    window_left, window_num_blocks = _window_block_config(
        window_size,
        block_size,
        kv_block_size,
        L // kv_block_size,
    )

    assert d == dq and Lv == L, (
        f"shape mismatch in K {K.shape}, Q {Q.shape} and V {V.shape}"
    )

    assert Hq % Hk == 0, "Hq must be divisible by Hk"
    assert Hv == Hk, "Hv must be equal to Hk"

    Hkv = Hk

    assert block_size is None or N % block_size == 0, (
        f"block_size must divide number of queries!"
    )
    assert L % kv_block_size == 0, (
        "kv_block_size must divide the number of keys and values!"
    )

    def attn_fn(block_idx_q_idx_Q):
        block_idx, q_idx, Q = block_idx_q_idx_Q
        return _attn_block_fn(
            block_idx,
            q_idx,
            Q,
            K,
            V,
            kv_block_size,
            mask_fn,
            kernel_fn,
            window_left,
            window_num_blocks,
            is_causal=is_causal,
        )

    Q = rearrange(Q, "B (blocks block_size) Hq d -> blocks B block_size Hq d", block_size=block_size)
    q_idx = jnp.reshape(jnp.arange(N), (N//block_size, block_size))
    block_idx = jnp.arange(N//block_size)

    values, log_normalizer = jax.lax.map(
        attn_fn,
        (block_idx, q_idx, Q),
    )

    values = rearrange(values, "blocks B block_size Hq dv -> B (blocks block_size) Hq dv")
    log_normalizer = rearrange(
        log_normalizer,
        "blocks B block_size Hq -> B (blocks block_size) Hq",
    )

    return values, log_normalizer


@partial(
    jax.custom_vjp,
    nondiff_argnames=[
        "kernel_fn",
        "mask_fn",
        "block_size",
        "kv_block_size",
        "window_size",
        "is_causal",
        "backward_strategy",
    ],
)
def _masked_attention_via_map(
    Q: Array,
    K: Array,
    V: Array,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[Tuple[int, int, int], Array], Array]] = None,
    block_size=None,
    kv_block_size=None,
    window_size=None,
    is_causal=False,
    backward_strategy="auto",
) -> Array:
    del backward_strategy
    values, _ = _masked_attention_via_map_impl(
        Q,
        K,
        V,
        kernel_fn=kernel_fn,
        mask_fn=mask_fn,
        block_size=block_size,
        kv_block_size=kv_block_size,
        window_size=window_size,
        is_causal=is_causal,
    )
    return values


def _masked_attention_via_map_fwd(
    Q: Array,
    K: Array,
    V: Array,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[int, Array], Array]] = None,
    block_size=None,
    kv_block_size=None,
    window_size=None,
    is_causal=False,
    backward_strategy="auto",
) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
    values, log_normalizer = _masked_attention_via_map_impl(
        Q,
        K,
        V,
        kernel_fn=kernel_fn,
        mask_fn=mask_fn,
        block_size=block_size,
        kv_block_size=kv_block_size,
        window_size=window_size,
        is_causal=is_causal,
    )
    del backward_strategy
    return values, (Q, K, V, values, log_normalizer)


@partial(
    jax.custom_vjp,
    nondiff_argnames=[
        "mask_fn",
        "block_size",
        "kv_block_size",
        "window_size",
        "is_causal",
        "backward_strategy",
    ],
)
def _masked_attention_via_mosaic(
    Q: Array,
    K: Array,
    V: Array,
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
    backward_strategy,
) -> Array:
    """Mosaic forward paired with the established tiled custom backward."""
    del block_size, kv_block_size, window_size, is_causal, backward_strategy
    from jaxmodules._mosaic_attention import mosaic_attention_forward

    values, _ = mosaic_attention_forward(Q, K, V, mask_fn)
    return values


def _masked_attention_via_mosaic_fwd(
    Q,
    K,
    V,
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
    backward_strategy,
):
    del block_size, kv_block_size, window_size, is_causal, backward_strategy
    from jaxmodules._mosaic_attention import mosaic_attention_forward

    values, log_normalizer = mosaic_attention_forward(Q, K, V, mask_fn)
    return values, (Q, K, V, values, log_normalizer)


def _standard_attention_tile_backward(
    q_block,
    k_block,
    v_block,
    grad_block,
    delta_block,
    lse_block,
    q_idx,
    k_idx,
    mask_fn,
    batch_size,
    query_heads,
    kv_heads,
    scale,
    accumulator_dtype,
    dot_precision,
):
    """Recompute one score tile and return its Q, K, and V contributions."""
    mask = _materialize_mask(
        mask_fn,
        batch_size,
        query_heads,
        kv_heads,
        q_idx,
        k_idx,
    )
    scores = jnp.einsum(
        "bqhgd,bkhd->bqhgk",
        q_block,
        k_block,
        precision=dot_precision,
        preferred_element_type=accumulator_dtype,
    )
    scores = jnp.where(mask, scores * scale, -jnp.inf)
    probabilities = jnp.exp(scores - lse_block)

    dP = jnp.einsum(
        "bqhge,bkhe->bqhgk",
        grad_block,
        v_block,
        precision=dot_precision,
        preferred_element_type=accumulator_dtype,
    )
    dS = probabilities * (dP - delta_block)

    dQ = scale * jnp.einsum(
        "bqhgk,bkhd->bqhgd",
        dS,
        k_block,
        precision=dot_precision,
        preferred_element_type=accumulator_dtype,
    )
    dK = scale * jnp.einsum(
        "bqhgk,bqhgd->bkhd",
        dS,
        q_block,
        precision=dot_precision,
        preferred_element_type=accumulator_dtype,
    )
    dV = jnp.einsum(
        "bqhgk,bqhge->bkhe",
        probabilities,
        grad_block,
        precision=dot_precision,
        preferred_element_type=accumulator_dtype,
    )
    return dQ, dK, dV


def _standard_attention_backward_query_major(
    Q,
    K,
    V,
    output,
    log_normalizer,
    upstream_grad,
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
):
    """Explicit tiled backward for standard scaled-dot-product attention."""
    B, N, Hq, d = Q.shape
    _, L, Hkv, _ = K.shape
    _, _, _, dv = V.shape
    group_size = Hq // Hkv
    accumulator_dtype = jnp.result_type(
        Q.dtype,
        K.dtype,
        V.dtype,
        upstream_grad.dtype,
        jnp.float32,
    )
    scale = jnp.asarray(d**-0.5, dtype=accumulator_dtype)
    dot_precision = jax.lax.Precision.HIGHEST

    q_blocks = rearrange(
        Q,
        "B (Q Qb) (Hkv G) d -> Q B Qb Hkv G d",
        Qb=block_size,
        Hkv=Hkv,
    )
    output_blocks = rearrange(
        output,
        "B (Q Qb) (Hkv G) dv -> Q B Qb Hkv G dv",
        Qb=block_size,
        Hkv=Hkv,
    )
    grad_blocks = rearrange(
        upstream_grad,
        "B (Q Qb) (Hkv G) dv -> Q B Qb Hkv G dv",
        Qb=block_size,
        Hkv=Hkv,
    )
    log_normalizer_blocks = rearrange(
        log_normalizer,
        "B (Q Qb) (Hkv G) -> Q B Qb Hkv G 1",
        Qb=block_size,
        Hkv=Hkv,
    )
    delta_blocks = jnp.sum(
        grad_blocks.astype(accumulator_dtype)
        * output_blocks.astype(accumulator_dtype),
        axis=-1,
        keepdims=True,
    )
    k_blocks = rearrange(
        K,
        "B (K Kb) Hkv d -> K B Kb Hkv d",
        Kb=kv_block_size,
    )
    v_blocks = rearrange(
        V,
        "B (K Kb) Hkv dv -> K B Kb Hkv dv",
        Kb=kv_block_size,
    )

    num_q_blocks = q_blocks.shape[0]
    num_kv_blocks = k_blocks.shape[0]
    q_indices = jnp.reshape(jnp.arange(N), (num_q_blocks, block_size))
    k_indices = jnp.reshape(jnp.arange(L), (num_kv_blocks, kv_block_size))

    window_left, total_window_blocks = _window_block_config(
        window_size,
        block_size,
        kv_block_size,
        num_kv_blocks,
    )

    dK_init = jnp.zeros(k_blocks.shape, dtype=accumulator_dtype)
    dV_init = jnp.zeros(v_blocks.shape, dtype=accumulator_dtype)

    def query_block_backward(dK_dV, query_data):
        dK, dV = dK_dV
        q_idx, q_block, grad_block, delta_block, lse_block = query_data
        dQ = jnp.zeros(q_block.shape, dtype=accumulator_dtype)

        if window_left is not None:
            kv_start = jnp.maximum(
                0,
                jnp.floor_divide(q_idx[0] - window_left, kv_block_size),
            )
            kv_start = jnp.minimum(
                kv_start,
                num_kv_blocks - total_window_blocks,
            )
            kv_stop = kv_start + total_window_blocks
        else:
            kv_start = 0
            kv_stop = num_kv_blocks

        if is_causal:
            causal_stop = jnp.minimum(
                jnp.floor_divide(q_idx[-1], kv_block_size) + 1,
                num_kv_blocks,
            )
            kv_stop = jnp.minimum(kv_stop, causal_stop)

        def kv_block_backward(kv_block_idx, state):
            dQ_acc, dK_acc, dV_acc = state
            k_block = k_blocks[kv_block_idx]
            v_block = v_blocks[kv_block_idx]
            dQ_block, dK_block, dV_block = _standard_attention_tile_backward(
                q_block,
                k_block,
                v_block,
                grad_block,
                delta_block,
                lse_block,
                q_idx,
                k_indices[kv_block_idx],
                mask_fn,
                B,
                Hq,
                Hkv,
                scale,
                accumulator_dtype,
                dot_precision,
            )
            dQ_acc = dQ_acc + dQ_block
            dK_block = dK_acc[kv_block_idx] + dK_block
            dV_block = dV_acc[kv_block_idx] + dV_block
            dK_acc = jax.lax.dynamic_update_slice_in_dim(
                dK_acc,
                dK_block[None],
                kv_block_idx,
                axis=0,
            )
            dV_acc = jax.lax.dynamic_update_slice_in_dim(
                dV_acc,
                dV_block[None],
                kv_block_idx,
                axis=0,
            )
            return dQ_acc, dK_acc, dV_acc

        dQ, dK, dV = jax.lax.fori_loop(
            kv_start,
            kv_stop,
            kv_block_backward,
            (dQ, dK, dV),
        )
        return (dK, dV), dQ.astype(Q.dtype)

    (dK, dV), dQ = jax.lax.scan(
        query_block_backward,
        init=(dK_init, dV_init),
        xs=(
            q_indices,
            q_blocks,
            grad_blocks,
            delta_blocks,
            log_normalizer_blocks,
        ),
    )

    dQ = rearrange(dQ, "Q B Qb Hkv G d -> B (Q Qb) (Hkv G) d")
    dK = rearrange(dK, "K B Kb Hkv d -> B (K Kb) Hkv d")
    dV = rearrange(dV, "K B Kb Hkv dv -> B (K Kb) Hkv dv")
    return dQ.astype(Q.dtype), dK.astype(K.dtype), dV.astype(V.dtype)


def _standard_attention_backward_key_major(
    Q,
    K,
    V,
    output,
    log_normalizer,
    upstream_grad,
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
):
    """Tiled backward carrying dQ and emitting each completed dK/dV tile."""
    B, N, Hq, d = Q.shape
    _, L, Hkv, _ = K.shape
    _, _, _, dv = V.shape
    accumulator_dtype = jnp.result_type(
        Q.dtype,
        K.dtype,
        V.dtype,
        upstream_grad.dtype,
        jnp.float32,
    )
    scale = jnp.asarray(d**-0.5, dtype=accumulator_dtype)
    dot_precision = jax.lax.Precision.HIGHEST

    q_blocks = rearrange(
        Q,
        "B (Q Qb) (Hkv G) d -> Q B Qb Hkv G d",
        Qb=block_size,
        Hkv=Hkv,
    )
    output_blocks = rearrange(
        output,
        "B (Q Qb) (Hkv G) dv -> Q B Qb Hkv G dv",
        Qb=block_size,
        Hkv=Hkv,
    )
    grad_blocks = rearrange(
        upstream_grad,
        "B (Q Qb) (Hkv G) dv -> Q B Qb Hkv G dv",
        Qb=block_size,
        Hkv=Hkv,
    )
    log_normalizer_blocks = rearrange(
        log_normalizer,
        "B (Q Qb) (Hkv G) -> Q B Qb Hkv G 1",
        Qb=block_size,
        Hkv=Hkv,
    )
    delta_blocks = jnp.sum(
        grad_blocks.astype(accumulator_dtype)
        * output_blocks.astype(accumulator_dtype),
        axis=-1,
        keepdims=True,
    )
    k_blocks = rearrange(
        K,
        "B (K Kb) Hkv d -> K B Kb Hkv d",
        Kb=kv_block_size,
    )
    v_blocks = rearrange(
        V,
        "B (K Kb) Hkv dv -> K B Kb Hkv dv",
        Kb=kv_block_size,
    )

    num_q_blocks = q_blocks.shape[0]
    num_kv_blocks = k_blocks.shape[0]
    q_indices = jnp.reshape(jnp.arange(N), (num_q_blocks, block_size))
    k_indices = jnp.reshape(jnp.arange(L), (num_kv_blocks, kv_block_size))
    kv_block_indices = jnp.arange(num_kv_blocks)

    window_left, total_window_blocks = _window_block_config(
        window_size,
        block_size,
        kv_block_size,
        num_kv_blocks,
    )
    dQ_init = jnp.zeros(q_blocks.shape, dtype=accumulator_dtype)

    def key_block_backward(dQ, key_data):
        kv_block_idx, k_idx, k_block, v_block = key_data
        dK = jnp.zeros(k_block.shape, dtype=accumulator_dtype)
        dV = jnp.zeros(v_block.shape, dtype=accumulator_dtype)

        if is_causal:
            q_start = jnp.minimum(
                jnp.floor_divide(k_idx[0], block_size),
                num_q_blocks,
            )
        else:
            q_start = 0

        def query_block_backward(query_block_idx, state):
            dQ_acc, dK_acc, dV_acc = state
            q_idx = q_indices[query_block_idx]

            def accumulate(block_state):
                dQ_carry, dK_carry, dV_carry = block_state
                dQ_block, dK_block, dV_block = (
                    _standard_attention_tile_backward(
                        q_blocks[query_block_idx],
                        k_block,
                        v_block,
                        grad_blocks[query_block_idx],
                        delta_blocks[query_block_idx],
                        log_normalizer_blocks[query_block_idx],
                        q_idx,
                        k_idx,
                        mask_fn,
                        B,
                        Hq,
                        Hkv,
                        scale,
                        accumulator_dtype,
                        dot_precision,
                    )
                )
                dQ_block = dQ_carry[query_block_idx] + dQ_block
                dQ_carry = jax.lax.dynamic_update_slice_in_dim(
                    dQ_carry,
                    dQ_block[None],
                    query_block_idx,
                    axis=0,
                )
                return (
                    dQ_carry,
                    dK_carry + dK_block,
                    dV_carry + dV_block,
                )

            if window_left is not None:
                window_start = jnp.maximum(
                    0,
                    jnp.floor_divide(
                        q_idx[0] - window_left,
                        kv_block_size,
                    ),
                )
                window_start = jnp.minimum(
                    window_start,
                    num_kv_blocks - total_window_blocks,
                )
                process_block = (kv_block_idx >= window_start) & (
                    kv_block_idx < window_start + total_window_blocks
                )
                return jax.lax.cond(
                    process_block,
                    accumulate,
                    lambda block_state: block_state,
                    (dQ_acc, dK_acc, dV_acc),
                )

            return accumulate((dQ_acc, dK_acc, dV_acc))

        dQ, dK, dV = jax.lax.fori_loop(
            q_start,
            num_q_blocks,
            query_block_backward,
            (dQ, dK, dV),
        )
        return dQ, (dK.astype(K.dtype), dV.astype(V.dtype))

    dQ, (dK, dV) = jax.lax.scan(
        key_block_backward,
        init=dQ_init,
        xs=(kv_block_indices, k_indices, k_blocks, v_blocks),
    )

    dQ = rearrange(dQ, "Q B Qb Hkv G d -> B (Q Qb) (Hkv G) d")
    dK = rearrange(dK, "K B Kb Hkv d -> B (K Kb) Hkv d")
    dV = rearrange(dV, "K B Kb Hkv dv -> B (K Kb) Hkv dv")
    return dQ.astype(Q.dtype), dK, dV


def _standard_attention_backward_two_pass(
    Q,
    K,
    V,
    output,
    log_normalizer,
    upstream_grad,
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
):
    """Minimize memory by completing Q and K/V gradients in separate passes."""
    B, N, Hq, d = Q.shape
    _, L, Hkv, _ = K.shape
    accumulator_dtype = jnp.result_type(
        Q.dtype,
        K.dtype,
        V.dtype,
        upstream_grad.dtype,
        jnp.float32,
    )
    scale = jnp.asarray(d**-0.5, dtype=accumulator_dtype)
    dot_precision = jax.lax.Precision.HIGHEST

    q_blocks = rearrange(
        Q,
        "B (Q Qb) (Hkv G) d -> Q B Qb Hkv G d",
        Qb=block_size,
        Hkv=Hkv,
    )
    output_blocks = rearrange(
        output,
        "B (Q Qb) (Hkv G) dv -> Q B Qb Hkv G dv",
        Qb=block_size,
        Hkv=Hkv,
    )
    grad_blocks = rearrange(
        upstream_grad,
        "B (Q Qb) (Hkv G) dv -> Q B Qb Hkv G dv",
        Qb=block_size,
        Hkv=Hkv,
    )
    log_normalizer_blocks = rearrange(
        log_normalizer,
        "B (Q Qb) (Hkv G) -> Q B Qb Hkv G 1",
        Qb=block_size,
        Hkv=Hkv,
    )
    delta_blocks = jnp.sum(
        grad_blocks.astype(accumulator_dtype)
        * output_blocks.astype(accumulator_dtype),
        axis=-1,
        keepdims=True,
    )
    k_blocks = rearrange(
        K,
        "B (K Kb) Hkv d -> K B Kb Hkv d",
        Kb=kv_block_size,
    )
    v_blocks = rearrange(
        V,
        "B (K Kb) Hkv dv -> K B Kb Hkv dv",
        Kb=kv_block_size,
    )

    num_q_blocks = q_blocks.shape[0]
    num_kv_blocks = k_blocks.shape[0]
    q_indices = jnp.reshape(jnp.arange(N), (num_q_blocks, block_size))
    k_indices = jnp.reshape(jnp.arange(L), (num_kv_blocks, kv_block_size))
    kv_block_indices = jnp.arange(num_kv_blocks)
    window_left, total_window_blocks = _window_block_config(
        window_size,
        block_size,
        kv_block_size,
        num_kv_blocks,
    )

    def query_gradient(query_data):
        q_idx, q_block, grad_block, delta_block, lse_block = query_data
        dQ = jnp.zeros(q_block.shape, dtype=accumulator_dtype)

        if window_left is not None:
            kv_start = jnp.maximum(
                0,
                jnp.floor_divide(q_idx[0] - window_left, kv_block_size),
            )
            kv_start = jnp.minimum(
                kv_start,
                num_kv_blocks - total_window_blocks,
            )
            kv_stop = kv_start + total_window_blocks
        else:
            kv_start = 0
            kv_stop = num_kv_blocks

        if is_causal:
            causal_stop = jnp.minimum(
                jnp.floor_divide(q_idx[-1], kv_block_size) + 1,
                num_kv_blocks,
            )
            kv_stop = jnp.minimum(kv_stop, causal_stop)

        def accumulate(kv_block_idx, dQ_acc):
            dQ_block, _, _ = _standard_attention_tile_backward(
                q_block,
                k_blocks[kv_block_idx],
                v_blocks[kv_block_idx],
                grad_block,
                delta_block,
                lse_block,
                q_idx,
                k_indices[kv_block_idx],
                mask_fn,
                B,
                Hq,
                Hkv,
                scale,
                accumulator_dtype,
                dot_precision,
            )
            return dQ_acc + dQ_block

        dQ = jax.lax.fori_loop(kv_start, kv_stop, accumulate, dQ)
        return dQ.astype(Q.dtype)

    dQ = jax.lax.map(
        query_gradient,
        (
            q_indices,
            q_blocks,
            grad_blocks,
            delta_blocks,
            log_normalizer_blocks,
        ),
    )

    def kv_gradients(key_data):
        kv_block_idx, k_idx, k_block, v_block = key_data
        dK = jnp.zeros(k_block.shape, dtype=accumulator_dtype)
        dV = jnp.zeros(v_block.shape, dtype=accumulator_dtype)

        if is_causal:
            q_start = jnp.minimum(
                jnp.floor_divide(k_idx[0], block_size),
                num_q_blocks,
            )
        else:
            q_start = 0

        def accumulate(query_block_idx, dK_dV):
            dK_acc, dV_acc = dK_dV
            q_idx = q_indices[query_block_idx]

            def update(block_state):
                dK_carry, dV_carry = block_state
                _, dK_block, dV_block = _standard_attention_tile_backward(
                    q_blocks[query_block_idx],
                    k_block,
                    v_block,
                    grad_blocks[query_block_idx],
                    delta_blocks[query_block_idx],
                    log_normalizer_blocks[query_block_idx],
                    q_idx,
                    k_idx,
                    mask_fn,
                    B,
                    Hq,
                    Hkv,
                    scale,
                    accumulator_dtype,
                    dot_precision,
                )
                return dK_carry + dK_block, dV_carry + dV_block

            if window_left is not None:
                window_start = jnp.maximum(
                    0,
                    jnp.floor_divide(
                        q_idx[0] - window_left,
                        kv_block_size,
                    ),
                )
                window_start = jnp.minimum(
                    window_start,
                    num_kv_blocks - total_window_blocks,
                )
                process_block = (kv_block_idx >= window_start) & (
                    kv_block_idx < window_start + total_window_blocks
                )
                return jax.lax.cond(
                    process_block,
                    update,
                    lambda block_state: block_state,
                    (dK_acc, dV_acc),
                )

            return update((dK_acc, dV_acc))

        dK, dV = jax.lax.fori_loop(
            q_start,
            num_q_blocks,
            accumulate,
            (dK, dV),
        )
        return dK.astype(K.dtype), dV.astype(V.dtype)

    dK, dV = jax.lax.map(
        kv_gradients,
        (kv_block_indices, k_indices, k_blocks, v_blocks),
    )

    dQ = rearrange(dQ, "Q B Qb Hkv G d -> B (Q Qb) (Hkv G) d")
    dK = rearrange(dK, "K B Kb Hkv d -> B (K Kb) Hkv d")
    dV = rearrange(dV, "K B Kb Hkv dv -> B (K Kb) Hkv dv")
    return dQ, dK, dV


def _standard_attention_backward(
    Q,
    K,
    V,
    output,
    log_normalizer,
    upstream_grad,
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
    backward_strategy,
):
    """Choose the traversal with the smaller full-precision gradient carry."""
    if backward_strategy == "minimal":
        return _standard_attention_backward_two_pass(
            Q,
            K,
            V,
            output,
            log_normalizer,
            upstream_grad,
            mask_fn,
            block_size,
            kv_block_size,
            window_size,
            is_causal,
        )

    query_carry_size = math.prod(Q.shape)
    kv_carry_size = math.prod(K.shape) + math.prod(V.shape)
    implementation = (
        _standard_attention_backward_key_major
        if query_carry_size <= kv_carry_size
        else _standard_attention_backward_query_major
    )
    return implementation(
        Q,
        K,
        V,
        output,
        log_normalizer,
        upstream_grad,
        mask_fn,
        block_size,
        kv_block_size,
        window_size,
        is_causal,
    )


def _masked_attention_via_map_bwd(
    kernel_fn: Callable[[Array, Array], float],
    mask_fn: Optional[Union[Callable[int, Array], Array]],
    block_size,
    kv_block_size,
    window_size,
    is_causal,
    backward_strategy,
    res,
    upstream_grad,
):
    Q, K, V, output, log_normalizer = res
    Bk, L, Hk, d = K.shape
    Bq, N, Hq, dq = Q.shape
    Bv, Lv, Hv, dv = V.shape

    if block_size is None:
        block_size = N
    if kv_block_size is None:
        kv_block_size = block_size

    window_left, window_num_blocks = _window_block_config(
        window_size,
        block_size,
        kv_block_size,
        L // kv_block_size,
    )

    assert d == dq and Lv == L, (
        f"shape mismatch in K {K.shape}, Q {Q.shape} and V {V.shape}"
    )
    assert Bk == Bq == Bv, "Q, K, and V must have the same batch dimension"

    assert Hq % Hk == 0, "Hq must be divisible by Hk"
    assert Hv == Hk, "Hv must be equal to Hk"

    assert block_size is None or N % block_size == 0, (
        f"block_size must divide number of queries!"
    )
    assert L % kv_block_size == 0, (
        "kv_block_size must divide the number of keys and values!"
    )

    if kernel_fn is default_kernel:
        return _standard_attention_backward(
            Q,
            K,
            V,
            output,
            log_normalizer,
            upstream_grad,
            mask_fn,
            block_size,
            kv_block_size,
            window_size,
            is_causal,
            backward_strategy,
        )

    def attn_fn(dK_dV, block_idx_q_idx_q_g):
        block_idx, q_idx, q, g = block_idx_q_idx_q_g
        dK_carry, dV_carry = dK_dV

        def get_values(q, K, V):
            values, _ = _attn_block_fn(
                block_idx,
                q_idx,
                q,
                K,
                V,
                kv_block_size,
                mask_fn,
                kernel_fn,
                window_left,
                window_num_blocks,
                is_causal=is_causal,
                # Disable block skipping in backward pass - fori_loop with
                # dynamic bounds doesn't support reverse-mode differentiation
                use_causal_block_skipping=False,
            )
            return values

        _, vjp_fn = jax.vjp(get_values, q, K, V)
        dq, dK, dV = vjp_fn(g)

        dK_carry = dK_carry + dK
        dV_carry = dV_carry + dV
        return (dK_carry, dV_carry), dq


    # break it up into blocks of size block_size
    g_blocks = rearrange(upstream_grad, "B (blocks block_size) Hq dv -> blocks B block_size Hq dv", block_size=block_size)



    Q = rearrange(Q, "B (blocks block_size) Hq dq -> blocks B block_size Hq dq", block_size=block_size)

    q_idx = jnp.reshape(jnp.arange(N), (N//block_size, block_size))
    block_idx = jnp.arange(N//block_size)
    
    (k_grad, v_grad), q_grad = jax.lax.scan(
        attn_fn, init=(jnp.zeros_like(K), jnp.zeros_like(V)), xs=(block_idx, q_idx, Q, g_blocks)
    )
    q_grad = rearrange(q_grad, "blocks B block_size Hq dq -> B (blocks block_size) Hq dq")
    return q_grad, k_grad, v_grad


def _masked_attention_via_mosaic_bwd(
    mask_fn,
    block_size,
    kv_block_size,
    window_size,
    is_causal,
    backward_strategy,
    res,
    upstream_grad,
):
    del (
        block_size,
        kv_block_size,
        window_size,
        is_causal,
        backward_strategy,
    )
    Q, K, V, output, log_normalizer = res
    from jaxmodules._mosaic_attention import mosaic_attention_backward

    return mosaic_attention_backward(
        Q,
        K,
        V,
        output,
        log_normalizer,
        upstream_grad,
        mask_fn,
    )


_masked_attention_via_map.defvjp(_masked_attention_via_map_fwd, _masked_attention_via_map_bwd)
_masked_attention_via_mosaic.defvjp(
    _masked_attention_via_mosaic_fwd,
    _masked_attention_via_mosaic_bwd,
)


def _canonicalize_mask_fn(mask_fn, is_causal):
    """Convert mask_fn to canonical 4-arg (b, h, q, k) form.

    Handles:
    - 3-arg mask functions (h, q, k) -> wrapped to ignore batch
    - is_causal=True -> causal mask (q >= k)
    - None -> no masking (always True)
    """
    if mask_fn is not None and mask_fn.__code__.co_argcount == 3:
        three_arg_mask_fn = mask_fn
        mask_fn = lambda b, h, q, k: three_arg_mask_fn(h, q, k)

    if is_causal and mask_fn is not None:
        raise ValueError("cannot specify both 'is_causal' and 'mask_fn'!")
    if is_causal:
        return _causal_mask
    if mask_fn is None:
        return _unmasked
    return mask_fn


def _pad_for_block_sizes(
    Q,
    K,
    V,
    query_block_size,
    kv_block_size,
    mask_fn,
):
    """Pad Q and K/V independently to their tile sizes, adjusting mask_fn.

    Returns (Q, K, V, mask_fn, padding_size_Q) where padding_size_Q is 0 if no padding.
    """
    N = Q.shape[1]
    L = K.shape[1]

    padding_size_Q = (-N) % query_block_size
    padding_size_KV = (-L) % kv_block_size
    if padding_size_Q == 0 and padding_size_KV == 0:
        return Q, K, V, mask_fn, 0

    if padding_size_Q:
        Q = jnp.pad(
            Q,
            ((0, 0), (0, padding_size_Q), (0, 0), (0, 0)),
            mode="constant",
        )
    if padding_size_KV:
        K = jnp.pad(
            K,
            ((0, 0), (0, padding_size_KV), (0, 0), (0, 0)),
            mode="constant",
        )
        V = jnp.pad(
            V,
            ((0, 0), (0, padding_size_KV), (0, 0), (0, 0)),
            mode="constant",
        )

    unpadded_mask_fn = mask_fn
    mask_fn = lambda b, h, q, k: unpadded_mask_fn(b, h, q, k) & (q < N) & (k < L)

    return Q, K, V, mask_fn, padding_size_Q


def _default_attention_block_sizes(Q, K, V, kv_block_size):
    """Choose tiles that keep the FP32 score tile near 32 MiB."""
    batch_size, query_length, query_heads, _ = Q.shape
    _, kv_length, kv_heads, _ = K.shape

    accumulator_dtype = jnp.result_type(
        Q.dtype,
        K.dtype,
        V.dtype,
        jnp.float32,
    )
    score_element_bytes = jnp.dtype(accumulator_dtype).itemsize
    target_score_tile_bytes = 32 * 1024 * 1024
    tile_element_capacity = max(
        1,
        target_score_tile_bytes
        // (batch_size * query_heads * score_element_bytes),
    )

    def bounded_power_of_two(capacity, upper_bound):
        block_size = 1 << (max(1, capacity).bit_length() - 1)
        return min(upper_bound, max(64, block_size))

    if kv_block_size is not None:
        query_capacity = tile_element_capacity // kv_block_size
        query_block_size = bounded_power_of_two(query_capacity, 2048)
        return min(query_length, query_block_size), kv_block_size

    query_carry_size = query_length * query_heads * Q.shape[-1]
    kv_carry_size = kv_length * kv_heads * (K.shape[-1] + V.shape[-1])
    if query_carry_size > kv_carry_size:
        # Query-major backward benefits from a balanced score tile.
        query_block_size = bounded_power_of_two(
            math.isqrt(tile_element_capacity),
            2048,
        )
        kv_capacity = tile_element_capacity // query_block_size
        kv_block_size = bounded_power_of_two(kv_capacity, 1024)
    else:
        # Key-major backward favors a wider query tile on the tested GPU.
        kv_block_size = min(kv_length, 1024)
        query_capacity = tile_element_capacity // kv_block_size
        query_block_size = bounded_power_of_two(query_capacity, 2048)
        if query_block_size < kv_block_size:
            query_block_size, kv_block_size = (
                kv_block_size,
                query_block_size,
            )

    query_block_size = min(query_length, query_block_size)
    kv_block_size = min(kv_length, kv_block_size)
    return query_block_size, kv_block_size


def _can_use_mosaic_attention(Q, K, V, kernel_fn, mask_fn, window_size):
    """Return whether this call can use the conservative Mosaic fast path."""
    if (
        jax.default_backend() != "gpu"
        or kernel_fn is not default_kernel
        or window_size is not None
    ):
        return False

    from jaxmodules._mosaic_attention import supports_mosaic_attention

    return supports_mosaic_attention(Q, K, V, mask_fn)


def masked_attention_via_map(
    Q: Array,
    K: Array,
    V: Array,
    *,
    is_causal: bool = False,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[int, Array], Array]] = None,
    block_size: Optional[int] = None,
    kv_block_size: Optional[int] = None,
    window_size: Optional[Tuple[int, int]] = None,
    backward_strategy: str = "auto",
) -> Array:
    """
    attention implementation that uses jax.lax.map to perform attention in a memory-efficient way
    analogous to flash attention, but written in pure jax, and with less tricks.

    K: array of key values, shape [B, L, Hkv, d] or [L, Hkv, d]
    Q: array of queries, shape [B, N, Hq, d] or [N, Hq, d]
    V: array of values, shape [B, L, Hkv, d] or [L, Hkv, d]
    is_causal: if true, apply a causal mask
    kernel_fn: the  unnormalized attention score is kernel_fn(Q, K).
        default is q, k -> jnp.exp( <q, k> / sqrt(d) )
        The default kernel uses ``jax.lax.Precision.HIGHEST`` contractions and
        accumulates low-precision inputs in at least FP32 in both passes.
    mask_fn: takes integers b, h, q, k or h, q, kand returns a boolean specifying
        the attention mask for the bth item in batch, hth head and the qth query and kth key.
        If is_causal is true, you cannot provide mask_fn; it will be generated automatically.
        If is_causal is False and mask_fn is None, then the default value of no masking will
        be used (equivalent to mask_fn = lambda b, h, q, k: True).
    block_size: Query tile size. If omitted, choose a bounded tile targeting an
        approximately 32 MiB score tile instead of materializing full attention.
        If specified and kv_block_size is omitted, use this size for both axes.
    kv_block_size: Optional independent K/V tile size. This makes it possible
        to tune the score-tile shape without changing the query tile size.
    window_size: Tuple (left, right) or None.If specified, apply a sliding window mask to the attention.
        window_size is the number of tokens to the left and right of the current block
        that are allowed to attend to the current block.
        NOTE: window_size is a *lower bound* on the enforced attention window: the true window size
        will be larger than the window size: it will be rounded to a whole number of blocks, and also
        will be even larger for keys or values near the edges of the sequence. This parameter is indended
        to control performance rather than for exact masking.
        Use the  mask_fn to explicitly enforce a constant window size if desired.
    backward_strategy: ``"auto"`` uses a one-pass backward and carries whichever
        full-precision gradient set is smaller. ``"minimal"`` recomputes score
        tiles in separate Q and K/V passes so no sequence-sized FP32 gradient
        is carried. This trades additional compute for less gradient-carry
        memory, although other live buffers can dominate the total peak.
        Neither strategy changes contraction or accumulation precision. The
        strategy applies to the optimized default kernel; custom kernels retain
        their generic custom-VJP path. Calls eligible for the Mosaic GPU fast
        path use its fixed two-pass, linear-memory backward and therefore do
        not use this fallback tuning option.
    """

    # Validate dimensions
    if K.ndim not in [3, 4]:
        raise ValueError("K must have 3 or 4 dimensions")
    if Q.ndim not in [3, 4]:
        raise ValueError("Q must have 3 or 4 dimensions")
    if V.ndim not in [3, 4]:
        raise ValueError("V must have 3 or 4 dimensions")
    if not (K.ndim == Q.ndim == V.ndim):
        raise ValueError("Q, K, and V must have the same number of dimensions")

    # Handle optional batch dimension
    added_batch_dim = False
    if Q.ndim == 3:
        added_batch_dim = True
        K = K[None, :, :, :]
        Q = Q[None, :, :, :]
        V = V[None, :, :, :]

    if not (K.shape[0] == Q.shape[0] == V.shape[0]):
        raise ValueError("Q, K, and V must have the same batch dimension")
    if K.shape[1] != V.shape[1]:
        raise ValueError("K and V must have the same sequence length")
    if K.shape[2] != V.shape[2]:
        raise ValueError("K and V must have the same number of heads")
    if K.shape[3] != Q.shape[3]:
        raise ValueError("Q and K must have the same feature dimension")
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive")
    if kv_block_size is not None and kv_block_size <= 0:
        raise ValueError("kv_block_size must be positive")
    if backward_strategy not in ("auto", "minimal"):
        raise ValueError("backward_strategy must be 'auto' or 'minimal'")

    # Canonicalize mask_fn to 4-arg form
    mask_fn = _canonicalize_mask_fn(mask_fn, is_causal)

    # Pad for block_size if needed
    N = Q.shape[1]
    if block_size is None:
        effective_block_size, effective_kv_block_size = (
            _default_attention_block_sizes(
                Q,
                K,
                V,
                kv_block_size,
            )
        )
    else:
        effective_block_size = block_size
        effective_kv_block_size = (
            block_size if kv_block_size is None else kv_block_size
        )
    Q, K, V, mask_fn, padding_size = _pad_for_block_sizes(
        Q,
        K,
        V,
        effective_block_size,
        effective_kv_block_size,
        mask_fn,
    )

    # Core computation. The callable mask API is unchanged: supported
    # coordinate-only JAX expressions use Mosaic GPU, while every other case
    # retains the established mapped implementation.
    if _can_use_mosaic_attention(
        Q,
        K,
        V,
        kernel_fn,
        mask_fn,
        window_size,
    ):
        result = _masked_attention_via_mosaic(
            Q,
            K,
            V,
            mask_fn=mask_fn,
            block_size=effective_block_size,
            kv_block_size=effective_kv_block_size,
            window_size=window_size,
            is_causal=is_causal,
            backward_strategy=backward_strategy,
        )
    else:
        result = _masked_attention_via_map(
            Q,
            K,
            V,
            kernel_fn=kernel_fn,
            mask_fn=mask_fn,
            block_size=effective_block_size,
            kv_block_size=effective_kv_block_size,
            window_size=window_size,
            is_causal=is_causal,
            backward_strategy=backward_strategy,
        )

    # Remove padding and batch dimension
    if padding_size:
        result = result[:, :N, :, :]
    if added_batch_dim:
        result = result[0]

    return result


def _flex_attention(
    query: Array,
    key: Array,
    value: Array,
    score_mod: Optional[Callable] = None,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[Array] = None,
    enable_gqa: bool = False,
    return_lse=False,
):
    """
    Flexible attention implementation that supports block-sparse attention patterns.
    This is a JAX implementation of the PyTorch flex_attention function.

    Args:
        query: Query tensor of shape (B, Hq, L, E)
        key: Key tensor of shape (B, Hkv, S, E)
        value: Value tensor of shape (B, Hkv, S, Ev)
        score_mod: Optional function to modify attention scores
        block_mask: Optional BlockMask to specify block-sparse attention pattern
        scale: Optional scaling factor for attention scores. If None, uses 1/sqrt(E)
        enable_gqa: If True, enables grouped-query attention where Hq can be larger than Hkv
        return_lse: If True, returns log-sum-exp of attention scores along with output (currently not supported)

    Returns:
        If return_lse is False:
            Output tensor of shape (B, Hq, L, Ev)
        If return_lse is True:
            Tuple of (output, lse) where:
            - output: Output tensor of shape (B, Hq, L, Ev)
            - lse: Log-sum-exp of attention scores
    """

    if return_lse:
        raise NotImplementedError("return_lse is not supported yet")

    B, Hq, L, E = query.shape
    Bk, Hkv, S, Ek = key.shape
    Bv, Hv, Sv, Ev = value.shape

    if scale is None:
        scale = 1.0 / jnp.sqrt(E)

    assert E == Ek, "query and key must have the same embedding dimension"
    assert B == Bk, "query and key must have the same batch dimension"
    assert Sv == S, "value and key must have the same sequence length"
    assert Bv == B, "value and query must have the same batch dimension"
    assert Hv == Hkv, "value and key must have the same head count"

    if block_mask is None:
        Q_BLOCK_SIZE = L
        KV_BLOCK_SIZE = S
        block_mask = BlockMask.full_mask(B, Hq, L, S, (Q_BLOCK_SIZE, KV_BLOCK_SIZE))
    else:
        Q_BLOCK_SIZE = block_mask.Q_BLOCK_SIZE
        KV_BLOCK_SIZE = block_mask.KV_BLOCK_SIZE
        assert L == block_mask.Q_LEN, "query length must match block mask"
        assert S == block_mask.KV_LEN, "key length must match block mask"

    # handle broadcasting the block mask over batch and head dimension
    # jax seems to allow out-of-bounds indexing by clipping the index, so
    # technically this would allow the broadcasting to work automatically
    # but this seems like non-obvious behavior so I don't want to rely on it.
    broadcast_mask_B = block_mask.B == 1
    broadcast_mask_H = block_mask.H == 1

    assert L % Q_BLOCK_SIZE == 0, "query length must be divisible by Q_BLOCK_SIZE"
    assert S % KV_BLOCK_SIZE == 0, "key length must be divisible by KV_BLOCK_SIZE"

    if not enable_gqa:
        assert Hq == Hkv, (
            "query and key must have the same head count, unless enable_gqa is True"
        )
    assert Hq % Hkv == 0, "kv head count must divide query head count"

    GROUP_SIZE = Hq // Hkv

    query = rearrange(
        query, "B (Hkv G) (L Qb) E -> B Hkv G L Qb E", Hkv=Hkv, Qb=Q_BLOCK_SIZE
    )
    key = rearrange(key, "B Hkv (S KVb) E -> B Hkv S KVb E", KVb=KV_BLOCK_SIZE)
    value = rearrange(value, "B Hvk (S KVb) Ev -> B Hvk S KVb Ev", KVb=KV_BLOCK_SIZE)

    def get_score_for_query_kv_block(b, h, g, l, s):
        score = einsum(query[b, h, g, l], key[b, h, s], "Qb E, KVb E -> Qb KVb") * scale
        if score_mod is not None:
            score = multi_vmap(
                lambda score, qidx, kidx: score_mod(
                    score, b, h, l * Q_BLOCK_SIZE + qidx, s * KV_BLOCK_SIZE + kidx
                ),
                in_axes=((0, 0, None), (1, None, 0)),
                out_axes=(0, 1),
            )(
                score,
                jnp.arange(Q_BLOCK_SIZE, dtype=jnp.int32),  # Q_BLOCK_SIZE
                jnp.arange(KV_BLOCK_SIZE, dtype=jnp.int32),  # KV_BLOCK_SIZE
            )
        return score

    def accumulate_value_for_query_block(
        b, h, g, l, s, accumulated, is_over_limit, do_mask=False
    ):
        return jax.lax.cond(
            is_over_limit,
            lambda b, h, g, l, s, accumulated: accumulated,
            lambda b, h, g, l, s, accumulated: _accumulate_value_for_query_block(
                b, h, g, l, s, accumulated, do_mask=do_mask
            ),
            b,
            h,
            g,
            l,
            s,
            accumulated,
        )

    def _accumulate_value_for_query_block(b, h, g, l, s, accumulated, do_mask=False):
        result_carry, sum_exp_score, max_score = accumulated
        score = get_score_for_query_kv_block(b, h, g, l, s)
        if broadcast_mask_B:
            block_b = 0
        else:
            block_b = b

        if broadcast_mask_H:
            block_h = 0
        else:
            block_h = h

        # using a block rather than just a scalar -jnp.inf maybe be slightly faster
        # when not jitted; haven't tested with jit though.
        inf_block = jnp.full_like(score, -jnp.inf)

        if do_mask:
            mask = block_mask.get_mask_for_partial_block(block_b, block_h, l, s)
            masked_score = jnp.where(mask, score, inf_block)
        else:
            mask = jnp.ones_like(score)
            masked_score = score

        next_max_score = jnp.maximum(
            max_score, jnp.max(masked_score, axis=-1, keepdims=True)
        )

        score_normalized = score - next_max_score
        score_normalized = jnp.where(mask, score_normalized, inf_block)
        value_for_block = einsum(
            jnp.exp(score_normalized), value[b, h, s], "Qb KVb, KVb Ev -> Qb Ev"
        )

        max_score_delta = max_score - jnp.where(
            next_max_score == -jnp.inf, 0.0, next_max_score
        )
        next_sum_exp_score = sum_exp_score * jnp.exp(max_score_delta) + jnp.sum(
            jnp.exp(score_normalized), axis=-1, keepdims=True
        )

        carry_multiplier = jnp.where(
            next_sum_exp_score == 0,
            0.0,
            jnp.exp(max_score_delta) * (sum_exp_score / next_sum_exp_score),
        )
        value_multiplier = jnp.where(
            next_sum_exp_score == 0, 0.0, 1.0 / next_sum_exp_score
        )

        next_result_carry = (
            result_carry * carry_multiplier + value_for_block * value_multiplier
        )
        return (next_result_carry, next_sum_exp_score, next_max_score)

    def get_value_from_full_masks_for_query_block(b, h, g, l):
        hq = h * GROUP_SIZE + g

        if broadcast_mask_B:
            block_b = 0
        else:
            block_b = b

        if broadcast_mask_H:
            block_hq = 0
        else:
            block_hq = hq

        full_block_limit = block_mask.full_kv_num_blocks[block_b, block_hq, l]
        full_kv_indices = block_mask.full_kv_indices[block_b, block_hq, l]

        result_carry = jnp.zeros((Q_BLOCK_SIZE, Ev))
        sum_exp_score = jnp.zeros((Q_BLOCK_SIZE, 1))
        max_score = jnp.full((Q_BLOCK_SIZE, 1), -jnp.inf)

        result_carry, sum_exp_score, max_score = jax.lax.fori_loop(
            lower=0,
            upper=full_kv_indices.shape[0],
            body_fun=lambda j, acc: accumulate_value_for_query_block(
                b,
                h,
                g,
                l,
                full_kv_indices[j],
                acc,
                j >= full_block_limit,
                do_mask=False,
            ),
            init_val=(result_carry, sum_exp_score, max_score),
        )

        partial_block_limit = block_mask.kv_num_blocks[block_b, block_hq, l]
        kv_indices = block_mask.kv_indices[block_b, block_hq, l]
        result_carry, sum_exp_score, max_score = jax.lax.fori_loop(
            lower=0,
            upper=kv_indices.shape[0],
            body_fun=lambda j, acc: accumulate_value_for_query_block(
                b, h, g, l, kv_indices[j], acc, j >= partial_block_limit, do_mask=True
            ),
            init_val=(result_carry, sum_exp_score, max_score),
        )

        return result_carry

    result = array_from_coords(
        shape=(B, Hkv, GROUP_SIZE, L // Q_BLOCK_SIZE),
        fn=get_value_from_full_masks_for_query_block,
    )

    result = rearrange(
        result, "B Hkv G L Qb Ev -> B (Hkv G) (L Qb) Ev", Qb=Q_BLOCK_SIZE
    )

    return result


flex_attention = jax.jit(
    _flex_attention, static_argnames=["score_mod", "enable_gqa", "return_lse"]
)


def _flex_attention_slow(
    query: Array,
    key: Array,
    value: Array,
    score_mod: Optional[Callable] = None,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[Array] = None,
    enable_gqa: bool = False,
    return_lse=False,
):
    """
    Slower but more slightly more straightforward implementation of flex_attention.
    This is used for testing and debugging purposes.

    Args:
        query: Query tensor of shape (B, Hq, L, E)
        key: Key tensor of shape (B, Hkv, S, E)
        value: Value tensor of shape (B, Hkv, S, Ev)
        score_mod: Optional function to modify attention scores
        block_mask: Optional BlockMask to specify block-sparse attention pattern
        scale: Optional scaling factor for attention scores. If None, uses 1/sqrt(E)
        enable_gqa: If True, enables grouped-query attention where Hq can be larger than Hkv
        return_lse: If True, returns log-sum-exp of attention scores along with output

    Returns:
        If return_lse is False:
            Output tensor of shape (B, Hq, L, Ev)
        If return_lse is True:
            Tuple of (output, lse) where:
            - output: Output tensor of shape (B, Hq, L, Ev)
            - lse: Log-sum-exp of attention scores
    """

    # first, let's do a naive implementation to make sure it's working

    B, Hq, L, E = query.shape
    Bk, Hkv, S, Ek = key.shape
    Bv, Hv, Sv, Ev = value.shape

    assert E == Ek, "query and key must have the same embedding dimension"
    assert B == Bk, "query and key must have the same batch dimension"
    assert Sv == S, "value and key must have the same sequence length"
    assert Bv == B, "value and query must have the same batch dimension"
    assert Hv == Hkv, "value and key must have the same head count"

    if scale is None:
        scale = 1.0 / jnp.sqrt(E)

    if block_mask is None:
        Q_BLOCK_SIZE = L
        KV_BLOCK_SIZE = S
    else:
        Q_BLOCK_SIZE = block_mask.Q_BLOCK_SIZE
        KV_BLOCK_SIZE = block_mask.KV_BLOCK_SIZE
        assert L == block_mask.Q_LEN, "query length must match block mask"
        assert S == block_mask.KV_LEN, "key length must match block mask"

    assert L % Q_BLOCK_SIZE == 0, "query length must be divisible by Q_BLOCK_SIZE"
    assert S % KV_BLOCK_SIZE == 0, "key length must be divisible by KV_BLOCK_SIZE"

    assert Hq % Hkv == 0, "kv head count must divide query head count"

    GROUP_SIZE = Hq // Hkv

    query = rearrange(
        query, "B (Hkv G) (L Qb) E -> B Hkv G L Qb E", Hkv=Hkv, Qb=Q_BLOCK_SIZE
    )
    key = rearrange(key, "B Hkv (S KVb) E -> B Hkv S KVb E", KVb=KV_BLOCK_SIZE)

    scores = (
        einsum(query, key, "B Hkv G L Qb E, B Hkv S KVb E -> B Hkv G L S Qb KVb")
        * scale
    )
    if score_mod is not None:

        def block_grouped_score_mod(score, b, h, g, l, s, qb, kb):
            h = h * GROUP_SIZE + g
            l = l * Q_BLOCK_SIZE + qb
            s = s * KV_BLOCK_SIZE + kb
            return score_mod(score, b, h, l, s)

        scores = multi_vmap(
            block_grouped_score_mod,
            in_axes=(
                (0, 0, None, None, None, None, None, None),
                (1, None, 0, None, None, None, None, None),
                (2, None, None, 0, None, None, None, None),
                (3, None, None, None, 0, None, None, None),
                (4, None, None, None, None, 0, None, None),
                (5, None, None, None, None, None, 0, None),
                (6, None, None, None, None, None, None, 0),
            ),
            out_axes=(0, 1, 2, 3, 4, 5, 6),
        )(
            scores,
            jnp.arange(B, dtype=jnp.int32),  # B
            jnp.arange(Hkv, dtype=jnp.int32),  # Hkv
            jnp.arange(GROUP_SIZE, dtype=jnp.int32),  # GROUP_SIZE
            jnp.arange(L // Q_BLOCK_SIZE, dtype=jnp.int32),  # L/Q_BLOCK_SIZE
            jnp.arange(S // KV_BLOCK_SIZE, dtype=jnp.int32),  # S/KV_BLOCK_SIZE
            jnp.arange(Q_BLOCK_SIZE, dtype=jnp.int32),  # Q_BLOCK_SIZE
            jnp.arange(KV_BLOCK_SIZE, dtype=jnp.int32),  # KV_BLOCK_SIZE
        )

    if block_mask is not None:
        mask = block_mask.materialize_mask()
        mask = rearrange(
            mask,
            "B (Hkv G) (L Qb) (S KVb) -> B Hkv G L S Qb KVb",
            Hkv=Hkv,
            Qb=Q_BLOCK_SIZE,
            KVb=KV_BLOCK_SIZE,
        )
        scores = jnp.where(mask, scores, jnp.full_like(scores, -jnp.inf))

    scores = rearrange(scores, "B Hkv G L S Qb KVb -> B Hkv G L Qb (S KVb)")
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)

    scores = jax.nn.softmax(scores, axis=-1)

    output_values = einsum(
        scores, value, "B Hkv G L Qb S, B Hkv S Ev -> B Hkv G L Qb Ev"
    )

    output_values = rearrange(output_values, "B Hkv G L Qb Ev -> B (Hkv G) (L Qb) Ev")

    return output_values

flex_attention_slow = jax.jit(
    _flex_attention_slow, static_argnames=["score_mod", "enable_gqa", "return_lse"]
)
