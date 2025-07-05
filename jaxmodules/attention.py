import jax
from jax import numpy as jnp
from typing import Callable, Union, Optional, Dict, Any, NamedTuple
from jaxtyping import Array, Float, UInt
import equinox as eqx
from einops import rearrange, repeat
from jaxmodules.vectorize import array_from_coords, multi_vmap, multi_vmap_transposed_in_axes, nested_fori_loop
from jaxmodules.block_mask import BlockMask

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


def masked_attention_via_scan(
    K: Array,
    Q: Array,
    V: Array,
    *,
    is_causal: bool = False,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[int, Array], Array]] = None,
    block_size=None,
) -> Array:
    """
    Same functionality and memory cost as attention_via_map by default, but also
    allows for user-specified masking as well as alternative kernels.

    K: array of key values, shape [L, d]
    Q: array of queries, shape [N, d]
    V: array of values, shape [L, d]
    is_causal: if true, apply a causal mask
    kernel_fn: the  unnormalized attention score is kernel_fn(Q, K).
        default is q, k -> jnp.exp( <q, k> / sqrt(d) )
    mask_fn: takes a integer index i and returns a size [d] array of booleans
        specifying the attention mask for the ith query.
        If is_causal is true, you cannot provide mask_fn; it will be generated automatically.
        If is_causal is False and mask_fn is None, then the default  value of no masking will
        be used.
    block_size: If specified, group the Q values [block_size,  d] sized blocks and
        perform attention on these blocks. Allows to trade-off the memory savings of scan.
    """

    L, d = K.shape
    N, dq = Q.shape
    Lv, dv = V.shape

    assert d == dq and Lv == L and dv == d, (
        f"shape mismatch in K {K.shape}, Q {Q.shape} and V {V.shape}"
    )

    assert block_size is None or N % block_size == 0, (
        f"block_size must divide number of queries!"
    )

    if is_causal and mask_fn is not None:
        raise ValueError("cannot specify both 'is_causal' and 'mask_fn'!")
    if is_causal:
        base_range = jnp.arange(L)
        mask_fn = lambda i: base_range <= i
    if mask_fn is None:
        mask_fn = lambda i: jnp.ones(L)

    def attn_fn(idx, q):
        # q is [block_size, d]
        # i is an integer or an array of integers
        if block_size is not None:
            mask = jax.vmap(mask_fn)(idx)  # [block_size, L]
        else:
            mask = mask_fn(idx)  # [L]

        if block_size is not None:
            # we want to make  a [block_size, L] matrix whose i,j entry is kernel_fn(q[i], K[j])
            # we do this with two vmaps. First, "inner_fn" will take input q, K[i] and vmap over
            # the first dimension of q.
            # The second vmap, "outer_fn", will take input q, K  and vmap over the first dimension of
            # K and place the vmapped dimension on the  *second* dimension of the output.
            inner_fn = jax.vmap(kernel_fn, in_axes=(0, None))
            outer_fn = jax.vmap(inner_fn, in_axes=(None, 0), out_axes=1)
            scores = outer_fn(
                q, K
            )  # [block_size, d], [L, d] -> [inner_fn([block_size, d], [d]), L] -> [block_size, L]
        else:
            scores = jax.vmap(kernel_fn, in_axes=(None, 0), out_axes=0)(
                q, K
            )  # [d], [L, d] -> [L]

        scores = scores * mask
        normalizer = jnp.sum(scores, axis=-1, keepdims=True)  # [block_size, 1]
        scores = scores / normalizer
        values = scores @ V  # [block_size, L] @ [L, d] -> [block_size, d]
        next_idx = idx + 1 if block_size is None else idx + block_size
        return next_idx, values

    if block_size is not None:
        Q = jnp.reshape(Q, (N // block_size, block_size, d))

    _, values = jax.lax.scan(
        attn_fn, init=0 if block_size is None else jnp.arange(block_size), xs=Q
    )

    if block_size is not None:
        values = jnp.reshape(values, (N, d))

    return values


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
