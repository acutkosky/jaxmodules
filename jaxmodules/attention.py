import jax
from jax import numpy as jnp
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

def _attn_kq_block_fn(
    normalizer, # [Lq, Hq]
    running_values, # [Lq, Hq, dv]
    q_idx, # [Lq]
    k_idx, # [Lk]
    q_block, # [Lq, Hq, dq]
    k_block, # [Lk, Hkv, dk]
    v_block, # [Lk, Hkv, dv]
    mask_fn,
    kernel_fn,
):
    Lq, Hq, dq = q_block.shape
    Lk, Hkv, _ = k_block.shape
    _, _, dv = v_block.shape
    MQA_factor = Hq // Hkv

    q_block = rearrange(q_block, "Lq (Hkv MQA) dq -> Lq Hkv MQA dq", Hkv=Hkv)

    mask = fancy_vmap(
        mask_fn,
        "mask[q, h, k] = mask_fn(Hq[h], q_idx[q], k_idx[k])"
    )(jnp.arange(Hq), q_idx, k_idx)

    scores = fancy_vmap(
        kernel_fn,
        'scores[q, h, m, k] = kernel_fn(q_block[q, h, m, :], K[k, h, :])',
    )(q_block, k_block)
    
    mask = rearrange(mask, "Lq (Hkv MQA) Lk -> Lq Hkv MQA Lk", Hkv=Hkv)
    scores = scores * mask
    
    local_normalizer = jnp.sum(scores, axis=-1, keepdims=True)  # [Lq, Hkv, MQA_factor, 1]

    normalizer = rearrange(normalizer, "Lq (Hkv MQA)-> Lq Hkv MQA 1", Hkv=Hkv)
    running_values = rearrange(running_values, "Lq (Hkv MQA) dv -> Lq Hkv MQA dv", Hkv=Hkv)


    unnormalized_values = einsum(scores, v_block, "Lq Hkv MQA Lk, Lk Hkv d -> Lq Hkv MQA d")

    new_normalizer = normalizer + local_normalizer
    running_values = jnp.where(new_normalizer > 0, running_values + (unnormalized_values - running_values * local_normalizer)/new_normalizer, jnp.zeros_like(running_values))
    normalizer = rearrange(new_normalizer, "Lq Hkv MQA 1 -> Lq (Hkv MQA)")
    running_values = rearrange(running_values, "Lq Hkv MQA dv -> Lq (Hkv MQA) dv")

    return normalizer, running_values

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
        normalizer, running_values = carry
        k_idx, k_block, v_block = blocks
        normalizer, running_values = _attn_kq_block_fn(
            normalizer,
            running_values,
            q_idx,
            k_idx,
            q_block,
            k_block,
            v_block,
            mask_fn,
            kernel_fn)
        return (normalizer, running_values), None
    return scan_fn




def _attn_block_fn(q_idx, q_block, K, V, mask_fn, kernel_fn, is_full_mask, is_causal):
    # idx is an array of integers
    # mask = jax.vmap(mask_fn)(idx)  # [block_size, L]
    Lq, Hq, dq = q_block.shape
    Lk, Hkv, _ = K.shape
    _, _, dv = V.shape

    MQA_factor = Hq // Hkv

    k_block = rearrange(K, "(blocks block_size) Hkv d -> blocks block_size Hkv d", block_size=Lq)
    v_block = rearrange(V, "(blocks block_size) Hkv d -> blocks block_size Hkv d", block_size=Lq)

    k_indices = rearrange(jnp.arange(Lk), "(blocks block_size) -> blocks block_size", block_size=Lq)

    kq_scanner = _make_attention_kq_scanner(mask_fn, kernel_fn, q_idx, q_block)
    (normalizer, running_values), _ = jax.lax.scan(
        kq_scanner,
        init=(jnp.zeros((Lq, Hq)), jnp.zeros((Lq, Hq, dv))),
        xs=(k_indices, k_block, v_block),
    )

    return running_values


@partial(jax.custom_vjp, nondiff_argnames=['is_causal', 'kernel_fn', 'mask_fn', 'block_size'])
def _masked_attention_via_map(
    Q: Array,
    K: Array,
    V: Array,
    is_causal: bool = False,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[Tuple[int,int,int], Array], Array]] = None,
    block_size=None,
) -> Array:
    """
    Same functionality and memory cost as attention_via_map by default, but also
    allows for user-specified masking as well as alternative kernels.

    K: array of key values, shape [L, Hkv, d]
    Q: array of queries, shape [N, Hq, d]
    V: array of values, shape [L, Hkv, d]
    is_causal: if true, apply a causal mask
    kernel_fn: the  unnormalized attention score is kernel_fn(Q, K).
        default is q, k -> jnp.exp( <q, k> / sqrt(d) )
    mask_fn: takes integers h, q, k and returns a boolean specifying
        the attention mask for the hth head and the qth query and kth key.
        If is_causal is true, you cannot provide mask_fn; it will be generated automatically.
        If is_causal is False and mask_fn is None, then the default value of no masking will
        be used.
    block_size: If specified, group the Q values into [block_size, d] sized blocks and
        perform attention on these blocks. Allows to trade-off the memory savings of scan.
    """
    L, Hk, d = K.shape
    N, Hq, dq = Q.shape
    Lv, Hv, dv = V.shape


    if block_size is None:
        block_size = N

    assert d == dq and Lv == L and dv == d, (
        f"shape mismatch in K {K.shape}, Q {Q.shape} and V {V.shape}"
    )

    assert Hq % Hk == 0, "Hq must be divisible by Hk"
    assert Hv == Hk, "Hv must be equal to Hk"

    Hkv = Hk

    assert block_size is None or N % block_size == 0, (
        f"block_size must divide number of queries!"
    )

    is_full_mask = False
    if is_causal and mask_fn is not None:
        raise ValueError("cannot specify both 'is_causal' and 'mask_fn'!")
    if is_causal:
        # mask_fn signature: (h, q, k) -> bool
        mask_fn = lambda h, q, k: q >= k
    if mask_fn is None:
        is_full_mask = True
        # mask_fn signature: (h, q, k) -> bool
        mask_fn = lambda h, q, k: True

    def attn_fn(idx_q):
        idx, q = idx_q
        return _attn_block_fn(idx, q, K, V, mask_fn, kernel_fn, is_full_mask, is_causal)

    Q = rearrange(Q, "(blocks block_size) Hq d -> blocks block_size Hq d", block_size=block_size)
    idx_blocks = jnp.reshape(jnp.arange(N), (N//block_size, block_size))

    values = jax.lax.map(jax.checkpoint(attn_fn), (idx_blocks, Q)) # [N//block_size, block_size, Hq, d]

    values = rearrange(values, "blocks block_size Hq dv -> (blocks block_size) Hq dv")

    return values


def _masked_attention_via_map_fwd(
    Q: Array,
    K: Array,
    V: Array,
    is_causal: bool = False,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[int, Array], Array]] = None,
    block_size=None,
) -> Array:

    values = _masked_attention_via_map(Q, K, V, is_causal=is_causal, kernel_fn=kernel_fn, mask_fn=mask_fn, block_size=block_size)
    return values, (Q, K, V)

def _masked_attention_via_map_bwd(
    is_causal: bool,
    kernel_fn: Callable[[Array, Array], float],
    mask_fn: Optional[Union[Callable[int, Array], Array]] ,
    block_size,  
    res,
    upstream_grad, 
):
    Q, K, V = res
    L, Hk, d = K.shape
    N, Hq, dq = Q.shape
    Lv, Hv, dv = V.shape


    if block_size is None:
        block_size = N

    assert d == dq and Lv == L and dv == d, (
        f"shape mismatch in K {K.shape}, Q {Q.shape} and V {V.shape}"
    )

    assert Hq % Hk == 0, "Hq must be divisible by Hk"
    assert Hv == Hk, "Hv must be equal to Hk"

    assert block_size is None or N % block_size == 0, (
        f"block_size must divide number of queries!"
    )


    is_full_mask = False
    if is_causal and mask_fn is not None:
        raise ValueError("cannot specify both 'is_causal' and 'mask_fn'!")
    if is_causal:
        # mask_fn signature: (h, q, k) -> bool
        mask_fn = lambda h, q, k: q >= k
    if mask_fn is None:
        is_full_mask = True
        # mask_fn signature: (h, q, k) -> bool
        mask_fn = lambda h, q, k: True

    def attn_fn(dK_dV, idx_q_g):
        idx, q, g = idx_q_g
        dK_carry, dV_carry = dK_dV

        def get_values(q, K, V):
            return _attn_block_fn(idx, q, K, V, mask_fn, kernel_fn, is_full_mask, is_causal)
        _, vjp_fn = jax.vjp(get_values, q, K, V)
        dq, qK, qV = vjp_fn(g)

        dq, qK, qV = vjp_fn(g)
        dK_carry = dK_carry + qK
        dV_carry = dV_carry + qV
        return (dK_carry, dV_carry), dq


    # break it up into blocks of size block_size
    g_blocks = rearrange(upstream_grad, "(blocks block_size) Hq dv -> blocks block_size Hq dv", block_size=block_size)



    Q = rearrange(Q, "(blocks block_size) Hq dq -> blocks block_size Hq dq", block_size=block_size)

    idx_blocks = jnp.reshape(jnp.arange(N), (N//block_size, block_size))
    
    (k_grad, v_grad), q_grad = jax.lax.scan(
        attn_fn, init=(jnp.zeros_like(K), jnp.zeros_like(V)), xs=(idx_blocks, Q, g_blocks)
    )
    q_grad = rearrange(q_grad, "blocks block_size Hq dq -> (blocks block_size) Hq dq")
    return q_grad, k_grad, v_grad


_masked_attention_via_map.defvjp(_masked_attention_via_map_fwd, _masked_attention_via_map_bwd)

def masked_attention_via_map(
    Q: Array,
    K: Array,
    V: Array,
    *,
    is_causal: bool = False,
    kernel_fn: Callable[[Array, Array], float] = default_kernel,
    mask_fn: Optional[Union[Callable[int, Array], Array]] = None,
    block_size=None,
) -> Array:
    return _masked_attention_via_map(Q, K, V, is_causal=is_causal, kernel_fn=kernel_fn, mask_fn=mask_fn, block_size=block_size)



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
