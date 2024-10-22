import jax
from jax import numpy as jnp
from typing import Callable, Union, Optional


def threshold_kernel(threshold: Optional[float] = None):
    def kernel_fn(q, k):
        dotp = jnp.dot(q,k)
        if threshold is not None:
            dotp = dotp - threshold/jnp.sqrt(q.shape[0])
        return jnp.exp(dotp)

    return kernel_fn

def default_kernel(q, k):
    return jnp.exp(jnp.dot(q,k)/jnp.sqrt(q.shape[-1]))

def masked_attention_via_scan(
    K: jax.Array,
    Q: jax.Array,
    V: jax.Array,
    *,
    is_causal: bool=False,
    # kernel_fn: Callable[float, float] = jnp.exp, 
    kernel_fn: Callable[float,  float] = default_kernel, #lambda q, k: jnp.exp(jnp.dot(q,k)/jnp.sqrt(3)),
    mask_fn: Optional[Union[Callable[int, jax.Array], jax.Array]]=None,
    block_size=None
):
    '''
    Same functionality and memory cost as attention_via_map by default, but also
    allows for user-specified masking as well as alternative kernels.
    
    K: array of key values, shape [L, d]
    Q: array of queries, shape [N, d]
    V: array of values, shape [L, d]
    is_causal: if true, apply a causal mask
    kernel_fn: the  unnnormalized attention score is kernel_fn(Q @ K.T/sqrt(d)).
        default is jnp.exp
    mask_fn: takes a integer index i and returns a size [d] array of booleans
        specifying the attention mask for the  ith query.
        If is_causal is true, you cannot provide mask_fn; it will be generated automatically.
        If is_causal is False and mask_fn is None, then the default  value of no masking will
        be used.
    block_size: If specified, group the Q values [block_size,  d] sized blocks and
        perform attention on these blocks. Allows to trade-off the memory savings of scan.
    '''

    L, d = K.shape
    N, dq = Q.shape
    Lv, dv = V.shape

    assert d==dq and Lv==L and dv==d, f"shape mismatch in K {K.shape}, Q {Q.shape} and V {V.shape}"

    assert block_size is None or N % block_size == 0, f"block_size must divide number of queries!"

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
            mask = jax.vmap(mask_fn)(idx) # [block_size, L]
        else:
            mask =  mask_fn(idx) # [L]
        
        if block_size is not None:
            # we want to make  a [block_size, L] matrix whose i,j entry is kernel_fn(q[i], K[j])
            # we do this with two vmaps. First, "inner_fn" will take input q, K[i] and vmap over
            # the first dimension of q.
            # The second vmap, "outer_fn", will take input q, K  and vmap over the first dimension of
            # K and place the vmapped dimension on the  *second* dimension of the output.
            inner_fn = jax.vmap(kernel_fn, in_axes=(0, None))
            outer_fn = jax.vmap(inner_fn, in_axes=(None, 0), out_axes=1)
            scores = outer_fn(q, K) # [block_size, d], [L, d] -> [inner_fn([block_size, d], [d]), L] -> [block_size, L] 
        else:
            scores = jax.vmap(kernel_fn, in_axes=(None, 0), out_axes=0)(q, K) # [d], [L, d] -> [L]

        scores =  scores * mask
        normalizer = jnp.sum(scores, axis=-1, keepdims=True) # [block_size, 1]
        scores = scores/normalizer
        values = scores @ V # [block_size, L] @ [L, d] -> [block_size, d]
        next_idx = idx + 1 if block_size is None else idx + block_size
        return next_idx, values

    if block_size is not None:
        Q = jnp.reshape(Q, (N//block_size, block_size, d))

    _, values = jax.lax.scan(
        attn_fn,
        init=0 if block_size is None else jnp.arange(block_size),
        xs=Q
    )

    if block_size is not None:
        values = jnp.reshape(values, (N, d))

    return values

