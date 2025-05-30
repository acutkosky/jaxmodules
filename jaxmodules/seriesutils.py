from collections.abc import Hashable, Sequence
from typing import Optional, Union, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

import equinox as eqx
from equinox.nn import StateIndex, StatefulLayer, State
from equinox import field

from einops import einsum, rearrange


def patch_series(
    series: jax.Array,
    patch_len: int,
    stride: int,
    padding: Optional[Union[str, int, Tuple[int, int]]] = None,
):
    """
    series: [L, ...] array

    splits serie into patches of length patch_len to return an array of shape:

    [(L-patch_len+1)//stride, patch_len, ...]

    result[i,j,...] is series[i*stride + j, ...]

    padding can be:
        "None" (no padding)
        int: (apply zero-padding of this amount to both sides of input
        "left": apply zero-padding of length k to left of input so that (L+k-patch_len) % stride == 0
        "right": apply zero-padding length  k to right of input
        tuple of int (left, right): apply this amount of zero-padding to the left and the right respetively.

    """

    if padding is None:
        padding = (0, 0)
    if isinstance(padding, int):
        padding = (padding, padding)
    k = (series.shape[0] - patch_len) % stride
    if k != 0:
        k = stride - k
    if padding == "left":
        padding = (k, 0)
    if padding == "right":
        padding = (0, k)

    if padding[0] != 0 or padding[1] != 0:
        pad_width = [padding] + [(0, 0) for _ in range(len(series.shape) - 1)]
        series = jnp.pad(series, pad_width, mode="constant", constant_values=0)

    L = series.shape[0]
    # Number of patches
    num_patches = 1 + (L - patch_len + 1) // stride

    # Indices along the first dimension to gather from
    patch_starts = jnp.arange(num_patches) * stride  # shape: [num_patches]
    patch_offsets = jnp.arange(patch_len)  # shape: [patch_len]

    # Use advanced indexing to gather patches
    # This will have shape [num_patches, patch_len, ...]
    result = series[patch_starts[:, None] + patch_offsets, ...]

    return result
