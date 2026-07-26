"""Mosaic GPU implementation details for memory-efficient attention.

The public attention API remains in :mod:`jaxmodules.attention`.  This module
contains a deliberately conservative GPU fast path for the standard
scaled-dot-product kernel and coordinate-based masks.

The kernel uses the nondeprecated Mosaic GPU entry point and the synchronous
warp-level ``mma.sync`` primitive.  That primitive is supported on consumer
Blackwell (SM120), unlike the Hopper-only WGMMA and datacenter-Blackwell
TCGEN05 interfaces.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


# This is intentionally narrower than Mosaic GPU's full lowering registry.
# Every accepted primitive must also be covered by a compiled mask test before
# it is added here. Nested ``jit`` calls are inspected recursively.
_SUPPORTED_MASK_PRIMITIVES = frozenset(
    {
        "abs",
        "add",
        "and",
        "broadcast_in_dim",
        "convert_element_type",
        "copy",
        "div",
        "eq",
        "ge",
        "gt",
        "integer_pow",
        "jit",
        "le",
        "lt",
        "max",
        "min",
        "mul",
        "ne",
        "neg",
        "not",
        "or",
        "rem",
        "reshape",
        "select_n",
        "squeeze",
        "sub",
        "xor",
    }
)


@dataclass(frozen=True)
class MosaicAttentionConfig:
    """Static tuning parameters for the first general-mask kernel."""

    block_q: int = 64
    block_kv: int = 64
    probability_components: int = 2


def _jaxpr_uses_supported_mask_primitives(jaxpr: jax_core.Jaxpr) -> bool:
    if jaxpr.effects:
        return False
    for equation in jaxpr.eqns:
        if equation.primitive.name not in _SUPPORTED_MASK_PRIMITIVES:
            return False
        if any(
            not _jaxpr_uses_supported_mask_primitives(nested)
            for nested in jax_core.jaxprs_in_params(equation.params)
        ):
            return False
    return True


def mask_is_mosaic_compatible(mask_fn: Callable[..., Any]) -> bool:
    """Return whether ``mask_fn`` belongs to the tested elementwise subset.

    This check is sufficient rather than complete: returning ``False`` only
    means that the mapped fallback should be used. It does not claim that
    Mosaic GPU would be fundamentally unable to lower the callable.
    """

    index = jax.ShapeDtypeStruct((), jnp.int32)
    try:
        mask_jaxpr = jax.make_jaxpr(mask_fn)(index, index, index, index)
    except Exception:  # noqa: BLE001 - an opaque callable is a fallback case.
        return False

    if len(mask_jaxpr.out_avals) != 1:
        return False
    output = mask_jaxpr.out_avals[0]
    if output.shape or output.dtype != jnp.dtype(jnp.bool_):
        return False

    # Array-valued closed-over constants would require arbitrary global-memory
    # gathers inside the kernel. Scalar configuration remains acceptable.
    if any(getattr(constant, "ndim", 0) != 0 for constant in mask_jaxpr.consts):
        return False
    return _jaxpr_uses_supported_mask_primitives(mask_jaxpr)


def _select_config(
    query: jax.Array,
    key: jax.Array,
) -> MosaicAttentionConfig | None:
    """Choose a conservative initial configuration for supported shapes."""

    if query.shape[1] % 64 or key.shape[1] % 64:
        return None
    probability_components = 2 if query.dtype == jnp.float16 else 3
    return MosaicAttentionConfig(
        probability_components=probability_components,
    )


def supports_mosaic_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask_fn: Callable[..., Any],
) -> bool:
    """Return whether the initial Mosaic implementation supports this call."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.dtype not in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        return False
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return False
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        return False
    if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
        return False
    if query.shape[2] % key.shape[2]:
        return False
    if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
        return False
    # Start with the overwhelmingly common D=64 case. Wider head dimensions
    # need separate register-pressure and tile-size tuning.
    if query.shape[3] != 64:
        return False
    if _select_config(query, key) is None:
        return False
    return mask_is_mosaic_compatible(mask_fn)


def _materialize_mask_tile(
    mask_fn: Callable[..., Any],
    *,
    batch: jax.Array,
    query_head: jax.Array,
    query_base: jax.Array,
    kv_step: jax.Array,
    block_q: int,
    block_kv: int,
    layout: Any,
) -> jax.Array:
    """Evaluate an accepted scalar mask expression over one score tile."""

    shape = (block_q, block_kv)
    query_indices = plgpu.broadcasted_iota(
        jnp.int32,
        shape,
        0,
        layout=layout,
    )
    key_indices = plgpu.broadcasted_iota(
        jnp.int32,
        shape,
        1,
        layout=layout,
    )
    mask = jnp.asarray(
        mask_fn(
            batch,
            query_head,
            query_indices + query_base,
            key_indices + kv_step * block_kv,
        ),
        dtype=jnp.bool_,
    )
    if not mask.shape:
        mask = lax.broadcast_in_dim(mask, shape, ())
    return mask


def mosaic_attention_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask_fn: Callable[..., Any],
) -> tuple[jax.Array, jax.Array]:
    """Compute standard attention and its natural-log normalizer on Mosaic GPU."""

    if not supports_mosaic_attention(query, key, value, mask_fn):
        raise ValueError("unsupported Mosaic attention configuration")
    config = _select_config(query, key)
    assert config is not None

    batch_size, query_length, query_heads, head_dim = query.shape
    kv_length, kv_heads = key.shape[1:3]
    query_heads_per_kv_head = query_heads // kv_heads
    dtype = query.dtype
    block_q = config.block_q
    block_kv = config.block_kv
    num_query_tiles = query_length // block_q
    num_kv_tiles = kv_length // block_kv
    scale = float(head_dim**-0.5)
    log2e = math.log2(math.e)
    accumulator_layout = plgpu.Layout.MMA_ACC(dtype)
    row_layout = accumulator_layout.reduce(1)

    def kernel(
        query_ref,
        key_ref,
        value_transposed_ref,
        output_ref,
        lse_ref,
        probability_smem,
    ):
        batch = lax.axis_index("batch")
        query_head = lax.axis_index("heads")
        kv_head = lax.div(
            query_head,
            jnp.asarray(query_heads_per_kv_head, query_head.dtype),
        )
        query_base = lax.axis_index("query_tiles") * block_q

        query_fragment = plgpu.load(
            query_ref.at[
                batch,
                pl.ds(query_base, block_q),
                query_head,
            ],
            layout=plgpu.Layout.MMA_LHS(dtype),
            optimized=False,
        )
        row_max = plgpu.layout_cast(
            jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
            row_layout,
        )
        row_sum = plgpu.layout_cast(
            jnp.zeros((block_q,), dtype=jnp.float32),
            row_layout,
        )
        output_accumulator = plgpu.layout_cast(
            jnp.zeros((block_q, head_dim), dtype=jnp.float32),
            accumulator_layout,
        )

        def kv_body(kv_step, carry):
            output_accumulator, row_max, row_sum = carry
            key_fragment = plgpu.load(
                plgpu.transpose_ref(
                    key_ref.at[
                        batch,
                        pl.ds(kv_step * block_kv, block_kv),
                        kv_head,
                    ],
                    (1, 0),
                ),
                layout=plgpu.Layout.MMA_RHS(dtype),
                optimized=False,
            )
            scores = plgpu.mma(
                plgpu.layout_cast(
                    jnp.zeros(
                        (block_q, block_kv),
                        dtype=jnp.float32,
                    ),
                    accumulator_layout,
                ),
                query_fragment,
                key_fragment,
            )
            scores *= scale
            mask = _materialize_mask_tile(
                mask_fn,
                batch=batch,
                query_head=query_head,
                query_base=query_base,
                kv_step=kv_step,
                block_q=block_q,
                block_kv=block_kv,
                layout=accumulator_layout,
            )
            scores = jnp.where(mask, scores, -jnp.inf)

            tile_max = plgpu.layout_cast(
                scores.max(axis=1) * log2e,
                row_layout,
            )
            next_row_max = jnp.maximum(row_max, tile_max)
            safe_next_row_max = jnp.where(
                next_row_max == -jnp.inf,
                jnp.zeros_like(next_row_max),
                next_row_max,
            )
            previous_scale = jnp.where(
                row_sum > 0,
                jnp.exp2(row_max - safe_next_row_max),
                jnp.zeros_like(row_sum),
            )
            safe_next_row_max_broadcast = plgpu.layout_cast(
                lax.broadcast_in_dim(
                    safe_next_row_max,
                    scores.shape,
                    (0,),
                ),
                accumulator_layout,
            )
            probabilities = jnp.where(
                mask,
                jnp.exp2(
                    scores * log2e - safe_next_row_max_broadcast
                ),
                jnp.zeros_like(scores),
            )
            previous_scale_broadcast = plgpu.layout_cast(
                lax.broadcast_in_dim(
                    previous_scale,
                    output_accumulator.shape,
                    (0,),
                ),
                accumulator_layout,
            )
            output_accumulator *= previous_scale_broadcast
            row_sum *= previous_scale
            row_max = next_row_max
            row_sum += plgpu.layout_cast(
                probabilities.sum(axis=1),
                row_layout,
            )

            value_fragment = plgpu.load(
                plgpu.transpose_ref(
                    value_transposed_ref.at[
                        batch,
                        kv_head,
                        :,
                        pl.ds(kv_step * block_kv, block_kv),
                    ],
                    (1, 0),
                ),
                layout=plgpu.Layout.MMA_RHS(dtype),
                optimized=False,
            )

            # A standard FlashAttention kernel casts ``probabilities`` to the
            # input dtype before PV. Preserve substantially all FP32
            # probability information by decomposing it into multiple
            # low-precision components, then accumulating each tensor-core
            # product into the same FP32 accumulator.
            probability_residual = probabilities
            for _ in range(config.probability_components):
                component = probability_residual.astype(dtype)
                probability_smem[...] = component
                probability_fragment = plgpu.load(
                    probability_smem,
                    layout=plgpu.Layout.MMA_LHS(dtype),
                    optimized=False,
                )
                output_accumulator = plgpu.mma(
                    output_accumulator,
                    probability_fragment,
                    value_fragment,
                )
                probability_residual -= component.astype(jnp.float32)

            return output_accumulator, row_max, row_sum

        output_accumulator, row_max, row_sum = lax.fori_loop(
            0,
            num_kv_tiles,
            kv_body,
            (output_accumulator, row_max, row_sum),
        )

        has_attention = row_sum > 0
        safe_row_sum = jnp.where(
            has_attention,
            row_sum,
            jnp.ones_like(row_sum),
        )
        safe_row_sum_broadcast = plgpu.layout_cast(
            lax.broadcast_in_dim(
                safe_row_sum,
                output_accumulator.shape,
                (0,),
            ),
            accumulator_layout,
        )
        has_attention_broadcast = plgpu.layout_cast(
            lax.broadcast_in_dim(
                has_attention,
                output_accumulator.shape,
                (0,),
            ),
            accumulator_layout,
        )
        normalized = jnp.where(
            has_attention_broadcast,
            output_accumulator / safe_row_sum_broadcast,
            jnp.zeros_like(output_accumulator),
        )
        output_ref[
            batch,
            pl.ds(query_base, block_q),
            query_head,
        ] = normalized

        # row_max is kept in base-2 units in the loop. The mapped backward
        # expects a natural-log normalizer.
        natural_lse = row_max / log2e + jnp.log(safe_row_sum)
        lse_ref[
            batch,
            query_head,
            pl.ds(query_base, block_q),
        ] = jnp.where(
            has_attention,
            natural_lse,
            jnp.zeros_like(natural_lse),
        )

    swizzle = 128
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    probability_scratch = plgpu.SMEM(
        (block_q, block_kv),
        dtype,
        transforms=(
            plgpu.TilingTransform((8, swizzle_elems)),
            plgpu.SwizzleTransform(swizzle),
        ),
    )
    output_type = jax.ShapeDtypeStruct(query.shape, jnp.float32)
    lse_type = jax.ShapeDtypeStruct(
        (batch_size, query_heads, query_length),
        jnp.float32,
    )

    # Warp-level MMA expects its right operand in column-major form. Keeping
    # this as a normal JAX transpose preserves jit/vmap composition and costs
    # O(B*H*L*D), not the O(L**2) storage that this kernel avoids.
    value_transposed = jnp.transpose(value, (0, 2, 3, 1))
    output, lse = plgpu.kernel(
        kernel,
        out_type=(output_type, lse_type),
        scratch_types=(probability_scratch,),
        compiler_params=plgpu.CompilerParams(
            approx_math=False,
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
        grid=(query_heads, num_query_tiles, batch_size),
        grid_names=("heads", "query_tiles", "batch"),
    )(query, key, value_transposed)
    return output, jnp.swapaxes(lse, 1, 2)
