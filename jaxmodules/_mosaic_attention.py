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
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, llvm, memref, nvvm


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


def _mma_layouts(
    dtype: jnp.dtype,
    block_rows: int,
) -> tuple[Any, Any, Any]:
    """Build matching synchronous-MMA layouts for 16, 32, or 64 rows."""

    if jnp.dtype(dtype).itemsize != 2:
        raise ValueError("MMA layouts require 16-bit operands")
    if block_rows not in (16, 32, 64):
        raise ValueError("MMA row tiles must be 16, 32, or 64")
    if block_rows == 64:
        return (
            plgpu.Layout.MMA_LHS(dtype),
            plgpu.Layout.MMA_RHS(dtype),
            plgpu.Layout.MMA_ACC(dtype),
        )

    m_warps = block_rows // 16
    n_warps = 4 // m_warps
    replicated_m = plgpu.Replicated(4 // m_warps)
    replicated_n = plgpu.Replicated(4 // n_warps)
    lhs_layout = plgpu.Layout.TILED(
        plgpu.Tiling(
            (
                (m_warps * 16, 16),
                (16, 8),
                (8, 8),
                (2,),
            )
        ),
        warp_dims=(-7, replicated_m),
        lane_dims=(-3, -2),
        vector_dim=-1,
    )
    rhs_layout = plgpu.Layout.TILED(
        plgpu.Tiling(
            (
                (16, n_warps * 8),
                (8, 8),
                (2, 1),
            )
        ),
        warp_dims=(replicated_n, -5),
        lane_dims=(-3, -4),
        vector_dim=-2,
    )
    accumulator_layout = plgpu.Layout.TILED(
        plgpu.Tiling(
            (
                (m_warps * 16, n_warps * 8),
                (16, 8),
                (8, 8),
                (2,),
            )
        ),
        warp_dims=(-7, -6),
        lane_dims=(-3, -2),
        vector_dim=-1,
    )
    return lhs_layout, rhs_layout, accumulator_layout


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


def _mask_is_always_true(mask_fn: Callable[..., Any]) -> bool:
    """Conservatively recognize a callable whose result is literal True."""
    index = jax.ShapeDtypeStruct((), jnp.int32)
    try:
        mask_jaxpr = jax.make_jaxpr(mask_fn)(index, index, index, index)
    except Exception:  # noqa: BLE001 - an opaque callable is not specialized.
        return False
    if mask_jaxpr.jaxpr.eqns or len(mask_jaxpr.jaxpr.outvars) != 1:
        return False
    output = mask_jaxpr.jaxpr.outvars[0]
    return isinstance(output, jax_core.Literal) and output.val is True


def _select_config(
    query: jax.Array,
    key: jax.Array,
) -> MosaicAttentionConfig | None:
    """Choose score tiles that keep wide-head register pressure bounded."""

    if query.shape[3] >= 80:
        tile_size = 32
    else:
        tile_size = 64
    if query.shape[1] % tile_size or key.shape[1] % tile_size:
        return None
    return MosaicAttentionConfig(block_q=tile_size, block_kv=tile_size)


def _is_supported_head_dimension(head_dim: int) -> bool:
    """Whether warp-level MMA can represent the shared attention dimension."""

    # ``mma.sync.m16n8k16`` consumes the feature dimension in groups of 16.
    # Keep the implementation parametric over all such widths instead of
    # selecting a kernel for one model-specific embedding shape.
    return 64 <= head_dim <= 2048 and head_dim % 16 == 0


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
    if not _is_supported_head_dimension(query.shape[3]):
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
    is_causal: bool = False,
) -> jax.Array:
    """Evaluate a user-mask tile and optionally intersect it with causality."""

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
    global_query_indices = query_indices + query_base
    global_key_indices = key_indices + kv_step * block_kv
    mask = jnp.asarray(
        mask_fn(
            batch,
            query_head,
            global_query_indices,
            global_key_indices,
        ),
        dtype=jnp.bool_,
    )
    if not mask.shape:
        mask = lax.broadcast_in_dim(mask, shape, ())
    if is_causal:
        mask &= global_query_indices >= global_key_indices
    return mask


def _tile_has_attention(
    mask: jax.Array,
    *,
    accumulator_layout: Any,
    row_layout: Any,
) -> jax.Array:
    """Reduce a mask tile without Mosaic GPU's unsupported boolean reduce-or."""
    numeric_mask = plgpu.layout_cast(mask.astype(jnp.int32), accumulator_layout)
    row_max = plgpu.layout_cast(numeric_mask.max(axis=1), row_layout)
    return row_max.max() != 0


def _can_use_warp_specialized_forward(
    query: jax.Array,
    key: jax.Array,
    *,
    is_unmasked: bool,
    is_causal: bool,
) -> bool:
    """Whether K/V sharing should replace the single-warpgroup forward path."""

    minimum_length = 1024 if is_causal else 4096
    return (
        is_unmasked
        and (not is_causal or query.shape[1] == key.shape[1])
        and query.dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))
        and query.shape[1] >= minimum_length
        and key.shape[1] >= minimum_length
        and query.shape[1] % 128 == 0
        and key.shape[1] % 64 == 0
        and query.shape[3] == 64
        and key.shape[3] == 64
    )


def _can_use_warp_specialized_causal_split_backward(
    query: jax.Array,
    key: jax.Array,
    *,
    is_unmasked: bool,
    is_causal: bool,
) -> bool:
    """Whether the non-atomic key-major causal backward should be selected."""

    return (
        is_causal
        and query.shape[2] == key.shape[2]
        and query.shape[1] >= 1024
        and _can_use_warp_specialized_forward(
            query,
            key,
            is_unmasked=is_unmasked,
            is_causal=is_causal,
        )
    )


def _prefer_non_atomic_generic_backward(query: jax.Array) -> bool:
    """Whether complete locally accumulated gradients beat tile atomics.

    Wide heads increase both the register footprint of the one-pass kernel and
    the amount of data committed through global atomics. The key-major split
    avoids those atomics and reduces compiler temporaries for every supported
    mask kind.
    """

    return query.shape[3] >= 80


def mosaic_attention_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask_fn: Callable[..., Any],
    *,
    is_causal: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Compute standard attention and its natural-log normalizer on Mosaic GPU."""

    if not supports_mosaic_attention(query, key, value, mask_fn):
        raise ValueError("unsupported Mosaic attention configuration")
    config = _select_config(query, key)
    assert config is not None
    is_unmasked = _mask_is_always_true(mask_fn)
    if _can_use_warp_specialized_forward(
        query,
        key,
        is_unmasked=is_unmasked,
        is_causal=is_causal,
    ):
        return _mosaic_attention_forward_warp_specialized(
            query,
            key,
            value,
            is_causal=is_causal,
        )

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
    lhs_layout, rhs_layout, accumulator_layout = _mma_layouts(
        dtype,
        block_q,
    )
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
        query_tile = lax.axis_index("query_tiles")
        if is_causal:
            query_tile = num_query_tiles - 1 - query_tile
        query_base = query_tile * block_q

        query_fragment = plgpu.load(
            query_ref.at[
                batch,
                pl.ds(query_base, block_q),
                query_head,
            ],
            layout=lhs_layout,
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

        def process_tile(kv_step, carry, mask):
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
                layout=rhs_layout,
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
            if mask is not None:
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
            probabilities = jnp.exp2(
                scores * log2e - safe_next_row_max_broadcast
            )
            if mask is not None:
                probabilities = jnp.where(
                    mask,
                    probabilities,
                    jnp.zeros_like(probabilities),
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
                layout=rhs_layout,
                optimized=False,
            )

            # Match the input precision used by standard FlashAttention:
            # softmax and accumulation remain FP32, while the tensor-core PV
            # operand uses the input dtype.
            probability_smem[...] = probabilities.astype(dtype)
            probability_fragment = plgpu.load(
                probability_smem,
                layout=lhs_layout,
                optimized=False,
            )
            output_accumulator = plgpu.mma(
                output_accumulator,
                probability_fragment,
                value_fragment,
            )

            return output_accumulator, row_max, row_sum

        def unmasked_kv_body(kv_step, carry):
            return process_tile(kv_step, carry, None)

        def materialize_and_process(kv_step, carry, *, apply_causal):
            mask = _materialize_mask_tile(
                mask_fn,
                batch=batch,
                query_head=query_head,
                query_base=query_base,
                kv_step=kv_step,
                block_q=block_q,
                block_kv=block_kv,
                layout=accumulator_layout,
                is_causal=apply_causal,
            )
            if is_unmasked:
                return process_tile(kv_step, carry, mask)
            return lax.cond(
                _tile_has_attention(
                    mask,
                    accumulator_layout=accumulator_layout,
                    row_layout=row_layout,
                ),
                lambda x: process_tile(kv_step, x, mask),
                lambda x: x,
                carry,
            )

        def user_masked_kv_body(kv_step, carry):
            return materialize_and_process(
                kv_step,
                carry,
                apply_causal=False,
            )

        def causal_masked_kv_body(kv_step, carry):
            return materialize_and_process(
                kv_step,
                carry,
                apply_causal=True,
            )

        carry = (output_accumulator, row_max, row_sum)
        if is_unmasked and not is_causal:
            output_accumulator, row_max, row_sum = lax.fori_loop(
                0,
                num_kv_tiles,
                unmasked_kv_body,
                carry,
            )
        elif is_causal:
            full_kv_stop = jnp.minimum(
                query_base // block_kv,
                num_kv_tiles,
            )
            kv_stop = jnp.minimum(
                (query_base + block_q + block_kv - 1) // block_kv,
                num_kv_tiles,
            )
            carry = lax.fori_loop(
                0,
                full_kv_stop,
                (
                    unmasked_kv_body
                    if is_unmasked
                    else user_masked_kv_body
                ),
                carry,
            )
            output_accumulator, row_max, row_sum = lax.fori_loop(
                full_kv_stop,
                kv_stop,
                causal_masked_kv_body,
                carry,
            )
        else:
            output_accumulator, row_max, row_sum = lax.fori_loop(
                0,
                num_kv_tiles,
                user_masked_kv_body,
                carry,
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
        ] = normalized.astype(dtype)

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

    swizzle = min(128, block_kv * jnp.dtype(dtype).itemsize)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    probability_scratch = plgpu.SMEM(
        (block_q, block_kv),
        dtype,
        transforms=(
            plgpu.TilingTransform((8, swizzle_elems)),
            plgpu.SwizzleTransform(swizzle),
        ),
    )
    output_type = jax.ShapeDtypeStruct(query.shape, dtype)
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
            lowering_semantics=(
                plgpu.LoweringSemantics.Lane
                if block_q < 64
                else plgpu.LoweringSemantics.Warpgroup
            ),
        ),
        grid=(query_heads, num_query_tiles, batch_size),
        grid_names=("heads", "query_tiles", "batch"),
    )(query, key, value_transposed)
    return output, jnp.swapaxes(lse, 1, 2)


def _mosaic_attention_forward_warp_specialized(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    is_causal: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Compute large dense attention with shared K/V producer staging.

    Two compute warpgroups process adjacent query tiles. A third warpgroup
    pipelines each K/V tile through shared memory once for both consumers.
    Maximal causal attention streams only the common visible K/V prefix and
    applies a triangular mask to each compute warpgroup's diagonal tile.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 arrays")
    if query.dtype not in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        raise ValueError("warp-specialized attention requires FP16 or BF16")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value batch sizes must match")
    if key.shape[1:3] != value.shape[1:3]:
        raise ValueError("key and value sequence/head shapes must match")
    if query.shape[2] % key.shape[2]:
        raise ValueError("query heads must be divisible by key/value heads")
    if query.shape[3] != 64 or key.shape[3] != 64 or value.shape[3] != 64:
        raise ValueError("warp-specialized attention currently requires D=64")

    batch_size, query_length, query_heads, head_dim = query.shape
    kv_length, kv_heads = key.shape[1:3]
    block_q = 64
    block_kv = 64
    num_compute_wgs = 2
    compute_registers = 232
    query_superblock = block_q * num_compute_wgs
    if query_length % query_superblock:
        raise ValueError(f"query length must be divisible by {query_superblock}")
    if kv_length % block_kv:
        raise ValueError(f"key/value length must be divisible by {block_kv}")

    query_heads_per_kv_head = query_heads // kv_heads
    num_query_supertiles = query_length // query_superblock
    num_kv_tiles = kv_length // block_kv
    max_concurrent_steps = min(2, num_kv_tiles)
    dtype = query.dtype
    scale = float(head_dim**-0.5)
    log2e = math.log2(math.e)
    accumulator_layout = plgpu.Layout.MMA_ACC(dtype)
    swizzle = 128
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    lhs_smem_transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(lhs_smem_transforms),),
        return_type=plgpu.ShapeDtypeStruct(
            (block_kv, head_dim),
            dtype,
            layout=plgpu.Layout.MMA_RHS(dtype),
        ),
    )
    def load_shared_rhs(_, smem_ref):
        # Store the logical RHS transposed in shared memory so its reduction
        # dimension is contiguous. The tiled transpose restores the matrix's
        # logical orientation and lets Mosaic lower the native MMA_RHS transfer
        # to ``ldmatrix`` instead of scalar loads plus local packing.
        transposed_ref = mgpu.memref_transpose(smem_ref, (1, 0, 3, 2))
        native_layout = plgpu.Layout.MMA_RHS(dtype).to_mgpu()
        return mgpu.FragmentedArray.load_tiled(
            transposed_ref,
            swizzle=swizzle,
            layout=native_layout,
            optimized=True,
            tiling_rank=2,
        )

    @plgpu.inline_mgpu(
        arg_types=(
            accumulator_layout,
            accumulator_layout,
            plgpu.Layout.MMA_RHS(dtype),
        ),
        return_type=plgpu.ShapeDtypeStruct(
            (block_q, head_dim),
            jnp.float32,
            layout=accumulator_layout,
        ),
    )
    def mma_with_accumulator_lhs(
        _,
        accumulator,
        matrix,
        rhs,
    ):
        # For a 64x64 tile, casting MMA_ACC to the input dtype leaves every
        # element in the same lane/register order expected by MMA_LHS. Rewrap
        # those registers directly instead of storing and reloading P through
        # shared memory.
        lhs_layout = plgpu.Layout.MMA_LHS(dtype).to_mgpu()
        lhs = mgpu.FragmentedArray(
            _registers=matrix.registers.reshape(
                lhs_layout.registers_shape(matrix.shape)
            ),
            _layout=lhs_layout,
            _is_signed=matrix.is_signed,
        )
        return mgpu.mma(accumulator, lhs, rhs)

    def kernel(
        query_ref,
        key_ref,
        value_transposed_ref,
        output_ref,
        lse_ref,
        smem_buffers,
        ready_barriers,
        consumed_barriers,
    ):
        batch = lax.axis_index("batch")
        query_head = lax.axis_index("heads")
        wg_index = lax.axis_index("wg")
        key_smem, value_smem = smem_buffers
        key_barriers, value_barriers = ready_barriers
        key_consumed_barriers, value_consumed_barriers = consumed_barriers
        kv_head = lax.div(
            query_head,
            jnp.asarray(query_heads_per_kv_head, query_head.dtype),
        )
        query_supertile = lax.axis_index("query_supertiles")
        if is_causal:
            query_supertile = num_query_supertiles - 1 - query_supertile
            producer_kv_stop = jnp.minimum(
                (query_supertile + 1) * (query_superblock // block_kv),
                num_kv_tiles,
            )
        else:
            producer_kv_stop = num_kv_tiles

        @pl.when(wg_index < num_compute_wgs)
        def compute_warpgroup():
            plgpu.set_max_registers(compute_registers, action="increase")
            query_tile = query_supertile * num_compute_wgs + wg_index
            query_base = query_tile * block_q
            query_fragment = plgpu.load(
                query_ref.at[
                    batch,
                    pl.ds(query_base, block_q),
                    query_head,
                ],
                layout=plgpu.Layout.MMA_LHS(dtype),
                optimized=False,
            )
            row_max = jnp.full(
                (block_q,),
                -jnp.inf,
                dtype=jnp.float32,
            )
            row_sum = jnp.zeros((block_q,), dtype=jnp.float32)
            output_accumulator = jnp.zeros(
                (block_q, head_dim),
                dtype=jnp.float32,
            )
            if is_causal:
                query_indices = plgpu.broadcasted_iota(
                    jnp.int32,
                    (block_q, block_kv),
                    0,
                    layout=accumulator_layout,
                )
                key_indices = plgpu.broadcasted_iota(
                    jnp.int32,
                    (block_q, block_kv),
                    1,
                    layout=accumulator_layout,
                )
                diagonal_mask = query_indices >= key_indices

            def process_kv_tile(kv_step, carry):
                slot = lax.rem(
                    kv_step,
                    jnp.asarray(max_concurrent_steps, kv_step.dtype),
                )
                plgpu.barrier_wait(key_barriers.at[slot])
                key_fragment = load_shared_rhs(key_smem.at[slot])
                scores = plgpu.mma(
                    jnp.zeros(
                        (block_q, block_kv),
                        dtype=jnp.float32,
                    ),
                    query_fragment,
                    key_fragment,
                )
                plgpu.barrier_arrive(key_consumed_barriers.at[slot])
                scores *= scale
                if is_causal:
                    scores = lax.cond(
                        kv_step == query_tile,
                        lambda matrix: jnp.where(
                            diagonal_mask,
                            matrix,
                            -jnp.inf,
                        ),
                        lambda matrix: matrix,
                        scores,
                    )

                def update_softmax(current):
                    output_accumulator, row_max, row_sum = current
                    tile_max = scores.max(axis=1) * log2e
                    next_row_max = jnp.maximum(row_max, tile_max)
                    previous_scale = jnp.exp2(row_max - next_row_max)
                    next_row_max_broadcast = lax.broadcast_in_dim(
                        next_row_max,
                        scores.shape,
                        (0,),
                    )
                    probabilities = jnp.exp2(
                        scores * log2e - next_row_max_broadcast
                    )
                    previous_scale_broadcast = lax.broadcast_in_dim(
                        previous_scale,
                        output_accumulator.shape,
                        (0,),
                    )
                    output_accumulator *= previous_scale_broadcast
                    row_sum *= previous_scale
                    row_sum += probabilities.sum(axis=1)
                    return (
                        output_accumulator,
                        next_row_max,
                        row_sum,
                    ), probabilities

                if is_causal:
                    carry, probabilities = lax.cond(
                        kv_step <= query_tile,
                        update_softmax,
                        lambda current: (
                            current,
                            jnp.zeros(
                                (block_q, block_kv),
                                dtype=jnp.float32,
                            ),
                        ),
                        carry,
                    )
                else:
                    carry, probabilities = update_softmax(carry)

                plgpu.barrier_wait(value_barriers.at[slot])
                value_fragment = load_shared_rhs(value_smem.at[slot])
                plgpu.barrier_arrive(value_consumed_barriers.at[slot])

                def accumulate_value(current):
                    output_accumulator, row_max, row_sum = current
                    output_accumulator = mma_with_accumulator_lhs(
                        output_accumulator,
                        probabilities.astype(dtype),
                        value_fragment,
                    )
                    return output_accumulator, row_max, row_sum

                if is_causal:
                    return lax.cond(
                        kv_step <= query_tile,
                        accumulate_value,
                        lambda current: current,
                        carry,
                    )
                return accumulate_value(carry)

            output_accumulator, row_max, row_sum = lax.fori_loop(
                0,
                producer_kv_stop,
                process_kv_tile,
                (output_accumulator, row_max, row_sum),
            )
            row_sum_broadcast = lax.broadcast_in_dim(
                row_sum,
                output_accumulator.shape,
                (0,),
            )
            normalized = output_accumulator / row_sum_broadcast
            natural_lse = row_max / log2e + jnp.log(row_sum)
            output_ref[
                batch,
                pl.ds(query_base, block_q),
                query_head,
            ] = normalized.astype(dtype)
            lse_ref[
                batch,
                query_head,
                pl.ds(query_base, block_q),
            ] = natural_lse

        @pl.when(wg_index == num_compute_wgs)
        def memory_warpgroup():
            plgpu.set_max_registers(40, action="decrease")
            for kv_step in range(max_concurrent_steps):
                kv_slice = (
                    batch,
                    pl.ds(kv_step * block_kv, block_kv),
                    kv_head,
                )
                value_slice = (
                    batch,
                    kv_head,
                    slice(None),
                    pl.ds(kv_step * block_kv, block_kv),
                )
                plgpu.copy_gmem_to_smem(
                    key_ref.at[kv_slice],
                    key_smem.at[kv_step],
                    key_barriers.at[kv_step],
                )
                plgpu.copy_gmem_to_smem(
                    value_transposed_ref.at[value_slice],
                    value_smem.at[kv_step],
                    value_barriers.at[kv_step],
                )

            @pl.loop(0, producer_kv_stop - max_concurrent_steps)
            def refill_pipeline(kv_step):
                next_kv_step = kv_step + max_concurrent_steps
                slot = lax.rem(
                    kv_step,
                    jnp.asarray(max_concurrent_steps, kv_step.dtype),
                )
                plgpu.barrier_wait(key_consumed_barriers.at[slot])
                next_kv_slice = (
                    batch,
                    pl.ds(next_kv_step * block_kv, block_kv),
                    kv_head,
                )
                next_value_slice = (
                    batch,
                    kv_head,
                    slice(None),
                    pl.ds(next_kv_step * block_kv, block_kv),
                )
                plgpu.copy_gmem_to_smem(
                    key_ref.at[next_kv_slice],
                    key_smem.at[slot],
                    key_barriers.at[slot],
                )
                plgpu.barrier_wait(value_consumed_barriers.at[slot])
                plgpu.copy_gmem_to_smem(
                    value_transposed_ref.at[next_value_slice],
                    value_smem.at[slot],
                    value_barriers.at[slot],
                )

    key_scratch = plgpu.SMEM(
        (max_concurrent_steps, block_kv, head_dim),
        dtype,
        transforms=lhs_smem_transforms,
    )
    value_scratch = plgpu.SMEM(
        (max_concurrent_steps, block_kv, head_dim),
        dtype,
        transforms=lhs_smem_transforms,
    )
    output_type = jax.ShapeDtypeStruct(query.shape, dtype)
    lse_type = jax.ShapeDtypeStruct(
        (batch_size, query_heads, query_length),
        jnp.float32,
    )
    value_transposed = jnp.transpose(value, (0, 2, 3, 1))
    output, lse = plgpu.kernel(
        kernel,
        out_type=(output_type, lse_type),
        scratch_types=(
            (
                key_scratch,
                value_scratch,
            ),
            (
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=max_concurrent_steps),
            ),
            (
                plgpu.Barrier(
                    num_arrivals=num_compute_wgs,
                    num_barriers=max_concurrent_steps,
                ),
                plgpu.Barrier(
                    num_arrivals=num_compute_wgs,
                    num_barriers=max_concurrent_steps,
                ),
            ),
        ),
        compiler_params=plgpu.CompilerParams(
            approx_math=False,
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
        grid=(query_heads, num_query_supertiles, batch_size),
        grid_names=("heads", "query_supertiles", "batch"),
        num_threads=num_compute_wgs + 1,
        thread_name="wg",
    )(query, key, value_transposed)
    return output, jnp.swapaxes(lse, 1, 2)


def _mosaic_attention_forward_warp_specialized_unmasked(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compatibility wrapper for the direct unmasked-kernel diagnostic."""
    return _mosaic_attention_forward_warp_specialized(
        query,
        key,
        value,
        is_causal=False,
    )


def _materialize_transposed_mask_tile(
    mask_fn: Callable[..., Any],
    *,
    batch: jax.Array,
    query_head: jax.Array,
    query_base: jax.Array,
    kv_base: jax.Array,
    block_q: int,
    block_kv: int,
    layout: Any,
    is_causal: bool = False,
) -> jax.Array:
    """Evaluate a user-mask tile in K-by-Q orientation."""

    shape = (block_kv, block_q)
    key_indices = plgpu.broadcasted_iota(
        jnp.int32,
        shape,
        0,
        layout=layout,
    )
    query_indices = plgpu.broadcasted_iota(
        jnp.int32,
        shape,
        1,
        layout=layout,
    )
    global_query_indices = query_indices + query_base
    global_key_indices = key_indices + kv_base
    mask = jnp.asarray(
        mask_fn(
            batch,
            query_head,
            global_query_indices,
            global_key_indices,
        ),
        dtype=jnp.bool_,
    )
    if not mask.shape:
        mask = lax.broadcast_in_dim(mask, shape, ())
    if is_causal:
        mask &= global_query_indices >= global_key_indices
    return mask


def _mosaic_attention_backward_two_pass(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
    mask_fn: Callable[..., Any],
    *,
    is_causal: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Memory-efficient general-mask backward using warp-level Mosaic MMA."""

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
    lhs_layout, rhs_layout, accumulator_layout = _mma_layouts(
        dtype,
        block_q,
    )
    row_layout = accumulator_layout.reduce(1)
    column_layout = accumulator_layout.reduce(0)

    gradient = upstream_gradient.astype(dtype)
    gradient_transposed = jnp.transpose(gradient, (0, 2, 3, 1))
    query_transposed = jnp.transpose(query, (0, 2, 3, 1))
    key_transposed = jnp.transpose(key, (0, 2, 3, 1))
    delta = jnp.sum(
        upstream_gradient.astype(jnp.float32) * output.astype(jnp.float32),
        axis=-1,
    )
    log_normalizer_transposed = jnp.transpose(log_normalizer, (0, 2, 1))
    delta_transposed = jnp.transpose(delta, (0, 2, 1))

    swizzle = min(128, block_kv * jnp.dtype(dtype).itemsize)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    matrix_scratch_type = plgpu.SMEM(
        (block_q, block_kv),
        dtype,
        transforms=(
            plgpu.TilingTransform((8, swizzle_elems)),
            plgpu.SwizzleTransform(swizzle),
        ),
    )

    def query_gradient_kernel(
        query_ref,
        key_ref,
        value_ref,
        key_transposed_ref,
        gradient_ref,
        lse_transposed_ref,
        delta_transposed_ref,
        query_gradient_ref,
        matrix_smem,
    ):
        batch = lax.axis_index("batch")
        query_head = lax.axis_index("heads")
        kv_head = lax.div(
            query_head,
            jnp.asarray(query_heads_per_kv_head, query_head.dtype),
        )
        query_tile = lax.axis_index("query_tiles")
        if is_causal:
            query_tile = num_query_tiles - 1 - query_tile
        query_base = query_tile * block_q
        query_slice = pl.ds(query_base, block_q)

        query_fragment = plgpu.load(
            query_ref.at[batch, query_slice, query_head],
            layout=lhs_layout,
            optimized=False,
        )
        gradient_fragment = plgpu.load(
            gradient_ref.at[
                batch,
                query_slice,
                query_head,
            ],
            layout=lhs_layout,
            optimized=False,
        )
        lse = plgpu.load(
            lse_transposed_ref.at[batch, query_head, query_slice],
            layout=row_layout,
            optimized=False,
        )
        delta_tile = plgpu.load(
            delta_transposed_ref.at[batch, query_head, query_slice],
            layout=row_layout,
            optimized=False,
        )
        lse_broadcast = plgpu.layout_cast(
            lax.broadcast_in_dim(
                lse,
                (block_q, block_kv),
                (0,),
            ),
            accumulator_layout,
        )
        delta_broadcast = plgpu.layout_cast(
            lax.broadcast_in_dim(
                delta_tile,
                (block_q, block_kv),
                (0,),
            ),
            accumulator_layout,
        )
        query_gradient = plgpu.layout_cast(
            jnp.zeros((block_q, head_dim), dtype=jnp.float32),
            accumulator_layout,
        )

        def kv_body(kv_step, query_gradient):
            kv_base = kv_step * block_kv
            kv_slice = pl.ds(kv_base, block_kv)
            mask = _materialize_mask_tile(
                mask_fn,
                batch=batch,
                query_head=query_head,
                query_base=query_base,
                kv_step=kv_step,
                block_q=block_q,
                block_kv=block_kv,
                layout=accumulator_layout,
                is_causal=is_causal,
            )

            def process_tile(query_gradient):
                key_for_scores = plgpu.load(
                    plgpu.transpose_ref(
                        key_ref.at[batch, kv_slice, kv_head],
                        (1, 0),
                    ),
                    layout=rhs_layout,
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
                    key_for_scores,
                )
                scores *= scale
                probabilities = jnp.where(
                    mask,
                    jnp.exp2((scores - lse_broadcast) * log2e),
                    jnp.zeros_like(scores),
                )

                value_for_dp = plgpu.load(
                    plgpu.transpose_ref(
                        value_ref.at[batch, kv_slice, kv_head],
                        (1, 0),
                    ),
                    layout=rhs_layout,
                    optimized=False,
                )
                dp = plgpu.layout_cast(
                    jnp.zeros(
                        (block_q, block_kv),
                        dtype=jnp.float32,
                    ),
                    accumulator_layout,
                )
                dp = plgpu.mma(dp, gradient_fragment, value_for_dp)
                ds = probabilities * (dp - delta_broadcast)

                key_for_dq = plgpu.load(
                    plgpu.transpose_ref(
                        key_transposed_ref.at[
                            batch,
                            kv_head,
                            :,
                            kv_slice,
                        ],
                        (1, 0),
                    ),
                    layout=rhs_layout,
                    optimized=False,
                )
                matrix_smem[...] = ds.astype(dtype)
                ds_fragment = plgpu.load(
                    matrix_smem,
                    layout=lhs_layout,
                    optimized=False,
                )
                query_gradient = plgpu.mma(
                    query_gradient,
                    ds_fragment,
                    key_for_dq,
                )
                return query_gradient

            if is_causal:
                return process_tile(query_gradient)
            return lax.cond(
                _tile_has_attention(
                    mask,
                    accumulator_layout=accumulator_layout,
                    row_layout=row_layout,
                ),
                process_tile,
                lambda x: x,
                query_gradient,
            )

        kv_stop = num_kv_tiles
        if is_causal:
            kv_stop = jnp.minimum(
                (query_base + block_q + block_kv - 1) // block_kv,
                num_kv_tiles,
            )
        query_gradient = lax.fori_loop(
            0,
            kv_stop,
            kv_body,
            query_gradient,
        )
        query_gradient_ref[
            batch,
            query_slice,
            query_head,
        ] = (query_gradient * scale).astype(dtype)

    query_gradient = plgpu.kernel(
        query_gradient_kernel,
        out_type=jax.ShapeDtypeStruct(query.shape, dtype),
        scratch_types=(matrix_scratch_type,),
        compiler_params=plgpu.CompilerParams(
            approx_math=False,
            lowering_semantics=(
                plgpu.LoweringSemantics.Lane
                if block_q < 64
                else plgpu.LoweringSemantics.Warpgroup
            ),
        ),
        grid=(query_heads, num_query_tiles, batch_size),
        grid_names=("heads", "query_tiles", "batch"),
    )(
        query,
        key,
        value,
        key_transposed,
        gradient,
        log_normalizer_transposed,
        delta_transposed,
    )

    def key_value_gradient_kernel(
        query_ref,
        key_ref,
        value_ref,
        query_transposed_ref,
        gradient_ref,
        gradient_transposed_ref,
        lse_transposed_ref,
        delta_transposed_ref,
        key_gradient_ref,
        value_gradient_ref,
        matrix_smem,
    ):
        batch = lax.axis_index("batch")
        kv_head = lax.axis_index("kv_heads")
        kv_base = lax.axis_index("kv_tiles") * block_kv
        kv_slice = pl.ds(kv_base, block_kv)

        key_fragment = plgpu.load(
            key_ref.at[batch, kv_slice, kv_head],
            layout=lhs_layout,
            optimized=False,
        )
        value_fragment = plgpu.load(
            value_ref.at[batch, kv_slice, kv_head],
            layout=lhs_layout,
            optimized=False,
        )
        key_gradient = plgpu.layout_cast(
            jnp.zeros((block_kv, head_dim), dtype=jnp.float32),
            accumulator_layout,
        )
        value_gradient = plgpu.layout_cast(
            jnp.zeros((block_kv, head_dim), dtype=jnp.float32),
            accumulator_layout,
        )

        for group_index in range(query_heads_per_kv_head):
            query_head = kv_head * query_heads_per_kv_head + group_index

            def query_body(query_step, gradients):
                query_base = query_step * block_q
                query_slice = pl.ds(query_base, block_q)
                mask_transposed = _materialize_transposed_mask_tile(
                    mask_fn,
                    batch=batch,
                    query_head=query_head,
                    query_base=query_base,
                    kv_base=kv_base,
                    block_q=block_q,
                    block_kv=block_kv,
                    layout=accumulator_layout,
                    is_causal=is_causal,
                )

                def process_tile(gradients):
                    key_gradient, value_gradient = gradients
                    query_for_scores = plgpu.load(
                        plgpu.transpose_ref(
                            query_ref.at[
                                batch,
                                query_slice,
                                query_head,
                            ],
                            (1, 0),
                        ),
                        layout=rhs_layout,
                        optimized=False,
                    )
                    scores_transposed = plgpu.mma(
                        plgpu.layout_cast(
                            jnp.zeros(
                                (block_kv, block_q),
                                dtype=jnp.float32,
                            ),
                            accumulator_layout,
                        ),
                        key_fragment,
                        query_for_scores,
                    )
                    scores_transposed *= scale
                    lse = plgpu.load(
                        lse_transposed_ref.at[
                            batch,
                            query_head,
                            query_slice,
                        ],
                        layout=column_layout,
                        optimized=False,
                    )
                    lse_broadcast = plgpu.layout_cast(
                        lax.broadcast_in_dim(
                            lse,
                            scores_transposed.shape,
                            (1,),
                        ),
                        accumulator_layout,
                    )
                    probabilities_transposed = jnp.where(
                        mask_transposed,
                        jnp.exp2((scores_transposed - lse_broadcast) * log2e),
                        jnp.zeros_like(scores_transposed),
                    )

                    dp_transposed = plgpu.layout_cast(
                        jnp.zeros(
                            (block_kv, block_q),
                            dtype=jnp.float32,
                        ),
                        accumulator_layout,
                    )
                    gradient_for_dp = plgpu.load(
                        plgpu.transpose_ref(
                            gradient_ref.at[
                                batch,
                                query_slice,
                                query_head,
                            ],
                            (1, 0),
                        ),
                        layout=rhs_layout,
                        optimized=False,
                    )
                    gradient_rhs = plgpu.load(
                        plgpu.transpose_ref(
                            gradient_transposed_ref.at[
                                batch,
                                query_head,
                                :,
                                query_slice,
                            ],
                            (1, 0),
                        ),
                        layout=rhs_layout,
                        optimized=False,
                    )
                    dp_transposed = plgpu.mma(
                        dp_transposed,
                        value_fragment,
                        gradient_for_dp,
                    )

                    delta_tile = plgpu.load(
                        delta_transposed_ref.at[
                            batch,
                            query_head,
                            query_slice,
                        ],
                        layout=column_layout,
                        optimized=False,
                    )
                    delta_broadcast = plgpu.layout_cast(
                        lax.broadcast_in_dim(
                            delta_tile,
                            dp_transposed.shape,
                            (1,),
                        ),
                        accumulator_layout,
                    )
                    ds_transposed = probabilities_transposed * (
                        dp_transposed - delta_broadcast
                    )

                    query_for_dk = plgpu.load(
                        plgpu.transpose_ref(
                            query_transposed_ref.at[
                                batch,
                                query_head,
                                :,
                                query_slice,
                            ],
                            (1, 0),
                        ),
                        layout=rhs_layout,
                        optimized=False,
                    )
                    matrix_smem[...] = ds_transposed.astype(dtype)
                    ds_fragment = plgpu.load(
                        matrix_smem,
                        layout=lhs_layout,
                        optimized=False,
                    )
                    key_gradient = plgpu.mma(
                        key_gradient,
                        ds_fragment,
                        query_for_dk,
                    )

                    matrix_smem[...] = probabilities_transposed.astype(dtype)
                    probability_fragment = plgpu.load(
                        matrix_smem,
                        layout=lhs_layout,
                        optimized=False,
                    )
                    value_gradient = plgpu.mma(
                        value_gradient,
                        probability_fragment,
                        gradient_rhs,
                    )

                    return key_gradient, value_gradient

                if is_causal:
                    return process_tile(gradients)
                return lax.cond(
                    _tile_has_attention(
                        mask_transposed,
                        accumulator_layout=accumulator_layout,
                        row_layout=row_layout,
                    ),
                    process_tile,
                    lambda x: x,
                    gradients,
                )

            query_start = 0
            if is_causal:
                query_start = jnp.minimum(
                    kv_base // block_q,
                    num_query_tiles,
                )
            key_gradient, value_gradient = lax.fori_loop(
                query_start,
                num_query_tiles,
                query_body,
                (key_gradient, value_gradient),
            )

        key_gradient_ref[
            batch,
            kv_slice,
            kv_head,
        ] = (key_gradient * scale).astype(dtype)
        value_gradient_ref[
            batch,
            kv_slice,
            kv_head,
        ] = value_gradient.astype(dtype)

    key_gradient, value_gradient = plgpu.kernel(
        key_value_gradient_kernel,
        out_type=(
            jax.ShapeDtypeStruct(key.shape, dtype),
            jax.ShapeDtypeStruct(value.shape, dtype),
        ),
        scratch_types=(matrix_scratch_type,),
        compiler_params=plgpu.CompilerParams(
            approx_math=False,
            lowering_semantics=(
                plgpu.LoweringSemantics.Lane
                if block_q < 64
                else plgpu.LoweringSemantics.Warpgroup
            ),
        ),
        grid=(kv_heads, num_kv_tiles, batch_size),
        grid_names=("kv_heads", "kv_tiles", "batch"),
    )(
        query,
        key,
        value,
        query_transposed,
        gradient,
        gradient_transposed,
        log_normalizer_transposed,
        delta_transposed,
    )
    return query_gradient, key_gradient, value_gradient


def _mosaic_attention_backward_one_pass(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
    mask_fn: Callable[..., Any],
    *,
    is_causal: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute all input gradients from one traversal of each score tile."""
    if not supports_mosaic_attention(query, key, value, mask_fn):
        raise ValueError("unsupported Mosaic attention configuration")
    config = _select_config(query, key)
    assert config is not None
    is_unmasked = _mask_is_always_true(mask_fn)

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
    lhs_layout, rhs_layout, accumulator_layout = _mma_layouts(
        dtype,
        block_q,
    )
    row_layout = accumulator_layout.reduce(1)

    gradient = upstream_gradient.astype(dtype)
    gradient_transposed = jnp.transpose(gradient, (0, 2, 3, 1))
    query_transposed = jnp.transpose(query, (0, 2, 3, 1))
    key_transposed = jnp.transpose(key, (0, 2, 3, 1))
    delta = jnp.sum(
        upstream_gradient.astype(jnp.float32) * output.astype(jnp.float32),
        axis=-1,
    )
    log_normalizer_transposed = jnp.transpose(log_normalizer, (0, 2, 1))
    delta_transposed = jnp.transpose(delta, (0, 2, 1))

    # Two 16-bit elements add one 32-bit shared-memory bank to every row.
    # That breaks the worst stride-64 conflicts in the transposed scalar load.
    matrix_scratch_type = plgpu.SMEM(
        (block_q, block_kv + 2),
        dtype,
    )
    mesh = plgpu.Mesh(
        grid=(query_heads, num_query_tiles, batch_size),
        grid_names=("heads", "query_tiles", "batch"),
    )
    compiler_params = plgpu.CompilerParams(
        approx_math=False,
        lowering_semantics=(
            plgpu.LoweringSemantics.Lane
            if block_q < 64
            else plgpu.LoweringSemantics.Warpgroup
        ),
    )

    # This shortcut is specific to the 64x64, four-warp MMA layouts.
    # After the FP32 accumulator is cast to the input dtype, MMA_ACC and
    # MMA_LHS assign every element to the same lane and register in the same
    # order. Smaller tiles use the general shared-memory conversion below.

    @plgpu.inline_mgpu(
        arg_types=(
            accumulator_layout,
            accumulator_layout,
            rhs_layout,
        ),
        return_type=plgpu.ShapeDtypeStruct(
            (block_q, head_dim),
            jnp.float32,
            layout=accumulator_layout,
        ),
    )
    def mma_with_accumulator_lhs(
        _,
        accumulator,
        matrix,
        rhs,
    ):
        lhs_layout = plgpu.Layout.MMA_LHS(dtype).to_mgpu()
        lhs = mgpu.FragmentedArray(
            _registers=matrix.registers.reshape(
                lhs_layout.registers_shape(matrix.shape)
            ),
            _layout=lhs_layout,
            _is_signed=matrix.is_signed,
        )
        return mgpu.mma(accumulator, lhs, rhs)

    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(),),
        return_type=plgpu.ShapeDtypeStruct(
            (block_kv, block_q),
            dtype,
            layout=lhs_layout,
        ),
    )
    def load_transposed_scratch(_, scratch_ref):
        return mgpu.FragmentedArray.build(
            (block_kv, block_q),
            lhs_layout.to_mgpu(),
            lambda row, column: memref.load(
                scratch_ref,
                (column, row),
            ),
        )

    def state_body(state):
        (
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            key_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
        ) = state

        @pl.kernel(
            mesh=mesh,
            out_type=(),
            compiler_params=compiler_params,
        )
        def kernel(
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            key_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
        ):
            batch = lax.axis_index("batch")
            query_head = lax.axis_index("heads")
            kv_head = lax.div(
                query_head,
                jnp.asarray(query_heads_per_kv_head, query_head.dtype),
            )
            query_tile = lax.axis_index("query_tiles")
            if is_causal:
                query_tile = num_query_tiles - 1 - query_tile
            query_base = query_tile * block_q
            query_slice = pl.ds(query_base, block_q)

            query_fragment = plgpu.load(
                query_ref.at[batch, query_slice, query_head],
                layout=lhs_layout,
                optimized=False,
            )
            query_for_dk = plgpu.load(
                plgpu.transpose_ref(
                    query_transposed_ref.at[
                        batch,
                        query_head,
                        :,
                        query_slice,
                    ],
                    (1, 0),
                ),
                layout=rhs_layout,
                optimized=False,
            )
            gradient_fragment = plgpu.load(
                gradient_ref.at[
                    batch,
                    query_slice,
                    query_head,
                ],
                layout=lhs_layout,
                optimized=False,
            )
            gradient_rhs = plgpu.load(
                plgpu.transpose_ref(
                    gradient_transposed_ref.at[
                        batch,
                        query_head,
                        :,
                        query_slice,
                    ],
                    (1, 0),
                ),
                layout=rhs_layout,
                optimized=False,
            )
            lse = plgpu.load(
                lse_transposed_ref.at[batch, query_head, query_slice],
                layout=row_layout,
                optimized=False,
            )
            delta_tile = plgpu.load(
                delta_transposed_ref.at[batch, query_head, query_slice],
                layout=row_layout,
                optimized=False,
            )
            lse_broadcast = plgpu.layout_cast(
                lax.broadcast_in_dim(
                    lse,
                    (block_q, block_kv),
                    (0,),
                ),
                accumulator_layout,
            )
            delta_broadcast = plgpu.layout_cast(
                lax.broadcast_in_dim(
                    delta_tile,
                    (block_q, block_kv),
                    (0,),
                ),
                accumulator_layout,
            )
            query_gradient = plgpu.layout_cast(
                jnp.zeros((block_q, head_dim), dtype=jnp.float32),
                accumulator_layout,
            )

            def process_tile(kv_step, query_gradient, mask):
                kv_base = kv_step * block_kv
                kv_slice = pl.ds(kv_base, block_kv)
                key_for_scores = plgpu.load(
                    plgpu.transpose_ref(
                        key_ref.at[batch, kv_slice, kv_head],
                        (1, 0),
                    ),
                    layout=rhs_layout,
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
                    key_for_scores,
                )
                scores *= scale
                probabilities = jnp.exp2(
                    (scores - lse_broadcast) * log2e
                )
                if mask is not None:
                    probabilities = jnp.where(
                        mask,
                        probabilities,
                        jnp.zeros_like(probabilities),
                    )

                value_for_dp = plgpu.load(
                    plgpu.transpose_ref(
                        value_ref.at[batch, kv_slice, kv_head],
                        (1, 0),
                    ),
                    layout=rhs_layout,
                    optimized=False,
                )
                dp = plgpu.layout_cast(
                    jnp.zeros(
                        (block_q, block_kv),
                        dtype=jnp.float32,
                    ),
                    accumulator_layout,
                )
                dp = plgpu.mma(dp, gradient_fragment, value_for_dp)
                ds = probabilities * (dp - delta_broadcast)

                key_for_dq = plgpu.load(
                    plgpu.transpose_ref(
                        key_transposed_ref.at[
                            batch,
                            kv_head,
                            :,
                            kv_slice,
                        ],
                        (1, 0),
                    ),
                    layout=rhs_layout,
                    optimized=False,
                )
                key_gradient = plgpu.layout_cast(
                    jnp.zeros(
                        (block_kv, head_dim),
                        dtype=jnp.float32,
                    ),
                    accumulator_layout,
                )
                ds_component = ds.astype(dtype)

                if block_q == 64:
                    def load_transposed_ds(matrix_smem):
                        matrix_smem[:, :block_kv] = ds_component
                        return load_transposed_scratch(matrix_smem)

                    ds_transposed_fragment = pl.run_scoped(
                        load_transposed_ds,
                        matrix_scratch_type,
                    )
                    query_gradient = mma_with_accumulator_lhs(
                        query_gradient,
                        ds_component,
                        key_for_dq,
                    )
                else:
                    def load_ds_fragments(matrix_smem):
                        matrix_smem[:, :block_kv] = ds_component
                        direct = plgpu.load(
                            matrix_smem.at[:, :block_kv],
                            layout=lhs_layout,
                            optimized=False,
                        )
                        return load_transposed_scratch(matrix_smem), direct

                    ds_transposed_fragment, ds_fragment = pl.run_scoped(
                        load_ds_fragments,
                        matrix_scratch_type,
                    )
                    query_gradient = plgpu.mma(
                        query_gradient,
                        ds_fragment,
                        key_for_dq,
                    )
                key_gradient = plgpu.mma(
                    key_gradient,
                    ds_transposed_fragment,
                    query_for_dk,
                )

                value_gradient = plgpu.layout_cast(
                    jnp.zeros(
                        (block_kv, head_dim),
                        dtype=jnp.float32,
                    ),
                    accumulator_layout,
                )
                probability_component = probabilities.astype(dtype)

                def load_probability_fragment(matrix_smem):
                    matrix_smem[:, :block_kv] = probability_component
                    return load_transposed_scratch(
                        matrix_smem,
                    )

                probability_transposed_fragment = pl.run_scoped(
                    load_probability_fragment,
                    matrix_scratch_type,
                )
                value_gradient = plgpu.mma(
                    value_gradient,
                    probability_transposed_fragment,
                    gradient_rhs,
                )

                plgpu.atomic_add(
                    key_gradient_ref.at[
                        batch,
                        kv_slice,
                        kv_head,
                    ],
                    key_gradient * scale,
                )
                plgpu.atomic_add(
                    value_gradient_ref.at[
                        batch,
                        kv_slice,
                        kv_head,
                    ],
                    value_gradient,
                )
                return query_gradient

            def unmasked_kv_body(kv_step, query_gradient):
                return process_tile(kv_step, query_gradient, None)

            def materialize_and_process(
                kv_step,
                query_gradient,
                *,
                apply_causal,
            ):
                mask = _materialize_mask_tile(
                    mask_fn,
                    batch=batch,
                    query_head=query_head,
                    query_base=query_base,
                    kv_step=kv_step,
                    block_q=block_q,
                    block_kv=block_kv,
                    layout=accumulator_layout,
                    is_causal=apply_causal,
                )
                if is_unmasked:
                    return process_tile(kv_step, query_gradient, mask)
                return lax.cond(
                    _tile_has_attention(
                        mask,
                        accumulator_layout=accumulator_layout,
                        row_layout=row_layout,
                    ),
                    lambda x: process_tile(kv_step, x, mask),
                    lambda x: x,
                    query_gradient,
                )

            def user_masked_kv_body(kv_step, query_gradient):
                return materialize_and_process(
                    kv_step,
                    query_gradient,
                    apply_causal=False,
                )

            def causal_masked_kv_body(kv_step, query_gradient):
                return materialize_and_process(
                    kv_step,
                    query_gradient,
                    apply_causal=True,
                )

            if is_unmasked and not is_causal:
                query_gradient = lax.fori_loop(
                    0,
                    num_kv_tiles,
                    unmasked_kv_body,
                    query_gradient,
                )
            elif is_causal:
                full_kv_stop = jnp.minimum(
                    query_base // block_kv,
                    num_kv_tiles,
                )
                kv_stop = jnp.minimum(
                    (query_base + block_q + block_kv - 1) // block_kv,
                    num_kv_tiles,
                )
                query_gradient = lax.fori_loop(
                    0,
                    full_kv_stop,
                    (
                        unmasked_kv_body
                        if is_unmasked
                        else user_masked_kv_body
                    ),
                    query_gradient,
                )
                query_gradient = lax.fori_loop(
                    full_kv_stop,
                    kv_stop,
                    causal_masked_kv_body,
                    query_gradient,
                )
            else:
                query_gradient = lax.fori_loop(
                    0,
                    num_kv_tiles,
                    user_masked_kv_body,
                    query_gradient,
                )
            query_gradient_ref[
                batch,
                query_slice,
                query_head,
            ] = (query_gradient * scale).astype(dtype)

        kernel(
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            key_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
        )

    initial_state = (
        query,
        key,
        value,
        query_transposed,
        key_transposed,
        gradient,
        gradient_transposed,
        log_normalizer_transposed,
        delta_transposed,
        jnp.zeros_like(query),
        jnp.zeros_like(key, dtype=jnp.float32),
        jnp.zeros_like(value, dtype=jnp.float32),
    )
    final_state = pl.run_state(state_body)(initial_state)
    query_gradient, key_gradient, value_gradient = final_state[-3:]
    return (
        query_gradient,
        key_gradient.astype(dtype),
        value_gradient.astype(dtype),
    )


def _mosaic_attention_backward_warp_specialized(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
    *,
    is_causal: bool = False,
    compute_kv_gradients: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute large dense gradients with shared, pipelined operands.

    Two compute warpgroups process adjacent query tiles. A memory warpgroup
    stages one row-major copy of each K tile; native matrix loads derive both
    the transposed score operand and the untransposed dQ operand. Query and
    output-gradient RHS views remain in shared memory rather than occupying
    registers for the entire K/V traversal. Maximal causal attention streams
    only the common visible K/V prefix and masks each diagonal tile.
    """

    if query.dtype not in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        raise ValueError("warp-specialized backward requires FP16 or BF16")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape != output.shape or query.shape != upstream_gradient.shape:
        raise ValueError("output and upstream gradient must match query shape")
    if query.shape[3] != 64 or key.shape[3] != 64 or value.shape[3] != 64:
        raise ValueError("warp-specialized backward currently requires D=64")
    if is_causal and query.shape[1] != key.shape[1]:
        raise ValueError(
            "warp-specialized causal backward requires square self-attention"
        )

    batch_size, query_length, query_heads, head_dim = query.shape
    kv_length, kv_heads = key.shape[1:3]
    block_q = 64
    block_kv = 64
    num_compute_wgs = 2
    compute_registers = 232 if compute_kv_gradients else 240
    producer_registers = 40 if compute_kv_gradients else 32
    query_superblock = block_q * num_compute_wgs
    if query_length % query_superblock:
        raise ValueError(
            f"query length must be divisible by {query_superblock}"
        )
    if kv_length % block_kv:
        raise ValueError(f"key/value length must be divisible by {block_kv}")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")

    query_heads_per_kv_head = query_heads // kv_heads
    num_query_supertiles = query_length // query_superblock
    num_kv_tiles = kv_length // block_kv
    key_pipeline_depth = 2
    dtype = query.dtype
    scale = float(head_dim**-0.5)
    log2e = math.log2(math.e)
    accumulator_layout = plgpu.Layout.MMA_ACC(dtype)
    row_layout = accumulator_layout.reduce(1)
    swizzle = 128
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    rhs_smem_transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    gradient = upstream_gradient.astype(dtype)
    query_transposed = jnp.transpose(query, (0, 2, 3, 1))
    gradient_transposed = jnp.transpose(gradient, (0, 2, 3, 1))
    delta = jnp.sum(
        upstream_gradient.astype(jnp.float32) * output.astype(jnp.float32),
        axis=-1,
    )
    log_normalizer_transposed = jnp.transpose(log_normalizer, (0, 2, 1))
    delta_transposed = jnp.transpose(delta, (0, 2, 1))

    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(rhs_smem_transforms),),
        return_type=plgpu.ShapeDtypeStruct(
            (block_kv, head_dim),
            dtype,
            layout=plgpu.Layout.MMA_RHS(dtype),
        ),
    )
    def load_shared_rhs(_, smem_ref):
        transposed_ref = mgpu.memref_transpose(smem_ref, (1, 0, 3, 2))
        return mgpu.FragmentedArray.load_tiled(
            transposed_ref,
            swizzle=swizzle,
            layout=plgpu.Layout.MMA_RHS(dtype).to_mgpu(),
            optimized=True,
            tiling_rank=2,
        )

    # ``mma.sync`` distributes its column-major RHS in pairs along K. The
    # shared K tile is row-major, so use the current NVVM ``ldmatrix.trans``
    # operation to form that register layout without a second global-memory
    # transpose. This mapping is specific to the validated 64x64, 16-bit tile.
    assert block_kv == head_dim == 64
    assert jnp.dtype(dtype).itemsize == 2

    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(rhs_smem_transforms),),
        return_type=plgpu.ShapeDtypeStruct(
            (block_kv, head_dim),
            dtype,
            layout=plgpu.Layout.MMA_RHS(dtype),
        ),
    )
    def load_shared_rhs_without_transpose(_, smem_ref):
        rhs_layout = plgpu.Layout.MMA_RHS(dtype).to_mgpu()
        ref_type = ir.MemRefType(smem_ref.type)
        register_type = ir.VectorType.get(
            (rhs_layout.vector_length,),
            ref_type.element_type,
        )
        registers = mgpu.FragmentedArray.splat(
            mgpu.c(0, ref_type.element_type),
            (block_kv, head_dim),
            rhs_layout,
        ).registers.copy()
        matrix_shape = ir.Attribute.parse(
            "#nvvm.ld_st_matrix_shape<m=8, n=8>"
        )
        int32 = ir.IntegerType.get_signless(32)
        lane = arith.remui(
            mgpu.thread_idx(),
            mgpu.c(32, int32),
        )
        lane_in_octet = arith.remui(lane, mgpu.c(8, int32))
        quadrant = arith.divui(lane, mgpu.c(8, int32))
        base_pointer = mgpu.utils.memref_ptr(smem_ref)
        for column_block in range(8):
            for row_block_base in (0, 4):
                source_row_block = arith.addi(
                    mgpu.c(row_block_base, int32),
                    quadrant,
                )
                source_row = arith.addi(
                    arith.muli(
                        source_row_block,
                        mgpu.c(8, int32),
                    ),
                    lane_in_octet,
                )
                vector_offset = arith.addi(
                    arith.muli(source_row, mgpu.c(32, int32)),
                    mgpu.c(column_block * 4, int32),
                )
                swizzled_offset = arith.xori(
                    vector_offset,
                    arith.muli(
                        lane_in_octet,
                        mgpu.c(4, int32),
                    ),
                )
                pointer = mgpu.utils.getelementptr(
                    base_pointer,
                    [swizzled_offset],
                    register_type,
                )
                loaded = nvvm.ldmatrix(
                    pointer,
                    num=4,
                    layout=nvvm.MMALayout.col,
                    shape=matrix_shape,
                    elt_type=nvvm.LdStMatrixEltType.B16,
                )
                for matrix_index in range(4):
                    row_block = row_block_base + matrix_index
                    registers[
                        (
                            row_block // 2,
                            column_block,
                            row_block % 2,
                            0,
                            0,
                            0,
                            0,
                            0,
                        )
                    ] = mgpu.utils.bitcast(
                        llvm.extractvalue(
                            int32,
                            loaded,
                            [matrix_index],
                        ),
                        register_type,
                    )
        return mgpu.FragmentedArray(
            _registers=registers,
            _layout=rhs_layout,
            _is_signed=None,
        )

    @plgpu.inline_mgpu(
        arg_types=(
            accumulator_layout,
            accumulator_layout,
            plgpu.Layout.MMA_RHS(dtype),
        ),
        return_type=plgpu.ShapeDtypeStruct(
            (block_q, head_dim),
            jnp.float32,
            layout=accumulator_layout,
        ),
    )
    def mma_with_accumulator_lhs(
        _,
        accumulator,
        matrix,
        rhs,
    ):
        lhs_layout = plgpu.Layout.MMA_LHS(dtype).to_mgpu()
        lhs = mgpu.FragmentedArray(
            _registers=matrix.registers.reshape(
                lhs_layout.registers_shape(matrix.shape)
            ),
            _layout=lhs_layout,
            _is_signed=matrix.is_signed,
        )
        return mgpu.mma(accumulator, lhs, rhs)

    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(rhs_smem_transforms),),
        return_type=plgpu.ShapeDtypeStruct(
            (head_dim, block_q),
            dtype,
            layout=plgpu.Layout.MMA_LHS(dtype),
        ),
    )
    def load_shared_lhs(_, scratch_ref):
        return mgpu.FragmentedArray.load_tiled(
            scratch_ref,
            swizzle=swizzle,
            layout=plgpu.Layout.MMA_LHS(dtype).to_mgpu(),
            optimized=True,
            tiling_rank=2,
        )

    mesh = plgpu.Mesh(
        grid=(query_heads, num_query_supertiles, batch_size),
        grid_names=("heads", "query_supertiles", "batch"),
        num_threads=num_compute_wgs + 1,
        thread_name="wg",
    )
    unused_scratch = plgpu.SMEM((1,), dtype)
    scratch_types = (
        # Two K slots preserve producer overlap while using the same 16 KiB
        # that the former one-slot K and one-slot transposed-K buffers used.
        plgpu.SMEM(
            (key_pipeline_depth, block_kv, head_dim),
            dtype,
            transforms=rhs_smem_transforms,
        ),
        plgpu.SMEM(
            (block_kv, head_dim),
            dtype,
            transforms=rhs_smem_transforms,
        ),
        (
            plgpu.SMEM(
                (num_compute_wgs, head_dim, block_q),
                dtype,
                transforms=rhs_smem_transforms,
            )
            if compute_kv_gradients
            else unused_scratch
        ),
        (
            plgpu.SMEM(
                (num_compute_wgs, head_dim, block_q),
                dtype,
                transforms=rhs_smem_transforms,
            )
            if compute_kv_gradients
            else unused_scratch
        ),
        (
            plgpu.SMEM(
                (num_compute_wgs, block_q, block_kv),
                dtype,
                transforms=rhs_smem_transforms,
            )
            if compute_kv_gradients
            else unused_scratch
        ),
        plgpu.Barrier(num_barriers=key_pipeline_depth),
        plgpu.Barrier(),
        plgpu.Barrier(num_barriers=num_compute_wgs),
        plgpu.Barrier(num_barriers=num_compute_wgs),
        plgpu.Barrier(
            num_arrivals=num_compute_wgs,
            num_barriers=key_pipeline_depth,
        ),
        plgpu.Barrier(num_arrivals=num_compute_wgs),
    )
    compiler_params = plgpu.CompilerParams(
        approx_math=False,
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
    )

    def state_body(state):
        (
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
        ) = state

        def kernel_with_scratch(
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
            key_for_scores_smem,
            value_for_dp_smem,
            query_for_dk_smem,
            gradient_for_dv_smem,
            transpose_smem,
            key_for_scores_ready,
            value_for_dp_ready,
            query_for_dk_ready,
            gradient_for_dv_ready,
            key_for_scores_consumed,
            value_for_dp_consumed,
        ):
            batch = lax.axis_index("batch")
            query_head = lax.axis_index("heads")
            kv_head = lax.div(
                query_head,
                jnp.asarray(query_heads_per_kv_head, query_head.dtype),
            )
            wg_index = lax.axis_index("wg")
            query_supertile = lax.axis_index("query_supertiles")
            if is_causal:
                query_supertile = num_query_supertiles - 1 - query_supertile
                producer_kv_stop = jnp.minimum(
                    (query_supertile + 1) * (query_superblock // block_kv),
                    num_kv_tiles,
                )
            else:
                producer_kv_stop = num_kv_tiles

            @pl.when(wg_index < num_compute_wgs)
            def compute_warpgroup():
                plgpu.set_max_registers(
                    compute_registers,
                    action="increase",
                )
                query_tile = query_supertile * num_compute_wgs + wg_index
                query_base = query_tile * block_q
                query_slice = pl.ds(query_base, block_q)
                matrix_smem = (
                    transpose_smem.at[wg_index]
                    if compute_kv_gradients
                    else None
                )

                query_fragment = plgpu.load(
                    query_ref.at[
                        batch,
                        query_slice,
                        query_head,
                    ],
                    layout=plgpu.Layout.MMA_LHS(dtype),
                    optimized=False,
                )
                gradient_fragment = plgpu.load(
                    gradient_ref.at[
                        batch,
                        query_slice,
                        query_head,
                    ],
                    layout=plgpu.Layout.MMA_LHS(dtype),
                    optimized=False,
                )
                lse = plgpu.load(
                    lse_transposed_ref.at[
                        batch,
                        query_head,
                        query_slice,
                    ],
                    layout=row_layout,
                    optimized=False,
                )
                delta_tile = plgpu.load(
                    delta_transposed_ref.at[
                        batch,
                        query_head,
                        query_slice,
                    ],
                    layout=row_layout,
                    optimized=False,
                )
                lse_broadcast = plgpu.layout_cast(
                    lax.broadcast_in_dim(
                        lse,
                        (block_q, block_kv),
                        (0,),
                    ),
                    accumulator_layout,
                )
                delta_broadcast = plgpu.layout_cast(
                    lax.broadcast_in_dim(
                        delta_tile,
                        (block_q, block_kv),
                        (0,),
                    ),
                    accumulator_layout,
                )
                query_gradient = plgpu.layout_cast(
                    jnp.zeros(
                        (block_q, head_dim),
                        dtype=jnp.float32,
                    ),
                    accumulator_layout,
                )
                if is_causal:
                    query_indices = plgpu.broadcasted_iota(
                        jnp.int32,
                        (block_q, block_kv),
                        0,
                        layout=accumulator_layout,
                    )
                    key_indices = plgpu.broadcasted_iota(
                        jnp.int32,
                        (block_q, block_kv),
                        1,
                        layout=accumulator_layout,
                    )
                    diagonal_mask = query_indices >= key_indices

                if compute_kv_gradients:
                    plgpu.barrier_wait(query_for_dk_ready.at[wg_index])
                    plgpu.barrier_wait(gradient_for_dv_ready.at[wg_index])

                def process_kv_tile(kv_step, query_gradient):
                    kv_base = kv_step * block_kv
                    kv_slice = pl.ds(kv_base, block_kv)
                    key_slot = lax.rem(
                        kv_step,
                        jnp.asarray(
                            key_pipeline_depth,
                            kv_step.dtype,
                        ),
                    )

                    plgpu.barrier_wait(
                        key_for_scores_ready.at[key_slot]
                    )
                    key_for_scores = load_shared_rhs(
                        key_for_scores_smem.at[key_slot]
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
                        key_for_scores,
                    )
                    scores *= scale
                    probabilities = jnp.exp2(
                        (scores - lse_broadcast) * log2e
                    )
                    if is_causal:
                        probabilities = lax.cond(
                            kv_step == query_tile,
                            lambda matrix: jnp.where(
                                diagonal_mask,
                                matrix,
                                jnp.zeros_like(matrix),
                            ),
                            lambda matrix: matrix,
                            probabilities,
                        )
                        probabilities = lax.cond(
                            kv_step <= query_tile,
                            lambda matrix: matrix,
                            jnp.zeros_like,
                            probabilities,
                        )

                    plgpu.barrier_wait(value_for_dp_ready)
                    value_for_dp = load_shared_rhs(value_for_dp_smem)
                    plgpu.barrier_arrive(value_for_dp_consumed)
                    dp = plgpu.mma(
                        plgpu.layout_cast(
                            jnp.zeros(
                                (block_q, block_kv),
                                dtype=jnp.float32,
                            ),
                            accumulator_layout,
                        ),
                        gradient_fragment,
                        value_for_dp,
                    )
                    ds = probabilities * (dp - delta_broadcast)
                    ds_component = ds.astype(dtype)

                    key_for_dq = load_shared_rhs_without_transpose(
                        key_for_scores_smem.at[key_slot]
                    )
                    plgpu.barrier_arrive(
                        key_for_scores_consumed.at[key_slot]
                    )
                    query_gradient = mma_with_accumulator_lhs(
                        query_gradient,
                        ds_component,
                        key_for_dq,
                    )
                    if not compute_kv_gradients:
                        return query_gradient

                    probability_component = probabilities.astype(dtype)
                    matrix_smem[...] = probability_component
                    probability_rhs = load_shared_rhs_without_transpose(
                        matrix_smem
                    )
                    gradient_for_dv = load_shared_lhs(
                        gradient_for_dv_smem.at[wg_index]
                    )
                    value_gradient_transposed = plgpu.mma(
                        plgpu.layout_cast(
                            jnp.zeros(
                                (head_dim, block_kv),
                                dtype=jnp.float32,
                            ),
                            accumulator_layout,
                        ),
                        gradient_for_dv,
                        probability_rhs,
                    )
                    plgpu.atomic_add(
                        value_gradient_ref.at[
                            batch,
                            kv_head,
                            :,
                            kv_slice,
                        ],
                        value_gradient_transposed,
                    )

                    matrix_smem[...] = ds_component
                    ds_rhs = load_shared_rhs_without_transpose(matrix_smem)
                    query_for_dk = load_shared_lhs(
                        query_for_dk_smem.at[wg_index]
                    )
                    key_gradient_transposed = plgpu.mma(
                        plgpu.layout_cast(
                            jnp.zeros(
                                (head_dim, block_kv),
                                dtype=jnp.float32,
                            ),
                            accumulator_layout,
                        ),
                        query_for_dk,
                        ds_rhs,
                    )
                    plgpu.atomic_add(
                        key_gradient_ref.at[
                            batch,
                            kv_head,
                            :,
                            kv_slice,
                        ],
                        key_gradient_transposed * scale,
                    )
                    return query_gradient

                query_gradient = lax.fori_loop(
                    0,
                    producer_kv_stop,
                    process_kv_tile,
                    query_gradient,
                )
                query_gradient_ref[
                    batch,
                    query_slice,
                    query_head,
                ] = (query_gradient * scale).astype(dtype)

            @pl.when(wg_index == num_compute_wgs)
            def memory_warpgroup():
                plgpu.set_max_registers(
                    producer_registers,
                    action="decrease",
                )
                query_super_base = query_supertile * query_superblock
                if compute_kv_gradients:
                    for consumer in range(num_compute_wgs):
                        query_base = query_super_base + consumer * block_q
                        query_slice = pl.ds(query_base, block_q)
                        query_transposed_slice = (
                            batch,
                            query_head,
                            slice(None),
                            query_slice,
                        )
                        plgpu.copy_gmem_to_smem(
                            query_transposed_ref.at[query_transposed_slice],
                            query_for_dk_smem.at[consumer],
                            query_for_dk_ready.at[consumer],
                        )
                        plgpu.copy_gmem_to_smem(
                            gradient_transposed_ref.at[
                                query_transposed_slice
                            ],
                            gradient_for_dv_smem.at[consumer],
                            gradient_for_dv_ready.at[consumer],
                        )

                for key_slot in range(key_pipeline_depth):
                    initial_key_slice = (
                        batch,
                        pl.ds(key_slot * block_kv, block_kv),
                        kv_head,
                    )
                    plgpu.copy_gmem_to_smem(
                        key_ref.at[initial_key_slice],
                        key_for_scores_smem.at[key_slot],
                        key_for_scores_ready.at[key_slot],
                    )
                first_kv_slice = (
                    batch,
                    pl.ds(0, block_kv),
                    kv_head,
                )
                plgpu.copy_gmem_to_smem(
                    value_ref.at[first_kv_slice],
                    value_for_dp_smem,
                    value_for_dp_ready,
                )

                @pl.loop(0, producer_kv_stop - key_pipeline_depth)
                def refill_pipeline(kv_step):
                    key_slot = lax.rem(
                        kv_step,
                        jnp.asarray(
                            key_pipeline_depth,
                            kv_step.dtype,
                        ),
                    )
                    next_key_step = kv_step + key_pipeline_depth
                    next_key_slice = (
                        batch,
                        pl.ds(next_key_step * block_kv, block_kv),
                        kv_head,
                    )
                    plgpu.barrier_wait(
                        key_for_scores_consumed.at[key_slot]
                    )
                    plgpu.copy_gmem_to_smem(
                        key_ref.at[next_key_slice],
                        key_for_scores_smem.at[key_slot],
                        key_for_scores_ready.at[key_slot],
                    )

                    next_value_step = kv_step + 1
                    next_value_slice = (
                        batch,
                        pl.ds(next_value_step * block_kv, block_kv),
                        kv_head,
                    )
                    plgpu.barrier_wait(value_for_dp_consumed)
                    plgpu.copy_gmem_to_smem(
                        value_ref.at[next_value_slice],
                        value_for_dp_smem,
                        value_for_dp_ready,
                    )

                final_value_slice = (
                    batch,
                    pl.ds(
                        (producer_kv_stop - 1) * block_kv,
                        block_kv,
                    ),
                    kv_head,
                )
                plgpu.barrier_wait(value_for_dp_consumed)
                plgpu.copy_gmem_to_smem(
                    value_ref.at[final_value_slice],
                    value_for_dp_smem,
                    value_for_dp_ready,
                )

        @pl.kernel(
            mesh=mesh,
            out_type=(),
            compiler_params=compiler_params,
        )
        def kernel(
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
        ):
            def run_with_scratch(*scratch_refs):
                kernel_with_scratch(
                    query_ref,
                    key_ref,
                    value_ref,
                    query_transposed_ref,
                    gradient_ref,
                    gradient_transposed_ref,
                    lse_transposed_ref,
                    delta_transposed_ref,
                    query_gradient_ref,
                    key_gradient_ref,
                    value_gradient_ref,
                    *scratch_refs,
                )

            pl.run_scoped(
                run_with_scratch,
                *scratch_types,
                collective_axes=("wg",),
            )

        kernel(
            query_ref,
            key_ref,
            value_ref,
            query_transposed_ref,
            gradient_ref,
            gradient_transposed_ref,
            lse_transposed_ref,
            delta_transposed_ref,
            query_gradient_ref,
            key_gradient_ref,
            value_gradient_ref,
        )

    initial_state = (
        query,
        key,
        value,
        query_transposed,
        gradient,
        gradient_transposed,
        log_normalizer_transposed,
        delta_transposed,
        jnp.zeros_like(query),
        jnp.zeros(
            (batch_size, kv_heads, head_dim, kv_length),
            dtype=jnp.float32,
        ),
        jnp.zeros(
            (batch_size, kv_heads, head_dim, kv_length),
            dtype=jnp.float32,
        ),
    )
    final_state = pl.run_state(state_body)(initial_state)
    (
        query_gradient,
        key_gradient_transposed,
        value_gradient_transposed,
    ) = final_state[-3:]
    key_gradient = jnp.transpose(key_gradient_transposed, (0, 3, 1, 2))
    value_gradient = jnp.transpose(value_gradient_transposed, (0, 3, 1, 2))
    return (
        query_gradient,
        key_gradient.astype(dtype),
        value_gradient.astype(dtype),
    )


def _mosaic_attention_backward_warp_specialized_causal_dkv(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute maximal-causal dK/dV without cross-program atomics.

    Two compute warpgroups own adjacent K/V tiles and accumulate their complete
    FP32 gradients while a producer streams the common suffix of transposed
    query and output-gradient tiles. Each result therefore needs one final
    store instead of one atomic contribution per active score tile.
    """

    if query.dtype not in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        raise ValueError("warp-specialized backward requires FP16 or BF16")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape != output.shape or query.shape != upstream_gradient.shape:
        raise ValueError("output and upstream gradient must match query shape")
    if query.shape[1] != key.shape[1]:
        raise ValueError(
            "warp-specialized causal backward requires square self-attention"
        )
    if query.shape[2] != key.shape[2]:
        raise ValueError(
            "non-atomic causal dK/dV currently requires equal Q and K/V heads"
        )
    if query.shape[3] != 64 or key.shape[3] != 64 or value.shape[3] != 64:
        raise ValueError("warp-specialized backward currently requires D=64")

    batch_size, query_length, query_heads, head_dim = query.shape
    kv_length, kv_heads = key.shape[1:3]
    block_q = 64
    block_kv = 64
    num_compute_wgs = 2
    compute_registers = 256
    producer_registers = 24
    kv_superblock = block_kv * num_compute_wgs
    if query_length % block_q:
        raise ValueError(f"query length must be divisible by {block_q}")
    if kv_length % kv_superblock:
        raise ValueError(
            f"key/value length must be divisible by {kv_superblock}"
        )

    num_query_tiles = query_length // block_q
    num_kv_supertiles = kv_length // kv_superblock
    pipeline_depth = 2
    dtype = query.dtype
    scale = float(head_dim**-0.5)
    log2e = math.log2(math.e)
    accumulator_layout = plgpu.Layout.MMA_ACC(dtype)
    column_layout = accumulator_layout.reduce(0)
    swizzle = 128
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    rhs_smem_transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    gradient = upstream_gradient.astype(dtype)
    query_transposed = jnp.transpose(query, (0, 2, 3, 1))
    gradient_transposed = jnp.transpose(gradient, (0, 2, 3, 1))
    delta = jnp.sum(
        upstream_gradient.astype(jnp.float32) * output.astype(jnp.float32),
        axis=-1,
    )
    log_normalizer_transposed = jnp.transpose(log_normalizer, (0, 2, 1))
    delta_transposed = jnp.transpose(delta, (0, 2, 1))

    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(rhs_smem_transforms),),
        return_type=plgpu.ShapeDtypeStruct(
            (block_q, head_dim),
            dtype,
            layout=plgpu.Layout.MMA_RHS(dtype),
        ),
    )
    def load_shared_rhs(_, smem_ref):
        transposed_ref = mgpu.memref_transpose(smem_ref, (1, 0, 3, 2))
        return mgpu.FragmentedArray.load_tiled(
            transposed_ref,
            swizzle=swizzle,
            layout=plgpu.Layout.MMA_RHS(dtype).to_mgpu(),
            optimized=True,
            tiling_rank=2,
        )

    # Load a row-major 64x64 shared tile directly into the column-major MMA
    # RHS register ownership. This is the same validated SM120 ldmatrix mapping
    # used by the one-pass warp-specialized backward.
    @plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(rhs_smem_transforms),),
        return_type=plgpu.ShapeDtypeStruct(
            (head_dim, block_q),
            dtype,
            layout=plgpu.Layout.MMA_RHS(dtype),
        ),
    )
    def load_shared_rhs_without_transpose(_, smem_ref):
        rhs_layout = plgpu.Layout.MMA_RHS(dtype).to_mgpu()
        ref_type = ir.MemRefType(smem_ref.type)
        register_type = ir.VectorType.get(
            (rhs_layout.vector_length,),
            ref_type.element_type,
        )
        registers = mgpu.FragmentedArray.splat(
            mgpu.c(0, ref_type.element_type),
            (head_dim, block_q),
            rhs_layout,
        ).registers.copy()
        matrix_shape = ir.Attribute.parse(
            "#nvvm.ld_st_matrix_shape<m=8, n=8>"
        )
        int32 = ir.IntegerType.get_signless(32)
        lane = arith.remui(
            mgpu.thread_idx(),
            mgpu.c(32, int32),
        )
        lane_in_octet = arith.remui(lane, mgpu.c(8, int32))
        quadrant = arith.divui(lane, mgpu.c(8, int32))
        base_pointer = mgpu.utils.memref_ptr(smem_ref)
        for column_block in range(8):
            for row_block_base in (0, 4):
                source_row_block = arith.addi(
                    mgpu.c(row_block_base, int32),
                    quadrant,
                )
                source_row = arith.addi(
                    arith.muli(
                        source_row_block,
                        mgpu.c(8, int32),
                    ),
                    lane_in_octet,
                )
                vector_offset = arith.addi(
                    arith.muli(source_row, mgpu.c(32, int32)),
                    mgpu.c(column_block * 4, int32),
                )
                swizzled_offset = arith.xori(
                    vector_offset,
                    arith.muli(
                        lane_in_octet,
                        mgpu.c(4, int32),
                    ),
                )
                pointer = mgpu.utils.getelementptr(
                    base_pointer,
                    [swizzled_offset],
                    register_type,
                )
                loaded = nvvm.ldmatrix(
                    pointer,
                    num=4,
                    layout=nvvm.MMALayout.col,
                    shape=matrix_shape,
                    elt_type=nvvm.LdStMatrixEltType.B16,
                )
                for matrix_index in range(4):
                    row_block = row_block_base + matrix_index
                    registers[
                        (
                            row_block // 2,
                            column_block,
                            row_block % 2,
                            0,
                            0,
                            0,
                            0,
                            0,
                        )
                    ] = mgpu.utils.bitcast(
                        llvm.extractvalue(
                            int32,
                            loaded,
                            [matrix_index],
                        ),
                        register_type,
                    )
        return mgpu.FragmentedArray(
            _registers=registers,
            _layout=rhs_layout,
            _is_signed=None,
        )

    @plgpu.inline_mgpu(
        arg_types=(
            accumulator_layout,
            accumulator_layout,
            plgpu.Layout.MMA_RHS(dtype),
        ),
        return_type=plgpu.ShapeDtypeStruct(
            (block_kv, head_dim),
            jnp.float32,
            layout=accumulator_layout,
        ),
    )
    def mma_with_accumulator_lhs(
        _,
        accumulator,
        matrix,
        rhs,
    ):
        lhs_layout = plgpu.Layout.MMA_LHS(dtype).to_mgpu()
        lhs = mgpu.FragmentedArray(
            _registers=matrix.registers.reshape(
                lhs_layout.registers_shape(matrix.shape)
            ),
            _layout=lhs_layout,
            _is_signed=matrix.is_signed,
        )
        return mgpu.mma(accumulator, lhs, rhs)

    def kernel(
        query_transposed_ref,
        key_ref,
        value_ref,
        gradient_transposed_ref,
        lse_transposed_ref,
        delta_transposed_ref,
        key_gradient_ref,
        value_gradient_ref,
        smem_buffers,
        ready_barriers,
        consumed_barriers,
    ):
        batch = lax.axis_index("batch")
        kv_head = lax.axis_index("kv_heads")
        wg_index = lax.axis_index("wg")
        kv_supertile = lax.axis_index("kv_supertiles")
        query_smem, gradient_smem = smem_buffers
        query_ready, gradient_ready = ready_barriers
        query_consumed, gradient_consumed = consumed_barriers
        query_start = kv_supertile * num_compute_wgs
        producer_query_count = num_query_tiles - query_start

        @pl.when(wg_index < num_compute_wgs)
        def compute_warpgroup():
            plgpu.set_max_registers(compute_registers, action="increase")
            kv_tile = kv_supertile * num_compute_wgs + wg_index
            kv_base = kv_tile * block_kv
            kv_slice = pl.ds(kv_base, block_kv)
            key_fragment = plgpu.load(
                key_ref.at[batch, kv_slice, kv_head],
                layout=plgpu.Layout.MMA_LHS(dtype),
                optimized=False,
            )
            value_fragment = plgpu.load(
                value_ref.at[batch, kv_slice, kv_head],
                layout=plgpu.Layout.MMA_LHS(dtype),
                optimized=False,
            )
            key_gradient = plgpu.layout_cast(
                jnp.zeros(
                    (block_kv, head_dim),
                    dtype=jnp.float32,
                ),
                accumulator_layout,
            )
            value_gradient = plgpu.layout_cast(
                jnp.zeros(
                    (block_kv, head_dim),
                    dtype=jnp.float32,
                ),
                accumulator_layout,
            )
            key_indices = plgpu.broadcasted_iota(
                jnp.int32,
                (block_kv, block_q),
                0,
                layout=accumulator_layout,
            )
            query_indices = plgpu.broadcasted_iota(
                jnp.int32,
                (block_kv, block_q),
                1,
                layout=accumulator_layout,
            )
            diagonal_mask = query_indices >= key_indices

            def process_query_tile(query_offset, gradients):
                key_gradient, value_gradient = gradients
                query_tile = query_start + query_offset
                query_base = query_tile * block_q
                query_slice = pl.ds(query_base, block_q)
                slot = lax.rem(
                    query_offset,
                    jnp.asarray(pipeline_depth, query_offset.dtype),
                )

                plgpu.barrier_wait(query_ready.at[slot])
                query_for_scores = load_shared_rhs_without_transpose(
                    query_smem.at[slot]
                )
                scores_transposed = plgpu.mma(
                    plgpu.layout_cast(
                        jnp.zeros(
                            (block_kv, block_q),
                            dtype=jnp.float32,
                        ),
                        accumulator_layout,
                    ),
                    key_fragment,
                    query_for_scores,
                )
                scores_transposed *= scale
                lse = plgpu.load(
                    lse_transposed_ref.at[
                        batch,
                        kv_head,
                        query_slice,
                    ],
                    layout=column_layout,
                    optimized=False,
                )
                lse_broadcast = plgpu.layout_cast(
                    lax.broadcast_in_dim(
                        lse,
                        scores_transposed.shape,
                        (1,),
                    ),
                    accumulator_layout,
                )
                probabilities_transposed = jnp.exp2(
                    (scores_transposed - lse_broadcast) * log2e
                )
                probabilities_transposed = lax.cond(
                    query_tile == kv_tile,
                    lambda matrix: jnp.where(
                        diagonal_mask,
                        matrix,
                        jnp.zeros_like(matrix),
                    ),
                    lambda matrix: matrix,
                    probabilities_transposed,
                )
                probabilities_transposed = lax.cond(
                    query_tile >= kv_tile,
                    lambda matrix: matrix,
                    jnp.zeros_like,
                    probabilities_transposed,
                )

                plgpu.barrier_wait(gradient_ready.at[slot])
                gradient_for_dp = load_shared_rhs_without_transpose(
                    gradient_smem.at[slot]
                )
                dp_transposed = plgpu.mma(
                    plgpu.layout_cast(
                        jnp.zeros(
                            (block_kv, block_q),
                            dtype=jnp.float32,
                        ),
                        accumulator_layout,
                    ),
                    value_fragment,
                    gradient_for_dp,
                )
                delta_tile = plgpu.load(
                    delta_transposed_ref.at[
                        batch,
                        kv_head,
                        query_slice,
                    ],
                    layout=column_layout,
                    optimized=False,
                )
                delta_broadcast = plgpu.layout_cast(
                    lax.broadcast_in_dim(
                        delta_tile,
                        dp_transposed.shape,
                        (1,),
                    ),
                    accumulator_layout,
                )
                ds_transposed = probabilities_transposed * (
                    dp_transposed - delta_broadcast
                )

                query_for_dk = load_shared_rhs(query_smem.at[slot])
                plgpu.barrier_arrive(query_consumed.at[slot])
                key_gradient = mma_with_accumulator_lhs(
                    key_gradient,
                    ds_transposed.astype(dtype),
                    query_for_dk,
                )
                gradient_for_dv = load_shared_rhs(gradient_smem.at[slot])
                plgpu.barrier_arrive(gradient_consumed.at[slot])
                value_gradient = mma_with_accumulator_lhs(
                    value_gradient,
                    probabilities_transposed.astype(dtype),
                    gradient_for_dv,
                )
                return key_gradient, value_gradient

            key_gradient, value_gradient = lax.fori_loop(
                0,
                producer_query_count,
                process_query_tile,
                (key_gradient, value_gradient),
            )
            key_gradient_ref[
                batch,
                kv_slice,
                kv_head,
            ] = (key_gradient * scale).astype(dtype)
            value_gradient_ref[
                batch,
                kv_slice,
                kv_head,
            ] = value_gradient.astype(dtype)

        @pl.when(wg_index == num_compute_wgs)
        def memory_warpgroup():
            plgpu.set_max_registers(producer_registers, action="decrease")
            for slot in range(pipeline_depth):
                query_tile = query_start + slot
                query_slice = pl.ds(query_tile * block_q, block_q)
                transposed_slice = (
                    batch,
                    kv_head,
                    slice(None),
                    query_slice,
                )
                plgpu.copy_gmem_to_smem(
                    query_transposed_ref.at[transposed_slice],
                    query_smem.at[slot],
                    query_ready.at[slot],
                )
                plgpu.copy_gmem_to_smem(
                    gradient_transposed_ref.at[transposed_slice],
                    gradient_smem.at[slot],
                    gradient_ready.at[slot],
                )

            @pl.loop(0, producer_query_count - pipeline_depth)
            def refill_pipeline(query_offset):
                slot = lax.rem(
                    query_offset,
                    jnp.asarray(pipeline_depth, query_offset.dtype),
                )
                next_query_tile = (
                    query_start + query_offset + pipeline_depth
                )
                next_query_slice = pl.ds(
                    next_query_tile * block_q,
                    block_q,
                )
                next_transposed_slice = (
                    batch,
                    kv_head,
                    slice(None),
                    next_query_slice,
                )
                plgpu.barrier_wait(query_consumed.at[slot])
                plgpu.copy_gmem_to_smem(
                    query_transposed_ref.at[next_transposed_slice],
                    query_smem.at[slot],
                    query_ready.at[slot],
                )
                plgpu.barrier_wait(gradient_consumed.at[slot])
                plgpu.copy_gmem_to_smem(
                    gradient_transposed_ref.at[next_transposed_slice],
                    gradient_smem.at[slot],
                    gradient_ready.at[slot],
                )

    query_scratch = plgpu.SMEM(
        (pipeline_depth, head_dim, block_q),
        dtype,
        transforms=rhs_smem_transforms,
    )
    gradient_scratch = plgpu.SMEM(
        (pipeline_depth, head_dim, block_q),
        dtype,
        transforms=rhs_smem_transforms,
    )
    key_gradient, value_gradient = plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct(key.shape, dtype),
            jax.ShapeDtypeStruct(value.shape, dtype),
        ),
        scratch_types=(
            (
                query_scratch,
                gradient_scratch,
            ),
            (
                plgpu.Barrier(num_barriers=pipeline_depth),
                plgpu.Barrier(num_barriers=pipeline_depth),
            ),
            (
                plgpu.Barrier(
                    num_arrivals=num_compute_wgs,
                    num_barriers=pipeline_depth,
                ),
                plgpu.Barrier(
                    num_arrivals=num_compute_wgs,
                    num_barriers=pipeline_depth,
                ),
            ),
        ),
        compiler_params=plgpu.CompilerParams(
            approx_math=False,
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
        grid=(kv_heads, num_kv_supertiles, batch_size),
        grid_names=("kv_heads", "kv_supertiles", "batch"),
        num_threads=num_compute_wgs + 1,
        thread_name="wg",
    )(
        query_transposed,
        key,
        value,
        gradient_transposed,
        log_normalizer_transposed,
        delta_transposed,
    )
    return key_gradient, value_gradient


def _mosaic_attention_backward_warp_specialized_causal_split(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute causal gradients with separate non-atomic dQ and dK/dV passes."""

    query_gradient, _, _ = _mosaic_attention_backward_warp_specialized(
        query,
        key,
        value,
        output,
        log_normalizer,
        upstream_gradient,
        is_causal=True,
        compute_kv_gradients=False,
    )
    key_gradient, value_gradient = (
        _mosaic_attention_backward_warp_specialized_causal_dkv(
            query,
            key,
            value,
            output,
            log_normalizer,
            upstream_gradient,
        )
    )
    return query_gradient, key_gradient, value_gradient


def _mosaic_attention_backward_warp_specialized_unmasked(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compatibility wrapper for the direct unmasked-kernel diagnostic."""
    return _mosaic_attention_backward_warp_specialized(
        query,
        key,
        value,
        output,
        log_normalizer,
        upstream_gradient,
        is_causal=False,
    )


def mosaic_attention_backward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output: jax.Array,
    log_normalizer: jax.Array,
    upstream_gradient: jax.Array,
    mask_fn: Callable[..., Any],
    *,
    is_causal: bool = False,
    backward_strategy: str = "auto",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the selected fast backward, or the lower-temporary two-pass variant.

    For large maximal-causal D=64 MHA, ``"auto"`` combines specialized
    query-major dQ and key-major dK/dV passes. Wide heads use the generic
    query-major/key-major split for causal, unmasked, and callable masks;
    complete dK/dV tiles are accumulated locally without global atomics.
    Other supported dense cases use a single-pass implementation with FP32
    atomic dK/dV accumulation. ``"minimal"`` always selects the generic split;
    ``"one_pass"`` explicitly selects the generic atomic implementation.
    """
    if backward_strategy == "minimal":
        return _mosaic_attention_backward_two_pass(
            query,
            key,
            value,
            output,
            log_normalizer,
            upstream_gradient,
            mask_fn,
            is_causal=is_causal,
        )
    if backward_strategy == "one_pass":
        return _mosaic_attention_backward_one_pass(
            query,
            key,
            value,
            output,
            log_normalizer,
            upstream_gradient,
            mask_fn,
            is_causal=is_causal,
        )
    if backward_strategy != "auto":
        raise ValueError(
            "backward_strategy must be 'auto', 'minimal', or 'one_pass'"
        )

    is_unmasked = _mask_is_always_true(mask_fn)
    use_warp_specialized = _can_use_warp_specialized_forward(
        query,
        key,
        is_unmasked=is_unmasked,
        is_causal=is_causal,
    )
    use_causal_split = _can_use_warp_specialized_causal_split_backward(
        query,
        key,
        is_unmasked=is_unmasked,
        is_causal=is_causal,
    )
    use_generic_split = _prefer_non_atomic_generic_backward(query)

    def run_selected(q, k, v, o, lse, do):
        if use_causal_split:
            return _mosaic_attention_backward_warp_specialized_causal_split(
                q,
                k,
                v,
                o,
                lse,
                do,
            )
        if use_generic_split:
            return _mosaic_attention_backward_two_pass(
                q,
                k,
                v,
                o,
                lse,
                do,
                mask_fn,
                is_causal=is_causal,
            )
        if use_warp_specialized:
            return _mosaic_attention_backward_warp_specialized(
                q,
                k,
                v,
                o,
                lse,
                do,
                is_causal=is_causal,
            )
        return _mosaic_attention_backward_one_pass(
            q,
            k,
            v,
            o,
            lse,
            do,
            mask_fn,
            is_causal=is_causal,
        )

    @jax.custom_batching.custom_vmap
    def run_one_pass(q, k, v, o, lse, do):
        return run_selected(q, k, v, o, lse, do)

    @run_one_pass.def_vmap
    def run_one_pass_vmap(axis_size, in_batched, *args):
        # run_state is not natively batchable, so preserve vmap semantics with
        # a compiled device loop over the mapped axis.
        def select_argument(argument, is_batched, index):
            return argument[index] if is_batched else argument

        def mapped_call(index):
            return run_selected(
                *(
                    select_argument(argument, batched, index)
                    for argument, batched in zip(
                        args,
                        in_batched,
                        strict=True,
                    )
                ),
            )

        outputs = lax.map(mapped_call, jnp.arange(axis_size))
        return outputs, (True, True, True)

    return run_one_pass(
        query,
        key,
        value,
        output,
        log_normalizer,
        upstream_gradient,
    )
