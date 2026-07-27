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
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir.dialects import memref


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
    """Choose a conservative initial configuration for supported shapes."""

    if query.shape[1] % 64 or key.shape[1] % 64:
        return None
    return MosaicAttentionConfig()


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

    return (
        is_unmasked
        and not is_causal
        and query.dtype
        in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))
        and query.shape[1] >= 4096
        and key.shape[1] >= 4096
        and query.shape[1] % 128 == 0
        and key.shape[1] % 64 == 0
        and query.shape[3] == 64
        and key.shape[3] == 64
    )


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
        return _mosaic_attention_forward_warp_specialized_unmasked(
            query,
            key,
            value,
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
                layout=plgpu.Layout.MMA_RHS(dtype),
                optimized=False,
            )

            # Match the input precision used by standard FlashAttention:
            # softmax and accumulation remain FP32, while the tensor-core PV
            # operand uses the input dtype.
            probability_smem[...] = probabilities.astype(dtype)
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
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
        grid=(query_heads, num_query_tiles, batch_size),
        grid_names=("heads", "query_tiles", "batch"),
    )(query, key, value_transposed)
    return output, jnp.swapaxes(lse, 1, 2)


def _mosaic_attention_forward_warp_specialized_unmasked(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute large unmasked attention with shared K/V producer staging.

    Two compute warpgroups process adjacent query tiles. A third warpgroup
    pipelines each K/V tile through shared memory once for both consumers.
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
        raise ValueError(
            f"query length must be divisible by {query_superblock}"
        )
    if kv_length % block_kv:
        raise ValueError(f"key/value length must be divisible by {block_kv}")

    query_heads_per_kv_head = query_heads // kv_heads
    num_query_supertiles = query_length // query_superblock
    num_kv_tiles = kv_length // block_kv
    max_concurrent_steps = min(2, num_kv_tiles)
    dtype = query.dtype
    scale = float(head_dim**-0.5)
    log2e = math.log2(math.e)
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
        # ``mma.sync`` expects pairs in each RHS register, but those pairs are
        # not contiguous in the TMA-compatible shared-memory swizzle. Load the
        # same lane ownership one scalar at a time, then pack each local pair.
        native_layout = plgpu.Layout.MMA_RHS(dtype).to_mgpu()
        scalar_rhs_layout = mgpu.TiledLayout(
            native_layout.tiling,
            warp_dims=native_layout.warp_dims,
            lane_dims=native_layout.lane_dims,
            vector_dim=-1,
        )
        scalar_fragment = mgpu.FragmentedArray.load_tiled(
            smem_ref,
            swizzle=swizzle,
            layout=scalar_rhs_layout,
            optimized=False,
            tiling_rank=2,
        )
        native_registers = np.empty(
            native_layout.registers_shape((block_kv, head_dim)),
            dtype=object,
        )
        for native_index in np.ndindex(native_registers.shape):
            scalar_prefix = native_index[:6]
            scalar_suffix = native_index[7]
            native_registers[native_index] = mgpu.utils.vector_concat(
                [
                    scalar_fragment.registers[
                        (*scalar_prefix, component, scalar_suffix)
                    ]
                    for component in range(2)
                ]
            )
        return mgpu.FragmentedArray(
            _registers=native_registers,
            _layout=native_layout,
            _is_signed=scalar_fragment.is_signed,
        )

    def kernel(
        query_ref,
        key_transposed_ref,
        value_ref,
        output_ref,
        lse_ref,
        smem_buffers,
        ready_barriers,
        consumed_barriers,
    ):
        batch = lax.axis_index("batch")
        query_head = lax.axis_index("heads")
        wg_index = lax.axis_index("wg")
        (
            key_smem,
            value_smem,
            probability_smem,
        ) = smem_buffers
        key_barriers, value_barriers = ready_barriers
        key_consumed_barriers, value_consumed_barriers = consumed_barriers
        kv_head = lax.div(
            query_head,
            jnp.asarray(query_heads_per_kv_head, query_head.dtype),
        )

        @pl.when(wg_index < num_compute_wgs)
        def compute_warpgroup():
            plgpu.set_max_registers(compute_registers, action="increase")
            query_base = (
                lax.axis_index("query_supertiles") * query_superblock
                + wg_index * block_q
            )
            wg_probability_smem = probability_smem.at[wg_index]
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

            def process_kv_tile(kv_step, carry):
                output_accumulator, row_max, row_sum = carry
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
                row_max = next_row_max
                row_sum += probabilities.sum(axis=1)

                plgpu.barrier_wait(value_barriers.at[slot])
                value_fragment = load_shared_rhs(value_smem.at[slot])
                plgpu.barrier_arrive(value_consumed_barriers.at[slot])
                wg_probability_smem[...] = probabilities.astype(dtype)
                probability_fragment = plgpu.load(
                    wg_probability_smem,
                    optimized=False,
                )
                output_accumulator = plgpu.mma(
                    output_accumulator,
                    probability_fragment,
                    value_fragment,
                )
                return output_accumulator, row_max, row_sum

            output_accumulator, row_max, row_sum = lax.fori_loop(
                0,
                num_kv_tiles,
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
                key_slice = (
                    batch,
                    kv_head,
                    slice(None),
                    pl.ds(kv_step * block_kv, block_kv),
                )
                plgpu.copy_gmem_to_smem(
                    key_transposed_ref.at[key_slice],
                    key_smem.at[kv_step],
                    key_barriers.at[kv_step],
                )
                plgpu.copy_gmem_to_smem(
                    value_ref.at[kv_slice],
                    value_smem.at[kv_step],
                    value_barriers.at[kv_step],
                )

            @pl.loop(0, num_kv_tiles - max_concurrent_steps)
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
                next_key_slice = (
                    batch,
                    kv_head,
                    slice(None),
                    pl.ds(next_kv_step * block_kv, block_kv),
                )
                plgpu.copy_gmem_to_smem(
                    key_transposed_ref.at[next_key_slice],
                    key_smem.at[slot],
                    key_barriers.at[slot],
                )
                plgpu.barrier_wait(value_consumed_barriers.at[slot])
                plgpu.copy_gmem_to_smem(
                    value_ref.at[next_kv_slice],
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
    probability_scratch = plgpu.SMEM(
        (num_compute_wgs, block_q, block_kv),
        dtype,
        transforms=lhs_smem_transforms,
    )
    output_type = jax.ShapeDtypeStruct(query.shape, dtype)
    lse_type = jax.ShapeDtypeStruct(
        (batch_size, query_heads, query_length),
        jnp.float32,
    )
    key_transposed = jnp.transpose(key, (0, 2, 3, 1))
    output, lse = plgpu.kernel(
        kernel,
        out_type=(output_type, lse_type),
        scratch_types=(
            (
                key_scratch,
                value_scratch,
                probability_scratch,
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
    )(query, key_transposed, value)
    return output, jnp.swapaxes(lse, 1, 2)


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
    accumulator_layout = plgpu.Layout.MMA_ACC(dtype)
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

    swizzle = 128
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
                    layout=plgpu.Layout.MMA_RHS(dtype),
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
                    layout=plgpu.Layout.MMA_RHS(dtype),
                    optimized=False,
                )
                matrix_smem[...] = ds.astype(dtype)
                ds_fragment = plgpu.load(
                    matrix_smem,
                    layout=plgpu.Layout.MMA_LHS(dtype),
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
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
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
            layout=plgpu.Layout.MMA_LHS(dtype),
            optimized=False,
        )
        value_fragment = plgpu.load(
            value_ref.at[batch, kv_slice, kv_head],
            layout=plgpu.Layout.MMA_LHS(dtype),
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
                        layout=plgpu.Layout.MMA_RHS(dtype),
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
                        layout=plgpu.Layout.MMA_RHS(dtype),
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
                        layout=plgpu.Layout.MMA_RHS(dtype),
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
                        layout=plgpu.Layout.MMA_RHS(dtype),
                        optimized=False,
                    )
                    matrix_smem[...] = ds_transposed.astype(dtype)
                    ds_fragment = plgpu.load(
                        matrix_smem,
                        layout=plgpu.Layout.MMA_LHS(dtype),
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
                        layout=plgpu.Layout.MMA_LHS(dtype),
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
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
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
    accumulator_layout = plgpu.Layout.MMA_ACC(dtype)
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
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
    )

    # This is specific to the 64x64, four-warp MMA layouts selected above.
    # After the FP32 accumulator is cast to the input dtype, MMA_ACC and
    # MMA_LHS assign every element to the same lane and register in the same
    # order. Rewrapping the registers therefore avoids a shared-memory reload;
    # changing the tile or warp layout requires revalidating that invariant.
    assert block_q == block_kv == 64

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
        arg_types=(plgpu.RefType(),),
        return_type=plgpu.ShapeDtypeStruct(
            (block_kv, block_q),
            dtype,
            layout=plgpu.Layout.MMA_LHS(dtype),
        ),
    )
    def load_transposed_scratch(_, scratch_ref):
        return mgpu.FragmentedArray.build(
            (block_kv, block_q),
            plgpu.Layout.MMA_LHS(dtype).to_mgpu(),
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
                layout=plgpu.Layout.MMA_LHS(dtype),
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
                layout=plgpu.Layout.MMA_RHS(dtype),
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
                layout=plgpu.Layout.MMA_RHS(dtype),
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
                    layout=plgpu.Layout.MMA_RHS(dtype),
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
                    layout=plgpu.Layout.MMA_RHS(dtype),
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
    """Run the one-pass backward, or the lower-temporary two-pass variant.

    ``"auto"`` traverses every active score tile once and combines query-major
    programs with FP32 atomic dK/dV accumulation. ``"minimal"`` retains the
    two-pass implementation for callers that prefer its smaller compiler
    temporary estimate.
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
    if backward_strategy != "auto":
        raise ValueError("backward_strategy must be 'auto' or 'minimal'")

    @jax.custom_batching.custom_vmap
    def run_one_pass(q, k, v, o, lse, do):
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

    @run_one_pass.def_vmap
    def run_one_pass_vmap(axis_size, in_batched, *args):
        # run_state is not natively batchable, so preserve vmap semantics with
        # a compiled device loop over the mapped axis.
        def select_argument(argument, is_batched, index):
            return argument[index] if is_batched else argument

        def mapped_call(index):
            return _mosaic_attention_backward_one_pass(
                *(
                    select_argument(argument, batched, index)
                    for argument, batched in zip(
                        args,
                        in_batched,
                        strict=True,
                    )
                ),
                mask_fn,
                is_causal=is_causal,
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
