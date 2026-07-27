"""Focused tests for the Mosaic GPU attention fast path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmodules._mosaic_attention import (
    _can_use_warp_specialized_forward,
    _mosaic_attention_forward_warp_specialized_unmasked,
    mask_is_mosaic_compatible,
    mosaic_attention_backward,
    mosaic_attention_forward,
)
from jaxmodules.attention import (
    _masked_attention_via_map,
    _masked_attention_via_map_impl,
    _masked_attention_via_mosaic,
    default_kernel,
)


def _complex_mask(batch, head, query_index, key_index):
    radius = 24 + 4 * (head % 2) + 2 * batch
    return (abs(query_index - key_index) <= radius) & (
        (query_index + 2 * key_index + head) % 5 != 1
    )


def _partly_fully_masked(batch, head, query_index, key_index):
    del batch, head
    return (query_index % 7 != 0) & (key_index <= query_index)


def _causal_mask(batch, head, query_index, key_index):
    del batch, head
    return query_index >= key_index


def _causal_complex_mask(batch, head, query_index, key_index):
    return (query_index >= key_index) & _complex_mask(
        batch,
        head,
        query_index,
        key_index,
    )


def _unmasked(batch, head, query_index, key_index):
    del batch, head, query_index, key_index
    return True


def _unmasked_expression(batch, head, query_index, key_index):
    del batch, head, key_index
    return query_index == query_index


def test_mask_compatibility_is_conservative():
    assert mask_is_mosaic_compatible(_complex_mask)
    assert mask_is_mosaic_compatible(
        lambda batch, head, query, key: (query >= key)
        & ((batch + head + query + key) % 3 != 0)
    )
    assert not mask_is_mosaic_compatible(
        lambda batch, head, query, key: jnp.sin(query + key) > 0
    )


def test_warp_specialized_forward_selection_is_large_scale_only():
    large = jax.ShapeDtypeStruct((1, 4096, 2, 64), jnp.float16)
    small = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.float16)
    fp32 = jax.ShapeDtypeStruct((1, 4096, 2, 64), jnp.float32)

    assert _can_use_warp_specialized_forward(
        large,
        large,
        is_unmasked=True,
        is_causal=False,
    )
    assert not _can_use_warp_specialized_forward(
        small,
        small,
        is_unmasked=True,
        is_causal=False,
    )
    assert not _can_use_warp_specialized_forward(
        large,
        large,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_forward(
        fp32,
        fp32,
        is_unmasked=True,
        is_causal=False,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (jnp.float16, 2e-3),
        (jnp.bfloat16, 2e-2),
    ],
)
def test_warp_specialized_forward_matches_xla(dtype, tolerance):
    keys = jax.random.split(jax.random.key(5), 3)
    query = jax.random.normal(keys[0], (1, 1024, 2, 64), dtype=dtype)
    key = jax.random.normal(keys[1], (1, 1024, 1, 64), dtype=dtype)
    value = jax.random.normal(keys[2], (1, 1024, 1, 64), dtype=dtype)

    output, log_normalizer = jax.jit(
        _mosaic_attention_forward_warp_specialized_unmasked
    )(query, key, value)
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        implementation="xla",
    )

    assert output.dtype == dtype
    assert log_normalizer.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(log_normalizer)))
    np.testing.assert_allclose(
        np.asarray(output),
        np.asarray(expected),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize("mask_fn", [_complex_mask, _partly_fully_masked])
def test_mosaic_forward_matches_mapped_for_general_masks(mask_fn):
    keys = jax.random.split(jax.random.key(7), 3)
    query = jax.random.normal(
        keys[0],
        (2, 128, 4, 64),
        dtype=jnp.float16,
    )
    key = jax.random.normal(
        keys[1],
        (2, 128, 2, 64),
        dtype=jnp.float16,
    )
    value = jax.random.normal(
        keys[2],
        (2, 128, 2, 64),
        dtype=jnp.float16,
    )

    mosaic = jax.jit(
        lambda q, k, v: mosaic_attention_forward(q, k, v, mask_fn)
    )
    mapped = jax.jit(
        lambda q, k, v: _masked_attention_via_map_impl(
            q,
            k,
            v,
            mask_fn=mask_fn,
            block_size=64,
            kv_block_size=64,
        )
    )
    mosaic_output, mosaic_lse = mosaic(query, key, value)
    mapped_output, mapped_lse = mapped(query, key, value)

    assert mosaic_output.dtype == query.dtype
    np.testing.assert_allclose(
        np.asarray(mosaic_output),
        np.asarray(mapped_output),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(mosaic_lse),
        np.asarray(mapped_lse),
        rtol=2e-4,
        atol=2e-4,
    )
    if mask_fn is _partly_fully_masked:
        np.testing.assert_array_equal(
            np.asarray(mosaic_output[:, ::7]),
            np.zeros_like(np.asarray(mosaic_output[:, ::7])),
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
def test_mosaic_forward_composes_with_jit_and_vmap():
    keys = jax.random.split(jax.random.key(11), 3)
    shape = (2, 1, 64, 2, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)

    vmapped = jax.jit(
        jax.vmap(
            lambda q, k, v: mosaic_attention_forward(
                q,
                k,
                v,
                _complex_mask,
            )[0]
        )
    )
    output = vmapped(query, key, value)
    expected = jnp.stack(
        [
            mosaic_attention_forward(query[i], key[i], value[i], _complex_mask)[
                0
            ]
            for i in range(query.shape[0])
        ]
    )
    np.testing.assert_allclose(
        np.asarray(output),
        np.asarray(expected),
        rtol=2e-3,
        atol=2e-3,
    )

    def loss(q, k, v):
        result = _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=_complex_mask,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=False,
            backward_strategy="auto",
        )
        return jnp.mean(result**2)

    vmapped_gradients = jax.jit(
        jax.vmap(jax.grad(loss, argnums=(0, 1, 2)))
    )(query, key, value)
    expected_gradients = tuple(
        jnp.stack(
            [
                jax.grad(loss, argnums=(0, 1, 2))(
                    query[i],
                    key[i],
                    value[i],
                )[gradient_index]
                for i in range(query.shape[0])
            ]
        )
        for gradient_index in range(3)
    )
    for vmapped_gradient, expected_gradient in zip(
        vmapped_gradients,
        expected_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(vmapped_gradient),
            np.asarray(expected_gradient),
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (jnp.float16, 3e-3),
        (jnp.bfloat16, 2e-2),
    ],
)
def test_mosaic_custom_vjp_matches_existing_tiled_backward(dtype, tolerance):
    keys = jax.random.split(jax.random.key(19), 4)
    query_shape = (1, 128, 4, 64)
    kv_shape = (1, 128, 2, 64)
    query = jax.random.normal(keys[0], query_shape, dtype=dtype)
    key = jax.random.normal(keys[1], kv_shape, dtype=dtype)
    value = jax.random.normal(keys[2], kv_shape, dtype=dtype)
    cotangent = jax.random.normal(
        keys[3],
        query_shape,
        dtype=jnp.float32,
    )

    def mosaic_loss(q, k, v):
        output = _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=_complex_mask,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=False,
            backward_strategy="auto",
        )
        return jnp.sum(output * cotangent)

    def mapped_loss(q, k, v):
        output = _masked_attention_via_map(
            q,
            k,
            v,
            kernel_fn=default_kernel,
            mask_fn=_complex_mask,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=False,
            backward_strategy="auto",
        )
        return jnp.sum(output * cotangent)

    mosaic_value, mosaic_gradients = jax.jit(
        jax.value_and_grad(mosaic_loss, argnums=(0, 1, 2))
    )(query, key, value)
    mapped_value, mapped_gradients = jax.jit(
        jax.value_and_grad(mapped_loss, argnums=(0, 1, 2))
    )(query, key, value)

    np.testing.assert_allclose(
        np.asarray(mosaic_value),
        np.asarray(mapped_value),
        rtol=tolerance,
        atol=tolerance,
    )
    for mosaic_gradient, mapped_gradient in zip(
        mosaic_gradients,
        mapped_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(mosaic_gradient),
            np.asarray(mapped_gradient),
            rtol=tolerance,
            atol=tolerance,
        )

    mosaic_output, mosaic_lse = mosaic_attention_forward(
        query,
        key,
        value,
        _complex_mask,
    )
    direct_gradients = jax.jit(
        lambda q, k, v, output, lse, gradient: mosaic_attention_backward(
            q,
            k,
            v,
            output,
            lse,
            gradient,
            _complex_mask,
            backward_strategy="minimal",
        )
    )(query, key, value, mosaic_output, mosaic_lse, cotangent)
    for direct_gradient, mapped_gradient in zip(
        direct_gradients,
        mapped_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(direct_gradient),
            np.asarray(mapped_gradient),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
def test_mosaic_custom_vjp_composes_with_jit_and_vmap():
    keys = jax.random.split(jax.random.key(29), 4)
    shape = (2, 1, 64, 2, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    def loss(attention_fn, q, k, v, gradient):
        output = attention_fn(
            q,
            k,
            v,
            mask_fn=_complex_mask,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=False,
            backward_strategy="auto",
        )
        return jnp.sum(output * gradient)

    mosaic_grad = jax.jit(
        jax.vmap(
            jax.grad(
                lambda q, k, v, gradient: loss(
                    _masked_attention_via_mosaic,
                    q,
                    k,
                    v,
                    gradient,
                ),
                argnums=(0, 1, 2),
            )
        )
    )
    mapped_grad = jax.jit(
        jax.vmap(
            jax.grad(
                lambda q, k, v, gradient: loss(
                    _masked_attention_via_map,
                    q,
                    k,
                    v,
                    gradient,
                ),
                argnums=(0, 1, 2),
            )
        )
    )

    mosaic_gradients = mosaic_grad(query, key, value, cotangent)
    mapped_gradients = mapped_grad(query, key, value, cotangent)
    for mosaic_gradient, mapped_gradient in zip(
        mosaic_gradients,
        mapped_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(mosaic_gradient),
            np.asarray(mapped_gradient),
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
def test_unmasked_specialization_matches_the_general_engine():
    keys = jax.random.split(jax.random.key(31), 4)
    shape = (1, 128, 2, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    def loss(q, k, v, mask_fn):
        result = _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=mask_fn,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=False,
            backward_strategy="auto",
        )
        return jnp.sum(result * cotangent)

    specialized_value, specialized_gradients = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(q, k, v, _unmasked),
            argnums=(0, 1, 2),
        )
    )(query, key, value)
    general_value, general_gradients = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(q, k, v, _unmasked_expression),
            argnums=(0, 1, 2),
        )
    )(query, key, value)

    np.testing.assert_allclose(
        np.asarray(specialized_value),
        np.asarray(general_value),
        rtol=2e-3,
        atol=2e-3,
    )
    for specialized_gradient, general_gradient in zip(
        specialized_gradients,
        general_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(specialized_gradient),
            np.asarray(general_gradient),
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize(
    ("hinted_mask", "explicit_mask"),
    [
        (_unmasked, _causal_mask),
        (_complex_mask, _causal_complex_mask),
    ],
)
def test_causal_block_pruning_matches_the_general_engine(
    hinted_mask,
    explicit_mask,
):
    keys = jax.random.split(jax.random.key(23), 4)
    shape = (1, 128, 2, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    def loss(q, k, v, *, prune):
        result = _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=hinted_mask if prune else explicit_mask,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=prune,
            backward_strategy="auto",
        )
        return jnp.sum(result * cotangent)

    pruned_value, pruned_gradients = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(q, k, v, prune=True),
            argnums=(0, 1, 2),
        )
    )(query, key, value)
    general_value, general_gradients = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(q, k, v, prune=False),
            argnums=(0, 1, 2),
        )
    )(query, key, value)

    np.testing.assert_allclose(
        np.asarray(pruned_value),
        np.asarray(general_value),
        rtol=2e-3,
        atol=2e-3,
    )
    for pruned_gradient, general_gradient in zip(
        pruned_gradients,
        general_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(pruned_gradient),
            np.asarray(general_gradient),
            rtol=3e-3,
            atol=3e-3,
        )
