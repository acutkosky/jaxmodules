"""Focused tests for the Mosaic GPU attention fast path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmodules._mosaic_attention import (
    mask_is_mosaic_compatible,
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


def test_mask_compatibility_is_conservative():
    assert mask_is_mosaic_compatible(_complex_mask)
    assert mask_is_mosaic_compatible(
        lambda batch, head, query, key: (query >= key)
        & ((batch + head + query + key) % 3 != 0)
    )
    assert not mask_is_mosaic_compatible(
        lambda batch, head, query, key: jnp.sin(query + key) > 0
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

    assert mosaic_output.dtype == jnp.float32
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


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
def test_mosaic_custom_vjp_matches_existing_tiled_backward():
    keys = jax.random.split(jax.random.key(19), 4)
    shape = (1, 128, 2, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

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
        rtol=2e-3,
        atol=2e-3,
    )
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
