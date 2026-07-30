"""Focused tests for the Mosaic GPU attention fast path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmodules._mosaic_attention import (
    _can_use_warp_specialized_causal_split_backward,
    _can_use_warp_specialized_forward,
    _mosaic_attention_backward_warp_specialized,
    _mosaic_attention_backward_warp_specialized_causal_split,
    _mosaic_attention_backward_warp_specialized_dkv,
    _mosaic_attention_backward_warp_specialized_dkv_split,
    _mosaic_attention_backward_warp_specialized_dq,
    _mosaic_attention_forward_warp_specialized,
    _prefer_non_atomic_generic_backward,
    mask_is_mosaic_compatible,
    mosaic_attention_backward,
    mosaic_attention_forward,
    supports_mosaic_attention,
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


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [
        (48, False),
        (64, True),
        (80, True),
        (96, True),
        (128, True),
        (192, True),
        (256, True),
        (512, True),
        (1024, True),
        (2048, True),
        (2064, False),
    ],
)
def test_mosaic_supports_flexible_head_dimensions(head_dim, expected):
    query = jax.ShapeDtypeStruct((1, 128, 2, head_dim), jnp.float16)
    key_value = jax.ShapeDtypeStruct((1, 128, 1, head_dim), jnp.float16)

    assert (
        supports_mosaic_attention(query, key_value, key_value, _unmasked)
        is expected
    )


def test_nondefault_head_dimensions_select_non_atomic_generic_backward():
    default_width = jax.ShapeDtypeStruct((1, 4096, 8, 64), jnp.float16)
    wider = jax.ShapeDtypeStruct((1, 4096, 8, 80), jnp.float16)

    assert not _prefer_non_atomic_generic_backward(default_width)
    assert _prefer_non_atomic_generic_backward(wider)


def test_warp_specialized_forward_selection_respects_supported_shapes():
    large = jax.ShapeDtypeStruct((1, 4096, 2, 64), jnp.float16)
    small = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.float16)
    subkilotoken = jax.ShapeDtypeStruct((1, 512, 2, 64), jnp.float16)
    fp32 = jax.ShapeDtypeStruct((1, 4096, 2, 64), jnp.float32)
    rectangular = jax.ShapeDtypeStruct((1, 8192, 2, 64), jnp.float16)
    d128 = jax.ShapeDtypeStruct((1, 4096, 2, 128), jnp.float16)
    d192 = jax.ShapeDtypeStruct((1, 4096, 2, 192), jnp.float16)
    d208 = jax.ShapeDtypeStruct((1, 4096, 2, 208), jnp.float16)
    d240 = jax.ShapeDtypeStruct((1, 4096, 2, 240), jnp.float16)
    d256 = jax.ShapeDtypeStruct((1, 4096, 2, 256), jnp.float16)

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
    assert _can_use_warp_specialized_forward(
        large,
        large,
        is_unmasked=True,
        is_causal=True,
    )
    assert _can_use_warp_specialized_forward(
        small,
        small,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_forward(
        subkilotoken,
        subkilotoken,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_forward(
        large,
        rectangular,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_forward(
        fp32,
        fp32,
        is_unmasked=True,
        is_causal=False,
    )
    for shape in (d128, d192):
        assert _can_use_warp_specialized_forward(
            shape,
            shape,
            is_unmasked=True,
            is_causal=False,
        )
        assert _can_use_warp_specialized_forward(
            shape,
            shape,
            is_unmasked=True,
            is_causal=True,
        )
    for shape in (d208, d240):
        assert _can_use_warp_specialized_forward(
            shape,
            shape,
            is_unmasked=True,
            is_causal=False,
        )
        assert not _can_use_warp_specialized_forward(
            shape,
            shape,
            is_unmasked=True,
            is_causal=True,
        )
    assert not _can_use_warp_specialized_forward(
        d256,
        d256,
        is_unmasked=True,
        is_causal=False,
    )


def test_causal_split_backward_selection_respects_supported_mha_cases():
    large_mha = jax.ShapeDtypeStruct((1, 4096, 2, 64), jnp.float16)
    small_mha = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.float16)
    subkilotoken_mha = jax.ShapeDtypeStruct((1, 512, 2, 64), jnp.float16)
    large_gqa_key = jax.ShapeDtypeStruct((1, 4096, 1, 64), jnp.float16)

    assert _can_use_warp_specialized_causal_split_backward(
        large_mha,
        large_mha,
        is_unmasked=True,
        is_causal=True,
    )
    assert _can_use_warp_specialized_causal_split_backward(
        small_mha,
        small_mha,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_causal_split_backward(
        subkilotoken_mha,
        subkilotoken_mha,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_causal_split_backward(
        large_mha,
        large_gqa_key,
        is_unmasked=True,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_causal_split_backward(
        large_mha,
        large_mha,
        is_unmasked=False,
        is_causal=True,
    )
    assert not _can_use_warp_specialized_causal_split_backward(
        large_mha,
        large_mha,
        is_unmasked=True,
        is_causal=False,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize(
    ("dtype", "tolerance", "is_causal", "head_dim"),
    [
        (jnp.float16, 2e-3, False, 64),
        (jnp.float16, 2e-3, True, 64),
        (jnp.bfloat16, 2e-2, False, 64),
        (jnp.bfloat16, 2e-2, True, 64),
        (jnp.float16, 2e-3, False, 128),
        (jnp.float16, 2e-3, True, 128),
        (jnp.bfloat16, 2e-2, False, 128),
        (jnp.bfloat16, 2e-2, True, 128),
        (jnp.float16, 2e-3, False, 80),
        (jnp.float16, 2e-3, True, 80),
        (jnp.float16, 2e-3, False, 192),
        (jnp.float16, 2e-3, True, 192),
        (jnp.float16, 2e-3, False, 240),
    ],
)
def test_warp_specialized_forward_matches_xla(
    dtype,
    tolerance,
    is_causal,
    head_dim,
):
    keys = jax.random.split(jax.random.key(5), 3)
    query = jax.random.normal(
        keys[0],
        (1, 1024, 2, head_dim),
        dtype=dtype,
    )
    key = jax.random.normal(
        keys[1],
        (1, 1024, 1, head_dim),
        dtype=dtype,
    )
    value = jax.random.normal(
        keys[2],
        (1, 1024, 1, head_dim),
        dtype=dtype,
    )

    output, log_normalizer = jax.jit(
        lambda q, k, v: _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    )(query, key, value)
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=is_causal,
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
def test_warp_specialized_causal_forward_composes_with_vmap():
    keys = jax.random.split(jax.random.key(59), 3)
    shape = (2, 1, 128, 1, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)

    def causal_attention(q, k, v):
        return _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=True,
        )[0]

    output = jax.jit(jax.vmap(causal_attention))(query, key, value)
    expected = jax.vmap(
        lambda q, k, v: jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            implementation="xla",
        )
    )(query, key, value)

    np.testing.assert_allclose(
        np.asarray(output),
        np.asarray(expected),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize(
    ("sequence_length", "query_heads", "kv_heads", "head_dim"),
    [
        (4096, 1, 1, 64),
        (1024, 2, 1, 64),
        (1024, 2, 2, 128),
    ],
)
def test_warp_specialized_causal_custom_vjp_matches_xla(
    sequence_length,
    query_heads,
    kv_heads,
    head_dim,
):
    keys = jax.random.split(jax.random.key(53), 4)
    query_shape = (1, sequence_length, query_heads, head_dim)
    kv_shape = (1, sequence_length, kv_heads, head_dim)
    query = jax.random.normal(keys[0], query_shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], kv_shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], kv_shape, dtype=jnp.float16)
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
            mask_fn=_unmasked,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=True,
            backward_strategy="auto",
        )
        return jnp.sum(output.astype(jnp.float32) * cotangent)

    def xla_loss(q, k, v):
        output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            implementation="xla",
        )
        return jnp.sum(output.astype(jnp.float32) * cotangent)

    mosaic_value, mosaic_gradients = jax.jit(
        jax.value_and_grad(mosaic_loss, argnums=(0, 1, 2))
    )(query, key, value)
    xla_value, xla_gradients = jax.jit(
        jax.value_and_grad(xla_loss, argnums=(0, 1, 2))
    )(query, key, value)

    np.testing.assert_allclose(
        np.asarray(mosaic_value),
        np.asarray(xla_value),
        rtol=3e-3,
        atol=3e-3,
    )
    for mosaic_gradient, xla_gradient in zip(
        mosaic_gradients,
        xla_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(mosaic_gradient),
            np.asarray(xla_gradient),
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
def test_warp_specialized_causal_split_backward_composes_with_vmap():
    keys = jax.random.split(jax.random.key(61), 4)
    shape = (2, 1, 4096, 1, 64)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    def mosaic_loss(q, k, v, gradient):
        output = _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=_unmasked,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=True,
            backward_strategy="auto",
        )
        return jnp.sum(output.astype(jnp.float32) * gradient)

    def xla_loss(q, k, v, gradient):
        output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            implementation="xla",
        )
        return jnp.sum(output.astype(jnp.float32) * gradient)

    mosaic_gradients = jax.jit(
        jax.vmap(jax.grad(mosaic_loss, argnums=(0, 1, 2)))
    )(query, key, value, cotangent)
    xla_gradients = jax.jit(
        jax.vmap(jax.grad(xla_loss, argnums=(0, 1, 2)))
    )(query, key, value, cotangent)

    for mosaic_gradient, xla_gradient in zip(
        mosaic_gradients,
        xla_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(mosaic_gradient),
            np.asarray(xla_gradient),
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize(
    ("dtype", "tolerance", "is_causal"),
    [
        (jnp.float16, 3e-3, False),
        (jnp.float16, 3e-3, True),
        (jnp.bfloat16, 2e-2, False),
        (jnp.bfloat16, 2e-2, True),
    ],
)
def test_warp_specialized_backward_matches_xla(
    dtype,
    tolerance,
    is_causal,
):
    keys = jax.random.split(jax.random.key(47), 4)
    query = jax.random.normal(keys[0], (1, 1024, 2, 64), dtype=dtype)
    kv_heads = 2 if is_causal else 1
    key = jax.random.normal(
        keys[1],
        (1, 1024, kv_heads, 64),
        dtype=dtype,
    )
    value = jax.random.normal(
        keys[2],
        (1, 1024, kv_heads, 64),
        dtype=dtype,
    )
    cotangent = jax.random.normal(
        keys[3],
        query.shape,
        dtype=jnp.float32,
    )

    output, log_normalizer = jax.jit(
        lambda q, k, v: _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    )(query, key, value)
    gradients = jax.jit(
        lambda q, k, v, output, lse, gradient: (
            _mosaic_attention_backward_warp_specialized_causal_split(
                q,
                k,
                v,
                output,
                lse,
                gradient,
            )
            if is_causal
            else _mosaic_attention_backward_warp_specialized(
                q,
                k,
                v,
                output,
                lse,
                gradient,
                is_causal=False,
            )
        )
    )(query, key, value, output, log_normalizer, cotangent)

    def xla_loss(q, k, v):
        attention = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            implementation="xla",
        )
        return jnp.sum(attention.astype(jnp.float32) * cotangent)

    expected = jax.jit(
        jax.grad(xla_loss, argnums=(0, 1, 2))
    )(query, key, value)
    for gradient, expected_gradient in zip(
        gradients,
        expected,
        strict=True,
    ):
        assert gradient.dtype == dtype
        np.testing.assert_allclose(
            np.asarray(gradient),
            np.asarray(expected_gradient),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize("head_dim", [80, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(jnp.float16, 3e-3), (jnp.bfloat16, 2e-2)],
)
def test_generated_warp_specialized_dq_matches_xla(
    head_dim,
    is_causal,
    dtype,
    tolerance,
):
    keys = jax.random.split(jax.random.key(71), 4)
    shape = (1, 1024, 2, head_dim)
    query = jax.random.normal(keys[0], shape, dtype=dtype)
    key = jax.random.normal(keys[1], shape, dtype=dtype)
    value = jax.random.normal(keys[2], shape, dtype=dtype)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    output, log_normalizer = jax.jit(
        lambda q, k, v: _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    )(query, key, value)
    query_gradient = jax.jit(
        lambda q, k, v, o, lse, do: (
            _mosaic_attention_backward_warp_specialized_dq(
                q,
                k,
                v,
                o,
                lse,
                do,
                is_causal=is_causal,
            )
        )
    )(query, key, value, output, log_normalizer, cotangent)

    def xla_loss(q):
        attention = jax.nn.dot_product_attention(
            q,
            key,
            value,
            is_causal=is_causal,
            implementation="xla",
        )
        return jnp.sum(attention.astype(jnp.float32) * cotangent)

    expected = jax.jit(jax.grad(xla_loss))(query)
    np.testing.assert_allclose(
        np.asarray(query_gradient),
        np.asarray(expected),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize("head_dim", [80, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(jnp.float16, 3e-3), (jnp.bfloat16, 2e-2)],
)
def test_generated_warp_specialized_dkv_matches_xla(
    head_dim,
    is_causal,
    dtype,
    tolerance,
):
    keys = jax.random.split(jax.random.key(73), 4)
    shape = (1, 1024, 2, head_dim)
    query = jax.random.normal(keys[0], shape, dtype=dtype)
    key = jax.random.normal(keys[1], shape, dtype=dtype)
    value = jax.random.normal(keys[2], shape, dtype=dtype)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    output, log_normalizer = jax.jit(
        lambda q, k, v: _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    )(query, key, value)
    gradients = jax.jit(
        lambda q, k, v, o, lse, do: (
            _mosaic_attention_backward_warp_specialized_dkv(
                q,
                k,
                v,
                o,
                lse,
                do,
                is_causal=is_causal,
            )
        )
    )(query, key, value, output, log_normalizer, cotangent)

    def xla_loss(k, v):
        attention = jax.nn.dot_product_attention(
            query,
            k,
            v,
            is_causal=is_causal,
            implementation="xla",
        )
        return jnp.sum(attention.astype(jnp.float32) * cotangent)

    expected = jax.jit(jax.grad(xla_loss, argnums=(0, 1)))(key, value)
    for gradient, expected_gradient in zip(
        gradients,
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(gradient),
            np.asarray(expected_gradient),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize("is_causal", [False, True])
def test_split_warp_specialized_dkv_matches_xla(is_causal):
    keys = jax.random.split(jax.random.key(79), 4)
    shape = (1, 1024, 2, 128)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    output, log_normalizer = jax.jit(
        lambda q, k, v: _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    )(query, key, value)
    gradients = jax.jit(
        lambda q, k, v, o, lse, do: (
            _mosaic_attention_backward_warp_specialized_dkv_split(
                q,
                k,
                v,
                o,
                lse,
                do,
                is_causal=is_causal,
            )
        )
    )(query, key, value, output, log_normalizer, cotangent)

    def xla_loss(k, v):
        attention = jax.nn.dot_product_attention(
            query,
            k,
            v,
            is_causal=is_causal,
            implementation="xla",
        )
        return jnp.sum(attention.astype(jnp.float32) * cotangent)

    expected = jax.jit(jax.grad(xla_loss, argnums=(0, 1)))(key, value)
    for gradient, expected_gradient in zip(
        gradients,
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(gradient),
            np.asarray(expected_gradient),
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires Mosaic GPU")
@pytest.mark.parametrize("mask_fn", [_complex_mask, _partly_fully_masked])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_mosaic_forward_matches_mapped_for_general_masks(mask_fn, head_dim):
    keys = jax.random.split(jax.random.key(7), 3)
    query = jax.random.normal(
        keys[0],
        (2, 128, 4, head_dim),
        dtype=jnp.float16,
    )
    key = jax.random.normal(
        keys[1],
        (2, 128, 2, head_dim),
        dtype=jnp.float16,
    )
    value = jax.random.normal(
        keys[2],
        (2, 128, 2, head_dim),
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
    ("dtype", "tolerance", "head_dim"),
    [
        (jnp.float16, 3e-3, 64),
        (jnp.float16, 3e-3, 128),
        (jnp.bfloat16, 2e-2, 64),
    ],
)
def test_mosaic_custom_vjp_matches_existing_tiled_backward(
    dtype,
    tolerance,
    head_dim,
):
    keys = jax.random.split(jax.random.key(19), 4)
    query_shape = (1, 128, 4, head_dim)
    kv_shape = (1, 128, 2, head_dim)
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
@pytest.mark.parametrize(
    ("head_dim", "mask_fn", "is_causal", "backward_strategy"),
    [
        (80, _complex_mask, False, "auto"),
        (96, _unmasked, True, "auto"),
        (128, _complex_mask, False, "one_pass"),
        (192, _complex_mask, False, "auto"),
        (256, _unmasked, False, "auto"),
        (512, _complex_mask, False, "auto"),
        (1024, _unmasked, False, "auto"),
        (2048, _unmasked, True, "auto"),
    ],
)
def test_wide_head_custom_vjp_matches_xla(
    head_dim,
    mask_fn,
    is_causal,
    backward_strategy,
):
    keys = jax.random.split(jax.random.key(67 + head_dim), 4)
    shape = (1, 64, 1, head_dim)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    value = jax.random.normal(keys[2], shape, dtype=jnp.float16)
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    def mosaic_loss(q, k, v):
        result = _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=mask_fn,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=is_causal,
            backward_strategy=backward_strategy,
        )
        return jnp.sum(result.astype(jnp.float32) * cotangent)

    def xla_loss(q, k, v):
        mask = None
        if mask_fn is _complex_mask:
            mask = _complex_mask(
                jnp.asarray(0),
                jnp.asarray(0),
                jnp.arange(shape[1])[:, None],
                jnp.arange(shape[1])[None, :],
            )
        result = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            implementation="xla",
        )
        return jnp.sum(result.astype(jnp.float32) * cotangent)

    mosaic_value, mosaic_gradients = jax.jit(
        jax.value_and_grad(mosaic_loss, argnums=(0, 1, 2))
    )(query, key, value)
    xla_value, xla_gradients = jax.jit(
        jax.value_and_grad(xla_loss, argnums=(0, 1, 2))
    )(query, key, value)

    np.testing.assert_allclose(
        np.asarray(mosaic_value),
        np.asarray(xla_value),
        rtol=3e-3,
        atol=3e-3,
    )
    for mosaic_gradient, xla_gradient in zip(
        mosaic_gradients,
        xla_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(mosaic_gradient),
            np.asarray(xla_gradient),
            rtol=3e-3,
            atol=3e-3,
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
def test_mosaic_custom_vjp_vmaps_value_only_and_reduces_shared_qk_gradients():
    keys = jax.random.split(jax.random.key(83), 5)
    shape = (1, 1024, 2, 80)
    query = jax.random.normal(keys[0], shape, dtype=jnp.float16)
    key = jax.random.normal(keys[1], shape, dtype=jnp.float16)
    values = jax.random.normal(keys[2], (3, *shape), dtype=jnp.float16)
    cotangents = jax.random.normal(
        keys[3],
        values.shape,
        dtype=jnp.float32,
    )

    def mosaic_attention(q, k, v):
        return _masked_attention_via_mosaic(
            q,
            k,
            v,
            mask_fn=_unmasked,
            block_size=64,
            kv_block_size=64,
            window_size=None,
            is_causal=True,
            backward_strategy="auto",
        )

    def xla_attention(q, k, v):
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            implementation="xla",
        )

    mosaic_vmapped = jax.jit(
        jax.vmap(mosaic_attention, in_axes=(None, None, 0))
    )
    xla_vmapped = jax.jit(
        jax.vmap(xla_attention, in_axes=(None, None, 0))
    )
    mosaic_output = mosaic_vmapped(query, key, values)
    xla_output = xla_vmapped(query, key, values)
    np.testing.assert_allclose(
        np.asarray(mosaic_output),
        np.asarray(xla_output),
        rtol=3e-3,
        atol=3e-3,
    )

    def loss(attention_fn, q, k, vs):
        outputs = jax.vmap(attention_fn, in_axes=(None, None, 0))(
            q,
            k,
            vs,
        )
        return jnp.sum(outputs.astype(jnp.float32) * cotangents)

    mosaic_gradients = jax.jit(
        jax.grad(
            lambda q, k, vs: loss(mosaic_attention, q, k, vs),
            argnums=(0, 1, 2),
        )
    )(query, key, values)
    xla_gradients = jax.jit(
        jax.grad(
            lambda q, k, vs: loss(xla_attention, q, k, vs),
            argnums=(0, 1, 2),
        )
    )(query, key, values)
    for mosaic_gradient, xla_gradient in zip(
        mosaic_gradients,
        xla_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(mosaic_gradient),
            np.asarray(xla_gradient),
            rtol=5e-3,
            atol=5e-3,
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
