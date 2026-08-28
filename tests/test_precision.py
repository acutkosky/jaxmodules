import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxmodules.attention import (
    _mosaic_supports_precision,
    attention,
    default_kernel,
    flex_attention,
    flex_attention_slow,
)
from jaxmodules.normalizers import CausalNorm, StandardizeNorm, matrix_inverse_sqrt


attention_module = importlib.import_module("jaxmodules.attention")
_HIGHEST = jax.lax.Precision.HIGHEST
_HIGHEST_STABLEHLO = "precision = [HIGHEST, HIGHEST]"


def _assert_all_dots_use_highest(lowered):
    compiler_ir = str(lowered.compiler_ir())
    dot_lines = [
        line for line in compiler_ir.splitlines() if "stablehlo.dot_general" in line
    ]
    assert dot_lines
    assert all(_HIGHEST_STABLEHLO in line for line in dot_lines)


def _elementwise_attention_kernel(query, key):
    """An arbitrary positive kernel with no dot product of its own."""
    return jnp.exp(query[0] + key[0])


@pytest.mark.parametrize("kernel_fn", [default_kernel, _elementwise_attention_kernel])
def test_attention_precision_reaches_forward_and_custom_vjp(kernel_fn):
    operand = jnp.ones((1, 8, 2, 4), dtype=jnp.float32)

    def loss(query, key, value):
        return attention(
            query,
            key,
            value,
            block_size=4,
            kernel_fn=kernel_fn,
            precision=_HIGHEST,
        ).sum()

    lowered = jax.jit(
        jax.grad(loss, argnums=(0, 1, 2)),
    ).lower(operand, operand, operand)
    _assert_all_dots_use_highest(lowered)


@pytest.mark.parametrize("implementation", [flex_attention, flex_attention_slow])
def test_legacy_attention_precision_reaches_all_contractions(implementation):
    operand = jnp.ones((1, 2, 8, 4), dtype=jnp.float32)
    lowered = implementation.lower(
        operand,
        operand,
        operand,
        precision=_HIGHEST,
    )
    _assert_all_dots_use_highest(lowered)


def test_normalizer_precision_reaches_all_contractions():
    matrix = jnp.eye(3, dtype=jnp.float32)
    inverse_sqrt = jax.jit(
        lambda value: matrix_inverse_sqrt(value, precision=_HIGHEST)
    ).lower(matrix)
    _assert_all_dots_use_highest(inverse_sqrt)

    causal_norm = CausalNorm(var_resolution="matrix", precision=_HIGHEST)
    causal = jax.jit(causal_norm).lower(jnp.ones((3, 2), dtype=jnp.float32))
    _assert_all_dots_use_highest(causal)

    standardize_norm = StandardizeNorm(
        2,
        "batch",
        full_matrix=True,
        precision=_HIGHEST,
    )
    state = eqx.nn.State(standardize_norm)

    def apply_standardize(value):
        def apply_one(element):
            result, _ = standardize_norm(element, state)
            return result

        return jax.vmap(apply_one, axis_name="batch")(value)

    standardize = jax.jit(apply_standardize).lower(
        jnp.ones((3, 4, 2), dtype=jnp.float32)
    )
    _assert_all_dots_use_highest(standardize)


@pytest.mark.parametrize(
    ("dtype", "precision", "expected"),
    [
        (jnp.float16, jax.lax.Precision.HIGHEST, True),
        (jnp.bfloat16, jax.lax.Precision.HIGHEST, True),
        (jnp.float32, jax.lax.Precision.DEFAULT, True),
        (jnp.float32, jax.lax.Precision.HIGH, True),
        (jnp.float32, jax.lax.Precision.HIGHEST, False),
        (jnp.float32, jax.lax.DotAlgorithmPreset.TF32_TF32_F32, True),
        (jnp.float32, jax.lax.DotAlgorithmPreset.F32_F32_F32, False),
    ],
)
def test_mosaic_precision_compatibility(dtype, precision, expected):
    assert _mosaic_supports_precision(dtype, precision) is expected


def test_auto_dispatch_errors_instead_of_silently_bypassing_mosaic(monkeypatch):
    monkeypatch.setattr(
        attention_module,
        "_can_use_single_tile_attention",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        attention_module,
        "_can_use_mosaic_attention",
        lambda *args, **kwargs: True,
    )
    operand = jnp.ones((1, 8, 2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="implementation='xla'"):
        attention(operand, operand, operand, precision=_HIGHEST)


def test_xla_implementation_explicitly_accepts_incompatible_mosaic_precision(
    monkeypatch,
):
    def fail_if_checked(*args, **kwargs):
        pytest.fail("an explicit XLA request must not probe Mosaic")

    monkeypatch.setattr(
        attention_module,
        "_can_use_mosaic_attention",
        fail_if_checked,
    )
    operand = jnp.ones((1, 8, 2, 4), dtype=jnp.float32)
    result = attention(
        operand,
        operand,
        operand,
        block_size=4,
        precision=_HIGHEST,
        implementation="xla",
    )

    assert result.shape == operand.shape
