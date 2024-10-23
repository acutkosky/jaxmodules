from collections.abc import Hashable, Sequence
from typing import Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

import equinox as eqx
from equinox.nn import StateIndex, StatefulLayer, State
from equinox import field

from einops import einsum, rearrange


def matrix_inverse_sqrt(M, eps=0):
    eig_vals, eig_vecs = jnp.linalg.eigh(M)

    inv_eig_vals = 1.0 / jnp.sqrt(jnp.maximum(eig_vals, eps))

    result = einsum(eig_vecs, inv_eig_vals, eig_vecs, '... d1 d2, ... d2, ... d3 d2 -> ... d1  d3')

    return result


class StandardizeNorm(StatefulLayer, strict=True):
    state_index: StateIndex[
        tuple[Float[Array, "input_size"], Float[Array, "input_size"]]
    ]
    axis_name: Union[Hashable, Sequence[Hashable]]
    inference: bool
    input_size: int = field(static=True)
    eps: float = field(static=True)
    momentum: float = field(static=True)
    full_matrix: bool = field(static=True)

    def __init__(
        self,
        input_size: int,
        axis_name: Union[Hashable, Sequence[Hashable]],
        full_matrix: bool = False,
        eps: float = 1e-5,
        momentum: float = 0.99,
        inference: bool = False,
        dtype=None,
    ):
        """**Arguments:**

        - `input_size`: The number of channels in the input array.
        - `axis_name`: The name of the batch axis to compute statistics over, as passed
            to `axis_name` in `jax.vmap` or `jax.pmap`. Can also be a sequence (e.g. a
            tuple or a list) of names, to compute statistics over multiple named axes.
        - full_magrix: whether to use full matrix standardization or just diagonal.
        - `eps`: Value added to the denominator for numerical stability.
        - `momentum`: The rate at which to update the running statistics. Should be a
            value in (0, 1]. If 1, then the behavior will be to keep track of a global
            running average for the statistics.
        - `inference`: If `False` then the batch means and variances will be calculated
            and used to update the running statistics. If `True` then the running
            statistics are directly used for normalisation. This may be toggled with
            [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.BatchNorm.__call__`][].
        - `dtype`: The dtype to use for the running statistics and the weight and bias
            if `channelwise_affine` is `True`. Defaults to either
            `jax.numpy.float32` or `jax.numpy.float64` depending on whether JAX is in
            64-bit mode.
        """
        dtype = jnp.float32 if dtype is None else dtype
        self.full_matrix = full_matrix
        if full_matrix:
            init_buffers = (
                jnp.empty((input_size,), dtype=dtype),
                jnp.empty((input_size, input_size), dtype=dtype),
                0,
            )
        else:
            init_buffers = (
                jnp.empty((input_size,), dtype=dtype),
                jnp.empty((input_size,), dtype=dtype),
                0,
            )
        self.state_index = StateIndex(init_buffers)
        self.inference = inference
        self.axis_name = axis_name
        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum

    @jax.named_scope("normalizers.StandardizeNorm")
    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
    ) -> tuple[Array, State]:
        """**Arguments:**

        - `x`: A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - `state`: An [`equinox.nn.State`][] object (which is used to store the
            running statistics).
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `inference`: As per [`equinox.nn.BatchNorm.__init__`][]. If
            `True` or `False` then it will take priority over `self.inference`. If
            `None` then the value from `self.inference` will be used.

        **Returns:**

        A 2-tuple of:

        - A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - An updated state object (storing the updated running statistics).

        **Raises:**

        A `NameError` if no `vmap`s are placed around this operation, or if this vmap
        does not have a matching `axis_name`.
        """

        x_flat = jnp.reshape(x, (-1, self.input_size))
        N, _ = x_flat.shape
        running_mean, running_cov, count = state.get(self.state_index)

        if inference is None:
            inference = self.inference

        if not inference:
            new_mean = jnp.mean(x_flat, axis=0)
            new_mean = lax.pmean(new_mean, self.axis_name)
            new_count = count + 1

            momentum = jnp.minimum(1.0 - 1.0 / new_count, self.momentum)

            running_mean = running_mean + (new_mean - running_mean) * (1.0 - momentum)

            centered_x = x_flat - running_mean

            if self.full_matrix:
                new_cov = (
                    centered_x.transpose() @ centered_x / N
                )  # [d, N] @ [d, N] -> [d, d]
                new_cov = lax.pmean(new_cov, self.axis_name)
            else:
                new_cov = jnp.mean(centered_x**2, axis=0)
                new_cov = lax.pmean(new_cov, self.axis_name)

            running_cov = running_cov + (new_cov - running_cov) * (1.0 - momentum)
            state = state.set(self.state_index, (running_mean, running_cov, new_count))
        else:
            centered_x = x_flat - running_mean

        if self.full_matrix:
            preconditioner = matrix_inverse_sqrt(running_cov, self.eps)

            normalized = centered_x @ preconditioner
        else:
            preconditioner = 1.0 / jnp.sqrt(running_cov + self.eps)
            normalized = centered_x * preconditioner
        normalized = jnp.reshape(normalized, x.shape)

        return normalized, state


class CausalNorm(StatefulLayer):
    eps: float = field(static=True)
    mean_resolution: str = field(static=True)
    var_resolution: str = field(static=True)

    def __init__(
        self,
        mean_resolution: str = "diag",
        var_resolution: str = "diag",
        eps: float = 1e-6,
    ):
        assert mean_resolution in ["diag", "scalar", "none"]
        assert var_resolution in ["diag", "scalar", "matrix"]

        self.mean_resolution = mean_resolution
        self.var_resolution = var_resolution
        self.eps = eps

    def __call__(self, x: jax.Array):
        T, C = x.shape


        if self.mean_resolution == 'none':
            centered_x = x
        else:
            means = jnp.cumsum(x, axis=0) / jnp.arange(1, T + 1).reshape((T, 1))

            if self.mean_resolution == "scalar":
                means = jnp.mean(means, axis=1, keepdims=True)
            centered_x = x  - means

        if self.var_resolution == "scalar":
            vars = jnp.cumsum(centered_x**2, axis=0) / jnp.arange(1, T + 1).reshape(
                (T, 1)
            )
            vars = jnp.sum(vars, axis=1, keepdims=True) + self.eps

            result = centered_x / jnp.sqrt(vars)

        elif self.var_resolution == "diag":
            vars = (
                jnp.cumsum(centered_x**2, axis=0)
                / jnp.arange(1, T + 1).reshape((T, 1))
                + self.eps
            )

            result = centered_x / jnp.sqrt(vars)
        elif self.var_resolution == "matrix":
            vars = einsum(centered_x, centered_x, "t c1, t c2 -> t c1 c2")
            vars = jnp.cumsum(vars, axis=0) / jnp.arange(1, T + 1).reshape((T, 1, 1))

            preconditioner = matrix_inverse_sqrt(vars, self.eps)

            result = einsum(centered_x, preconditioner, "t c, t c c2 -> t c2")

        return result


