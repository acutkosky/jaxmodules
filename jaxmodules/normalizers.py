from collections.abc import Hashable, Sequence
from typing import Any, Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

import equinox as eqx
from equinox.nn import StateIndex, StatefulLayer, State
from equinox import field

from einops import rearrange


def matrix_inverse_sqrt(M, eps=0, *, precision=None):
    eig_vals, eig_vecs = jnp.linalg.eigh(M)

    inv_eig_vals = 1.0 / jnp.sqrt(jnp.maximum(eig_vals, eps))

    result = jnp.einsum(
        "...ij,...j,...kj->...ik",
        eig_vecs,
        inv_eig_vals,
        eig_vecs,
        precision=precision,
    )

    return result


class StandardizeNorm(StatefulLayer, strict=True):
    state_index: StateIndex[
        tuple[Float[Array, "input_size"], Float[Array, "input_size"]]
    ]
    axis_name: Union[Hashable, Sequence[Hashable]] = field(static=True)
    inference: bool
    input_size: int = field(static=True)
    eps: float = field(static=True)
    momentum: float = field(static=True)
    full_matrix: bool = field(static=True)
    precision: Any = field(static=True)

    def __init__(
        self,
        input_size: int,
        axis_name: Union[Hashable, Sequence[Hashable]],
        full_matrix: bool = False,
        eps: float = 1e-5,
        momentum: float = 0.99,
        inference: bool = False,
        dtype=None,
        precision=None,
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
        - `precision`: Optional JAX contraction precision for full-matrix statistics.
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
        self.precision = precision

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
                    jnp.matmul(
                        centered_x.transpose(),
                        centered_x,
                        precision=self.precision,
                    )
                    / N
                )
                new_cov = lax.pmean(new_cov, self.axis_name)
            else:
                new_cov = jnp.mean(centered_x**2, axis=0)
                new_cov = lax.pmean(new_cov, self.axis_name)

            running_cov = running_cov + (new_cov - running_cov) * (1.0 - momentum)
            state = state.set(self.state_index, (running_mean, running_cov, new_count))
        else:
            centered_x = x_flat - running_mean

        if self.full_matrix:
            preconditioner = matrix_inverse_sqrt(
                running_cov,
                self.eps,
                precision=self.precision,
            )

            normalized = jnp.matmul(
                centered_x,
                preconditioner,
                precision=self.precision,
            )
        else:
            preconditioner = 1.0 / jnp.sqrt(running_cov + self.eps)
            normalized = centered_x * preconditioner
        normalized = jnp.reshape(normalized, x.shape)

        return normalized, state


class CausalNorm(StatefulLayer):
    """Normalize rows using statistics from their causal prefixes.

    The prefix mean minimizes average squared L2 residual subject to the selected
    mean resolution: a per-channel vector, one scalar shared by every channel, or
    the fixed zero vector. Scalar variance is the resulting squared L2 value;
    diagonal and matrix variance retain the corresponding residual moments.
    """

    eps: float = field(static=True)
    mean_resolution: str = field(static=True)
    var_resolution: str = field(static=True)
    precision: Any = field(static=True)

    def __init__(
        self,
        mean_resolution: str = "diag",
        var_resolution: str = "diag",
        eps: float = 1e-6,
        precision=None,
    ):
        assert mean_resolution in ["diag", "scalar", "none"]
        assert var_resolution in ["diag", "scalar", "matrix"]

        self.mean_resolution = mean_resolution
        self.var_resolution = var_resolution
        self.eps = eps
        self.precision = precision

    def __call__(self, x: jax.Array, return_stats=False):
        T, _ = x.shape

        counts = jnp.arange(1, T + 1, dtype=x.dtype).reshape((T, 1))
        if self.mean_resolution == "none":
            means = jnp.zeros_like(x)
            vector_residuals = x
            correction = jnp.ones_like(counts)
            mean_offsets = jnp.zeros_like(x)
        else:
            vector_means = jnp.cumsum(x, axis=0) / counts
            if self.mean_resolution == "diag":
                means = vector_means
            else:
                means = jnp.mean(vector_means, axis=1, keepdims=True)

            # Welford's update can be expressed in terms of the residual against
            # the updated mean as
            #
            #   M2[t] = M2[t - 1]
            #           + t / (t - 1) * (x[t] - mean[t]) ** 2.
            #
            # This form retains the parallel cumulative sums used here while
            # avoiding the cancellation in E[x**2] - E[x]**2.
            vector_residuals = x - vector_means
            correction = jnp.where(
                counts > 1,
                counts / jnp.maximum(counts - 1, 1),
                jnp.zeros_like(counts),
            )
            mean_offsets = vector_means - means

        centered_x = x - means

        if self.var_resolution == "scalar":
            diagonal_vars = (
                jnp.cumsum(correction * vector_residuals**2, axis=0) / counts
                + mean_offsets**2
            )
            vars = jnp.sum(diagonal_vars, axis=1, keepdims=True) + self.eps

            result = centered_x / jnp.sqrt(vars)

        elif self.var_resolution == "diag":
            vars = (
                jnp.cumsum(correction * vector_residuals**2, axis=0) / counts
                + mean_offsets**2
                + self.eps
            )

            result = centered_x / jnp.sqrt(vars)
        elif self.var_resolution == "matrix":
            vars = jnp.einsum(
                "ti,tj->tij",
                vector_residuals,
                vector_residuals,
                precision=self.precision,
            )
            vars = jnp.cumsum(vars * correction[..., None], axis=0) / counts[
                ..., None
            ] + jnp.einsum(
                "ti,tj->tij",
                mean_offsets,
                mean_offsets,
                precision=self.precision,
            )

            preconditioner = matrix_inverse_sqrt(
                vars,
                self.eps,
                precision=self.precision,
            )

            result = jnp.einsum(
                "ti,tij->tj",
                centered_x,
                preconditioner,
                precision=self.precision,
            )

        if return_stats:
            return result, means, vars
        else:
            return result
