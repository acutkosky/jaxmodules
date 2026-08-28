import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from jaxmodules.normalizers import CausalNorm


def _expected_prefix_stats(x, mean_resolution, var_resolution, eps):
    means = []
    variances = []

    for end in range(1, x.shape[0] + 1):
        prefix = x[:end]
        if mean_resolution == "diag":
            mean = jnp.mean(prefix, axis=0)
            returned_mean = mean
        elif mean_resolution == "scalar":
            mean = jnp.mean(prefix)
            returned_mean = mean[None]
        else:
            mean = jnp.zeros((x.shape[1],), dtype=x.dtype)
            returned_mean = mean

        residuals = prefix - mean
        covariance = residuals.T @ residuals / end

        if var_resolution == "scalar":
            variance = jnp.trace(covariance)[None] + eps
        elif var_resolution == "diag":
            variance = jnp.diag(covariance) + eps
        else:
            variance = covariance

        means.append(returned_mean)
        variances.append(variance)

    return jnp.stack(means), jnp.stack(variances)


@pytest.mark.parametrize(
    ("mean_resolution", "var_resolution"),
    itertools.product(
        ("diag", "scalar", "none"),
        ("diag", "scalar", "matrix"),
    ),
)
def test_causal_norm_returns_true_prefix_statistics(
    mean_resolution,
    var_resolution,
):
    x = jnp.array(
        [
            [0.0, 3.0, -1.0],
            [6.0, 2.0, 4.0],
            [-2.0, 8.0, 5.0],
            [7.0, -3.0, 1.0],
        ]
    )
    eps = 1e-6
    normalizer = CausalNorm(
        mean_resolution=mean_resolution,
        var_resolution=var_resolution,
        eps=eps,
    )

    _, means, variances = normalizer(x, return_stats=True)
    expected_means, expected_variances = _expected_prefix_stats(
        x,
        mean_resolution,
        var_resolution,
        eps,
    )

    np.testing.assert_allclose(means, expected_means, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        variances,
        expected_variances,
        rtol=1e-6,
        atol=5e-6,
    )
