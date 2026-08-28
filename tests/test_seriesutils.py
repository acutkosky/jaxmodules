import jax.numpy as jnp
import numpy as np
import pytest

from jaxmodules.seriesutils import patch_series


def test_patch_series_only_returns_complete_patches():
    result = patch_series(jnp.arange(5), patch_len=3, stride=1)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
            ]
        ),
    )


def test_patch_series_stride_only_returns_complete_patches():
    result = patch_series(jnp.arange(6), patch_len=3, stride=2)

    np.testing.assert_array_equal(result, np.array([[0, 1, 2], [2, 3, 4]]))


def test_patch_series_right_padding_completes_last_patch():
    result = patch_series(
        jnp.arange(6),
        patch_len=3,
        stride=2,
        padding="right",
    )

    np.testing.assert_array_equal(
        result,
        np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0]]),
    )


@pytest.mark.parametrize(
    ("patch_len", "stride", "match"),
    [
        (0, 1, "patch_len must be positive"),
        (1, 0, "stride must be positive"),
        (6, 1, "cannot exceed"),
    ],
)
def test_patch_series_rejects_invalid_windows(patch_len, stride, match):
    with pytest.raises(ValueError, match=match):
        patch_series(jnp.arange(5), patch_len=patch_len, stride=stride)
