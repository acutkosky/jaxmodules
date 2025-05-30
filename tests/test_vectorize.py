import pytest
import jax
import jax.numpy as jnp
from jaxmodules.vectorize import array_from_coords, multi_vmap, nested_fori_loop


def test_array_from_coords_basic():
    """Test basic functionality of array_from_coords"""
    # Test creating a 2x2 array with simple index addition
    shape = (2, 2)
    fn = lambda i, j: i + j
    result = array_from_coords(shape, fn)
    expected = jnp.array([[0, 1], [1, 2]])
    assert jnp.array_equal(result, expected)


def test_array_from_coords_3d():
    """Test array_from_coords with 3D array"""
    shape = (2, 2, 2)
    fn = lambda i, j, k: i + j + k
    result = array_from_coords(shape, fn)
    expected = jnp.array([[[0, 1], [1, 2]], [[1, 2], [2, 3]]])
    assert jnp.array_equal(result, expected)


def test_multi_vmap_basic():
    """Test basic functionality of multi_vmap"""

    def fn(x, y):
        return x + y

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])

    # Test mapping over first axis of both inputs
    result = multi_vmap(fn, in_axes=[(0, 0)], out_axes=(0,))(x, y)
    expected = jnp.array([5, 7, 9])
    assert jnp.array_equal(result, expected)


def test_multi_vmap_complex():
    """Test multi_vmap with more complex axis mappings"""

    def fn(x, y, z):
        return x + y + z

    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    z = jnp.array([[9, 10], [11, 12]])

    # Test mapping over different axes
    result = multi_vmap(fn, in_axes=[(0, None, 1), (1, 0, None)], out_axes=(0, 1))(
        x, y, z
    )

    # build expected output using simple for loops.
    expected = []
    for i in range(x.shape[0]):
        expected.append([])
        for j in range(x.shape[1]):
            expected[i].append(fn(x[i, j], y[j, :], z[:, i]))
    expected = jnp.array(expected)

    # expected = jnp.array([[15, 18], [21, 24]])
    assert jnp.array_equal(result, expected)


def test_multi_vmap_unmapped_inputs():
    """Test multi_vmap with some inputs having no mapped axes"""

    def fn(x, y, z):
        # x and z will be mapped, y will be unmapped
        return jnp.concatenate([x + y, z * y])

    x = jnp.array([1, 2, 3])  # Will be mapped
    y = jnp.array([10])  # Will not be mapped
    z = jnp.array([4, 5, 6])  # Will be mapped

    result = multi_vmap(
        fn,
        in_axes=[(0, None, 0)],  # Map x and z, leave y unmapped
        out_axes=(0,),
    )(x, y, z)

    # Expected: for each i:
    # result[i] = [x[i] + y[0], z[i] * y[0]]
    expected = jnp.array(
        [
            [11, 40],  # [1+10, 4*10]
            [12, 50],  # [2+10, 5*10]
            [13, 60],  # [3+10, 6*10]
        ]
    )
    assert jnp.array_equal(result, expected)


def test_multi_vmap_multiple_axes():
    """Test multi_vmap with multiple axes being mapped simultaneously and vector mapping function output"""

    def fn(x, y, z):
        # x will be mapped over first axis, z over second axis
        return x + y + z

    x = jnp.array([[1, 2], [3, 4]])  # Will be mapped over first axis
    y = jnp.array([10, 5])  # Will not be mapped
    z = jnp.array([[5, 6], [7, 8]])  # Will be mapped over second axis

    result = multi_vmap(
        fn,
        in_axes=[
            (0, None, 1),
            (1, None, None),
        ],  # Map x over first axis, z over second axis
        out_axes=(0, 1),
    )(x, y, z)

    # Expected: for each i,j:
    # result[i,j] = x[i,j] + y + z[:, i]
    expected = jnp.array(
        [
            [
                [1 + 10 + 5, 1 + 5 + 7],  # i=0, j=0
                [2 + 10 + 5, 2 + 5 + 7],  # i=0, j=1
            ],
            [
                [3 + 10 + 6, 3 + 5 + 8],  # i=1, j=0
                [4 + 10 + 6, 4 + 5 + 8],  # i=1, j=1
            ],
        ]
    )
    assert jnp.array_equal(result, expected)


def test_multi_vmap_matrix_output():
    """Test multi_vmap with matrix output"""

    def fn(x, y):
        # Returns a 2x2 matrix for each input pair
        return jnp.array([[x, y], [x + y, x * y]])

    x = jnp.array([1, 2])
    y = jnp.array([3, 4])

    result = multi_vmap(fn, in_axes=[(0, 0)], out_axes=(0,))(x, y)

    # Expected: for each i:
    # result[i] = [[x[i], y[i]], [x[i]+y[i], x[i]*y[i]]]
    expected = jnp.array(
        [
            [[1, 3], [4, 3]],  # For x=1, y=3
            [[2, 4], [6, 8]],  # For x=2, y=4
        ]
    )
    assert jnp.array_equal(result, expected)


def test_multi_vmap_mixed_mapping():
    """Test multi_vmap with mixed mapping patterns and non-scalar output"""

    def fn(x, y, z):
        # x will be mapped, y will be partially mapped, z will be unmapped
        return jnp.stack(
            [
                x + y[0],  # Uses first element of y
                z[0] * y[1],  # Uses second element of y
                x * z[0] + y[2],  # Uses third element of y
            ]
        )

    x = jnp.array([1, 2, 3])  # Will be fully mapped
    y = jnp.array(
        [
            [4, 5, 6],  # Will be partially mapped
            [7, 8, 9],
            [10, 11, 12],
        ]
    )
    z = jnp.array([2])  # Will not be mapped

    result = multi_vmap(
        fn,
        in_axes=[(0, 1, None)],  # Map x, map second axis of y, leave z unmapped
        out_axes=(0,),
    )(x, y, z)

    # Expected: for each i:
    # result[i] = [x[i] + y[0,i], z[0] * y[1,i], x[i] * z[0] + y[2,i]]
    expected = jnp.array(
        [
            [5, 14, 12],  # [1+4, 2*7, 1*2+10]
            [7, 16, 15],  # [2+5, 2*8, 2*2+11]
            [9, 18, 18],  # [3+6, 2*9, 3*2+12]
        ]
    )
    assert jnp.array_equal(result, expected)


def test_nested_fori_loop_basic():
    """Test basic functionality of nested_fori_loop with fixed bounds"""

    def body_fun(i, j, val):
        return val + i + j

    result = nested_fori_loop(
        lowers=(0, 0), uppers=(2, 2), body_fun=body_fun, init_val=0
    )
    # Expected: sum of all i+j for i,j in range(2)
    expected = 4  # (0+0) + (0+1) + (1+0) + (1+1)
    assert result == expected


def test_nested_fori_loop_dynamic_bounds():
    """Test nested_fori_loop with dynamic bounds"""

    def body_fun(i, j, val):
        return val + i + j

    result = nested_fori_loop(
        lowers=(0, lambda i: i),
        uppers=(3, lambda i: i + 2),
        body_fun=body_fun,
        init_val=0,
    )
    # Expected: sum of all i+j where j ranges from i to i+1
    # i=0: j=0,1
    # i=1: j=1,2
    # i=2: j=2,3
    expected = 15  # (0+0) + (0+1) + (1+1) + (1+2) + (2+2) + (2+3)
    assert result == expected


def test_nested_fori_loop_array_accumulation():
    """Test nested_fori_loop with array accumulation"""

    def body_fun(i, j, val):
        return val.at[i, j].set(i + j)

    result = nested_fori_loop(
        lowers=(0, 0), uppers=(2, 2), body_fun=body_fun, init_val=jnp.zeros((2, 2))
    )
    expected = jnp.array([[0, 1], [1, 2]])
    assert jnp.array_equal(result, expected)


def test_nested_fori_loop_single_level():
    """Test nested_fori_loop with single level (no nesting)"""

    def body_fun(i, val):
        return val + i

    result = nested_fori_loop(lowers=(0,), uppers=(3,), body_fun=body_fun, init_val=0)
    expected = 3  # 0 + 1 + 2
    assert result == expected


def test_nested_fori_loop_empty_range():
    """Test nested_fori_loop with empty range"""

    def body_fun(i, j, val):
        return val + i + j

    result = nested_fori_loop(
        lowers=(0, 0),
        uppers=(0, 0),  # Empty range
        body_fun=body_fun,
        init_val=0,
    )
    assert result == 0  # No iterations performed
