import pytest
import jax
import jax.numpy as jnp
from jaxmodules.vectorize import array_from_coords, multi_vmap, nested_fori_loop, fancy_vmap
import numpy as np


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

def test_multi_vmap_out_axes():
    """Test multi_vmap with complex out_axes"""

    # If in_axes = ((0, None, 1), (1, 1, None)) and out_axes = (2, 0), then:
    # output[i0, :, i1,...] = fn(A[i1, i0, ...], B[:, i0, ...], C[:, i1, ...])

    def fn(x, y, z):
        return x + jnp.sum(y, axis=0) * jnp.sum(z, axis=0)


    x = jnp.arange(2*2*2).reshape((2,2,2))
    y = jnp.arange(5, 5+ 2*2*2).reshape((2,2,2))
    z = jnp.arange(10, 10+ 2*2*2).reshape((2,2,2))

    result = multi_vmap(
        fn,
        in_axes=[(1, 1, None), (0, None, 1)],
        out_axes=(0, 2),
    )(x, y, z)

    expected = np.zeros((2,2,2))
    for i0 in range(2):
        for i1 in range(2):
            expected[i0, :, i1] = fn(x[i1, i0, :], y[:, i0, :], z[:, i1, :])

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


def test_fancy_vmap_basic():
    """Test basic functionality of fancy_vmap with simple mapping"""
    def fn(x, y):
        return x + y
    
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    
    # Test basic mapping: output[i] = fn(x[i], y[i])
    vectorized_fn = fancy_vmap(fn, "i, i -> i")
    result = vectorized_fn(x, y)
    expected = jnp.array([5, 7, 9])
    assert jnp.array_equal(result, expected)


def test_fancy_vmap_broadcasting():
    """Test fancy_vmap with broadcasting (using ':')"""
    def fn(x, y, z):
        return x + y + z
    
    x = jnp.array([1, 2, 3])
    y = jnp.array([10, 20, 30])
    z = jnp.array([100])  # Will be broadcasted
    
    # Test broadcasting: output[i] = fn(x[i], y[i], z[:])
    vectorized_fn = fancy_vmap(fn, "i, i, : -> i")
    result = vectorized_fn(x, y, z)
    # there is a subtlety: the z argument to fn will be [100], so the output will be a 1, shape array
    # rather than a 0 shape scalar. This means the final output will be a 3,1 shape array.
    expected = jnp.array([[111], [122], [133]])  # [[1+10+100], [2+20+100], [3+30+100]]
    assert jnp.array_equal(result, expected)


def test_fancy_vmap_multiple_outputs():
    """Test fancy_vmap with multiple output dimensions"""
    def fn(x, y):
        return x + y
    
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    
    # Test multiple outputs: output[i, j] = fn(x[i, j], y[i, j])
    vectorized_fn = fancy_vmap(fn, "i j, i j -> i j")
    result = vectorized_fn(x, y)
    expected = jnp.array([[6, 8], [10, 12]])
    assert jnp.array_equal(result, expected)


def test_fancy_vmap_complex_pattern():
    """Test fancy_vmap with complex axis mapping patterns"""
    def fn(x, y, z, w):
        return x + y + z + w
    
    x = jnp.array([[1, 2], [3, 4]])  # Shape: (2, 2)
    y = jnp.array([10, 20])          # Shape: (2,)
    z = jnp.array([100])             # Shape: (1,)
    w = jnp.array([[5, 6], [7, 8]])  # Shape: (2, 2)
    
    # Complex pattern: output[i, j] = fn(x[i, j], y[i], z[:], w[j, i])
    vectorized_fn = fancy_vmap(fn, "i j, i, :, j i -> i j")
    result = vectorized_fn(x, y, z, w)
    
    # Expected: for each i, j:
    # result[i,j] = x[i,j] + y[i] + z[0] + w[j,i]
    expected = jnp.array([
        [[1+10+100+5], [2+10+100+7]],  # i=0: [x[0,0]+y[0]+z[0]+w[0,0], x[0,1]+y[0]+z[0]+w[1,0]]
        [[3+20+100+6], [4+20+100+8]]   # i=1: [x[1,0]+y[1]+z[0]+w[0,1], x[1,1]+y[1]+z[0]+w[1,1]]
    ])
    assert jnp.array_equal(result, expected)




def test_fancy_vmap_complex_pattern_pretty_input():
    """Test fancy_vmap with complex axis mapping patterns and pretty input format"""
    def fn(x, y, z, w):
        return x + y + z + w
    
    x = jnp.array([[1, 2], [3, 4]])  # Shape: (2, 2)
    y = jnp.array([10, 20])          # Shape: (2,)
    z = jnp.array([100])             # Shape: (1,)
    w = jnp.array([[5, 6], [7, 8]])  # Shape: (2, 2)
    
    # Complex pattern: output[i, j] = fn(x[i, j], y[i], z[:], w[j, i])
    vectorized_fn = fancy_vmap(fn, "out_arr[i_0, i_1] = map_fn(x_arr[i_0, i_1], y[i_0], z[:], w[i_1, i_0])")
    result = vectorized_fn(x, y, z, w)
    
    # Expected: for each i, j:
    # result[i,j] = x[i,j] + y[i] + z[0] + w[j,i]
    expected = jnp.array([
        [[1+10+100+5], [2+10+100+7]],  # i=0: [x[0,0]+y[0]+z[0]+w[0,0], x[0,1]+y[0]+z[0]+w[1,0]]
        [[3+20+100+6], [4+20+100+8]]   # i=1: [x[1,0]+y[1]+z[0]+w[0,1], x[1,1]+y[1]+z[0]+w[1,1]]
    ])
    assert jnp.array_equal(result, expected)


def test_fancy_vmap_reverse_direction():
    """Test fancy_vmap with reverse direction syntax (<-)"""
    def fn(x, y):
        return x + y
    
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    
    # Test reverse direction: output[i] = fn(x[i], y[i])
    vectorized_fn = fancy_vmap(fn, "i <- i, i")
    result = vectorized_fn(x, y)
    expected = jnp.array([5, 7, 9])
    assert jnp.array_equal(result, expected)


def test_fancy_vmap_matrix_output():
    """Test fancy_vmap with matrix output from function"""
    def fn(x, y):
        # Returns a 2x2 matrix for each input pair
        return jnp.array([[x, y], [x + y, x * y]])
    
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    
    # Test matrix output: output[i] = fn(x[i], y[i]) where fn returns a matrix
    vectorized_fn = fancy_vmap(fn, "i, i -> i")
    result = vectorized_fn(x, y)
    
    expected = jnp.array([
        [[1, 3], [4, 3]],  # For x=1, y=3
        [[2, 4], [6, 8]]   # For x=2, y=4
    ])
    assert jnp.array_equal(result, expected)


def test_fancy_vmap_invalid_format():
    """Test fancy_vmap with invalid format strings"""
    def fn(x, y):
        return x + y
    
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    
    # Test invalid format (no separator)
    with pytest.raises(ValueError, match="invalid format string"):
        fancy_vmap(fn, "i, i i")
    
    # Test ellipsis in output (not allowed)
    with pytest.raises(ValueError, match="ellipsis not allowed in output format"):
        fancy_vmap(fn, "i, i -> ...")
    
    # Test ellipsis in input (not allowed)
    with pytest.raises(ValueError, match="ellipsis not allowed in input format"):
        fancy_vmap(fn, "..., i -> i")


def test_fancy_vmap_mixed_broadcasting():
    """Test fancy_vmap with mixed broadcasting patterns"""
    def fn(x, y, z):
        return x + y + z
    
    x = jnp.array([1, 2, 3])
    y = jnp.array([10, 20, 30])
    z = jnp.array([100, 200, 300])
    
    # Test mixed broadcasting: output[i] = fn(x[i], y[i], z[:])
    vectorized_fn = fancy_vmap(fn, "i, i, : -> i")
    result = vectorized_fn(x, y, z)
    
    # Expected: for each i, z is broadcasted (only first element used)
    expected = jnp.array([[111, 211, 311], [122, 222, 322], [133, 233, 333]])
    assert jnp.array_equal(result, expected)
