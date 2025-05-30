import jax
from jax import numpy as jnp
from typing import Callable, List, Tuple, Dict, Any, Union
from jaxtyping import Array, Float, Int, PyTree


def array_from_coords(shape: Tuple[int, ...], fn: Callable[..., Array]) -> Array:
    """Creates a new array by applying a function to each position's indices.

    This function creates a new array of specified shape where each element is computed
    by applying the given function to its position indices. For example, if shape is (2,3),
    then result[i,j] = fn(i,j) for i in range(2) and j in range(3).

    Args:
        shape: A tuple of integers specifying the dimensions of the output array.
        fn: A callable that takes integer indices as arguments and returns an array element.
            The number of arguments should match the length of shape.

    Returns:
        Array: A new array of shape `shape` where each element is computed by applying `fn`
        to its position indices.
    """
    in_axes = []
    for i in range(len(shape)):
        in_axis = [None] * len(shape)
        in_axis[i] = 0
        in_axes.append(tuple(in_axis))

    args = [jnp.arange(shape[i]) for i in range(len(shape))]

    return multi_vmap(fn, in_axes=in_axes, out_axes=tuple(range(len(shape))))(*args)


def multi_vmap(
    fn: Callable[..., Array],
    in_axes: List[Tuple[Union[int, None], ...]],
    out_axes: Tuple[int, ...],
) -> Callable[..., Array]:
    """Vectorizes a function over multiple axes using JAX's vmap.

    This is a convenience wrapper over jax.vmap that supports mapping over multiple axes
    simultaneously. It handles axis swapping to ensure proper broadcasting and mapping
    behavior.

    Args:
        fn: The function to vectorize. Should return an array.
        in_axes: List of tuples specifying which axes to map over for each input argument.
            Each tuple should have length equal to the number of output axes.
            Use None to indicate that an axis should not be mapped over.
        out_axes: Tuple specifying the order of axes in the output array.

    Returns:
        Callable: A vectorized version of the input function that can handle batched inputs
        according to the specified in_axes and out_axes.

    Example:
        If in_axes = ((0, None, 1), (1, 1, None)) and out_axes = (0, 1), then:
        output[i1, i2] = fn(A[i1, i2, ...], B[:, i2, ...], C[:, i1, ...])
    """

    if len(in_axes) == 1:
        return jax.vmap(fn, in_axes=in_axes[0], out_axes=out_axes[0])

    fn = jax.vmap(fn, in_axes=in_axes[0], out_axes=out_axes[0])

    return multi_vmap(fn, in_axes[1:], out_axes[1:])


def multi_vmap_transposed_in_axes(
    fn: Callable[..., Array],
    in_axes: List[Tuple[Union[int, None], ...]],
    out_axes: Tuple[int, ...],
) -> Callable[..., Array]:
    """Vectorizes a function over multiple axes with transposed input axis specifications.

    Similar to multi_vmap, but with a different interpretation of in_axes. Here, in_axes
    specifies mapping axes for each input argument separately, rather than for each output axis.

    Args:
        fn: The function to vectorize. Should return an array.
        in_axes: List of tuples where each tuple corresponds to one input argument.
            Each tuple specifies which axes to map over for that input.
            Use None to indicate that an axis should not be mapped over.
        out_axes: Tuple specifying the order of axes in the output array.

    Returns:
        Callable: A vectorized version of the input function that can handle batched inputs
        according to the transposed in_axes specification.

    Example:
        If in_axes = ((0,1), (None,1), (1,None)) and out_axes = (0,1), then:
        output[i1, i2] = fn(A[i1, i2, ...], B[:, i2, ...], C[:, i1, ...])
    """
    in_axes = [tuple([axis[i] for axis in in_axes]) for i in range(len(in_axes[0]))]
    return multi_vmap(fn, in_axes, out_axes)


def _process_limit_fn(limit_fn: Callable | int) -> Callable:
    if callable(limit_fn):
        return limit_fn
    return lambda *args: limit_fn


def partial(fn, arg):
    return lambda *args: fn(arg, *args)


def nested_fori_loop(
    lowers: Tuple[int], uppers: Tuple[int], body_fun: Callable, init_val: Array
) -> Array:
    """
    Execute a nested sequence of for loops using JAX's fori_loop.

    This function implements nested loops by recursively applying JAX's fori_loop.
    Each level of nesting corresponds to one element in the lowers and uppers tuples.
    The body_fun is called at the innermost level with all loop indices.

    Args:
        lowers: Tuple of lower bounds for each loop level. Each element can be either:
            - An integer for a fixed lower bound
            - A callable that takes the outer loop indices and returns the lower bound
        uppers: Tuple of upper bounds for each loop level. Each element can be either:
            - An integer for a fixed upper bound
            - A callable that takes the outer loop indices and returns the upper bound
        body_fun: Function to execute at each iteration. Takes loop indices as arguments
            followed by the accumulated value. The first argument is the outermost loop index,
            the second is the next level, and so on.
        init_val: Initial value for the accumulated result

    Returns:
        The final accumulated value after executing all nested loops

    Examples:
        # Example 1: Fixed bounds
        # Equivalent to:
        # for i in range(2):
        #     for j in range(3):
        #         result = body_fun(i, j, result)
        result = nested_fori_loop(
            lowers=(0, 0),
            uppers=(2, 3),
            body_fun=lambda i, j, val: body_fun(i, j, val),
            init_val=initial_value
        )

        # Example 2: Dynamic bounds using functions
        # Equivalent to:
        # for i in range(2):
        #     for j in range(i, i + 3):  # inner loop range depends on i
        #         result = body_fun(i, j, result)
        result = nested_fori_loop(
            lowers=(0, lambda i: i),  # inner loop starts at i
            uppers=(2, lambda i: i + 3),  # inner loop ends at i + 3
            body_fun=lambda i, j, val: body_fun(i, j, val),
            init_val=initial_value
        )
    """
    nest_count = len(lowers)

    lowers = [_process_limit_fn(limit) for limit in lowers]
    uppers = [_process_limit_fn(limit) for limit in uppers]
    lower = lowers[0]()
    upper = uppers[0]()

    # jax.debug.print("nest count: {}, lower: {}, upper: {}", nest_count, lower, upper)

    if nest_count == 1:
        return jax.lax.fori_loop(
            lower=lower, upper=upper, body_fun=body_fun, init_val=init_val
        )

    lowers = lowers[1:]
    uppers = uppers[1:]

    return jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=lambda i, val: nested_fori_loop(
            lowers=[partial(lower_fn, i) for lower_fn in lowers],
            uppers=[partial(upper_fn, i) for upper_fn in uppers],
            body_fun=lambda *args: body_fun(i, *args),
            init_val=val,
        ),
        init_val=init_val,
    )
