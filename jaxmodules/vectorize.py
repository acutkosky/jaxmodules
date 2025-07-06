import jax
from jax import numpy as jnp
from typing import Callable, List, Tuple, Dict, Any, Union, Sequence
from jaxtyping import Array, Float, Int, PyTree
import numpy as np
import einops


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
    in_axes: Sequence[Sequence[Union[int, None]]],
    out_axes: Sequence[int],
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
        If in_axes = ((0, None, 1), (1, 1, None)) and out_axes = (2, 0), then:
        output[i0, :, i2,...] = fn(A[i2, i0, ...], B[:, i0, ...], C[:, i2, ...])
    """

    # need to sort out_axes
    out_axes_sorted_idx = np.argsort(out_axes)
    in_axes_sorted = [list(in_axes[i]) for i in out_axes_sorted_idx]
    out_axes_sorted = tuple(out_axes[i] for i in out_axes_sorted_idx)
    # now, we need to address a subtle issue that arises in the following example
    # in_axes = ((1, 1, None), (0, None, 1)) and out_axes = (0, 2)
    # in this case, the outer vmap will take in_axis=(1,1,None) and out_axis=0
    # however, axis 1 of the input 1 is now actually axis 2 of the original input
    # because the input to the outer vmap will be produced by the inner vmap
    # which has already consumed the zeroth axis of the original input.

    # to fix this, we need to replace in_axes[i][j] with in_axes[i][j] - k where
    # k is the number of times that in_axes[n][j]<in_axes[i][j] for n>i.

    # this is why we converted to lists above; it will make this editing a bit easier.

    max_length = 0
    for axes in in_axes_sorted:
        if isinstance(axes, int) or axes is None:
            continue
        max_length = max(max_length, len(axes))
    
    for j in range(max_length):
        
        indices = []
        for axes in in_axes_sorted:
            if axes is None:
                indices.append(None)
            elif isinstance(axes, int):
                indices.append(axes)
            else:
                indices.append(axes[j])
        for i in reversed(range(len(in_axes_sorted))):
            if in_axes_sorted[i] is None:
                continue
            if not isinstance(in_axes_sorted[i], int):
                if in_axes_sorted[i][j] is None:
                    continue
            if not isinstance(in_axes_sorted[i], int):
                assert in_axes_sorted[i][j] == indices[i]
            for inner_idx in indices[i:]:
                if inner_idx is not None and inner_idx < indices[i]:
                    if isinstance(in_axes_sorted[i], int):
                        in_axes_sorted[i] -= 1
                    else:
                        in_axes_sorted[i][j] -= 1


    return _multi_vmap_pre_sorted(fn, in_axes_sorted, out_axes_sorted)

def _multi_vmap_pre_sorted(
    fn: Callable[..., Array],
    in_axes: Sequence[Sequence[Union[int, None]]],
    out_axes: Sequence[int],
) -> Callable[..., Array]:
    if len(in_axes) == 1:
        return jax.vmap(fn, in_axes=in_axes[0], out_axes=out_axes[0])

    fn = jax.vmap(fn, in_axes=in_axes[0], out_axes=out_axes[0])

    return _multi_vmap_pre_sorted(fn, in_axes[1:], out_axes[1:])


def multi_vmap_transposed_in_axes(
    fn: Callable[..., Array],
    in_axes: Sequence[Sequence[Union[int, None]]],
    out_axes: Sequence[int],
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

        If in_axes = ((0,1), (None,1), (1,None)) and out_axes = (1,0), then:
        output[i2, i1] = fn(A[i1, i2, ...], B[:, i2, ...], C[:, i1, ...])
    """
    in_axes = [tuple([axis[i] for axis in in_axes]) for i in range(len(in_axes[0]))]
    return multi_vmap(fn, in_axes, out_axes)


def _process_limit_fn(limit_fn: Union[Callable, int]) -> Callable:
    if callable(limit_fn):
        return limit_fn
    return lambda *args: limit_fn


def partial(fn, arg):
    return lambda *args: fn(arg, *args)


def nested_fori_loop(
    lowers: Tuple[Union[int, Callable], ...], 
    uppers: Tuple[Union[int, Callable], ...], 
    body_fun: Callable, 
    init_val: Array
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

    lowers_processed = [_process_limit_fn(limit) for limit in lowers]
    uppers_processed = [_process_limit_fn(limit) for limit in uppers]
    lower = lowers_processed[0]()
    upper = uppers_processed[0]()

    if nest_count == 1:
        return jax.lax.fori_loop(
            lower=lower, upper=upper, body_fun=body_fun, init_val=init_val
        )

    lowers_remaining = lowers_processed[1:]
    uppers_remaining = uppers_processed[1:]

    return jax.lax.fori_loop(
        lower=lower,
        upper=upper,
        body_fun=lambda i, val: nested_fori_loop(
            lowers=tuple(partial(lower_fn, i) for lower_fn in lowers_remaining),
            uppers=tuple(partial(upper_fn, i) for upper_fn in uppers_remaining),
            body_fun=lambda *args: body_fun(i, *args),
            init_val=val,
        ),
        init_val=init_val,
    )

def _remove_repeated_elements(lst):
    return [x for i, x in enumerate(lst) if x not in lst[:i]]

'''
This is an implementation of einsum that uses the same pattern parsing as einops.
However, einops relies on jnp.einsum as a backend. This sounds reasonable,
but for some reason jnp.einops introduces a lot of floating point error that
is not present in this version that just relies on jax.vmap.
'''
def einsum(*args):

    pattern = args[-1]
    args = list(args[:-1])



    in_patterns, out_patterns = pattern.split("->")
    in_patterns = in_patterns.split(",")
    in_patterns = [p.strip() for p in in_patterns]
    in_patterns = [
        [p.strip() for p in in_pattern.split(" ") if p != ""] for in_pattern in in_patterns
    ]
    out_patterns = out_patterns.strip()
    out_patterns = out_patterns.split(" ")
    out_patterns = [p.strip() for p in out_patterns if p != ""]


    assert len(in_patterns) == len(args), "number of input patterns must match number of arguments"


    # first, let's preprocess out all the repeated indices in input patterns. Hopefully
    # we can rely on baseline einsum for this since it's just memory routing and no floating point operations.

    for idx in range(len(in_patterns)):
        in_pattern = in_patterns[idx]
        unique_pattern = _remove_repeated_elements(in_pattern)
        if len(unique_pattern) < len(in_pattern):
            in_patterns[idx] = unique_pattern
            args[idx] = einops.einsum(args[idx], " ".join(in_pattern) + " -> " + " ".join(unique_pattern))
            
        
    # handle ellipses
    max_ellipsis_shape = []

    for idx in range(len(in_patterns)):
        in_pattern = in_patterns[idx]
        arg = args[idx]
        if arg.ndim != len(in_pattern):
            # there must be exactly one ellipsis in this pattern
            ellipsis_count = sum(p == "..." for p in in_pattern)
            if ellipsis_count != 1:
                raise ValueError(f"pattern {in_pattern} has {ellipsis_count} ellipses, but there must be exactly one.")
            ellipsis_idx = in_pattern.index("...")

            ellipsis_shape = arg.shape[ellipsis_idx:len(arg.shape)-len(in_pattern[ellipsis_idx+1:])]
            if len(max_ellipsis_shape) < len(ellipsis_shape):
                assert all([x == y for x, y in zip(max_ellipsis_shape, ellipsis_shape[-len(max_ellipsis_shape):])]), "ellipsis shapes must broadcast"
                max_ellipsis_shape = ellipsis_shape
            else:
                assert all([x == y for x, y in zip(max_ellipsis_shape[-len(ellipsis_shape):], ellipsis_shape)]), "ellipsis shapes must broadcast"

            new_in_pattern = in_pattern[:ellipsis_idx] + list(range(-len(ellipsis_shape), 0)) + in_pattern[ellipsis_idx+1:]
            in_patterns[idx] = new_in_pattern

    out_axes = []
    idx = 0
    if '...' in out_patterns:
        out_patterns = out_patterns[:out_patterns.index('...')] + list(range(-len(max_ellipsis_shape), 0)) + out_patterns[out_patterns.index('...')+1:]

    out_axes = list(range(len(out_patterns)))


    def get_in_axis(p):
        out_axes_map = {
            name: i for i, name in enumerate(p)
        }
        axes = list(out_axes_map.get(out, None) for out in out_patterns)
        return axes

    in_axes=[get_in_axis(p) for p in in_patterns]

    summed_in_axes = [
        [p for p in in_pattern if p not in out_patterns] for in_pattern in in_patterns
    ]

    max_summed_in_axis = []
    for summed_in_axis in summed_in_axes:
        if len(max_summed_in_axis) < len(summed_in_axis):
            max_summed_in_axis = summed_in_axis

    def broadcast_and_multiply(axes_prod, axes_factor, ar_prod, ar_factor):
        axes_broadcast = axes_prod + [ax for ax in axes_factor if ax not in axes_prod]

        prod_shape_map = {ax: shape for ax, shape in zip(axes_prod, ar_prod.shape)}
        factor_shape_map = {ax: shape for ax, shape in zip(axes_factor, ar_factor.shape)}

        prod_shape = [prod_shape_map.get(ax, 1) for ax in axes_broadcast]
        ar_prod = jnp.reshape(ar_prod, prod_shape)

        factor_position_map = {ax: i for i, ax in enumerate(axes_factor)}
        ar_factor = jnp.reshape(ar_factor, list(ar_factor.shape) + [1] * (len(axes_broadcast) - len(ar_factor.shape)))
        idx = len(axes_factor)
        for ax in axes_broadcast:
            if ax in factor_position_map:
                continue
            else:
                factor_position_map[ax] = idx
                idx += 1

        ar_factor = jnp.transpose(ar_factor, [factor_position_map[ax] for ax in axes_broadcast])

        return ar_factor * ar_prod, axes_broadcast

    def sum_prod(*args):
        prod = args[0]
        prod_shape = summed_in_axes[0]
        for arg, factor_shape in zip(args[1:], summed_in_axes[1:]):
            prod, prod_shape = broadcast_and_multiply(prod_shape, factor_shape, prod, arg)
        return jnp.sum(prod)

    if len(out_axes) == 0:
        return sum_prod(*args)

    return multi_vmap_transposed_in_axes(
        sum_prod,
        in_axes=in_axes,
        out_axes=out_axes
    )(*args)
