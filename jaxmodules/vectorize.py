import jax
from jax import numpy as jnp
from typing import Callable, List, Tuple, Dict, Any, Union, Sequence, Optional
from jaxtyping import Array, Float, Int, PyTree
import numpy as np
import einops
import re

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
    out_axes: Optional[Sequence[int]]=None,
) -> Callable[..., Array]:
    """Vectorizes a function over multiple axes using JAX's vmap.

    This is a convenience wrapper over jax.vmap that supports mapping over multiple axes
    simultaneously. It handles axis swapping to ensure proper broadcasting and mapping
    behavior.

    Args:
        fn: The function to vectorize. Should return an array.
        in_axes: List of tuples specifying which axes to map over for each input argument.
            Each tuple should have length equal to the number of inputs.
            Use None to indicate that an axis should not be mapped over.
        out_axes: Tuple specifying the order of axes in the output array.

    Returns:
        Callable: A vectorized version of the input function that can handle batched inputs
        according to the specified in_axes and out_axes.

    Example:
        If in_axes = ((0, None, 1), (1, 1, None)) and out_axes = (2, 0), then:
        output[i0, :, i2,...] = fn(A[i2, i0, ...], B[:, i0, ...], C[:, i2, ...])
    """

    if out_axes is None:
        out_axes = tuple(range(len(in_axes)))

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
    out_axes: Optional[Sequence[int]]=None,
) -> Callable[..., Array]:
    if len(in_axes) == 1:
        return jax.vmap(fn, in_axes=in_axes[0], out_axes=out_axes[0])

    fn = jax.vmap(fn, in_axes=in_axes[0], out_axes=out_axes[0])

    return _multi_vmap_pre_sorted(fn, in_axes[1:], out_axes[1:])


def multi_vmap_transposed_in_axes(
    fn: Callable[..., Array],
    in_axes: Sequence[Sequence[Union[int, None]]],
    out_axes: Sequence[int] = None,
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
    if out_axes is None:
        out_axes = tuple(range(len(in_axes[0])))

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


def _split_string_no_whitespace(s: str) -> List[str]:
    return [x for x in s.split(' ') if x != ""]

'''
A wrapper around multi_vmap that allows for specifying the in and out axes
via format strings in a manner similar to einsum.

Conceptually, I think about mapping as asking the question:
"what is the value of output[i1, i2, ..., in]?"

So, this function tries to encode this:

i1 i2 : : i3 <- map_fn(A1[i1, :, i2], A2[:, i3], A3[i1], A4[i3, i2, :])

should translate to axis tuples:

out_axes = (0, 1, 24)

in_axes = (
(0, None, 0, None),
(1, None, None, 1),
(0, 1, None, 0),
)

So, this would be a string like:

i1 i2 : : i3 <- i1 : i2, : i3, i1, i3 i2 :
(the last : is optional)

We'll allow parsing in two directions:
i1 : i2, : i3, i1, i3 i2 : -> i1 i2 : : i3

is also valid.

The output pattern is then a space-separated list of names.
The input string is a comma-separated list of patterns, which of which
is a space-separated list of names.

NEW:
we'll also allow for putting dummy function and variable names with parentheses or square brackets, like so:

out[i1, i2, i3] <- fn(A[i1, i2], B[i2, i3], C[i3, :, i1])

This will be parsed as:

i1 i2 i3 <- i1 i2, i2 i3, i3 : i1

note that in this format, there are commas between dimension as well as between patterns.

the actual names of the functions and variables are ignored.
'''

def _parse_format(fmt: str) -> str:
    '''
    converts a format string with dummy function and variable names into
    a simpler format string that can be parsed by the rest of the code.
    '''
    if '<-' in fmt:
        assert '->' not in fmt, "cannot have both '<-' and '->' in format string"
        output_fmt, input_fmt = fmt.split('<-')
    elif '->' in fmt:
        input_fmt, output_fmt = fmt.split('->')
    else:
        raise ValueError(f"invalid format string: {fmt}")
    
    # clean up the output_fmt:
    # it should match ^\s*[\w\d]*\[([\w\d\s,:]*)\]\s*$
    output_fmt = re.match(r'^\s*[\w\d]*\[([\w\d\s,:]*)\]\s*$', output_fmt)


def _parse_dummy_format(output_fmt: str, input_fmt: str) -> Tuple[str, str]:
    # now output_fmt is the comma-separated list of axis names. Just replace the commas with spaces.
    output_fmt = output_fmt.replace(',', ' ')

    # now let's grab all the axis specifications:
    in_axes = re.match(r'[\w\d]*\[([\w\d\s,:]*)\]', input_fmt)
    if in_axes is None:
        raise ValueError(f"invalid input format string: {input_fmt}")
    in_axes = in_axes.group(1)

    # now let's grab all the axis specifications:
    in_axes = re.findall(r'[\w\d]*\[([\w\d\s,:]*)\]', input_fmt)
    in_axes = [x.replace(',', ' ') for x in in_axes]
    input_fmt = ', '.join(in_axes)

    # now let's put it all together:
    return output_fmt, input_fmt
    

def fancy_vmap(fn: Callable, fmt: str) -> Callable:
    """Vectorizes a function using a format string similar to einsum notation.
    
    A wrapper around multi_vmap that allows for specifying the in and out axes
    via format strings in a manner similar to einsum. This provides a more intuitive
    way to specify complex vectorization patterns.
    
    The format string uses a pattern where you specify how output indices map to
    input indices. Conceptually, this answers the question: "what is the value of 
    output[i1, i2, ..., in]?" by specifying which input indices contribute to each
    output index.
    
    Args:
        fn: The function to vectorize. Should take multiple array arguments and return an array.
        fmt: A format string specifying the mapping between input and output axes.
            Can use either '->' or '<-' as the separator between input and output patterns.
            
            Format: "input_patterns -> output_pattern" or "output_pattern <- input_patterns"
            
            - input_patterns: Comma-separated list of space-separated axis names for each input
            - output_pattern: Space-separated list of axis names for the output
            - Use ':' to indicate axes that should not be mapped over (these will appear in the input to fn)
            - Axis names must match between input and output to establish the mapping
            
    Returns:
        Callable: A vectorized version of the input function that can handle batched inputs
        according to the specified format string.
        
    Raises:
        ValueError: If the format string is invalid, contains ellipsis, or has mismatched patterns.
        
    Examples:
        # Example 1: Basic mapping
        # output[i1, i2, i3] = fn(A[i1, i2], B[i2, i3])
        vectorized_fn = fancy_vmap(fn, "i1 i2, i2 i3 -> i1 i2 i3")

        # Example 2: More complex mapping
        # output[:, i1, i2] = fn(A[i1, :, i2], B[:, i2], C[i1], D[i2, i1, :])
        vectorized_fn = fancy_vmap(fn, "i1 : i2, : i2, i1, i2 i1 : -> :i1 i2")
        
        # Example 3: Reverse direction (equivalent to above)
        vectorized_fn = fancy_vmap(fn, ": i1 i2 <- i1 : i2, : i2, i1, i2 i1 :")
    """
    # find the output specification

    if '<-' in fmt:
        assert '->' not in fmt, "cannot have both '<-' and '->' in format string"
        output_fmt, input_fmt = fmt.split('<-')
    elif '->' in fmt:
        input_fmt, output_fmt = fmt.split('->')
    else:
        raise ValueError(f"invalid format string: {fmt}")
    
    # clean up the output_fmt:
    # it should match ^\s*[\w\d]*\[([\w\d\s,:]*)\]\s*$
    output_match = re.match(r'^\s*[\w\d]*\[([\w\d\s,:]*)\]\s*$', output_fmt)
    input_match = re.match(r'^\s*[\w\d]*\(([\w\d\s,:\[\]]*)\)\s*$', input_fmt)
    if output_match is not None and input_match is not None:
        output_fmt, input_fmt = _parse_dummy_format(output_match.group(1), input_match.group(1))


    # parse the output

    output_fmt = _split_string_no_whitespace(output_fmt)
    

    out_name_to_idx = {}
    out_idx_to_name = {}
    out_axes_list = []
    for idx, x in enumerate(output_fmt):
        if x == '...':
            raise ValueError("ellipsis not allowed in output format")
        if x != ':':
            out_name_to_idx[x] = idx
            out_idx_to_name[idx] = x
            out_axes_list.append(idx)

    
    # parse the input
    input_fmt = [x for x in input_fmt.split(',')]
    input_fmt = [_split_string_no_whitespace(in_pattern) for in_pattern in input_fmt]   

    for in_pattern in input_fmt:
        if '...' in in_pattern:
            raise ValueError("ellipsis not allowed in input format")


    def make_in_axes(out_idx):
        in_axes = []
        out_name = out_idx_to_name[out_idx]
        for in_pattern in input_fmt:
            if out_name not in in_pattern:
                in_axes.append(None)
            else:
                in_axes.append(in_pattern.index(out_name))
        return in_axes

    in_axes = [make_in_axes(out_idx) for out_idx in out_axes_list]

    return multi_vmap(fn, in_axes=in_axes, out_axes=out_axes_list)

