# jaxmodules

some random jax tools

## fancy_vmap

`fancy_vmap` is a vectorization utility that extends JAX's `vmap` with an intuitive format string syntax similar to einsum notation. It allows you to specify complex vectorization patterns by describing how output indices map to input indices, making it easy to vectorize functions over multiple axes simultaneously.

The format string supports two styles:
- **einsum-like format**: `"input_patterns -> output_pattern"` where you specify axis names for each input and output
- **dummy function format**: `"output[axes] = fn(input1[axes], input2[axes], ...)"` for more readable specifications

Use `:` to indicate axes that should not be mapped over (these will be passed directly to the function).

### Examples

```python
from jaxmodules.vectorize import fancy_vmap
import jax.numpy as jnp

# Example 1: Simple outer product pattern
# output[i, j] = fn(A[i], B[j])
def multiply(a, b):
    return a * b

vectorized = fancy_vmap(multiply, "i, j -> i j")
vectorized = fancy_vmap(multiply, "output[i, j] = multiply(A[i], B[j])") # same as previous line
A = jnp.array([1, 2, 3])
B = jnp.array([4, 5])
result = vectorized(A, B)  # shape: (3, 2)

# Example 2: Complex mapping with unmapped dimensions
# output[:, i, j] = fn(A[i, :, j], B[:, j], C[i])
def complex_fn(a, b, c):
    return jnp.sum(a) + jnp.sum(b) + c

vectorized = fancy_vmap(complex_fn, "i : j, : j, i -> : i j")
vectorized = fancy_vmap(complex_fn, "output[:, i, j] = complex_fn(A[i, :, j], B[:, j], C[i])") # same as previous line
A = jnp.array([1, 2, 3])
A = jnp.ones((3, 5, 4))  # (i=3, :, j=4)
B = jnp.ones((7, 4))     # (:, j=4)
C = jnp.ones(3)          # (i=3)
result = vectorized(A, B, C)  # shape: (5, 3, 4) - 5 comes from unmapped dimension
```

## masked_attention_via_map

`masked_attention_via_map` is a memory-efficient attention implementation written in pure JAX, similar to Flash Attention. It uses `jax.lax.map` to process attention in blocks, enabling efficient computation for long sequences while supporting flexible masking patterns.

Key features:
- Memory-efficient block-wise processing using `jax.lax.map`
- Customizable attention masks via `mask_fn`
- Support for causal masking
- Optional windowing for local attention patterns
- Configurable block sizes for memory/performance trade-offs

### Examples

```python
from jaxmodules.attention import masked_attention_via_map
import jax.numpy as jnp

# Example 1: Basic causal attention
B, L, H, d = 2, 128, 8, 64
Q = jnp.ones((B, L, H, d))
K = jnp.ones((B, L, H, d))
V = jnp.ones((B, L, H, d))

# Causal masking: each position can only attend to previous positions
output = masked_attention_via_map(Q, K, V, is_causal=True)

# Example 2: Custom mask with windowing
# Custom mask function: (batch, head, query_idx, key_idx) -> bool
def window_mask(b, h, q, k):
    # Allow attention within a window of 10 positions
    return jnp.abs(q - k) <= 10

output = masked_attention_via_map(
    Q, K, V,
    mask_fn=window_mask,
    block_size=32,  # Process in blocks of 32 for memory efficiency
    window_size=(10, 10)  # Approximate window bounds
)
```

## log_state

The `log_state` module provides utilities for extracting logging data from JAX-compiled functions. When working with JIT-compiled training loops, it's often difficult to extract intermediate values for logging. This module solves this by providing a `Log` wrapper class that can be embedded in your pytree, and helper functions to extract and process all logged values.

### Examples

```python
from jaxmodules.logstate import Log, map_logs, list_of_logs
import jax.numpy as jnp

# Example 1: Embedding logs in a training state
class TrainState:
    def __init__(self, params, optimizer_state):
        self.params = params
        self.optimizer_state = optimizer_state
        # Embed log data in the state
        self.activation_norm = Log(jnp.array(0.5))  # Log the activation norm
        self.gradient_norm = Log(jnp.array(0.1))   # Log the gradient norm

state = TrainState(params={}, optimizer_state={})

# Extract all logged values
logs = list_of_logs(state)
# logs = [0.5, 0.1]

# Example 2: Processing logs after training step
def process_logs(log_value):
    # Send to logging system, print, etc.
    print(f"Logged value: {log_value}")
    return log_value

# Apply a function to all logs while preserving the structure
updated_state = map_logs(process_logs, state)
# This will print both logged values and return state with updated Log objects

# Example 3: Clearing logs for next iteration
def clear_logs(tree):
    return map_logs(lambda x: None, tree)

state = clear_logs(state)  # All Log objects now contain None
```
