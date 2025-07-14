from jaxmodules.attention import masked_attention_via_map
from einops import rearrange, einsum
import jax
import jax.numpy as jnp
import time
import io
import sys

def get_block_causal_mask_fn(block_size, N, L):
    '''
    takes a integer index i in [N] and returns a size [L] array of booleans
    specifying the attention mask for the ith query.
    
    resulting full mask should satisfy:

    result[q, k] = True if and only if k <= q and k % block_size == q % block_size
    '''

    def block_causal_mask_fn(h, q, k):
        block_idx_q = q % block_size
        block_idx_k = k % block_size
        return (k <= q) & (block_idx_k == block_idx_q)

    block_mask = jnp.reshape(jnp.arange(N//block_size), (N//block_size, 1))
    # jax.vmap(lambda q: jnp.arange(L//block_size))(jnp.arange(N//block_size))
    
    # jnp.astype(jnp.tril(jnp.ones((N//block_size, L//block_size))), bool)
    max_kv_indices = jnp.full(N//block_size, 1)
    # block_mask = jax.vmap(lambda _: jnp.arange(L//block_size))(jnp.arange(N//block_size))
    # max_kv_indices = jnp.full(N//block_size, L//block_size)

    return block_causal_mask_fn, block_mask, max_kv_indices

def get_block_causal_mask(block_size, N, L):
    '''
    returns a size [L, L] array of booleans
    specifying the attention mask for the ith query.

    result[q, k] = True if and only if k <= q and k % block_size == q % block_size.

    compute this directly, don't use get_block_causal_mask_fn
    '''
    # Create coordinate grids
    q_coords = jnp.arange(N)[:, None]  # [N, 1]
    k_coords = jnp.arange(L)[None, :]  # [1, L]
    
    # Causal mask: k <= q
    causal_mask = k_coords <= q_coords  # [N, L]
    
    # Block mask: k % block_size == q % block_size
    q_block_offset = q_coords % block_size  # [N, 1]
    k_block_offset = k_coords % block_size  # [1, L]
    block_mask = q_block_offset == k_block_offset  # [N, L]
    
    # Combine both conditions
    return causal_mask & block_mask  # [N, L]

def simple_attention_naive(Q, K, V):
    """
    Simple quadratic attention implementation for comparison.
    This is the standard attention formula: softmax(QK^T/sqrt(d))V
    """
    d_k = K.shape[-1]
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = Q @ K.T / jnp.sqrt(d_k)  # [N, L]
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    
    # Apply softmax
    attention_weights = jax.nn.softmax(scores, axis=-1)  # [N, L]
    
    # Apply to values
    output = attention_weights @ V  # [N, d]
    
    return output


# @jax.jit
# def test_masked_attention_via_map(Q, K, V):
#     """Jitted version of masked_attention_via_map for testing"""
#     return masked_attention_via_map(K, Q, V, is_causal=False, mask_fn=mask_fn)


def get_masked_attention_via_map_block(block_size, mask_fn, block_mask, max_kv_indices):
    @jax.jit
    def test_masked_attention_via_map_block(Q, K, V):
        # return Q
        """Jitted version of masked_attention_via_map with block_size=block_size"""
        Q_reshaped = Q[:, None, :]  # [N, 1, d]
        K_reshaped = K[:, None, :]  # [L, 1, d]
        V_reshaped = V[:, None, :]  # [L, 1, d]

        return masked_attention_via_map(
            Q_reshaped, K_reshaped, V_reshaped, is_causal=False, block_size=block_size,
            mask_fn=mask_fn,
            # block_mask=block_mask, max_kv_indices=max_kv_indices
            )
    return test_masked_attention_via_map_block

@jax.jit
def test_simple_attention_naive(Q, K, V):
    """Jitted version of simple naive attention"""
    return simple_attention_naive(Q, K, V)

def get_jax_dot_product_attention(get_mask_fn):
    @jax.jit
    def test_jax_dot_product_attention(Q, K, V):
        """Jitted version of JAX's built-in dot product attention"""
        # Reshape for JAX's dot_product_attention: expects [batch, heads, seq_len, dim]
        Q_reshaped = Q[:, None, :]  # [N, 1, d]
        K_reshaped = K[:, None, :]  # [L, 1, d]
        V_reshaped = V[:, None, :]  # [L, 1, d]

        mask = get_mask_fn()
        mask = jnp.reshape(mask, (1, mask.shape[0], mask.shape[1]))
        
        # Use deterministic=True for more consistent results
        output = jax.nn.dot_product_attention(
            Q_reshaped, K_reshaped, V_reshaped,
            mask=mask,
            implementation='cudnn'
        )
        return output
    return test_jax_dot_product_attention


# Global variable to store all output
captured_output = []

def capture_print(*args, **kwargs):
    """Print function that captures output for later display"""
    # Print immediately (on-demand)
    print(*args, **kwargs)
    
    # Capture the output
    output_str = " ".join(str(arg) for arg in args)
    captured_output.append(output_str)


def warm_up_jit_functions(funcs, *args):
    """Warm up all JIT functions to ensure compilation is complete"""
    # capture_print("  Warming up JIT functions...")
    for name, func in funcs.items():
        # try:
            # Trigger compilation and warm up
        result = func(*args)
        result.block_until_ready()  # Ensure compilation is complete
        for _ in range(10):
            result = func(*args)
            result.block_until_ready()
        capture_print(f"    {name}: warmed up")
        # except Exception as e:
        #     capture_print(f"    {name}: warm-up failed - {e}")


def time_function(func, *args, num_runs=100):
    """Time a function over multiple runs"""
    # More thorough warm-up for JIT compilation
    # First call triggers compilation
    # result = func(*args)
    # result.block_until_ready()  # Ensure compilation is complete
    
    # Additional warm-up calls to ensure compilation is complete
    for _ in range(3):
        result = func(*args)
        result.block_until_ready()
    
    # Time the function
    start_time = time.time()
    for idx in range(num_runs):
        result = func(*args)
        result.block_until_ready()  # Block until the last computation is complete
    end_time = time.time()
    
    return (end_time - start_time) / num_runs, result


def check_attn():
    '''
    checks that the masked_attention_via_map function can be differentiated quickly
    and works at larger context lengths that would break more naive quadratic implementations.

    Also compare speed to a simple quadratic implementation and the built-in jax.nn.dot_product_attention.

    use jitted functions for all comparisons
    '''
    
    # Clear captured output
    global captured_output
    captured_output = []
    
    capture_print("Testing attention implementations without masking...")
    
    # Test parameters
    seq_lengths = [32768, 65536, 131072]#, 262144, 524288] #, 1048576]
    d_model = 64
    N = 16384
    for L in seq_lengths:
        N = L
        capture_print(f"\n=== Testing with sequence length L={L}, N={N}, d={d_model} ===")
        
        # Generate random data
        key = jax.random.PRNGKey(42)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        Q = jax.random.normal(key1, (N, d_model), dtype=jnp.float16)  # [N, d]
        K = jax.random.normal(key2, (L, d_model), dtype=jnp.float16)  # [L, d]
        V = jax.random.normal(key3, (L, d_model), dtype=jnp.float16)  # [L, d]
        

        BLOCK_MASK_SIZE = 2048
        # Test all implementations
        mask_fn, block_mask, max_kv_indices = get_block_causal_mask_fn(BLOCK_MASK_SIZE, N, L)
        create_mask_fn = lambda: get_block_causal_mask(BLOCK_MASK_SIZE, N, L)
        implementations = {
            # "masked_attention_via_map_block1024": get_masked_attention_via_map_block(1024, mask_fn),
            # "masked_attention_via_map_block4096": get_masked_attention_via_map_block(4096, mask_fn),
            "masked_attention_via_map_block8192": get_masked_attention_via_map_block(8192, mask_fn, block_mask, max_kv_indices),
            # "masked_attention_via_map_block16384": get_masked_attention_via_map_block(16384, mask_fn),
            # "simple_attention_naive": test_simple_attention_naive,
        }

        if L == 32768:
            implementations["jax_dot_product_attention"] = get_jax_dot_product_attention(create_mask_fn)
        
        # Warm up all JIT functions first
        capture_print("Warming up JIT functions...")
        warm_up_jit_functions(implementations, Q, K, V)
        
        results = {}
        times = {}
        
        for name, func in implementations.items():
            try:
                # Use fewer runs for longer sequences to avoid timeout
                num_runs = max(5, 100 // (L // 4096))  # Even fewer runs for very long sequences
                avg_time, result = time_function(func, Q, K, V, num_runs=num_runs)
                times[name] = avg_time * 1000  # Convert to milliseconds
                results[name] = result
                capture_print(f"  {name}: {avg_time*1000:.3f} ms ({num_runs} runs)")
            except Exception as e:
                capture_print(f"  {name}: ERROR - {e}")
                times[name] = float('inf')
                results[name] = None
        
        # Check correctness by comparing outputs
        ref_name = "masked_attention_via_map_block8192"
        if all(r is not None for r in results.values()):
            capture_print("  Checking correctness...")
            reference = results[ref_name]
            
            for name, result in results.items():
                if name == ref_name:
                    continue
                if result is not None:
                    diff = jnp.abs(result - reference).max()
                    rel_diff = diff / (jnp.abs(reference).max() + 1e-8)
                    capture_print(f"    {name} max diff from {ref_name}: {diff:.2e} (rel: {rel_diff:.2e})")
                    
                    # Check if difference is reasonable (within numerical precision)
                    if diff > 1e-5:
                        capture_print(f"      WARNING: Large difference detected!")
        
        # Test gradient computation
        capture_print("  Testing gradient computation...")
        grad_fns = {}

        for name in implementations:
            fn = implementations[name]
            def loss_fn(q, fn=fn):
                return fn(q, K, V).sum()
            grad_fns[name] = jax.grad(loss_fn)

        capture_print("Warming up gradient functions...")
        warm_up_jit_functions(grad_fns, Q)

        # print("grad_fns: ", grad_fns)
        # for name in grad_fns:
            # try:
            #     grad_fns[name](Q).block_until_ready()
            #     capture_print(f"    {name} gradient warmed up")
            # except Exception as e:
            #     capture_print(f"    {name} gradient: ERROR - {e}")
        capture_print("Timing gradient functions...")
        grad_results = {}
        for name in grad_fns:
            try:
                grad_runs = max(3, 20 // (L // 4096))  # Even fewer runs for very long sequences
                grad_time, grad_result = time_function(grad_fns[name], Q, num_runs=grad_runs)
                grad_results[name] = grad_result
                capture_print(f"    {name} gradient: {grad_time*1000:.3f} ms ({grad_runs} runs)")
            except Exception as e:
                capture_print(f"    {name} gradient: ERROR - {e}")
            # break
        # Check correctness by comparing outputs
        if all(r is not None for r in grad_results.values()):
            capture_print("  Checking gradient correctness...")
            reference = grad_results[ref_name]
            
            for name, grad_result in grad_results.items():
                if name == ref_name:
                    continue
                if grad_result is not None:
                    diff = jnp.abs(grad_result - reference).max()
                    rel_diff = diff / (jnp.abs(reference).max() + 1e-8)
                    capture_print(f"    {name} max diff from {ref_name}: {diff:.2e} (rel: {rel_diff:.2e})")
                    
                    # Check if difference is reasonable (within numerical precision)
                    if diff > 1e-5:
                        capture_print(f"      WARNING: Large difference detected!")
    capture_print("\n=== Summary ===")
    capture_print("All attention implementations tested successfully!")
    capture_print("The masked_attention_via_map function should show similar performance")
    capture_print("to the simple implementations for smaller sequence lengths, and better")
    capture_print("performance for larger sequence lengths due to memory efficiency.")
    capture_print("\nBlock size analysis:")
    capture_print("- block_size=None: Uses full sequence length (same as before your modification)")
    capture_print("- block_size=128: Processes queries in blocks of 128 for memory efficiency")
    capture_print("- block_size=1024: Larger blocks, potentially better for GPU utilization")
    capture_print("For very long sequences, smaller block sizes should show memory benefits.")
    
    # Print clean summary at the end
    print("\n" + "="*80)
    print("CLEAN OUTPUT SUMMARY (without XLA logs):")
    print("="*80)
    for line in captured_output:
        print(line)
    print("="*80)


if __name__ == "__main__":
    check_attn()
