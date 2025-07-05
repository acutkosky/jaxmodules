import pytest
import jax
import jax.numpy as jnp
import numpy as np
import einops
from jaxmodules.vectorize import einsum as jaxmodules_einsum


def test_einsum_basic_operations():
    """Test basic einsum operations comparing jaxmodules.einsum to einops.einsum"""
    # Set JAX to use CPU with 64-bit precision
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    
    # Test cases with different einsum patterns
    test_cases = [
        # Matrix multiplication
        {
            'pattern': 'i j, j k -> i k',
            'shapes': [(3, 4), (4, 5)],
            'description': 'Matrix multiplication'
        },
        # Batch matrix multiplication
        {
            'pattern': 'b i j, b j k -> b i k',
            'shapes': [(2, 3, 4), (2, 4, 5)],
            'description': 'Batch matrix multiplication'
        },
        # Outer product
        {
            'pattern': 'i, j -> i j',
            'shapes': [(3,), (4,)],
            'description': 'Outer product'
        },
        # Tensor contraction
        {
            'pattern': 'i j k, j l m -> i k l m',
            'shapes': [(2, 3, 4), (3, 5, 6)],
            'description': 'Tensor contraction'
        },
        # Missing output axes
        {
            'pattern': 'i j, k l -> i l',
            'shapes': [(2, 3), (3, 4)],
            'description': 'Missing output axes'
        },
        # Diagonal extraction
        {
            'pattern': 'i i -> i',
            'shapes': [(4, 4)],
            'description': 'Diagonal extraction'
        },
        # Trace
        {
            'pattern': 'i i ->',
            'shapes': [(3, 3)],
            'description': 'Trace'
        },
        # Element-wise multiplication with broadcasting
        {
            'pattern': 'i j, i -> i j',
            'shapes': [(3, 4), (3,)],
            'description': 'Element-wise multiplication with broadcasting'
        },
        # Complex contraction
        {
            'pattern': 'i j k, k l, l m -> i j m',
            'shapes': [(2, 3, 4), (4, 5), (5, 6)],
            'description': 'Complex contraction'
        },
        # Transpose
        {
            'pattern': 'i j -> j i',
            'shapes': [(3, 4)],
            'description': 'Transpose'
        },
        # Sum over specific dimensions
        {
            'pattern': 'i j k -> i k',
            'shapes': [(2, 3, 4)],
            'description': 'Sum over middle dimension'
        }
    ]

    # test_cases = [test_cases[1]]
    
    for test_case in test_cases:
        pattern = test_case['pattern']
        shapes = test_case['shapes']
        description = test_case['description']
        
        # Generate random inputs with appropriate shapes
        key = jax.random.PRNGKey(42)
        inputs = []
        for i, shape in enumerate(shapes):
            key, subkey = jax.random.split(key)
            # Use smaller values to avoid numerical issues
            input_array = jax.random.uniform(subkey, shape, minval=-1.0, maxval=1.0)
            inputs.append(input_array)
        
        # Compute with jaxmodules.einsum
        try:
            result_jaxmodules = jaxmodules_einsum(*inputs, pattern)
        except Exception as e:
            pytest.fail(f"jaxmodules.einsum failed for {description}: {e}")
        
        # Compute with einops.einsum
        try:
            result_einops = einops.einsum(*inputs, pattern)
        except Exception as e:
            pytest.fail(f"einops.einsum failed for {description}: {e}")
        
        # Compare results
        assert result_jaxmodules.shape == result_einops.shape, \
            f"Shape mismatch for {description}: {result_jaxmodules.shape} vs {result_einops.shape}"
        
        # Use more lenient tolerance for floating point comparisons
        assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10), \
            f"Result mismatch for {description}"


def test_einsum_edge_cases():
    """Test edge cases and special patterns"""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    
    # Test with single element
    a = jnp.array([[1.0]])
    result_jaxmodules = jaxmodules_einsum(a, 'i j -> i j')
    result_einops = einops.einsum(a, 'i j -> i j')
    assert jnp.allclose(result_jaxmodules, result_einops)
    
    # Test with zero dimensions (scalar)
    a = jnp.array(5.0)
    result_jaxmodules = jaxmodules_einsum(a, '->')
    result_einops = einops.einsum(a, '->')
    assert jnp.allclose(result_jaxmodules, result_einops)
    
    # Test with ellipsis
    a = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, 5))
    b = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 4, 5))
    
    result_jaxmodules = jaxmodules_einsum(a, b, '... i, ... i -> ...')
    result_einops = einops.einsum(a, b, '... i, ... i -> ...')
    assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10)


def test_einsum_numerical_stability():
    """Test numerical stability with various input ranges"""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    
    # Test with very small values
    a = jnp.array([[1e-10, 1e-12], [1e-14, 1e-16]])
    b = jnp.array([[1e-8, 1e-10], [1e-12, 1e-14]])
    
    result_jaxmodules = jaxmodules_einsum(a, b, 'i j, j k -> i k')
    result_einops = einops.einsum(a, b, 'i j, j k -> i k')
    assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10)


    # Test with very large values
    a = jnp.array([[1e10, 1e12], [1e14, 1e16]])
    b = jnp.array([[1e8, 1e10], [1e12, 1e14]])
    
    result_jaxmodules = jaxmodules_einsum(a, b, 'i j, j k -> i k')
    result_einops = einops.einsum(a, b, 'i j, j k -> i k')
    assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10)

    


def test_einsum_complex_patterns():
    """Test more complex einsum patterns"""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    
    # Test with multiple contractions
    a = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    b = jax.random.normal(jax.random.PRNGKey(1), (3, 5, 6))
    c = jax.random.normal(jax.random.PRNGKey(2), (4, 6, 7))
    
    result_jaxmodules = jaxmodules_einsum(a, b, c, 'i j k, j l m, k m n -> i l n')
    result_einops = einops.einsum(a, b, c, 'i j k, j l m, k m n -> i l n')
    assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10)
    
    # Test with repeated indices
    a = jax.random.normal(jax.random.PRNGKey(3), (3, 4))
    b = jax.random.normal(jax.random.PRNGKey(4), (4, 3))
    
    result_jaxmodules = jaxmodules_einsum(a, b, 'i j, j i ->')
    result_einops = einops.einsum(a, b, 'i j, j i ->')
    assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10)


def test_einsum_error_cases():
    """Test that appropriate errors are raised for invalid patterns"""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    
    a = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    b = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    a = jnp.astype(a, jnp.float64)
    
    # Test with mismatched dimensions
    with pytest.raises(Exception):
        jaxmodules_einsum(a, b, 'i j, j l -> i l')
    
    # Test with invalid pattern syntax
    with pytest.raises(Exception):
        jaxmodules_einsum(a, b, 'invalid pattern')


def test_einsum_performance_comparison():
    """Test performance comparison between the two implementations"""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    # Use larger tensors to see performance differences
    a = jax.random.normal(jax.random.PRNGKey(0), (10, 10, 10))
    b = jax.random.normal(jax.random.PRNGKey(1), (10, 10, 10))
    
    # Time both implementations
    import time
    
    # Warm up
    jmein = jax.jit(jaxmodules_einsum, static_argnums=2)
    jeein = jax.jit(einops.einsum, static_argnums=2)
    _ = jmein(a, b, 'i j k, j k l -> i l')
    _ = jeein(a, b, 'i j k, j k l -> i l')
    
    # Time jaxmodules.einsum
    start_time = time.time()
    result_jaxmodules = jmein(a, b, 'i j k, j k l -> i l')
    jaxmodules_time = time.time() - start_time
    
    # Time einops.einsum
    start_time = time.time()
    result_einops = jeein(a, b, 'i j k, j k l -> i l')
    einops_time = time.time() - start_time

    print("result_jaxmodules: ", jnp.sum((result_jaxmodules-result_einops)**2))
    
    
    print(f"jaxmodules.einsum time: {jaxmodules_time:.4f}s")
    print(f"einops.einsum time: {einops_time:.4f}s")
    print(f"Speedup: {einops_time / jaxmodules_time:.2f}x") 


    # Verify results match
    assert jnp.allclose(result_jaxmodules, result_einops, rtol=1e-10, atol=1e-10)


def test_einsum_accuracy_comparison():
    """Test accuracy comparison between the two implementations"""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_default_dtype_bits', '64')
    jax.config.update('jax_enable_x64', True)
    # Use larger tensors to see performance differences
    a = jax.random.normal(jax.random.PRNGKey(0), (10, 10, 10))
    b = jax.random.normal(jax.random.PRNGKey(1), (10, 10, 10))

    # Warm up
    jmein = jax.jit(jaxmodules_einsum, static_argnums=2)
    jeein = jax.jit(einops.einsum, static_argnums=2)
    _ = jmein(a, b, 'i j k, j k l -> i l')
    _ = jeein(a, b, 'i j k, j k l -> i l')
    
    result_einops = jeein(a, b, 'i j k, j k l -> i l')

    jax.config.update('jax_platform_name', 'cuda')
    jax.config.update('jax_default_dtype_bits', '32')
    jax.config.update('jax_enable_x64', False)
    gpus = jax.devices("gpu")

    a = jax.device_put(jnp.astype(a, jnp.float32), gpus[0])
    b = jax.device_put(jnp.astype(b, jnp.float32), gpus[0])

    jmein = jax.jit(jaxmodules_einsum, static_argnums=2)
    jeein = jax.jit(einops.einsum, static_argnums=2)
    _ = jmein(a, b, 'i j k, j k l -> i l')
    _ = jeein(a, b, 'i j k, j k l -> i l')

    jmein = jax.jit(jaxmodules_einsum, static_argnums=2)
    jeein = jax.jit(einops.einsum, static_argnums=2)
    result_jaxmodules_32 = jmein(a, b, 'i j k, j k l -> i l')
    result_einops_32 = jeein(a, b, 'i j k, j k l -> i l')
    assert jnp.sum((result_jaxmodules_32 - result_einops)**2) < jnp.sum((result_einops_32 - result_einops)**2)


