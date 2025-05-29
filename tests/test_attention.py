import pytest
import jax
import jax.numpy as jnp
import torch
from jaxmodules.attention import flex_attention, flex_attention_slow
from jaxmodules.block_mask import BlockMask, create_block_mask

def test_flex_attention_basic():
    """Test basic functionality of flex_attention with simple inputs"""
    B, H, L, E = 2, 4, 8, 16
    BLOCK_SIZE = 4
    
    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))
    
    # Test without block mask
    output = flex_attention(query, key, value)
    output_slow = flex_attention_slow(query, key, value)
    
    # Check shapes
    assert output.shape == (B, H, L, E)
    assert output_slow.shape == (B, H, L, E)
    
    # Check that fast and slow implementations match
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)

def test_flex_attention_gqa():
    """Test flex_attention with grouped query attention (GQA)"""
    B, Hkv, Hq, L, E = 2, 2, 4, 8, 16
    BLOCK_SIZE = 4
    
    # Create random inputs with different number of heads
    key = jax.random.normal(jax.random.PRNGKey(0), (B, Hkv, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, Hq, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, Hkv, L, E))
    
    # Test with GQA enabled
    output = flex_attention(query, key, value, enable_gqa=True)
    output_slow = flex_attention_slow(query, key, value, enable_gqa=True)
    
    # Check shapes
    assert output.shape == (B, Hq, L, E)
    assert output_slow.shape == (B, Hq, L, E)
    
    # Check that fast and slow implementations match
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)

def test_flex_attention_with_block_mask():
    """Test flex_attention with different block mask patterns"""
    B, H, L, E = 2, 4, 8, 16
    BLOCK_SIZE = 4
    
    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))
    
    # Create different block mask patterns
    def causal_mask(b, h, q_idx, k_idx):
        return q_idx >= k_idx
    
    def sliding_window_mask(b, h, q_idx, k_idx):
        return abs(q_idx - k_idx) <= 2
    
    def alternating_mask(b, h, q_idx, k_idx):
        return (q_idx + k_idx) % 2 == 0
    
    # Test with different mask patterns
    for mask_fn in [causal_mask, sliding_window_mask, alternating_mask]:
        block_mask = create_block_mask(mask_fn, B, H, L, L, BLOCK_SIZE)
        
        output = flex_attention(query, key, value, block_mask=block_mask)
        output_slow = flex_attention_slow(query, key, value, block_mask=block_mask)
        
        # Check shapes
        assert output.shape == (B, H, L, E)
        assert output_slow.shape == (B, H, L, E)
        
        # Check that fast and slow implementations match
        assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)

def test_flex_attention_with_score_mod():
    """Test flex_attention with score modification function"""
    B, H, L, E = 2, 4, 8, 16
    BLOCK_SIZE = 4
    
    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))
    
    # Create a simple score modification function
    def score_mod(score, b, h, q_idx, k_idx):
        return score + 0.1 * (q_idx - k_idx)
    
    output = flex_attention(query, key, value, score_mod=score_mod)
    output_slow = flex_attention_slow(query, key, value, score_mod=score_mod)
    
    # Check shapes
    assert output.shape == (B, H, L, E)
    assert output_slow.shape == (B, H, L, E)
    
    # Check that fast and slow implementations match
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)

def test_flex_attention_edge_cases():
    """Test flex_attention with edge cases"""
    B, H, L, E = 2, 4, 8, 16
    BLOCK_SIZE = 4
    
    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))
    
    # Test with zero inputs
    zero_query = jnp.zeros_like(query)
    zero_key = jnp.zeros_like(key)
    zero_value = jnp.zeros_like(value)
    
    output = flex_attention(zero_query, zero_key, zero_value)
    output_slow = flex_attention_slow(zero_query, zero_key, zero_value)
    
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)
    
    # Test with very large inputs
    large_query = query * 1000
    large_key = key * 1000
    large_value = value * 1000
    
    output = flex_attention(large_query, large_key, large_value)
    output_slow = flex_attention_slow(large_query, large_key, large_value)
    
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)

def test_flex_attention_error_cases():
    """Test flex_attention with error cases"""
    B, H, L, E = 2, 4, 8, 16
    
    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))
    
    # Test with mismatched dimensions
    with pytest.raises(AssertionError):
        flex_attention(query, key[:, :, :L-1, :], value)
    
    # Test with invalid block size
    with pytest.raises(AssertionError):
        flex_attention(query, key, value, block_mask=BlockMask(
            B=B, H=H, Q_LEN=L, KV_LEN=L, Q_BLOCK_SIZE=3, KV_BLOCK_SIZE=4,
            kv_num_blocks=jnp.zeros((B, H, L//4)), kv_indices=jnp.zeros((B, H, L//4, 2)),
            q_num_blocks=jnp.zeros((B, H, L//4)), q_indices=jnp.zeros((B, H, L//4, 2)),
            full_kv_num_blocks=jnp.zeros((B, H, L//4)), full_kv_indices=jnp.zeros((B, H, L//4, 2)),
            full_q_num_blocks=jnp.zeros((B, H, L//4)), full_q_indices=jnp.zeros((B, H, L//4, 2)),
            mask_mod=lambda b, h, q, k: True
        )) 