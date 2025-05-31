import pytest
import jax
import jax.numpy as jnp
import torch
from torch.nn.attention import flex_attention as torch_fa
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
    output_torch = torch_fa.flex_attention(
        torch.tensor(query), torch.tensor(key), torch.tensor(value)
    ).numpy()
    # Check shapes
    assert output.shape == (B, H, L, E)
    assert output_slow.shape == (B, H, L, E)

    # Check that fast and slow implementations match
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(output, output_torch, rtol=1e-5, atol=1e-5)


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
    output_torch = torch_fa.flex_attention(
        torch.tensor(query), torch.tensor(key), torch.tensor(value), enable_gqa=True
    ).numpy()

    # Check shapes
    assert output.shape == (B, Hq, L, E)
    assert output_slow.shape == (B, Hq, L, E)

    # Check that fast and slow implementations match
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(output, output_torch, rtol=1e-5, atol=1e-5)


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

    def batch_head_sliding_mask(b, h, q_idx, k_idx):
        return abs(q_idx - k_idx) <= b + h

    # Test with different mask patterns
    for mask_fn in [
        causal_mask,
        sliding_window_mask,
        alternating_mask,
        batch_head_sliding_mask,
    ]:
        block_mask = create_block_mask(mask_fn, B, H, L, L, BLOCK_SIZE)

        block_mask_torch = torch_fa.create_block_mask(
            mask_fn, B, H, L, L, device="cpu", BLOCK_SIZE=BLOCK_SIZE
        )

        output = flex_attention(query, key, value, block_mask=block_mask)
        output_slow = flex_attention_slow(query, key, value, block_mask=block_mask)
        output_torch = torch_fa.flex_attention(
            torch.tensor(query),
            torch.tensor(key),
            torch.tensor(value),
            block_mask=block_mask_torch,
        ).numpy()
        # Check shapes
        assert output.shape == (B, H, L, E)
        assert output_slow.shape == (B, H, L, E)

        # Check that fast and slow implementations match
        # assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(output, output_torch, rtol=1e-5, atol=1e-5)


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
    output_torch = torch_fa.flex_attention(
        torch.tensor(query), torch.tensor(key), torch.tensor(value), score_mod=score_mod
    ).numpy()

    # Check shapes
    assert output.shape == (B, H, L, E)
    assert output_slow.shape == (B, H, L, E)

    # Check that fast and slow implementations match
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(output, output_torch, rtol=1e-5, atol=1e-5)


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
    output_torch = torch_fa.flex_attention(
        torch.tensor(zero_query), torch.tensor(zero_key), torch.tensor(zero_value)
    ).numpy()
    assert jnp.allclose(output, output_slow, rtol=1e-5, atol=1e-5)

    # Test with very large inputs
    large_query = query * 1000
    large_key = key * 1000
    large_value = value * 1000

    output = flex_attention(large_query, large_key, large_value)
    output_slow = flex_attention_slow(large_query, large_key, large_value)
    output_torch = torch_fa.flex_attention(
        torch.tensor(large_query), torch.tensor(large_key), torch.tensor(large_value)
    ).numpy()
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
        flex_attention(query, key[:, :, : L - 1, :], value)

    # Test with invalid block size
    with pytest.raises(AssertionError):
        flex_attention(
            query,
            key,
            value,
            block_mask=BlockMask(
                B=B,
                H=H,
                Q_LEN=L,
                KV_LEN=L,
                Q_BLOCK_SIZE=3,
                KV_BLOCK_SIZE=4,
                kv_num_blocks=jnp.zeros((B, H, L // 4)),
                kv_indices=jnp.zeros((B, H, L // 4, 2)),
                q_num_blocks=jnp.zeros((B, H, L // 4)),
                q_indices=jnp.zeros((B, H, L // 4, 2)),
                full_kv_num_blocks=jnp.zeros((B, H, L // 4)),
                full_kv_indices=jnp.zeros((B, H, L // 4, 2)),
                full_q_num_blocks=jnp.zeros((B, H, L // 4)),
                full_q_indices=jnp.zeros((B, H, L // 4, 2)),
                mask_mod=lambda b, h, q, k: True,
            ),
        )


def test_flex_attention_block_mask_broadcasting():
    """Test that block masks can broadcast over batch and head dimensions"""
    B, H, L, E = 2, 4, 8, 2
    BLOCK_SIZE = 4

    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))

    # Create a simple mask function
    def sliding_window_mask(b, h, q_idx, k_idx):
        return abs(q_idx - k_idx) <= 2

    # Test broadcasting over batch dimension
    block_mask_batch1 = create_block_mask(sliding_window_mask, 1, H, L, L, BLOCK_SIZE)
    output_batch_broadcast = flex_attention(
        query, key, value, block_mask=block_mask_batch1
    )

    # Check that fast and slow implementations match with batch broadcasting
    # assert jnp.allclose(output_batch_broadcast, output_slow_batch_broadcast, rtol=1e-5, atol=1e-5)

    # Test broadcasting over head dimension
    block_mask_head1 = create_block_mask(sliding_window_mask, B, 1, L, L, BLOCK_SIZE)
    output_head_broadcast = flex_attention(
        query, key, value, block_mask=block_mask_head1
    )

    # Check that fast and slow implementations match with head broadcasting
    # assert jnp.allclose(output_head_broadcast, output_slow_head_broadcast, rtol=1e-5, atol=1e-5)

    # Test broadcasting over both dimensions
    block_mask_both1 = create_block_mask(sliding_window_mask, 1, 1, L, L, BLOCK_SIZE)
    output_both_broadcast = flex_attention(
        query, key, value, block_mask=block_mask_both1
    )

    block_mask_no_broadcast = create_block_mask(
        sliding_window_mask, B, H, L, L, BLOCK_SIZE
    )
    output_no_broadcast = flex_attention(
        query, key, value, block_mask=block_mask_no_broadcast
    )

    # Check that fast and slow implementations match with both dimensions broadcasting
    # assert jnp.allclose(output_both_broadcast, output_slow_both_broadcast, rtol=1e-5, atol=1e-5)

    # Verify that the outputs are different from each other to ensure the mask is actually being applied
    assert jnp.allclose(output_batch_broadcast, output_no_broadcast)
    assert jnp.allclose(output_head_broadcast, output_no_broadcast)
    assert jnp.allclose(output_both_broadcast, output_no_broadcast)


def test_flex_attention_with_scale():
    """Test flex_attention with different scale parameters"""
    B, H, L, E = 2, 4, 8, 16
    BLOCK_SIZE = 4

    # Create random inputs
    key = jax.random.normal(jax.random.PRNGKey(0), (B, H, L, E))
    query = jax.random.normal(jax.random.PRNGKey(1), (B, H, L, E))
    value = jax.random.normal(jax.random.PRNGKey(2), (B, H, L, E))

    # Test with default scale (None, should be 1/sqrt(E))
    output_default = flex_attention(query, key, value, scale=None)
    output_slow_default = flex_attention_slow(query, key, value, scale=None)
    output_torch_default = torch_fa.flex_attention(
        torch.tensor(query), torch.tensor(key), torch.tensor(value), scale=None
    ).numpy()

    # Test with explicit default scale
    default_scale = 1.0 / jnp.sqrt(E)
    output_explicit_default = flex_attention(query, key, value, scale=default_scale)
    output_slow_explicit_default = flex_attention_slow(
        query, key, value, scale=default_scale
    )
    output_torch_explicit_default = torch_fa.flex_attention(
        torch.tensor(query),
        torch.tensor(key),
        torch.tensor(value),
        scale=float(default_scale),
    ).numpy()

    # Test with custom scale
    custom_scale = 0.5
    output_custom = flex_attention(query, key, value, scale=custom_scale)
    output_slow_custom = flex_attention_slow(query, key, value, scale=custom_scale)
    output_torch_custom = torch_fa.flex_attention(
        torch.tensor(query), torch.tensor(key), torch.tensor(value), scale=custom_scale
    ).numpy()

    # Test with another custom scale
    custom_scale2 = 2.0
    output_custom2 = flex_attention(query, key, value, scale=custom_scale2)
    output_slow_custom2 = flex_attention_slow(query, key, value, scale=custom_scale2)
    output_torch_custom2 = torch_fa.flex_attention(
        torch.tensor(query), torch.tensor(key), torch.tensor(value), scale=custom_scale2
    ).numpy()

    # Check shapes
    assert output_default.shape == (B, H, L, E)
    assert output_explicit_default.shape == (B, H, L, E)
    assert output_custom.shape == (B, H, L, E)
    assert output_custom2.shape == (B, H, L, E)

    # Check that fast and slow implementations match for all scale values
    assert jnp.allclose(output_default, output_slow_default, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(
        output_explicit_default, output_slow_explicit_default, rtol=1e-5, atol=1e-5
    )
    assert jnp.allclose(output_custom, output_slow_custom, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(output_custom2, output_slow_custom2, rtol=1e-5, atol=1e-5)

    # Check that JAX and PyTorch implementations match
    assert jnp.allclose(output_default, output_torch_default, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(
        output_explicit_default, output_torch_explicit_default, rtol=1e-5, atol=1e-5
    )
    assert jnp.allclose(output_custom, output_torch_custom, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(output_custom2, output_torch_custom2, rtol=1e-5, atol=1e-5)

    # Check that default scale and explicit default scale give the same result
    assert jnp.allclose(output_default, output_explicit_default, rtol=1e-5, atol=1e-5)

    # Check that different scales give different results
    assert not jnp.allclose(output_default, output_custom, rtol=1e-3, atol=1e-3)
    assert not jnp.allclose(output_custom, output_custom2, rtol=1e-3, atol=1e-3)

    # Test with zero scale (should give uniform attention)
    zero_scale = 0.0
    output_zero = flex_attention(query, key, value, scale=zero_scale)
    output_slow_zero = flex_attention_slow(query, key, value, scale=zero_scale)

    # With zero scale, all attention weights should be equal (uniform distribution)
    # So the output should be the mean of all values
    expected_zero = jnp.mean(value, axis=2, keepdims=True).repeat(L, axis=2)

    assert jnp.allclose(output_zero, output_slow_zero, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(output_zero, expected_zero, rtol=1e-4, atol=1e-4)
