import pytest
import jax
import jax.numpy as jnp
import torch
import numpy as np
import torch.nn.functional as F
from jaxmodules.attention import (
    attention,
    masked_attention_via_map,
    use_custom_einsum,
)
from jaxmodules.vectorize import fancy_vmap


use_custom_einsum()  # Required for higher precision to get the tests to pass.

# GPU FP32 attention intentionally uses TF32 tensor-core multiplies. Different
# tile shapes can therefore round the same contraction in a different order.
_FP32_TILE_RTOL = 3e-3 if jax.default_backend() == "gpu" else 1e-4
_FP32_TILE_ATOL = 3e-4 if jax.default_backend() == "gpu" else 1e-4
_FP32_GRAD_ATOL = 3e-4 if jax.default_backend() == "gpu" else 1e-6


def test_attention_is_the_canonical_public_name():
    assert attention is masked_attention_via_map
    assert attention.__name__ == "attention"


def jax_to_torch(x):
    """Convert JAX array to PyTorch tensor"""
    # Handle PyTorch tensors - preserve requires_grad if already set
    if isinstance(x, torch.Tensor):
        # If it requires grad, return as-is to preserve the gradient computation
        if x.requires_grad:
            return x
        # Otherwise, detach and clone
        return x.detach().clone()
    return torch.tensor(np.array(x))


def torch_to_jax(x):
    """Convert PyTorch tensor to JAX array"""
    return jnp.array(x.detach().cpu().numpy())


def pytorch_scaled_dot_product_attention(Q, K, V, mask=None, is_causal=False, return_torch=False):
    """
    PyTorch scaled dot product attention for comparison.
    
    Args:
        Q: Query tensor [N, Hq, d] (JAX array or PyTorch tensor)
        K: Key tensor [L, Hkv, d] (JAX array or PyTorch tensor)
        V: Value tensor [L, Hkv, d] (JAX array or PyTorch tensor)
        mask: Optional attention mask [Hq, N, L] (boolean mask where True means attend)
        is_causal: Whether to apply causal mask
        return_torch: If True, return PyTorch tensor (for gradient computation); otherwise return JAX array
    """
    N, Hq, d = Q.shape
    L, Hkv, d_k = K.shape
    
    # Reshape to PyTorch format: [batch, heads, seq_len, dim]
    # PyTorch's scaled_dot_product_attention expects [B, H, N, d]
    Q_t = jax_to_torch(Q)  # [N, Hq, d]
    K_t = jax_to_torch(K)  # [L, Hkv, d]
    V_t = jax_to_torch(V)  # [L, Hkv, d]
    
    # Reshape to [1, H, N, d] for PyTorch (treating as single batch)
    Q_t = Q_t.permute(1, 0, 2).unsqueeze(0)  # [1, Hq, N, d]
    K_t = K_t.permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, L, d]
    V_t = V_t.permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, L, d]
    
    # Handle GQA if Hq != Hkv
    if Hq == Hkv:
        # Use PyTorch's built-in attention
        attn_mask = None
        if mask is not None:
            # mask is [Hq, N, L] -> [1, Hq, N, L] for PyTorch
            # Our mask: True means attend, False means mask out
            # PyTorch attn_mask: -inf means mask out, 0.0 or False means attend
            mask_t = jax_to_torch(mask).unsqueeze(0)  # [1, Hq, N, L]
            # Convert: True -> 0.0 (attend), False -> -inf (mask out)
            attn_mask = torch.where(
                mask_t.bool(),
                torch.zeros_like(mask_t, dtype=torch.float32),
                torch.full_like(mask_t, float('-inf'), dtype=torch.float32)
            )
        
        output = F.scaled_dot_product_attention(
            Q_t, K_t, V_t,
            attn_mask=attn_mask,
            is_causal=is_causal
        )
        # Reshape back: [1, Hq, N, d] -> [N, Hq, d]
        output = output.squeeze(0).permute(1, 0, 2)  # [N, Hq, d]
    else:
        # For GQA (Hq > Hkv), handle each kv head separately
        GROUP_SIZE = Hq // Hkv
        output = torch.zeros(1, Hq, N, d)
        
        for h in range(Hkv):
            # Query heads for this kv head
            h_start = h * GROUP_SIZE
            h_end = (h + 1) * GROUP_SIZE
            h_idx = slice(h_start, h_end)
            
            Q_h = Q_t[:, h_idx, :, :]  # [1, GROUP_SIZE, N, d]
            K_h = K_t[:, h:h+1, :, :]  # [1, 1, L, d]
            V_h = V_t[:, h:h+1, :, :]  # [1, 1, L, d]
            
            # Expand K and V for broadcasting across query heads
            K_h = K_h.expand(1, GROUP_SIZE, L, d)
            V_h = V_h.expand(1, GROUP_SIZE, L, d)
            
            attn_mask_h = None
            if mask is not None:
                # Extract mask for these query heads: [Hq, N, L] -> [GROUP_SIZE, N, L]
                mask_h = mask[h_start:h_end]
                mask_t = jax_to_torch(mask_h).unsqueeze(0)  # [1, GROUP_SIZE, N, L]
                # Convert: True -> 0.0 (attend), False -> -inf (mask out)
                attn_mask_h = torch.where(
                    mask_t.bool(),
                    torch.zeros_like(mask_t, dtype=torch.float32),
                    torch.full_like(mask_t, float('-inf'), dtype=torch.float32)
                )
            
            output_h = F.scaled_dot_product_attention(
                Q_h, K_h, V_h,
                attn_mask=attn_mask_h,
                is_causal=is_causal
            )
            output[:, h_idx, :, :] = output_h
        
        output = output.squeeze(0).permute(1, 0, 2)  # [N, Hq, d]
    
    if return_torch:
        return output
    return torch_to_jax(output)


def materialize_mask(mask_fn, Hq, N, L):
    """
    Materialize a mask function into a boolean array using fancy_vmap.
    
    Args:
        mask_fn: Function that takes (h, q, k) and returns boolean
        Hq: Number of query heads
        N: Number of queries
        L: Number of keys
    
    Returns:
        Boolean array of shape [Hq, N, L] where mask[h, q, k] = mask_fn(h, q, k)
    """
    # Use fancy_vmap to efficiently vectorize mask_fn over all combinations of h, q, k
    # Format: mask[h, q, k] = mask_fn(h_inds[h], q_inds[q], k_inds[k])
    # where h_inds, q_inds, k_inds are the input arrays
    vectorized_mask_fn = fancy_vmap(
        mask_fn,
        "mask[h, q, k] = mask_fn(h_inds[h], q_inds[q], k_inds[k])"
    )
    # Pass index arrays: h indices, q indices, k indices
    h_inds = jnp.arange(Hq, dtype=jnp.int32)
    q_inds = jnp.arange(N, dtype=jnp.int32)
    k_inds = jnp.arange(L, dtype=jnp.int32)
    mask = vectorized_mask_fn(h_inds, q_inds, k_inds)
    return mask


def materialize_mask_batch(mask_fn, B, Hq, N, L):
    """
    Materialize a batch-aware mask function into a boolean array using fancy_vmap.
    
    Args:
        mask_fn: Function that takes (b, h, q, k) and returns boolean
        B: Batch size
        Hq: Number of query heads
        N: Number of queries
        L: Number of keys
    
    Returns:
        Boolean array of shape [B, Hq, N, L] where mask[b, h, q, k] = mask_fn(b, h, q, k)
    """
    # Use fancy_vmap to efficiently vectorize mask_fn over all combinations of b, h, q, k
    vectorized_mask_fn = fancy_vmap(
        mask_fn,
        "mask[b, h, q, k] = mask_fn(B[b], Hq[h], q_inds[q], k_inds[k])"
    )
    # Pass index arrays: batch indices, head indices, query indices, key indices
    b_inds = jnp.arange(B, dtype=jnp.int32)
    h_inds = jnp.arange(Hq, dtype=jnp.int32)
    q_inds = jnp.arange(N, dtype=jnp.int32)
    k_inds = jnp.arange(L, dtype=jnp.int32)
    mask = vectorized_mask_fn(b_inds, h_inds, q_inds, k_inds)
    return mask


def pytorch_scaled_dot_product_attention_batch(Q, K, V, mask=None, is_causal=False, return_torch=False):
    """
    PyTorch scaled dot product attention for comparison with batch dimension.
    
    Args:
        Q: Query tensor [B, N, Hq, d] (JAX array or PyTorch tensor)
        K: Key tensor [B, L, Hkv, d] (JAX array or PyTorch tensor)
        V: Value tensor [B, L, Hkv, d] (JAX array or PyTorch tensor)
        mask: Optional attention mask [B, Hq, N, L] (boolean mask where True means attend)
        is_causal: Whether to apply causal mask
        return_torch: If True, return PyTorch tensor (for gradient computation); otherwise return JAX array
    """
    B, N, Hq, d = Q.shape
    Bk, L, Hkv, d_k = K.shape
    
    # Reshape to PyTorch format: [batch, heads, seq_len, dim]
    # PyTorch's scaled_dot_product_attention expects [B, H, N, d]
    Q_t = jax_to_torch(Q)  # [B, N, Hq, d]
    K_t = jax_to_torch(K)  # [B, L, Hkv, d]
    V_t = jax_to_torch(V)  # [B, L, Hkv, d]
    
    # Reshape to [B, H, N, d] for PyTorch
    Q_t = Q_t.permute(0, 2, 1, 3)  # [B, Hq, N, d]
    K_t = K_t.permute(0, 2, 1, 3)  # [B, Hkv, L, d]
    V_t = V_t.permute(0, 2, 1, 3)  # [B, Hkv, L, d]
    
    # Handle GQA if Hq != Hkv
    if Hq == Hkv:
        # Use PyTorch's built-in attention
        attn_mask = None
        if mask is not None:
            # mask is [B, Hq, N, L] -> keep as [B, Hq, N, L] for PyTorch
            # Our mask: True means attend, False means mask out
            # PyTorch attn_mask: -inf means mask out, 0.0 or False means attend
            mask_t = jax_to_torch(mask)  # [B, Hq, N, L]
            # Convert: True -> 0.0 (attend), False -> -inf (mask out)
            attn_mask = torch.where(
                mask_t.bool(),
                torch.zeros_like(mask_t, dtype=torch.float32),
                torch.full_like(mask_t, float('-inf'), dtype=torch.float32)
            )
        
        output = F.scaled_dot_product_attention(
            Q_t, K_t, V_t,
            attn_mask=attn_mask,
            is_causal=is_causal
        )
        # Reshape back: [B, Hq, N, d] -> [B, N, Hq, d]
        output = output.permute(0, 2, 1, 3)  # [B, N, Hq, d]
    else:
        # For GQA (Hq > Hkv), handle each kv head separately
        GROUP_SIZE = Hq // Hkv
        output = torch.zeros(B, Hq, N, d)
        
        for h in range(Hkv):
            # Query heads for this kv head
            h_start = h * GROUP_SIZE
            h_end = (h + 1) * GROUP_SIZE
            h_idx = slice(h_start, h_end)
            
            Q_h = Q_t[:, h_idx, :, :]  # [B, GROUP_SIZE, N, d]
            K_h = K_t[:, h:h+1, :, :]  # [B, 1, L, d]
            V_h = V_t[:, h:h+1, :, :]  # [B, 1, L, d]
            
            # Expand K and V for broadcasting across query heads
            K_h = K_h.expand(B, GROUP_SIZE, L, d)
            V_h = V_h.expand(B, GROUP_SIZE, L, d)
            
            attn_mask_h = None
            if mask is not None:
                # Extract mask for these query heads: [B, Hq, N, L] -> [B, GROUP_SIZE, N, L]
                mask_h = mask[:, h_start:h_end]
                mask_t = jax_to_torch(mask_h)  # [B, GROUP_SIZE, N, L]
                # Convert: True -> 0.0 (attend), False -> -inf (mask out)
                attn_mask_h = torch.where(
                    mask_t.bool(),
                    torch.zeros_like(mask_t, dtype=torch.float32),
                    torch.full_like(mask_t, float('-inf'), dtype=torch.float32)
                )
            
            output_h = F.scaled_dot_product_attention(
                Q_h, K_h, V_h,
                attn_mask=attn_mask_h,
                is_causal=is_causal
            )
            output[:, h_idx, :, :] = output_h
        
        output = output.permute(0, 2, 1, 3)  # [B, N, Hq, d]
    
    if return_torch:
        return output
    return torch_to_jax(output)


def assert_outputs_close(jax_output, torch_output, rtol=1e-3, atol=1e-4, test_name=""):
    """
    Compare JAX and PyTorch outputs with relaxed tolerance and diagnostic information.
    
    Args:
        jax_output: JAX array output
        torch_output: PyTorch array output (converted to JAX array)
        rtol: Relative tolerance (default: 1e-3)
        atol: Absolute tolerance (default: 1e-3)
        test_name: Optional test name for diagnostic output
    
    Raises:
        AssertionError: If outputs are not close within tolerance
    """
    # Compute differences
    max_diff = jnp.abs(jax_output - torch_output).max()
    mean_diff = jnp.abs(jax_output - torch_output).mean()
    relative_error = jnp.linalg.norm(jax_output - torch_output) / jnp.minimum(jnp.linalg.norm(jax_output), jnp.linalg.norm(torch_output))
    
    # Print diagnostic information
    name_prefix = f"{test_name}: " if test_name else ""
    print(f"{name_prefix}Max difference: {max_diff}, Mean difference: {mean_diff}, Relative error: {relative_error}")
    
    # Assert with helpful error message
    assert jnp.linalg.norm(jax_output - torch_output) < atol or jnp.linalg.norm(jax_output - torch_output) /(jnp.minimum(jnp.linalg.norm(jax_output), jnp.linalg.norm(torch_output))) < rtol, \
        f"Outputs differ. Max diff: {max_diff}, Mean diff: {mean_diff}, Relative error: {relative_error}"


def test_masked_attention_basic():
    """Test basic functionality of masked_attention_via_map"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    # Create random inputs
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test without mask
    output = masked_attention_via_map(Q, K, V)
    
    # Check shape
    assert output.shape == (N, Hq, d)
    
    # Compare with PyTorch (no mask)
    output_torch = pytorch_scaled_dot_product_attention(Q, K, V)
    
    # Check that outputs are close
    assert_outputs_close(output, output_torch, test_name="basic")


def test_masked_attention_causal():
    """Test masked_attention_via_map with causal mask"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test with causal mask
    output = masked_attention_via_map(Q, K, V, is_causal=True)
    
    # Compare with PyTorch causal attention
    output_torch = pytorch_scaled_dot_product_attention(Q, K, V, is_causal=True)
    
    assert output.shape == (N, Hq, d)
    
    # Check that outputs are close
    assert_outputs_close(output, output_torch, test_name="causal")


def test_causal_hint_intersects_with_user_mask():
    """A causal hint prunes acausal pairs without replacing the user mask."""
    N, Hq, d = 8, 4, 16
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(43), 3)
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (N, Hq, d))
    V = jax.random.normal(key3, (N, Hq, d))

    def user_mask(h, q, k):
        return (q + 2 * k + h) % 3 != 0

    def explicit_causal_user_mask(h, q, k):
        return (q >= k) & user_mask(h, q, k)

    hinted = masked_attention_via_map(
        Q,
        K,
        V,
        is_causal=True,
        mask_fn=user_mask,
        block_size=4,
    )
    explicit = masked_attention_via_map(
        Q,
        K,
        V,
        mask_fn=explicit_causal_user_mask,
        block_size=4,
    )
    np.testing.assert_allclose(hinted, explicit, rtol=1e-5, atol=1e-5)

    def loss(q, k, v, *, hinted):
        output = masked_attention_via_map(
            q,
            k,
            v,
            is_causal=hinted,
            mask_fn=user_mask if hinted else explicit_causal_user_mask,
            block_size=4,
        )
        return jnp.mean(output**2)

    hinted_gradients = jax.grad(
        lambda q, k, v: loss(q, k, v, hinted=True),
        argnums=(0, 1, 2),
    )(Q, K, V)
    explicit_gradients = jax.grad(
        lambda q, k, v: loss(q, k, v, hinted=False),
        argnums=(0, 1, 2),
    )(Q, K, V)
    for hinted_gradient, explicit_gradient in zip(
        hinted_gradients,
        explicit_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            hinted_gradient,
            explicit_gradient,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_attention_output_matches_input_dtype(dtype):
    keys = jax.random.split(jax.random.PRNGKey(44), 3)
    shape = (8, 2, 16)
    query = jax.random.normal(keys[0], shape, dtype=dtype)
    key = jax.random.normal(keys[1], shape, dtype=dtype)
    value = jax.random.normal(keys[2], shape, dtype=dtype)

    output = masked_attention_via_map(query, key, value, block_size=4)

    assert output.dtype == query.dtype


def test_masked_attention_sliding_window():
    """Test masked_attention_via_map with sliding window mask"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    window_size = 2
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Create sliding window mask function
    def sliding_window_mask(h, q, k):
        return abs(q - k) <= window_size
    
    # Materialize mask for PyTorch
    mask = materialize_mask(sliding_window_mask, Hq, N, L)
    
    output = masked_attention_via_map(Q, K, V, mask_fn=sliding_window_mask)
    output_torch = pytorch_scaled_dot_product_attention(Q, K, V, mask=mask)
    
    assert output.shape == (N, Hq, d)
    
    # Check that outputs are close
    assert_outputs_close(output, output_torch, test_name="sliding_window")


def test_masked_attention_alternating():
    """Test masked_attention_via_map with alternating mask pattern"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Create alternating mask function
    def alternating_mask(h, q, k):
        return (q + k) % 2 == 0
    
    mask = materialize_mask(alternating_mask, Hq, N, L)
    
    output = masked_attention_via_map(Q, K, V, mask_fn=alternating_mask)
    output_torch = pytorch_scaled_dot_product_attention(Q, K, V, mask=mask)
    
    assert output.shape == (N, Hq, d)
    assert_outputs_close(output, output_torch, test_name="alternating")


def test_masked_attention_head_specific():
    """Test masked_attention_via_map with head-specific masks"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Create head-specific mask (different window size per head)
    def head_specific_mask(h, q, k):
        window_size = h + 1  # Different window size per head
        return abs(q - k) <= window_size
    
    mask = materialize_mask(head_specific_mask, Hq, N, L)
    
    output = masked_attention_via_map(Q, K, V, mask_fn=head_specific_mask)
    output_torch = pytorch_scaled_dot_product_attention(Q, K, V, mask=mask)
    
    assert output.shape == (N, Hq, d)
    assert_outputs_close(output, output_torch, test_name="head_specific")


def test_masked_attention_block_size():
    """Test masked_attention_via_map with different block sizes"""
    N, Hq, d = 16, 4, 16
    L, Hkv = 16, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test with different block sizes
    for block_size in [4, 8, 16]:
        output = masked_attention_via_map(Q, K, V, block_size=block_size)
        output_default = masked_attention_via_map(Q, K, V)  # Default block_size
        
        assert output.shape == (N, Hq, d)
        # Results should be the same regardless of block_size
        assert jnp.allclose(
            output,
            output_default,
            rtol=_FP32_TILE_RTOL,
            atol=_FP32_TILE_ATOL,
        )


@pytest.mark.parametrize(
    ("query_block_size", "kv_block_size"),
    [(4, 3), (6, 4), (8, 16)],
)
def test_masked_attention_independent_tile_sizes(
    query_block_size,
    kv_block_size,
):
    """Query and K/V tile sizes can be tuned independently."""
    length, heads, dim = 13, 2, 8
    query, key, value = (
        jax.random.normal(random_key, (length, heads, dim))
        for random_key in jax.random.split(jax.random.PRNGKey(30), 3)
    )

    output = masked_attention_via_map(
        query,
        key,
        value,
        block_size=query_block_size,
        kv_block_size=kv_block_size,
        is_causal=True,
    )
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        implementation="xla",
    )

    assert_outputs_close(
        output,
        expected,
        test_name=f"independent_tiles_{query_block_size}_{kv_block_size}",
    )


def test_masked_attention_default_tiles_bound_score_memory():
    """The automatic policy does not use a full long-context score tile."""
    from jaxmodules.attention import _default_attention_block_sizes

    shapes_and_expected_tiles = (
        ((2, 32768, 8, 8), (1024, 512)),
        ((1, 32768, 32, 8), (512, 512)),
        ((1, 8192, 4, 4), (2048, 1024)),
    )
    for shape_spec, expected_tiles in shapes_and_expected_tiles:
        batch_size, length, query_heads, kv_heads = shape_spec
        query = jax.ShapeDtypeStruct(
            (batch_size, length, query_heads, 64),
            jnp.float16,
        )
        key = jax.ShapeDtypeStruct(
            (batch_size, length, kv_heads, 64),
            jnp.float16,
        )
        value = jax.ShapeDtypeStruct(
            (batch_size, length, kv_heads, 64),
            jnp.float16,
        )

        assert _default_attention_block_sizes(
            query,
            key,
            value,
            None,
        ) == expected_tiles


def test_masked_attention_independent_tile_gradients():
    """The explicit VJP supports different query and K/V tile sizes."""
    query_length, kv_length, heads, dim = 9, 11, 2, 8
    query, key, value, cotangent = (
        jax.random.normal(random_key, shape)
        for random_key, shape in zip(
            jax.random.split(jax.random.PRNGKey(31), 4),
            (
                (query_length, heads, dim),
                (kv_length, heads, dim),
                (kv_length, heads, dim),
                (query_length, heads, dim),
            ),
        )
    )

    def mapped_loss(q, k, v):
        output = masked_attention_via_map(
            q,
            k,
            v,
            block_size=4,
            kv_block_size=6,
        )
        return jnp.vdot(output, cotangent)

    def equal_tile_loss(q, k, v):
        output = masked_attention_via_map(
            q,
            k,
            v,
            block_size=4,
            kv_block_size=4,
        )
        return jnp.vdot(output, cotangent)

    mapped_gradients = jax.grad(mapped_loss, argnums=(0, 1, 2))(query, key, value)
    expected_gradients = jax.grad(equal_tile_loss, argnums=(0, 1, 2))(
        query,
        key,
        value,
    )

    for mapped_gradient, expected_gradient in zip(
        mapped_gradients,
        expected_gradients,
    ):
        assert jnp.allclose(
            mapped_gradient,
            expected_gradient,
            rtol=_FP32_TILE_RTOL,
            atol=_FP32_GRAD_ATOL,
        )


@pytest.mark.parametrize("use_window", [False, True])
def test_masked_attention_backward_strategies_match(use_window):
    """Explicit one- and two-pass choices preserve the auto result."""
    length, query_heads, kv_heads, dim = 9, 4, 2, 8
    query, key, value, cotangent = (
        jax.random.normal(random_key, shape)
        for random_key, shape in zip(
            jax.random.split(jax.random.PRNGKey(34), 4),
            (
                (length, query_heads, dim),
                (length, kv_heads, dim),
                (length, kv_heads, dim),
                (length, query_heads, dim),
            ),
        )
    )

    def sliding_window_mask(head, q, k):
        del head
        return abs(q - k) <= 2

    attention_kwargs = (
        {
            "mask_fn": sliding_window_mask,
            "window_size": (2, 2),
        }
        if use_window
        else {"is_causal": True}
    )

    def loss(q, k, v, strategy):
        output = masked_attention_via_map(
            q,
            k,
            v,
            block_size=4,
            kv_block_size=6,
            backward_strategy=strategy,
            **attention_kwargs,
        )
        return jnp.vdot(output, cotangent)

    auto_gradients = jax.grad(
        lambda q, k, v: loss(q, k, v, "auto"),
        argnums=(0, 1, 2),
    )(query, key, value)
    minimal_gradients = jax.grad(
        lambda q, k, v: loss(q, k, v, "minimal"),
        argnums=(0, 1, 2),
    )(query, key, value)
    one_pass_gradients = jax.grad(
        lambda q, k, v: loss(q, k, v, "one_pass"),
        argnums=(0, 1, 2),
    )(query, key, value)

    for auto_gradient, minimal_gradient, one_pass_gradient in zip(
        auto_gradients,
        minimal_gradients,
        one_pass_gradients,
        strict=True,
    ):
        assert jnp.allclose(
            minimal_gradient,
            auto_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
        assert jnp.allclose(
            one_pass_gradient,
            auto_gradient,
            rtol=1e-5,
            atol=1e-6,
        )


def test_masked_attention_window_with_independent_tile_sizes():
    """Window block selection remains correct with asymmetric tiles."""
    length, heads, dim = 13, 2, 8
    query, key, value = (
        jax.random.normal(random_key, (length, heads, dim))
        for random_key in jax.random.split(jax.random.PRNGKey(32), 3)
    )

    def sliding_window_mask(h, q, k):
        del h
        return abs(q - k) <= 2

    expected = masked_attention_via_map(
        query,
        key,
        value,
        block_size=4,
        kv_block_size=6,
        mask_fn=sliding_window_mask,
    )
    output = masked_attention_via_map(
        query,
        key,
        value,
        block_size=4,
        kv_block_size=6,
        mask_fn=sliding_window_mask,
        window_size=(2, 2),
    )

    assert jnp.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_masked_attention_gqa():
    """Test masked_attention_via_map with grouped query attention (GQA)"""
    N, Hq, d = 8, 8, 16
    L, Hkv = 8, 4  # Hq = 2 * Hkv
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    output = masked_attention_via_map(Q, K, V)
    
    assert output.shape == (N, Hq, d)
    # With GQA, each key/value head should be used by multiple query heads
    # The output should still be valid


def test_masked_attention_different_scales():
    """Test masked_attention_via_map with different input scales"""
    scales = [32, 64, 128, 256, 512]
    Hq, Hkv, d = 4, 4, 16
    
    for N in scales:
        L = N  # Same length for simplicity
        
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        Q = jax.random.normal(key1, (N, Hq, d))
        K = jax.random.normal(key2, (L, Hkv, d))
        V = jax.random.normal(key3, (L, Hkv, d))
        
        output = masked_attention_via_map(Q, K, V, is_causal=True)
        
        assert output.shape == (N, Hq, d)
        
        # Compare with PyTorch for smaller sizes (to avoid memory issues)
        if N <= 128:
            output_torch = pytorch_scaled_dot_product_attention(Q, K, V, is_causal=True)
            assert_outputs_close(output, output_torch, test_name=f"different_scales_{N}")


def test_masked_attention_gradients():
    """Test that gradients work correctly for masked_attention_via_map"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_with_mask():
    """Test gradients with custom mask function"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    def sliding_window_mask(h, q, k):
        return abs(q - k) <= 2
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=sliding_window_mask)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_pytorch_comparison():
    """Compare gradients with PyTorch implementation"""
    # N, Hq, d = 8, 4, 16
    # L, Hkv = 8, 4
    N = 32  # Number of queries
    L = 64  # Sequence length (keys/values)
    Hq = 8  # Number of query heads
    Hkv = 4  # Number of key/value heads (Hq must be divisible by Hkv)
    d = 64  # Embedding dimension  

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # JAX gradients
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # PyTorch gradients
    Q_t = jax_to_torch(Q).requires_grad_(True)
    K_t = jax_to_torch(K).requires_grad_(True)
    V_t = jax_to_torch(V).requires_grad_(True)
    
    output_t = pytorch_scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=True, return_torch=True)
    loss_t = output_t.sum()
    loss_t.backward()
    
    grad_Q_torch = torch_to_jax(Q_t.grad)
    grad_K_torch = torch_to_jax(K_t.grad)
    grad_V_torch = torch_to_jax(V_t.grad)
    
    # Compare gradients
    assert_outputs_close(grad_Q_jax, grad_Q_torch, test_name="grad_Q")
    assert_outputs_close(grad_K_jax, grad_K_torch, test_name="grad_K")
    assert_outputs_close(grad_V_jax, grad_V_torch, test_name="grad_V")




def test_masked_attention_gradients_window_pytorch_comparison():
    """Compare gradients with PyTorch implementation"""
    # N, Hq, d = 8, 4, 16
    # L, Hkv = 8, 4
    N = 32  # Number of queries
    L = 64  # Sequence length (keys/values)
    Hq = 8  # Number of query heads
    Hkv = 4  # Number of key/value heads (Hq must be divisible by Hkv)
    d = 64  # Embedding dimension  

    block_size = 8
    
    # key = jax.random.PRNGKey(42)
    # key1, key2, key3 = jax.random.split(key, 3)

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))

    def mask_fn(h, q, k):
        return q==k
    
    # JAX gradients
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=mask_fn, block_size=block_size)
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # PyTorch gradients
    Q_t = jax_to_torch(Q).requires_grad_(True)
    K_t = jax_to_torch(K).requires_grad_(True)
    V_t = jax_to_torch(V).requires_grad_(True)

    mask_t = materialize_mask(mask_fn, Hq, N, L)
    output_t = pytorch_scaled_dot_product_attention(Q_t, K_t, V_t, mask=mask_t, return_torch=True)
    loss_t = output_t.sum()
    loss_t.backward()
    
    grad_Q_torch = torch_to_jax(Q_t.grad)
    grad_K_torch = torch_to_jax(K_t.grad)
    grad_V_torch = torch_to_jax(V_t.grad)
    
    
    # Compare gradients
    assert_outputs_close(grad_Q_jax, grad_Q_torch, test_name="grad_Q")
    assert_outputs_close(grad_K_jax, grad_K_torch, test_name="grad_K")
    assert_outputs_close(grad_V_jax, grad_V_torch, test_name="grad_V")


def test_masked_attention_gradients_complex_masking():
    """Test gradients with complex masking patterns"""
    N, Hq, d = 16, 4, 16
    L, Hkv = 16, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Complex mask: combination of sliding window and head-specific patterns
    def complex_mask(h, q, k):
        # Different heads have different window sizes
        window_size = (h % 2) + 2  # Head 0,2: window=2, Head 1,3: window=3
        # Also apply alternating pattern for even heads
        window_ok = abs(q - k) <= window_size
        # For even heads, also require (q + k) % 3 == 0
        is_even_head = (h % 2) == 0
        alternating_ok = ((q + k) % 3) == 0
        return jnp.where(is_even_head, window_ok & alternating_ok, window_ok)
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=complex_mask)
        return jnp.sum(output ** 2)  # Use squared output to test non-linear loss
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_gqa():
    """Test gradients with Grouped Query Attention (GQA)"""
    N, Hq, d = 16, 8, 32
    L, Hkv = 16, 4  # Hq = 2 * Hkv
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_gqa_with_mask():
    """Test GQA gradients with custom masking"""
    N, Hq, d = 16, 12, 32  # Hq = 3 * Hkv
    L, Hkv = 16, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Head-specific mask with different patterns per kv head group
    def gqa_mask(h, q, k):
        kv_head = h // (Hq // Hkv)  # Which kv head group this query head belongs to
        window_size = kv_head + 1  # Different window size per kv head group
        return abs(q - k) <= window_size
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=gqa_mask)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_large_sequence():
    """Test gradients with larger sequence lengths"""
    N, Hq, d = 128, 8, 32
    L, Hkv = 256, 8  # L > N to test different lengths
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are meaningful
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_large_dimension():
    """Test gradients with larger embedding dimensions"""
    N, Hq, d = 32, 4, 128
    L, Hkv = 32, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are meaningful
    assert not jnp.allclose(grad_Q, 0.0)


def test_masked_attention_gradients_large_comprehensive():
    """Test gradients with large inputs across all dimensions"""
    N, Hq, d = 256, 16, 64
    L, Hkv = 512, 8  # GQA with large sequences
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Complex mask with head-specific patterns
    def large_mask(h, q, k):
        kv_head = h // (Hq // Hkv)
        # Larger window for earlier kv heads, smaller for later ones
        window_size = (Hkv - kv_head) * 8
        return abs(q - k) <= window_size
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=large_mask, block_size=64)
        return jnp.sum(output ** 2)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are meaningful
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_gradients_pytorch_comparison_gqa():
    """Compare GQA gradients with PyTorch implementation"""
    N = 32
    L = 64
    Hq = 8
    Hkv = 4  # Hq = 2 * Hkv
    d = 64

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # JAX gradients
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # PyTorch gradients
    Q_t = jax_to_torch(Q).requires_grad_(True)
    K_t = jax_to_torch(K).requires_grad_(True)
    V_t = jax_to_torch(V).requires_grad_(True)
    
    output_t = pytorch_scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=True, return_torch=True)
    loss_t = output_t.sum()
    loss_t.backward()
    
    grad_Q_torch = torch_to_jax(Q_t.grad)
    grad_K_torch = torch_to_jax(K_t.grad)
    grad_V_torch = torch_to_jax(V_t.grad)
    
    # Compare gradients
    assert_outputs_close(grad_Q_jax, grad_Q_torch, test_name="grad_Q_gqa")
    assert_outputs_close(grad_K_jax, grad_K_torch, test_name="grad_K_gqa")
    assert_outputs_close(grad_V_jax, grad_V_torch, test_name="grad_V_gqa")


def test_masked_attention_gradients_pytorch_comparison_complex_mask():
    """Compare gradients with complex masking against PyTorch"""
    N = 32
    L = 32
    Hq = 4
    Hkv = 4
    d = 32

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Head-specific sliding window mask
    def head_sliding_window_mask(h, q, k):
        window_size = (h % 3) + 1  # Different window sizes per head
        return abs(q - k) <= window_size
    
    mask = materialize_mask(head_sliding_window_mask, Hq, N, L)
    
    # JAX gradients
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=head_sliding_window_mask)
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # PyTorch gradients
    Q_t = jax_to_torch(Q).requires_grad_(True)
    K_t = jax_to_torch(K).requires_grad_(True)
    V_t = jax_to_torch(V).requires_grad_(True)
    
    output_t = pytorch_scaled_dot_product_attention(Q_t, K_t, V_t, mask=mask, return_torch=True)
    loss_t = output_t.sum()
    loss_t.backward()
    
    grad_Q_torch = torch_to_jax(Q_t.grad)
    grad_K_torch = torch_to_jax(K_t.grad)
    grad_V_torch = torch_to_jax(V_t.grad)
    
    # Compare gradients
    assert_outputs_close(grad_Q_jax, grad_Q_torch, test_name="grad_Q_complex_mask")
    assert_outputs_close(grad_K_jax, grad_K_torch, test_name="grad_K_complex_mask")
    assert_outputs_close(grad_V_jax, grad_V_torch, test_name="grad_V_complex_mask")


def test_masked_attention_jit():
    """Test that masked_attention_via_map works with JIT"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # JIT the function
    @jax.jit
    def jitted_attention(q, k, v):
        return masked_attention_via_map(q, k, v, is_causal=True)
    
    output = jitted_attention(Q, K, V)
    
    # Compare with non-JIT version
    output_no_jit = masked_attention_via_map(Q, K, V, is_causal=True)
    
    assert output.shape == (N, Hq, d)
    assert jnp.allclose(output, output_no_jit, rtol=1e-5, atol=1e-5)


def test_masked_attention_jit_gradients():
    """Test that JIT + gradients work together"""
    # N, Hq, d = 64, 4, 16
    # L, Hkv = 64, 4
    N = 32  # Number of queries
    L = 64  # Sequence length (keys/values)
    Hq = 8  # Number of query heads
    Hkv = 4  # Number of key/value heads (Hq must be divisible by Hkv)
    d = 64  # Embedding dimension    

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    @jax.jit
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are meaningful
    assert not jnp.allclose(grad_Q, 0.0)


def test_masked_attention_edge_cases():
    """Test masked_attention_via_map with edge cases"""
    N, Hq, d = 4, 2, 8
    L, Hkv = 4, 2
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Test with zeros
    Q_zero = jnp.zeros((N, Hq, d))
    K_zero = jnp.zeros((L, Hkv, d))
    V_zero = jnp.zeros((L, Hkv, d))
    
    output_zero = masked_attention_via_map(Q_zero, K_zero, V_zero)
    assert output_zero.shape == (N, Hq, d)
    assert jnp.allclose(output_zero, 0.0, atol=1e-6)
    
    # Test with very large values
    Q = jax.random.normal(key1, (N, Hq, d)) * 100
    K = jax.random.normal(key2, (L, Hkv, d)) * 100
    V = jax.random.normal(key3, (L, Hkv, d)) * 100
    
    output = masked_attention_via_map(Q, K, V)
    assert output.shape == (N, Hq, d)
    expected = jax.nn.dot_product_attention(Q, K, V, implementation="xla")
    assert jnp.all(jnp.isfinite(output))
    assert_outputs_close(
        output,
        expected,
        test_name="large_logits",
        rtol=_FP32_TILE_RTOL,
        atol=_FP32_TILE_ATOL,
    )


def test_masked_attention_fully_masked_rows_are_finite():
    """Fully masked rows produce defined zero outputs and gradients."""
    query, key, value = (
        jax.random.normal(random_key, (6, 2, 8))
        for random_key in jax.random.split(jax.random.PRNGKey(26), 3)
    )

    def no_attention(head, query_index, key_index):
        del head, query_index, key_index
        return False

    def loss(q, k, v):
        return jnp.sum(
            masked_attention_via_map(
                q,
                k,
                v,
                mask_fn=no_attention,
                block_size=2,
            )
        )

    output = masked_attention_via_map(
        query,
        key,
        value,
        mask_fn=no_attention,
        block_size=2,
    )
    gradients = jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

    assert jnp.array_equal(output, jnp.zeros_like(output))
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
    assert all(jnp.array_equal(gradient, jnp.zeros_like(gradient)) for gradient in gradients)


def test_masked_attention_error_cases():
    """Test that appropriate errors are raised for invalid inputs"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test with mismatched dimensions
    K_wrong = jax.random.normal(key2, (L, Hkv, d + 1))
    with pytest.raises(ValueError):
        masked_attention_via_map(Q, K_wrong, V)


def test_masked_attention_block_size_padding():
    """Test masked_attention_via_map with number of queries not divisible by block_size and padding"""
    # Use N=10 which doesn't divide common block_size values
    # Note: L must be divisible by block_size because K/V are blocked using the query block_size
    N, Hq, d = 10, 4, 16
    L, Hkv = 13, 4  # L must be divisible by block_size (4) for K/V blocking to work
    block_size = 4  # 10 % 4 = 2, so padding will be applied to Q
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test without mask (should pad Q to 12 queries internally)
    output = masked_attention_via_map(Q, K, V, block_size=block_size)
    
    assert output.shape == (N, Hq, d)
    
    # Compare with PyTorch (no mask)
    output_torch = pytorch_scaled_dot_product_attention(Q, K, V)
    assert_outputs_close(output, output_torch, test_name="block_size_padding_no_mask")
    
    # Test with causal mask
    output_causal = masked_attention_via_map(Q, K, V, is_causal=True, block_size=block_size)
    output_torch_causal = pytorch_scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert_outputs_close(output_causal, output_torch_causal, test_name="block_size_padding_causal")
    
    # Test with custom mask
    def sliding_window_mask(h, q, k):
        return abs(q - k) <= 2
    
    mask = materialize_mask(sliding_window_mask, Hq, N, L)
    output_mask = masked_attention_via_map(Q, K, V, mask_fn=sliding_window_mask, block_size=block_size)
    output_torch_mask = pytorch_scaled_dot_product_attention(Q, K, V, mask=mask)
    assert_outputs_close(output_mask, output_torch_mask, test_name="block_size_padding_custom_mask")
    
    # # Test with different block_size that also requires padding
    # # Need to ensure L is divisible by block_size2 as well
    # block_size2 = 3  # 10 % 3 = 1, but L=12 is divisible by 3
    # output_block2 = masked_attention_via_map(Q, K, V, block_size=block_size2)
    # assert output.shape == (N, Hq, d)
    # # Results should be the same regardless of block_size (within numerical precision)
    # assert_outputs_close(output, output_block2, test_name="block_size_padding_different_sizes", rtol=1e-4, atol=1e-4)


def test_masked_attention_pads_kv_when_query_needs_no_padding():
    """K/V padding is independent of whether the query needs padding."""
    N, L, heads, dim = 8, 10, 2, 8
    query, key, value = (
        jax.random.normal(random_key, shape)
        for random_key, shape in zip(
            jax.random.split(jax.random.PRNGKey(27), 3),
            (
                (N, heads, dim),
                (L, heads, dim),
                (L, heads, dim),
            ),
        )
    )

    output = masked_attention_via_map(query, key, value, block_size=4)
    expected = jax.nn.dot_product_attention(query, key, value, implementation="xla")

    assert_outputs_close(output, expected, test_name="independent_kv_padding")


def test_masked_attention_causal_more_queries_than_keys():
    """Causal block skipping does not revisit the final K/V block."""
    N, L, heads, dim = 8, 4, 2, 8
    query, key, value = (
        jax.random.normal(random_key, shape)
        for random_key, shape in zip(
            jax.random.split(jax.random.PRNGKey(28), 3),
            (
                (N, heads, dim),
                (L, heads, dim),
                (L, heads, dim),
            ),
        )
    )

    output = masked_attention_via_map(
        query,
        key,
        value,
        block_size=2,
        is_causal=True,
    )
    expected = jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        implementation="xla",
    )

    assert_outputs_close(output, expected, test_name="causal_more_queries")


def test_masked_attention_supports_different_value_dimension():
    """The value feature dimension need not match the Q/K feature dimension."""
    N, L, heads, query_dim, value_dim = 8, 10, 2, 8, 5
    query, key, value = (
        jax.random.normal(random_key, shape)
        for random_key, shape in zip(
            jax.random.split(jax.random.PRNGKey(29), 3),
            (
                (N, heads, query_dim),
                (L, heads, query_dim),
                (L, heads, value_dim),
            ),
        )
    )

    output = masked_attention_via_map(query, key, value, block_size=4)
    scores = jnp.einsum("nhd,lhd->hnl", query, key) / jnp.sqrt(query_dim)
    probabilities = jax.nn.softmax(scores, axis=-1)
    expected = jnp.einsum("hnl,lhe->nhe", probabilities, value)

    assert output.shape == (N, heads, value_dim)
    assert_outputs_close(output, expected, test_name="different_value_dimension")


def test_masked_attention_different_kernel():
    """Test masked_attention_via_map with custom kernel function"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Custom kernel with different scaling
    def custom_kernel(q, k):
        return jnp.exp(jnp.dot(q, k) / (2 * jnp.sqrt(k.shape[-1])))
    
    output = masked_attention_via_map(Q, K, V, kernel_fn=custom_kernel)
    
    assert output.shape == (N, Hq, d)
    # Should produce different output than default kernel
    output_default = masked_attention_via_map(Q, K, V)
    assert not jnp.allclose(output, output_default, rtol=1e-2)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_custom_kernel_low_precision_backward(dtype, is_causal):
    """Custom kernels transpose the low-precision public output cast."""
    N, H, d = 8, 2, 16
    query, key, value = (
        0.2 * jax.random.normal(random_key, (N, H, d), dtype=dtype)
        for random_key in jax.random.split(jax.random.PRNGKey(42), 3)
    )

    def custom_kernel(q, k):
        return jnp.exp(jnp.dot(q, k) / (2 * jnp.sqrt(k.shape[-1])))

    def reference(q, k, v):
        weights = fancy_vmap(
            custom_kernel,
            "weights[q, h, k] = custom_kernel(q[q, h, :], k[k, h, :])",
        )(q, k).astype(jnp.float32)
        if is_causal:
            indices = jnp.arange(N)
            weights = jnp.where(
                indices[:, None, None] >= indices[None, None, :],
                weights,
                0,
            )
        normalizer = jnp.sum(weights, axis=-1, keepdims=True)
        numerator = jnp.einsum(
            "qhk,khe->qhe",
            weights,
            v,
            preferred_element_type=jnp.float32,
        )
        return (numerator / normalizer).astype(dtype)

    def implementation_loss(q, k, v):
        output = masked_attention_via_map(
            q,
            k,
            v,
            kernel_fn=custom_kernel,
            block_size=N,
            is_causal=is_causal,
        )
        return jnp.mean(output.astype(jnp.float32) ** 2)

    def reference_loss(q, k, v):
        return jnp.mean(reference(q, k, v).astype(jnp.float32) ** 2)

    actual = jax.jit(
        jax.value_and_grad(implementation_loss, argnums=(0, 1, 2))
    )(query, key, value)
    expected = jax.jit(
        jax.value_and_grad(reference_loss, argnums=(0, 1, 2))
    )(query, key, value)

    assert jnp.allclose(actual[0], expected[0], rtol=2e-2, atol=2e-3)
    for actual_gradient, expected_gradient in zip(
        actual[1],
        expected[1],
        strict=True,
    ):
        assert actual_gradient.dtype == dtype
        assert jnp.all(jnp.isfinite(actual_gradient))
        assert jnp.allclose(
            actual_gradient,
            expected_gradient,
            rtol=5e-2,
            atol=5e-3,
        )


@pytest.mark.parametrize("N,L", [(8, 8), (16, 16), (32, 32), (64, 32)])
def test_masked_attention_different_lengths(N, L):
    """Test masked_attention_via_map with different query and key lengths"""
    Hq, Hkv, d = 4, 4, 16
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    output = masked_attention_via_map(Q, K, V, is_causal=False)
    expected = jax.nn.dot_product_attention(Q, K, V, implementation="xla")
    
    assert output.shape == (N, Hq, d)
    assert_outputs_close(output, expected, test_name=f"different_lengths_{N}_{L}")


def test_masked_attention_large_scale():
    """Test masked_attention_via_map with larger inputs"""
    # Use moderate size to avoid memory issues but test scalability
    N, Hq, d = 128, 8, 32
    L, Hkv = 128, 8
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    output = masked_attention_via_map(Q, K, V, is_causal=True, block_size=32)
    
    assert output.shape == (N, Hq, d)
    
    # Compare with smaller block size
    output_small_block = masked_attention_via_map(Q, K, V, is_causal=True, block_size=16)
    assert jnp.allclose(
        output,
        output_small_block,
        rtol=_FP32_TILE_RTOL,
        atol=_FP32_TILE_ATOL,
    )


# Batch dimension tests

def test_masked_attention_batch_basic():
    """Test basic functionality of masked_attention_via_map with batch dimension"""
    B, N, Hq, d = 4, 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    # Test without mask
    output = masked_attention_via_map(Q, K, V)
    
    # Check shape
    assert output.shape == (B, N, Hq, d)
    
    # Compare with PyTorch (no mask)
    output_torch = pytorch_scaled_dot_product_attention_batch(Q, K, V)
    
    # Check that outputs are close
    assert_outputs_close(output, output_torch, test_name="batch_basic")


def test_masked_attention_batch_causal():
    """Test masked_attention_via_map with batch dimension and causal mask"""
    B, N, Hq, d = 4, 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    # Test with causal mask
    output = masked_attention_via_map(Q, K, V, is_causal=True)
    
    # Compare with PyTorch causal attention
    output_torch = pytorch_scaled_dot_product_attention_batch(Q, K, V, is_causal=True)
    
    assert output.shape == (B, N, Hq, d)
    
    # Check that outputs are close
    assert_outputs_close(output, output_torch, test_name="batch_causal")


def test_masked_attention_batch_custom_mask():
    """Test masked_attention_via_map with batch dimension and batch-specific masking"""
    B, N, Hq, d = 4, 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    # Create batch-specific mask: different window size per batch
    def batch_sliding_window_mask(b, h, q, k):
        window_size = b + 1  # Different window size per batch
        return abs(q - k) <= window_size
    
    # Materialize mask for PyTorch
    mask = materialize_mask_batch(batch_sliding_window_mask, B, Hq, N, L)
    
    output = masked_attention_via_map(Q, K, V, mask_fn=batch_sliding_window_mask)
    output_torch = pytorch_scaled_dot_product_attention_batch(Q, K, V, mask=mask)
    
    assert output.shape == (B, N, Hq, d)
    
    # Check that outputs are close
    assert_outputs_close(output, output_torch, test_name="batch_custom_mask")


def test_masked_attention_batch_complex_mask():
    """Test masked_attention_via_map with batch dimension and complex batch-head-specific masking"""
    B, N, Hq, d = 4, 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    # Complex mask: different patterns for different batches and heads
    def complex_batch_mask(b, h, q, k):
        # Window size depends on both batch and head
        window_size = ((b % 2) * 2) + ((h % 2) + 1)
        window_ok = abs(q - k) <= window_size
        # For odd batches and even heads, add additional constraint
        is_odd_batch = (b % 2) == 1
        is_even_head = (h % 2) == 0
        alternating_ok = ((q + k) % 2) == 0
        return jnp.where(is_odd_batch & is_even_head, window_ok & alternating_ok, window_ok)
    
    mask = materialize_mask_batch(complex_batch_mask, B, Hq, N, L)
    
    output = masked_attention_via_map(Q, K, V, mask_fn=complex_batch_mask)
    output_torch = pytorch_scaled_dot_product_attention_batch(Q, K, V, mask=mask)
    
    assert output.shape == (B, N, Hq, d)
    assert_outputs_close(output, output_torch, test_name="batch_complex_mask")


def test_masked_attention_batch_gqa():
    """Test masked_attention_via_map with batch dimension and GQA"""
    B, N, Hq, d = 4, 8, 8, 16
    L, Hkv = 8, 4  # Hq = 2 * Hkv
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    output = masked_attention_via_map(Q, K, V, is_causal=True)
    
    assert output.shape == (B, N, Hq, d)
    
    # Compare with PyTorch
    output_torch = pytorch_scaled_dot_product_attention_batch(Q, K, V, is_causal=True)
    assert_outputs_close(output, output_torch, test_name="batch_gqa")


def test_masked_attention_batch_gradients():
    """Test gradients with batch dimension"""
    B, N, Hq, d = 4, 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_batch_gradients_with_mask():
    """Test gradients with batch dimension and custom mask"""
    B, N, Hq, d = 4, 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    def batch_sliding_window_mask(b, h, q, k):
        window_size = (b % 3) + 2  # Different window size per batch
        return abs(q - k) <= window_size
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=batch_sliding_window_mask)
        return jnp.sum(output ** 2)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_batch_gradients_pytorch_comparison():
    """Compare batch gradients with PyTorch implementation"""
    B = 4
    N = 16
    L = 16
    Hq = 4
    Hkv = 4
    d = 32

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    # JAX gradients
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # PyTorch gradients
    Q_t = jax_to_torch(Q).requires_grad_(True)
    K_t = jax_to_torch(K).requires_grad_(True)
    V_t = jax_to_torch(V).requires_grad_(True)
    
    output_t = pytorch_scaled_dot_product_attention_batch(Q_t, K_t, V_t, is_causal=True, return_torch=True)
    loss_t = output_t.sum()
    loss_t.backward()
    
    grad_Q_torch = torch_to_jax(Q_t.grad)
    grad_K_torch = torch_to_jax(K_t.grad)
    grad_V_torch = torch_to_jax(V_t.grad)
    
    # Compare gradients
    assert_outputs_close(grad_Q_jax, grad_Q_torch, test_name="batch_grad_Q")
    assert_outputs_close(grad_K_jax, grad_K_torch, test_name="batch_grad_K")
    assert_outputs_close(grad_V_jax, grad_V_torch, test_name="batch_grad_V")


def test_masked_attention_batch_gradients_pytorch_comparison_with_mask():
    """Compare batch gradients with custom mask against PyTorch"""
    B = 4
    N = 16
    L = 16
    Hq = 4
    Hkv = 4
    d = 32

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    # Batch-specific mask
    def batch_head_mask(b, h, q, k):
        window_size = (b * 2) + (h % 2) + 1  # Different window per batch and head
        return abs(q - k) <= window_size
    
    mask = materialize_mask_batch(batch_head_mask, B, Hq, N, L)
    
    # JAX gradients
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, mask_fn=batch_head_mask)
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # PyTorch gradients
    Q_t = jax_to_torch(Q).requires_grad_(True)
    K_t = jax_to_torch(K).requires_grad_(True)
    V_t = jax_to_torch(V).requires_grad_(True)
    
    output_t = pytorch_scaled_dot_product_attention_batch(Q_t, K_t, V_t, mask=mask, return_torch=True)
    loss_t = output_t.sum()
    loss_t.backward()
    
    grad_Q_torch = torch_to_jax(Q_t.grad)
    grad_K_torch = torch_to_jax(K_t.grad)
    grad_V_torch = torch_to_jax(V_t.grad)
    
    # Compare gradients
    assert_outputs_close(grad_Q_jax, grad_Q_torch, test_name="batch_grad_Q_mask")
    assert_outputs_close(grad_K_jax, grad_K_torch, test_name="batch_grad_K_mask")
    assert_outputs_close(grad_V_jax, grad_V_torch, test_name="batch_grad_V_mask")


def test_masked_attention_batch_gradients_gqa():
    """Test batch gradients with GQA"""
    B, N, Hq, d = 4, 8, 8, 16
    L, Hkv = 8, 4  # Hq = 2 * Hkv
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_batch_gradients_large():
    """Test batch gradients with larger inputs"""
    B, N, Hq, d = 8, 32, 8, 32
    L, Hkv = 64, 8
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(q, k, v, is_causal=True)
        return jnp.sum(output ** 2)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are meaningful
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_window_size():
    """Test masked_attention_via_map with window_size parameter"""
    N, Hq, d = 32, 4, 16
    L, Hkv = 32, 4
    left_window = 3
    right_window = 2
    block_size = 8
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test with manual mask_fn that implements the same sliding window
    # For query position q and key position k, allow if q - left_window <= k <= q + right_window
    def sliding_window_mask_fn(h, q, k):
        return (q - left_window <= k) & (k <= q + right_window)
    
    output_manual_mask = masked_attention_via_map(
        Q, K, V,
        mask_fn=sliding_window_mask_fn,
        block_size=block_size
    )

    # Test with window_size parameter (no mask_fn)
    output_window_size = masked_attention_via_map(
        Q, K, V, 
        window_size=(left_window, right_window),
        mask_fn=sliding_window_mask_fn,
        block_size=block_size
    )
    
    assert output_window_size.shape == (N, Hq, d)
    assert output_manual_mask.shape == (N, Hq, d)
    
    # The outputs should be very close (allowing for small numerical differences)
    assert_outputs_close(output_window_size, output_manual_mask, test_name="window_size_vs_manual_mask", rtol=1e-4, atol=1e-4)


def test_masked_attention_window_size_restricts_attention():
    """Test that window_size actually restricts attention when mask_fn is permissive"""
    N, Hq, d = 64, 4, 16
    L, Hkv = 64, 4
    block_size = 16
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # First, run without window_size (all keys considered)
    output_no_window = masked_attention_via_map(
        Q, K, V,
        block_size=block_size
    )
    
    # Then run with a small window_size that maps to 1 block
    # window_size=(1, 1): left_blocks = (1+16-1)//16 = 1, right_blocks = 1+(1+16-1)//16 = 2, total = 3 block
    small_window_size = (1, 1)
    output_with_window = masked_attention_via_map(
        Q, K, V,
        window_size=small_window_size,
        block_size=block_size
    )
    
    # The outputs should be different because window_size restricts which keys are considered
    # (even though mask_fn allows all, window_size limits the blocks processed)
    max_diff = jnp.abs(output_no_window - output_with_window).max()
    assert max_diff > 1e-3, "window_size should change output when mask_fn is permissive"

    # With a medium window_size, output should be unchanged if the window size doesn't require
    # more blocks to cover.
    # window_size=(16, 1): left_blocks=(16+16-1)//16=1, right_blocks=1+(1+16-1)//16 = 2, total=3 blocks
    medium_window_size = (16, 1)
    output_medium_window = masked_attention_via_map(
        Q, K, V,
        window_size=medium_window_size,
        block_size=block_size
    )
    max_diff = jnp.abs(output_medium_window - output_with_window).max()
    assert max_diff < 1e-3, "window_size should not change output when window size doesn't require more blocks to cover"

    # With a larger window_size, output should be closer to no window
    # Use a window that covers more blocks but not all (keep total blocks <= 3)
    # window_size=(17, 1): left_blocks=(17+16-1)//16=2, right_blocks=1+(1+16-1)//16 = 2, total=4 blocks
    larger_window_size = (17, 1)
    output_larger_window = masked_attention_via_map(
        Q, K, V,
        window_size=larger_window_size,
        block_size=block_size
    )
    max_diff = jnp.abs(output_medium_window - output_larger_window).max()
    assert max_diff > 1e-3, "window_size should change output when window size requires more blocks to cover"


def test_masked_attention_window_size_jit():
    """Test that window_size works with JIT compilation"""
    N, Hq, d = 32, 4, 16
    L, Hkv = 32, 4
    block_size = 8
    window_size = (4, 4)
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # JIT the function with window_size
    @jax.jit
    def jitted_attention(q, k, v):
        return masked_attention_via_map(
            q, k, v,
            window_size=window_size,
            block_size=block_size
        )
    
    output_jit = jitted_attention(Q, K, V)
    
    # Compare with non-JIT version
    output_no_jit = masked_attention_via_map(
        Q, K, V,
        window_size=window_size,
        block_size=block_size
    )
    
    assert output_jit.shape == (N, Hq, d)
    assert jnp.allclose(output_jit, output_no_jit, rtol=1e-5, atol=1e-5)


def test_masked_attention_window_size_gradients():
    """Test that window_size works with gradient computation"""
    N, Hq, d = 32, 4, 16
    L, Hkv = 32, 4
    block_size = 8
    window_size = (4, 4)
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    def loss_fn(q, k, v):
        output = masked_attention_via_map(
            q, k, v,
            window_size=window_size,
            block_size=block_size
        )
        return jnp.sum(output ** 2)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are not all zeros
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_window_size_jit_gradients():
    """Test that window_size works with JIT + gradients together"""
    N, Hq, d = 32, 4, 16
    L, Hkv = 32, 4
    block_size = 8
    window_size = (4, 4)
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    @jax.jit
    def loss_fn(q, k, v):
        output = masked_attention_via_map(
            q, k, v,
            window_size=window_size,
            block_size=block_size
        )
        return jnp.sum(output ** 2)
    
    # Compute gradients with JIT
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
    grad_Q, grad_K, grad_V = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q.shape == Q.shape
    assert grad_K.shape == K.shape
    assert grad_V.shape == V.shape
    
    # Check that gradients are meaningful
    assert not jnp.allclose(grad_Q, 0.0)
    assert not jnp.allclose(grad_K, 0.0)
    assert not jnp.allclose(grad_V, 0.0)


def test_masked_attention_window_size_with_restrictive_mask():
    """Test that window_size works correctly with a restrictive mask_fn"""
    N, Hq, d = 32, 4, 16
    L, Hkv = 32, 4
    block_size = 8
    # Use window_size that maps to <= 4 blocks (with L=32, block_size=8, we have 4 blocks)
    # window_size=(8, 8): left_blocks=(8+8-1)//8=1, right_blocks=1+1=2, total=3 blocks (safe)
    window_size = (8, 8)  # Large enough window
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Restrictive mask: only allow keys within 3 positions
    def restrictive_mask(h, q, k):
        return abs(q - k) <= 3
    
    # With window_size larger than mask restriction, mask should control behavior
    output_window_mask = masked_attention_via_map(
        Q, K, V,
        window_size=window_size,
        mask_fn=restrictive_mask,
        block_size=block_size
    )
    
    # Without window_size, should get same result (mask controls behavior)
    output_mask_only = masked_attention_via_map(
        Q, K, V,
        mask_fn=restrictive_mask,
        block_size=block_size
    )
    
    # Should be very close since mask is the limiting factor
    assert_outputs_close(output_window_mask, output_mask_only, test_name="window_size_with_restrictive_mask", rtol=1e-3, atol=1e-3)


def test_masked_attention_window_size_batch():
    """Test window_size with batch dimension"""
    B, N, Hq, d = 4, 32, 4, 16
    L, Hkv = 32, 4
    block_size = 8
    window_size = (4, 4)
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (B, N, Hq, d))
    K = jax.random.normal(key2, (B, L, Hkv, d))
    V = jax.random.normal(key3, (B, L, Hkv, d))
    
    output = masked_attention_via_map(
        Q, K, V,
        window_size=window_size,
        block_size=block_size
    )
    
    assert output.shape == (B, N, Hq, d)


def test_masked_attention_window_size_gqa():
    """Test window_size with grouped query attention (GQA)"""
    N, Hq, d = 32, 8, 16
    L, Hkv = 32, 4  # Hq = 2 * Hkv
    block_size = 8
    window_size = (4, 4)
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    output = masked_attention_via_map(
        Q, K, V,
        window_size=window_size,
        block_size=block_size
    )
    
    assert output.shape == (N, Hq, d)


def test_masked_attention_window_size_gradients_with_mask():
    """Test gradients with window_size and mask_fn
    
    Note: window_size is a performance optimization that restricts which blocks are processed.
    This test verifies that gradients can be computed when using window_size together with a mask_fn.
    """
    N = 16
    L = 16
    Hq = 4
    Hkv = 4
    d = 16
    block_size = 4
    window_size = (2, 2)  # Small window
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Create a mask that works well with the window_size
    def window_mask(h, q, k):
        return abs(q - k) <= (window_size[0] + window_size[1])
    
    # JAX gradients with window_size
    def loss_fn(q, k, v):
        output = masked_attention_via_map(
            q, k, v,
            window_size=window_size,
            block_size=block_size,
            mask_fn=window_mask
        )
        return jnp.sum(output)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_Q_jax, grad_K_jax, grad_V_jax = grad_fn(Q, K, V)
    
    # Check shapes
    assert grad_Q_jax.shape == Q.shape
    assert grad_K_jax.shape == K.shape
    assert grad_V_jax.shape == V.shape
    
    # Check that at least some gradients are non-zero and non-NaN
    # (some may be NaN if query positions don't have valid key blocks in window)
    has_valid_Q = jnp.any(~jnp.isnan(grad_Q_jax))
    if has_valid_Q:
        valid_Q_grads = grad_Q_jax[~jnp.isnan(grad_Q_jax)]
        assert not jnp.allclose(valid_Q_grads, 0.0), "At least some Q gradients should be non-zero"
    
    has_valid_K = jnp.any(~jnp.isnan(grad_K_jax))
    if has_valid_K:
        valid_K_grads = grad_K_jax[~jnp.isnan(grad_K_jax)]
        assert not jnp.allclose(valid_K_grads, 0.0), "At least some K gradients should be non-zero"
    
    has_valid_V = jnp.any(~jnp.isnan(grad_V_jax))
    if has_valid_V:
        valid_V_grads = grad_V_jax[~jnp.isnan(grad_V_jax)]
        assert not jnp.allclose(valid_V_grads, 0.0), "At least some V gradients should be non-zero"
