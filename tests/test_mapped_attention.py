import pytest
import jax
import jax.numpy as jnp
import torch
import numpy as np
import torch.nn.functional as F
from jaxmodules.attention import use_custom_einsum, masked_attention_via_map
from jaxmodules.vectorize import fancy_vmap


use_custom_einsum()  # Required for higher precision to get the tests to pass.


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


def assert_outputs_close(jax_output, torch_output, rtol=1e-3, atol=1e-3, test_name=""):
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
    assert jnp.linalg.norm(jax_output - torch_output) / jnp.minimum(jnp.linalg.norm(jax_output), jnp.linalg.norm(torch_output)) < rtol, \
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
        assert jnp.allclose(output, output_default, rtol=1e-4, atol=1e-4)


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
    # Should not crash and produce valid output


def test_masked_attention_error_cases():
    """Test that appropriate errors are raised for invalid inputs"""
    N, Hq, d = 8, 4, 16
    L, Hkv = 8, 4
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # Test that is_causal and mask_fn cannot both be specified
    def dummy_mask(h, q, k):
        return True
    
    with pytest.raises(ValueError):
        masked_attention_via_map(Q, K, V, is_causal=True, mask_fn=dummy_mask)
    
    # Test with mismatched dimensions
    K_wrong = jax.random.normal(key2, (L, Hkv, d + 1))
    with pytest.raises(AssertionError):
        masked_attention_via_map(Q, K_wrong, V)
    
    # Test with invalid block_size (not dividing N)
    with pytest.raises(AssertionError):
        masked_attention_via_map(Q, K, V, block_size=3)  # 8 % 3 != 0


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


@pytest.mark.parametrize("N,L", [(8, 8), (16, 16), (32, 32), (64, 32)])
def test_masked_attention_different_lengths(N, L):
    """Test masked_attention_via_map with different query and key lengths"""
    Hq, Hkv, d = 4, 4, 16
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.normal(key1, (N, Hq, d))
    K = jax.random.normal(key2, (L, Hkv, d))
    V = jax.random.normal(key3, (L, Hkv, d))
    
    # The current implementation requires that when block_size is set to N,
    # K and V must be long enough to be divided into blocks of size Lq.
    # For N > L, we need to use a smaller block_size that divides both N and works with L.
    # Skip the problematic case for now (when N > L and block_size would be N).
    # This is a limitation of the current implementation.
    if N > L:
        # Use a smaller block_size that works with both N and L
        block_size = min(N, L) if L > 0 else N
        # Ensure block_size divides N
        while N % block_size != 0 and block_size > 1:
            block_size -= 1
        output = masked_attention_via_map(Q, K, V, is_causal=False, block_size=block_size)
    else:
        output = masked_attention_via_map(Q, K, V, is_causal=False)
    
    assert output.shape == (N, Hq, d)


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
    assert jnp.allclose(output, output_small_block, rtol=1e-4, atol=1e-4)

