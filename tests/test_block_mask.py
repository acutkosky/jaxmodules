import pytest
import jax
import jax.numpy as jnp
import torch
from jaxmodules.block_mask import (
    BlockMask,
    create_block_mask,
    get_partial_block,
    get_sparse_kv_data_from_blocks,
    get_sparse_q_data_from_blocks,
    get_dense_from_kv_blocks,
)


def test_block_mask_basic():
    """Test basic functionality of BlockMask with a simple causal mask"""
    B, H, L = 2, 2, 8
    BLOCK_SIZE = 4

    def mask_mod(b, h, q_idx, k_idx):
        return q_idx >= k_idx  # Causal mask

    # Create block mask
    block_mask = create_block_mask(mask_mod, B, H, L, L, BLOCK_SIZE)

    # Test basic properties
    assert block_mask.B == B
    assert block_mask.H == H
    assert block_mask.Q_LEN == L
    assert block_mask.KV_LEN == L
    assert block_mask.Q_BLOCK_SIZE == BLOCK_SIZE
    assert block_mask.KV_BLOCK_SIZE == BLOCK_SIZE

    # Test materialized mask matches expected causal pattern
    dense_mask = block_mask.materialize_mask()
    expected_mask = jnp.array(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ]
    )
    expected_mask = jnp.broadcast_to(expected_mask, (B, H, L, L))
    assert jnp.array_equal(dense_mask, expected_mask)


def test_block_mask_partial_blocks():
    """Test BlockMask with partial blocks (blocks that are neither all True nor all False)"""
    B, H, L = 1, 1, 8
    BLOCK_SIZE = 4

    def mask_mod(b, h, q_idx, k_idx):
        # Create a pattern where some blocks are partial
        return (q_idx + k_idx) % 3 == 0

    block_mask = create_block_mask(mask_mod, B, H, L, L, BLOCK_SIZE)

    # Test that partial blocks are correctly identified
    dense_mask = block_mask.materialize_mask()

    # Verify that the mask matches the expected pattern
    for q_idx in range(L):
        for k_idx in range(L):
            expected = (q_idx + k_idx) % 3 == 0
            assert dense_mask[0, 0, q_idx, k_idx] == expected


def test_block_mask_from_kv_blocks():
    """Test BlockMask.from_kv_blocks with a known pattern"""
    B, H, L = 1, 1, 8
    BLOCK_SIZE = 2

    # Create a simple pattern where each query block attends to the next two key blocks
    kv_num_blocks = jnp.array([[[2, 2, 2, 2]]])  # Each row has 2 blocks
    kv_indices = jnp.array(
        [[[[0, 1], [1, 2], [2, 3], [3, 4]]]]
    )  # Indices of the blocks

    # Create full blocks (all-ones blocks)
    full_kv_num_blocks = jnp.array([[[0, 0]]])  # No full blocks
    full_kv_indices = jnp.array([[[[0, 0], [0, 0]]]])  # Dummy indices

    def mask_mod(b, h, q_idx, k_idx):
        # This should match the pattern specified by kv_num_blocks and kv_indices
        block_q = q_idx // BLOCK_SIZE
        block_k = k_idx // BLOCK_SIZE
        return (block_k >= block_q) & (block_k <= block_q + 1)

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(L, L),
    )

    # Test that the mask matches the expected pattern
    dense_mask = block_mask.materialize_mask()
    for q_idx in range(L):
        for k_idx in range(L):
            block_q = q_idx // BLOCK_SIZE
            block_k = k_idx // BLOCK_SIZE
            expected = (block_k >= block_q) & (block_k < block_q + 2)
            assert dense_mask[0, 0, q_idx, k_idx] == expected


def test_get_partial_block():
    """Test get_partial_block function"""
    B, H, L = 1, 1, 8
    BLOCK_SIZE = 4

    def mask_mod(b, h, q_idx, k_idx):
        return (q_idx + k_idx) % 2 == 0

    # Test a specific block
    block = get_partial_block(mask_mod, 0, 0, 0, 0, BLOCK_SIZE, BLOCK_SIZE, B, H)

    # Verify the block pattern
    expected = jnp.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert jnp.array_equal(block, expected)


def test_get_sparse_kv_data_from_blocks():
    """Test get_sparse_kv_data_from_blocks function"""
    B, H = 1, 1
    num_blocks_in_col = 2
    num_blocks_in_row = 2

    # Create a simple block pattern
    blocks = jnp.array([[[[1, 0], [1, 1]]]])

    kv_num_blocks, kv_indices = get_sparse_kv_data_from_blocks(blocks)

    blocks = jnp.array([[1, 0], [1, 1]])

    broadcasted_kv_num_blocks, broadcasted_kv_indices = get_sparse_kv_data_from_blocks(blocks)

    # Verify the output
    expected_num_blocks = jnp.array([[[1, 2]]])  # First row has 1 block, second has 2
    expected_indices = jnp.array([[[[0, 2], [0, 1]]]])  # Indices of the blocks

    relevant_indices = jnp.array([[True, False], [True, True]])

    assert jnp.array_equal(kv_num_blocks, expected_num_blocks)
    assert jnp.array_equal(
        kv_indices * relevant_indices, expected_indices * relevant_indices
    )

    assert jnp.array_equal(broadcasted_kv_num_blocks, expected_num_blocks)
    assert jnp.array_equal(
        broadcasted_kv_indices * relevant_indices, expected_indices * relevant_indices
    )


def test_get_sparse_q_data_from_blocks():
    """Test get_sparse_q_data_from_blocks function"""
    B, H = 1, 1
    num_blocks_in_col = 2
    num_blocks_in_row = 2

    # Create a simple block pattern
    blocks = jnp.array([[[[1, 0], [1, 1]]]])

    q_num_blocks, q_indices = get_sparse_q_data_from_blocks(blocks)

    # Verify the output
    expected_num_blocks = jnp.array(
        [[[2, 1]]]
    )  # First column has 2 blocks, second has 1
    expected_indices = jnp.array([[[[0, 1], [1, 2]]]])  # Indices of the blocks
    relevant_indices = jnp.array([[True, True], [True, False]])

    assert jnp.array_equal(q_num_blocks, expected_num_blocks)
    assert jnp.array_equal(
        q_indices * relevant_indices, expected_indices * relevant_indices
    )


def test_get_dense_from_kv_blocks():
    """Test get_dense_from_kv_blocks function"""
    B, H = 1, 1
    NUM_Q_BLOCKS = 2
    NUM_KV_BLOCKS = 2

    # Create sparse block data
    kv_num_blocks = jnp.array([[[1, 2]]])  # First row has 1 block, second has 2
    kv_indices = jnp.array([[[[0, 0], [0, 1]]]])  # Indices of the blocks

    dense = get_dense_from_kv_blocks(
        B, H, NUM_Q_BLOCKS, NUM_KV_BLOCKS, kv_num_blocks, kv_indices
    )

    # Verify the output
    expected = jnp.array([[[[1, 0], [1, 1]]]])
    assert jnp.array_equal(dense, expected)


def test_block_mask_causal_mask():
    """Test BlockMask.causal_mask class method"""
    B, H, L = 2, 2, 8
    BLOCK_SIZE = 4

    # Create causal mask
    block_mask = BlockMask.causal_mask(B, H, L, L, BLOCK_SIZE)

    # Test basic properties
    assert block_mask.B == B
    assert block_mask.H == H
    assert block_mask.Q_LEN == L
    assert block_mask.KV_LEN == L
    assert block_mask.Q_BLOCK_SIZE == BLOCK_SIZE
    assert block_mask.KV_BLOCK_SIZE == BLOCK_SIZE

    # Test materialized mask matches expected causal pattern
    dense_mask = block_mask.materialize_mask()
    expected_mask = jnp.array(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ]
    )
    expected_mask = jnp.broadcast_to(expected_mask, (B, H, L, L))
    assert jnp.array_equal(dense_mask, expected_mask)


def test_block_mask_full_mask():
    """Test BlockMask.full_mask class method"""
    B, H, L = 2, 2, 8
    BLOCK_SIZE = 4

    # Create full mask
    block_mask = BlockMask.full_mask(B, H, L, L, BLOCK_SIZE)

    # Test basic properties
    assert block_mask.B == B
    assert block_mask.H == H
    assert block_mask.Q_LEN == L
    assert block_mask.KV_LEN == L
    assert block_mask.Q_BLOCK_SIZE == BLOCK_SIZE
    assert block_mask.KV_BLOCK_SIZE == BLOCK_SIZE

    # Test materialized mask is all ones
    dense_mask = block_mask.materialize_mask()
    expected_mask = jnp.ones((B, H, L, L), dtype=jnp.int32)
    assert jnp.array_equal(dense_mask, expected_mask)
