import jax
import jax.numpy as jnp
from jax import random as jr
import pytest
from jaxmodules.prng_utils import PRNGContext


def test_prng_context_basic():
    """Test basic context manager functionality."""
    key = jr.PRNGKey(42)

    with PRNGContext(key) as rng:
        assert rng is not None
        assert hasattr(rng, "new")
        assert hasattr(rng, "key")


def test_single_key_generation():
    """Test generating single keys."""
    key = jr.PRNGKey(42)

    with PRNGContext(key) as rng:
        key1 = rng.new()
        key2 = rng.new()

        # Keys should be different
        assert not jnp.array_equal(key1, key2)

        # Keys should be valid PRNG keys
        assert key1.shape == key2.shape
        assert key1.dtype == key2.dtype


def test_pytree_key_generation():
    """Test generating PyTree of keys."""
    key = jr.PRNGKey(42)

    # Test with a simple PyTree structure
    tree_structure = {"a": 1, "b": [2, 3], "c": {"d": 4}}

    with PRNGContext(key) as rng:
        key_tree = rng.new(tree_structure)

        # Should have same structure as input
        assert set(key_tree.keys()) == set(tree_structure.keys())
        assert len(key_tree["b"]) == len(tree_structure["b"])
        assert set(key_tree["c"].keys()) == set(tree_structure["c"].keys())

        # All leaves should be PRNG keys
        leaves, _ = jax.tree.flatten(key_tree)
        for leaf in leaves:
            assert leaf.shape == key.shape
            assert leaf.dtype == key.dtype


def test_key_independence():
    """Test that generated keys produce different random values."""
    key = jr.PRNGKey(42)

    with PRNGContext(key) as rng:
        key1 = rng.new()
        key2 = rng.new()

        # Generate random values with each key
        values1 = jr.normal(key1, (10,))
        values2 = jr.normal(key2, (10,))

        # Values should be different (extremely unlikely to be identical)
        assert not jnp.allclose(values1, values2)


def test_multiple_calls_consistency():
    """Test that multiple calls to new() with same tree structure work."""
    key = jr.PRNGKey(42)
    tree_structure = [1, 2, 3]

    with PRNGContext(key) as rng:
        keys1 = rng.new(tree_structure)
        keys2 = rng.new(tree_structure)

        # Should have same structure
        assert len(keys1) == len(keys2) == len(tree_structure)

        # But keys should be different
        for k1, k2 in zip(keys1, keys2):
            assert not jnp.array_equal(k1, k2)


def test_context_manager_exit():
    """Test that context manager exits cleanly."""
    key = jr.PRNGKey(42)

    try:
        with PRNGContext(key) as rng:
            _ = rng.new()
            # Simulate an exception
            raise ValueError("Test exception")
    except ValueError:
        pass  # Expected

    # Should exit cleanly without issues


if __name__ == "__main__":
    # Simple test runner
    test_prng_context_basic()
    test_single_key_generation()
    test_pytree_key_generation()
    test_key_independence()
    test_multiple_calls_consistency()
    test_context_manager_exit()
    print("All tests passed!")
