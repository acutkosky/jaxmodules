import jax
from jax import random as jr
from jaxtyping import PyTree
from jax import Array


class PRNGContext:
    def __init__(self, key: Array):
        self.key = key

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context manager cleanup - nothing special needed for PRNG keys
        return False

    def new(self, tree: PyTree | None = None) -> Array:
        """Split the current key in two and return one of the splits.

        The current key is updated to use the other split, ensuring
        that subsequent calls to new() produce independent random streams.

        Args:
            tree: Optional PyTree structure. If provided, returns a PyTree of
                  PRNG keys with the same structure as the input tree, where
                  each leaf is replaced with a unique PRNG key. If None,
                  returns a single PRNG key.

        Returns:
            If tree is None: A single PRNG key that can be used for random operations.
            If tree is provided: A PyTree of PRNG keys with the same structure as
            the input tree, where each leaf position contains an independent PRNG key.
        """
        self.key, new_key = jr.split(self.key)
        if tree is None:
            return new_key
        leaves, treedef = jax.tree.flatten(tree)
        new_key = jr.split(new_key, len(leaves))
        return jax.tree.unflatten(treedef, new_key)
