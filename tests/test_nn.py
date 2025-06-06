import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import numpy as np
import pytest
from jaxmodules.nn import softmax_cross_entropy


def to_numpy(x):
    """Helper to convert torch tensors or jax arrays to numpy"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, jnp.ndarray):
        return np.array(x)
    return x


class TestSoftmaxCrossEntropy:
    def test_basic_functionality(self):
        """Test basic cross entropy computation"""
        # Simple 2D case: batch_size=3, num_classes=4
        key = jax.random.PRNGKey(42)
        logits_jax = jax.random.normal(key, (3, 4))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        targets = jnp.array([0, 3, 1])
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        # Test default reduction='mean'
        loss_jax = softmax_cross_entropy(logits_jax, targets)
        loss_torch = F.cross_entropy(logits_torch, targets_torch)

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-6)

    def test_different_reductions(self):
        """Test different reduction modes"""
        key = jax.random.PRNGKey(123)
        logits_jax = jax.random.normal(key, (4, 5))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        targets = jnp.array([1, 0, 3, 2])
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        # Test reduction='none'
        loss_jax_none = softmax_cross_entropy(logits_jax, targets, reduction="none")
        loss_torch_none = F.cross_entropy(logits_torch, targets_torch, reduction="none")
        np.testing.assert_allclose(
            to_numpy(loss_jax_none), to_numpy(loss_torch_none), rtol=1e-6
        )

        # Test reduction='sum'
        loss_jax_sum = softmax_cross_entropy(logits_jax, targets, reduction="sum")
        loss_torch_sum = F.cross_entropy(logits_torch, targets_torch, reduction="sum")
        np.testing.assert_allclose(
            to_numpy(loss_jax_sum), to_numpy(loss_torch_sum), rtol=1e-6
        )

        # Test reduction='mean'
        loss_jax_mean = softmax_cross_entropy(logits_jax, targets, reduction="mean")
        loss_torch_mean = F.cross_entropy(logits_torch, targets_torch, reduction="mean")
        np.testing.assert_allclose(
            to_numpy(loss_jax_mean), to_numpy(loss_torch_mean), rtol=1e-6
        )

    def test_ignore_index(self):
        """Test ignore_index functionality"""
        key = jax.random.PRNGKey(456)
        logits_jax = jax.random.normal(key, (5, 3))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        # Include some ignored indices
        targets = jnp.array([0, -100, 1, 2, -100])
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        loss_jax = softmax_cross_entropy(logits_jax, targets, ignore_index=-100)
        loss_torch = F.cross_entropy(logits_torch, targets_torch, ignore_index=-100)

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-6)

    def test_class_weights(self):
        """Test class weighting functionality"""
        key = jax.random.PRNGKey(789)
        logits_jax = jax.random.normal(key, (4, 3))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        weights = jnp.array([0.5, 1.0, 2.0])
        weights_torch = torch.tensor(to_numpy(weights), dtype=torch.float32)

        targets = jnp.array([0, 1, 2, 1])
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        loss_jax = softmax_cross_entropy(logits_jax, targets, weight=weights)
        loss_torch = F.cross_entropy(logits_torch, targets_torch, weight=weights_torch)

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-5)

    def test_label_smoothing(self):
        """Test label smoothing functionality"""
        key = jax.random.PRNGKey(101112)
        logits_jax = jax.random.normal(key, (3, 4))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        targets = jnp.array([0, 2, 1])
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        label_smoothing = 0.1

        loss_jax = softmax_cross_entropy(
            logits_jax, targets, label_smoothing=label_smoothing
        )
        loss_torch = F.cross_entropy(
            logits_torch, targets_torch, label_smoothing=label_smoothing
        )

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-5)

    def test_different_axes(self):
        """Test with different axis specifications"""
        key = jax.random.PRNGKey(131415)
        # Shape: (2, 3, 4) - batch, classes, sequence
        logits_jax = jax.random.normal(key, (2, 3, 4))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        targets = jnp.array([[0, 1, 2, 1], [2, 0, 1, 0]])
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        # Test with axis=1 (class dimension)
        loss_jax = softmax_cross_entropy(logits_jax, targets, axis=1)
        loss_torch = F.cross_entropy(logits_torch, targets_torch, reduction="mean")

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-6)

    def test_multidimensional_targets(self):
        """Test with multidimensional inputs and targets"""
        key = jax.random.PRNGKey(161718)
        # Shape: (2, 5, 3, 3) - batch, classes, height, width
        logits_jax = jax.random.normal(key, (2, 5, 3, 3))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        # Shape: (2, 3, 3) - batch, height, width
        targets = jax.random.randint(jax.random.PRNGKey(192021), (2, 3, 3), 0, 5)
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        loss_jax = softmax_cross_entropy(logits_jax, targets, axis=1)
        loss_torch = F.cross_entropy(logits_torch, targets_torch)

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-6)

    def test_combined_features(self):
        """Test combination of multiple features"""
        key = jax.random.PRNGKey(222324)
        logits_jax = jax.random.normal(key, (4, 5))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        weights = jnp.array([1.0, 0.5, 2.0, 1.5, 0.8])
        weights_torch = torch.tensor(to_numpy(weights), dtype=torch.float32)

        targets = jnp.array([1, -100, 3, 2])  # Include ignored index
        targets_torch = torch.tensor(to_numpy(targets), dtype=torch.long)

        # Combine weight, ignore_index, label_smoothing, and reduction
        loss_jax = softmax_cross_entropy(
            logits_jax,
            targets,
            weight=weights,
            ignore_index=-100,
            label_smoothing=0.05,
            reduction="sum",
        )
        loss_torch = F.cross_entropy(
            logits_torch,
            targets_torch,
            weight=weights_torch,
            ignore_index=-100,
            label_smoothing=0.05,
            reduction="sum",
        )

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-5)

    def test_probabilistic_targets(self):
        """Test with probabilistic (soft) targets"""
        key = jax.random.PRNGKey(252627)
        logits_jax = jax.random.normal(key, (3, 4))
        logits_torch = torch.tensor(to_numpy(logits_jax), dtype=torch.float32)

        # Soft targets (probabilities)
        target_probs = jax.random.dirichlet(
            jax.random.PRNGKey(282930), jnp.ones(4), (3,)
        )
        target_probs_torch = torch.tensor(to_numpy(target_probs), dtype=torch.float32)

        # JAX implementation with probabilistic targets
        loss_jax = softmax_cross_entropy(logits_jax, target_probs)

        # PyTorch equivalent using manual computation
        log_probs_torch = F.log_softmax(logits_torch, dim=1)
        loss_torch = -torch.sum(target_probs_torch * log_probs_torch, dim=1).mean()

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-6)

    def test_edge_cases(self):
        """Test edge cases"""
        # Single sample, single class
        logits_jax = jnp.array([[2.0]])
        targets = jnp.array([0])
        loss_jax = softmax_cross_entropy(logits_jax, targets)

        logits_torch = torch.tensor([[2.0]])
        targets_torch = torch.tensor([0])
        loss_torch = F.cross_entropy(logits_torch, targets_torch)

        np.testing.assert_allclose(to_numpy(loss_jax), to_numpy(loss_torch), rtol=1e-6)

        # All targets ignored
        logits_jax = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([-100, -100])
        loss_jax = softmax_cross_entropy(logits_jax, targets, ignore_index=-100)

        logits_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets_torch = torch.tensor([-100, -100])
        loss_torch = F.cross_entropy(logits_torch, targets_torch, ignore_index=-100)

        # Both should be 0 or nan - PyTorch returns nan for this case
        # Our implementation should handle this gracefully
        assert jnp.isnan(loss_jax) or loss_jax == 0.0


if __name__ == "__main__":
    # Run a quick test
    test = TestSoftmaxCrossEntropy()
    test.test_basic_functionality()
    print("Basic functionality test passed!")

    test.test_different_reductions()
    print("Reduction tests passed!")

    test.test_ignore_index()
    print("Ignore index test passed!")

    print("All tests completed successfully!")
