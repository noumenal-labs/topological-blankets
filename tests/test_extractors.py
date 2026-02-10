"""
Tests for topological_blankets.extractors module.
"""

import numpy as np
import pytest
from topological_blankets.extractors import (
    extract_gradients_callable,
    extract_gradients_from_scores,
)


class TestExtractGradientsCallable:

    def test_quadratic_energy(self):
        """For E = 0.5 * x^T x, grad E = x."""
        def energy(x):
            return 0.5 * np.dot(x, x)

        np.random.seed(42)
        data = np.random.randn(20, 4)
        grads = extract_gradients_callable(energy, data)

        assert grads.shape == (20, 4)
        assert np.allclose(grads, data, atol=1e-3)

    def test_linear_energy(self):
        """For E = c^T x, grad E = c."""
        c = np.array([1.0, 2.0, 3.0])
        def energy(x):
            return np.dot(c, x)

        data = np.random.randn(10, 3)
        grads = extract_gradients_callable(energy, data)

        assert grads.shape == (10, 3)
        for i in range(10):
            assert np.allclose(grads[i], c, atol=1e-3)

    def test_custom_eps(self):
        def energy(x):
            return 0.5 * np.dot(x, x)

        data = np.random.randn(5, 3)
        grads = extract_gradients_callable(energy, data, eps=1e-7)
        assert np.allclose(grads, data, atol=1e-5)

    def test_output_shape(self):
        def energy(x):
            return np.sum(x**2)

        data = np.random.randn(15, 7)
        grads = extract_gradients_callable(energy, data)
        assert grads.shape == (15, 7)


class TestExtractGradientsFromScores:

    def test_negation(self):
        """Score = -grad E, so energy gradients = -score."""
        np.random.seed(42)
        data = np.random.randn(20, 4)

        def score_fn(x):
            return -x  # score = -grad E for E = 0.5*x^Tx

        grads = extract_gradients_from_scores(score_fn, data)
        assert grads.shape == (20, 4)
        assert np.allclose(grads, data)  # -(-x) = x

    def test_batch_score(self):
        """Score function that processes batch at once."""
        np.random.seed(42)
        data = np.random.randn(10, 3)

        def score_fn(x):
            return -2 * x  # score for E = x^Tx

        grads = extract_gradients_from_scores(score_fn, data)
        assert grads.shape == (10, 3)
        assert np.allclose(grads, 2 * data)


class TestExtractGradientsPyTorch:
    """Tests for PyTorch gradient extraction (skipped if torch not available)."""

    @pytest.fixture
    def torch_available(self):
        pytest.importorskip("torch")

    def test_pytorch_linear(self, torch_available):
        import torch
        from topological_blankets.extractors import extract_gradients_pytorch

        model = torch.nn.Linear(4, 1, bias=False)
        np.random.seed(42)
        data = np.random.randn(10, 4).astype(np.float32)
        targets = np.random.randn(10, 1).astype(np.float32)

        grads = extract_gradients_pytorch(model, data,
                                           loss_fn=torch.nn.MSELoss(),
                                           targets=targets)
        assert grads.shape == (10, 4)
        assert not np.all(grads == 0)

    def test_pytorch_no_loss(self, torch_available):
        import torch
        from topological_blankets.extractors import extract_gradients_pytorch

        model = torch.nn.Linear(3, 1, bias=False)
        data = np.random.randn(5, 3).astype(np.float32)

        grads = extract_gradients_pytorch(model, data)
        assert grads.shape == (5, 3)
