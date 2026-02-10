"""
Shared fixtures for topological_blankets test suite.
"""

import numpy as np
import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def rng():
    """Fixed random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_gradients(rng):
    """Simple gradient samples: 2 clusters + blanket region, 9 variables."""
    n_samples = 500
    n_vars = 9

    # Object 1 (vars 0-2): strongly correlated gradients
    g1 = rng.randn(n_samples, 3) @ np.array([[3, 0.1, 0.1],
                                               [0.1, 3, 0.1],
                                               [0.1, 0.1, 3]])

    # Object 2 (vars 3-5): strongly correlated, different direction
    g2 = rng.randn(n_samples, 3) @ np.array([[3, 0.1, 0.1],
                                               [0.1, 3, 0.1],
                                               [0.1, 0.1, 3]])

    # Blanket (vars 6-8): weakly coupled to both objects
    g_blanket = 0.3 * (g1 + g2) / 2 + 0.2 * rng.randn(n_samples, 3)

    gradients = np.hstack([g1, g2, g_blanket])
    return gradients


@pytest.fixture
def quadratic_ebm_gradients():
    """Gradients from actual quadratic EBM with known ground truth."""
    from experiments.quadratic_toy_comparison import (
        QuadraticEBMConfig, build_precision_matrix, langevin_sampling, get_ground_truth
    )

    cfg = QuadraticEBMConfig(n_objects=2, vars_per_object=3,
                              vars_per_blanket=3, blanket_strength=0.8)
    Theta = build_precision_matrix(cfg)
    samples, gradients = langevin_sampling(Theta, n_samples=1000,
                                            n_steps=200, step_size=0.01, temp=0.1)
    truth = get_ground_truth(cfg)
    return gradients, truth, cfg


@pytest.fixture
def high_dim_gradients(rng):
    """Higher-dimensional gradient samples: 20 variables, 4 objects."""
    n_samples = 1000
    n_vars = 20
    gradients = rng.randn(n_samples, n_vars)
    return gradients
