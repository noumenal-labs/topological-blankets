"""
Tests for topological_blankets.detection module.
"""

import numpy as np
import pytest
from topological_blankets.detection import (
    detect_blankets_otsu,
    detect_blankets_gradient,
    detect_blankets_spectral,
    detect_blankets_hybrid,
    detect_blankets_coupling,
)
from topological_blankets.features import compute_geometric_features


class TestDetectBlanketsOtsu:

    def test_returns_tuple(self, simple_gradients):
        features = compute_geometric_features(simple_gradients)
        is_blanket, tau = detect_blankets_otsu(features)
        assert isinstance(is_blanket, np.ndarray)
        assert is_blanket.dtype == bool
        assert isinstance(tau, float)

    def test_blanket_is_minority(self, simple_gradients):
        features = compute_geometric_features(simple_gradients)
        is_blanket, _ = detect_blankets_otsu(features)
        # Blanket should be minority (or equal)
        assert np.sum(is_blanket) <= len(is_blanket) / 2 + 1


class TestDetectBlanketsGradient:

    def test_otsu_method(self, simple_gradients):
        is_blanket, tau = detect_blankets_gradient(simple_gradients, method='otsu')
        assert is_blanket.shape == (simple_gradients.shape[1],)

    def test_percentile_method(self, simple_gradients):
        is_blanket, tau = detect_blankets_gradient(simple_gradients, method='percentile')
        assert is_blanket.shape == (simple_gradients.shape[1],)

    def test_median_method(self, simple_gradients):
        is_blanket, tau = detect_blankets_gradient(simple_gradients, method='median')
        assert is_blanket.shape == (simple_gradients.shape[1],)


class TestDetectBlanketsSpectral:

    def test_returns_dict(self, simple_gradients):
        features = compute_geometric_features(simple_gradients)
        result = detect_blankets_spectral(features['hessian_est'])
        assert 'is_blanket' in result
        assert 'eigengap' in result
        assert 'eigvals' in result
        assert result['is_blanket'].dtype == bool


class TestDetectBlanketsHybrid:

    def test_returns_dict(self, simple_gradients):
        features = compute_geometric_features(simple_gradients)
        result = detect_blankets_hybrid(simple_gradients, features['hessian_est'])
        assert 'is_blanket' in result
        assert 'method_used' in result
        assert result['method_used'] in ('spectral', 'gradient', 'gradient_fallback')

    def test_eigengap_threshold(self, simple_gradients):
        features = compute_geometric_features(simple_gradients)
        # Very high threshold forces gradient method
        result = detect_blankets_hybrid(simple_gradients, features['hessian_est'],
                                         eigengap_threshold=1000.0)
        assert result['method_used'] == 'gradient'


class TestDetectBlanketsCoupling:

    def test_basic_functionality(self):
        np.random.seed(42)
        n_vars = 9
        # Build a Hessian-like matrix with block structure
        H_est = np.random.rand(n_vars, n_vars) * 0.1
        H_est = (H_est + H_est.T) / 2

        # 2 objects strongly coupled internally + blanket
        for i in range(3):
            for j in range(3):
                H_est[i, j] = 6.0 if i != j else 18.0
        for i in range(3, 6):
            for j in range(3, 6):
                H_est[i, j] = 6.0 if i != j else 18.0
        # Blanket vars (6-8) weakly coupled internally, moderate to objects
        for i in range(6, 9):
            H_est[i, i] = 3.0
            for j in range(6, 9):
                if i != j:
                    H_est[i, j] = 1.0
            for j in range(6):
                H_est[i, j] = H_est[j, i] = 0.8

        # Build normalized coupling from H_est
        diag = np.sqrt(np.abs(np.diag(H_est)) + 1e-10)
        coupling = np.abs(H_est) / np.outer(diag, diag)
        np.fill_diagonal(coupling, 1.0)

        is_blanket = detect_blankets_coupling(H_est, coupling, n_objects=2)
        assert is_blanket.shape == (9,)
        assert is_blanket.dtype == bool

    def test_too_few_variables(self):
        H_est = np.eye(3)
        coupling = np.eye(3)
        is_blanket = detect_blankets_coupling(H_est, coupling, n_objects=2)
        assert np.sum(is_blanket) == 0  # Too few vars for 3 clusters

    def test_no_blanket_when_all_internal(self):
        # All variables belong to clusters with no cross-coupling
        n_vars = 6
        H_est = np.zeros((n_vars, n_vars))
        for i in range(3):
            for j in range(3):
                H_est[i, j] = 6.0 if i != j else 18.0
        for i in range(3, 6):
            for j in range(3, 6):
                H_est[i, j] = 6.0 if i != j else 18.0

        diag = np.sqrt(np.abs(np.diag(H_est)) + 1e-10)
        coupling = np.abs(H_est) / np.outer(diag, diag)
        np.fill_diagonal(coupling, 1.0)

        is_blanket = detect_blankets_coupling(H_est, coupling, n_objects=2)
        assert is_blanket.shape == (6,)
