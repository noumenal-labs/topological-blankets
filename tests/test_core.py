"""
Tests for topological_blankets.core module.
"""

import numpy as np
import pytest
from topological_blankets import TopologicalBlankets, topological_blankets


class TestTopologicalBlanketsFunctional:
    """Tests for the topological_blankets() functional API."""

    def test_basic_output_shape(self, simple_gradients):
        result = topological_blankets(simple_gradients, n_objects=2)
        assert result['assignment'].shape == (9,)
        assert result['is_blanket'].shape == (9,)
        assert 'features' in result
        assert 'threshold' in result

    def test_gradient_method(self, simple_gradients):
        result = topological_blankets(simple_gradients, n_objects=2, method='gradient')
        assert result['assignment'].shape == (9,)
        # Should have some blanket and some internal variables
        assert np.sum(result['is_blanket']) > 0
        assert np.sum(~result['is_blanket']) > 0

    def test_spectral_method(self, simple_gradients):
        result = topological_blankets(simple_gradients, n_objects=2, method='spectral')
        assert result['assignment'].shape == (9,)

    def test_hybrid_method(self, simple_gradients):
        result = topological_blankets(simple_gradients, n_objects=2, method='hybrid')
        assert result['assignment'].shape == (9,)

    def test_coupling_method(self, simple_gradients):
        result = topological_blankets(simple_gradients, n_objects=2, method='coupling')
        assert result['assignment'].shape == (9,)

    def test_invalid_method_raises(self, simple_gradients):
        with pytest.raises(ValueError, match="Unknown method"):
            topological_blankets(simple_gradients, n_objects=2, method='invalid')

    def test_consistency_with_original(self, quadratic_ebm_gradients):
        """Verify the packaged version matches the original experiment code."""
        from experiments.quadratic_toy_comparison import (
            topological_blankets as tb_original, compute_metrics
        )

        gradients, truth, cfg = quadratic_ebm_gradients

        old_result = tb_original(gradients, n_objects=2)
        new_result = topological_blankets(gradients, n_objects=2, method='gradient')

        old_metrics = compute_metrics(old_result, truth)
        new_metrics = compute_metrics(new_result, truth)

        assert abs(old_metrics['object_ari'] - new_metrics['object_ari']) < 0.01

    def test_perfect_recovery_strong_signal(self, quadratic_ebm_gradients):
        """At strength=0.8, TB should achieve near-perfect ARI."""
        from experiments.quadratic_toy_comparison import compute_metrics
        gradients, truth, cfg = quadratic_ebm_gradients

        result = topological_blankets(gradients, n_objects=2, method='gradient')
        metrics = compute_metrics(result, truth)
        assert metrics['object_ari'] > 0.9


class TestTopologicalBlanketsClass:
    """Tests for the TopologicalBlankets class API."""

    def test_instantiation_defaults(self):
        tb = TopologicalBlankets()
        assert tb.method == 'hybrid'
        assert tb.n_objects is None

    def test_instantiation_custom(self):
        tb = TopologicalBlankets(method='gradient', n_objects=3, clustering_method='kmeans')
        assert tb.method == 'gradient'
        assert tb.n_objects == 3
        assert tb.clustering_method == 'kmeans'

    def test_fit_returns_self(self, simple_gradients):
        tb = TopologicalBlankets(n_objects=2)
        result = tb.fit(simple_gradients)
        assert result is tb

    def test_get_objects(self, simple_gradients):
        tb = TopologicalBlankets(n_objects=2).fit(simple_gradients)
        objects = tb.get_objects()
        assert isinstance(objects, dict)
        assert len(objects) >= 1

    def test_get_blankets(self, simple_gradients):
        tb = TopologicalBlankets(n_objects=2).fit(simple_gradients)
        blankets = tb.get_blankets()
        assert isinstance(blankets, np.ndarray)

    def test_get_coupling_matrix(self, simple_gradients):
        tb = TopologicalBlankets(n_objects=2).fit(simple_gradients)
        coupling = tb.get_coupling_matrix()
        assert coupling.shape == (9, 9)
        # Coupling should be symmetric
        assert np.allclose(coupling, coupling.T)
        # Diagonal should be zero
        assert np.allclose(np.diag(coupling), 0)

    def test_get_graph(self, simple_gradients):
        tb = TopologicalBlankets(n_objects=2).fit(simple_gradients)
        graph = tb.get_graph()
        assert 'adjacency' in graph
        assert 'laplacian' in graph
        assert 'eigenvalues' in graph
        assert 'eigenvectors' in graph

    def test_get_assignment(self, simple_gradients):
        tb = TopologicalBlankets(n_objects=2).fit(simple_gradients)
        assign = tb.get_assignment()
        assert assign.shape == (9,)
        # Blanket vars should have assignment -1
        blankets = tb.get_blankets()
        for b in blankets:
            assert assign[b] == -1

    def test_not_fitted_raises(self):
        tb = TopologicalBlankets()
        with pytest.raises(RuntimeError, match="not been fitted"):
            tb.get_objects()

    def test_hierarchical_detection(self, simple_gradients):
        tb = TopologicalBlankets()
        hierarchy = tb.fit_hierarchical(simple_gradients, max_levels=2)
        assert isinstance(hierarchy, list)
        assert len(hierarchy) >= 1
        assert 'level' in hierarchy[0]
        assert 'internals' in hierarchy[0]
        assert 'blanket' in hierarchy[0]

    def test_all_methods_produce_valid_output(self, simple_gradients):
        for method in ['gradient', 'spectral', 'hybrid', 'coupling']:
            tb = TopologicalBlankets(method=method, n_objects=2)
            tb.fit(simple_gradients)
            objects = tb.get_objects()
            blankets = tb.get_blankets()
            assert len(objects) > 0 or len(blankets) > 0


class TestEdgeCases:
    """Edge case tests."""

    def test_single_variable(self):
        gradients = np.random.randn(100, 1)
        # Should not crash
        result = topological_blankets(gradients, n_objects=1, method='gradient')
        assert result['assignment'].shape == (1,)

    def test_two_variables(self):
        gradients = np.random.randn(100, 2)
        result = topological_blankets(gradients, n_objects=1, method='gradient')
        assert result['assignment'].shape == (2,)

    def test_high_dimensional(self, high_dim_gradients):
        result = topological_blankets(high_dim_gradients, n_objects=4, method='gradient')
        assert result['assignment'].shape == (20,)

    def test_many_samples(self):
        np.random.seed(42)
        gradients = np.random.randn(5000, 6)
        result = topological_blankets(gradients, n_objects=2, method='gradient')
        assert result['assignment'].shape == (6,)

    def test_few_samples(self):
        np.random.seed(42)
        gradients = np.random.randn(10, 6)
        result = topological_blankets(gradients, n_objects=2, method='gradient')
        assert result['assignment'].shape == (6,)
