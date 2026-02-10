"""
Tests for topological_blankets.spectral module.
"""

import numpy as np
import pytest
from topological_blankets.spectral import (
    build_adjacency_from_hessian,
    build_graph_laplacian,
    compute_eigengap,
    spectral_partition,
    identify_blanket_from_spectrum,
    schur_complement_reduction,
    recursive_spectral_detection,
)


class TestBuildAdjacency:

    def test_symmetric_output(self):
        H = np.array([[1.0, 0.5, 0.1],
                       [0.5, 1.0, 0.3],
                       [0.1, 0.3, 1.0]])
        A = build_adjacency_from_hessian(H)
        assert np.allclose(A, A.T)

    def test_no_self_loops(self):
        H = np.eye(5) * 2.0
        A = build_adjacency_from_hessian(H)
        assert np.allclose(np.diag(A), 0)

    def test_threshold_effect(self):
        H = np.array([[1.0, 0.5, 0.01],
                       [0.5, 1.0, 0.01],
                       [0.01, 0.01, 1.0]])
        A_low = build_adjacency_from_hessian(H, threshold=0.001)
        A_high = build_adjacency_from_hessian(H, threshold=0.9)
        assert A_low.sum() >= A_high.sum()

    def test_binary_output(self):
        H = np.random.randn(5, 5)
        H = H @ H.T  # Make symmetric
        A = build_adjacency_from_hessian(H)
        assert set(np.unique(A)).issubset({0.0, 1.0})


class TestBuildLaplacian:

    def test_unnormalized(self):
        A = np.array([[0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0]], dtype=float)
        L = build_graph_laplacian(A)
        # L = D - A, so L should have row sums = 0
        assert np.allclose(L.sum(axis=1), 0, atol=1e-10)

    def test_positive_semidefinite(self):
        A = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=float)
        L = build_graph_laplacian(A)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)

    def test_normalized_laplacian(self):
        A = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=float)
        L = build_graph_laplacian(A, normalized=True)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)
        assert np.all(eigvals <= 2.0 + 1e-10)


class TestEigengap:

    def test_clear_gap(self):
        eigvals = np.array([0.0, 0.01, 0.02, 2.0, 2.1])
        n_clusters, gap = compute_eigengap(eigvals)
        assert gap > 1.0  # Gap between 0.02 and 2.0
        assert n_clusters == 3

    def test_no_gap(self):
        eigvals = np.array([0.0, 1.0, 2.0, 3.0])
        n_clusters, gap = compute_eigengap(eigvals)
        assert gap == 1.0

    def test_single_eigenvalue(self):
        eigvals = np.array([0.0])
        n_clusters, gap = compute_eigengap(eigvals)
        assert n_clusters == 1
        assert gap == 0.0


class TestSpectralPartition:

    def test_output_shapes(self):
        np.random.seed(42)
        H = np.random.randn(10, 10)
        H = H @ H.T
        A = build_adjacency_from_hessian(H, threshold=0.1)
        L = build_graph_laplacian(A)

        labels, eigvals, eigvecs = spectral_partition(L, n_partitions=3)
        assert labels.shape == (10,)
        assert len(eigvals) > 0
        assert eigvecs.shape[0] == 10

    def test_correct_number_of_clusters(self):
        np.random.seed(42)
        H = np.random.randn(10, 10)
        H = H @ H.T
        A = build_adjacency_from_hessian(H, threshold=0.1)
        L = build_graph_laplacian(A)

        labels, _, _ = spectral_partition(L, n_partitions=3)
        assert len(np.unique(labels)) <= 3


class TestIdentifyBlanket:

    def test_returns_boolean_mask(self):
        eigvals = np.array([0, 0.1, 0.5, 2.0, 2.5])
        eigvecs = np.random.randn(5, 5)
        labels = np.array([0, 0, 1, 2, 2])

        is_blanket = identify_blanket_from_spectrum(eigvals, eigvecs, labels)
        assert is_blanket.dtype == bool
        assert is_blanket.shape == (5,)


class TestSchurComplement:

    def test_preserves_shape(self):
        H = np.random.randn(6, 6)
        H = H @ H.T
        keep = np.array([0, 1, 2, 3])
        elim = np.array([4, 5])

        H_eff = schur_complement_reduction(H, keep, elim)
        assert H_eff.shape == (4, 4)

    def test_empty_elimination(self):
        H = np.random.randn(4, 4)
        H = H @ H.T
        keep = np.array([0, 1, 2, 3])
        elim = np.array([], dtype=int)

        H_eff = schur_complement_reduction(H, keep, elim)
        assert np.allclose(H_eff, H)


class TestRecursiveDetection:

    def test_returns_list(self):
        np.random.seed(42)
        H = np.random.randn(10, 10)
        H = H @ H.T
        hierarchy = recursive_spectral_detection(H, max_levels=2)
        assert isinstance(hierarchy, list)

    def test_max_levels_respected(self):
        np.random.seed(42)
        H = np.random.randn(20, 20)
        H = H @ H.T
        hierarchy = recursive_spectral_detection(H, max_levels=2)
        assert len(hierarchy) <= 2

    def test_hierarchy_structure(self):
        np.random.seed(42)
        H = np.random.randn(12, 12)
        H = H @ H.T
        hierarchy = recursive_spectral_detection(H, max_levels=3)
        if len(hierarchy) > 0:
            level = hierarchy[0]
            assert 'level' in level
            assert 'internals' in level
            assert 'blanket' in level
            assert 'external' in level
            assert 'eigengap' in level
