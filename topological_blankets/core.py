"""
Core Topological Blankets pipeline.

Provides the main TopologicalBlankets class and the topological_blankets()
function for running the full detection pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .features import compute_geometric_features
from .detection import (
    detect_blankets_otsu,
    detect_blankets_gradient,
    detect_blankets_spectral,
    detect_blankets_hybrid,
    detect_blankets_coupling,
    detect_blankets_persistence,
    detect_blankets_pcca,
)
from .clustering import cluster_internals
from .spectral import recursive_spectral_detection


def topological_blankets(gradients: np.ndarray,
                         n_objects: int = 2,
                         method: str = 'gradient',
                         covariance_method: str = 'pearson',
                         sparsify: str = 'threshold') -> Dict:
    """
    Full Topological Blankets pipeline (functional interface).

    Matches the original experiments/quadratic_toy_comparison.py signature.

    Args:
        gradients: Gradient samples of shape (N, n_vars).
        n_objects: Expected number of objects.
        method: Detection method ('gradient', 'spectral', 'hybrid',
                'coupling', 'persistence', 'pcca').
        covariance_method: Covariance estimation method ('pearson' or 'rank').
        sparsify: Sparsification method for adjacency construction.
            One of 'threshold' (default), 'l1', 'stability'.

    Returns:
        Dictionary with:
            - assignment: Object labels per variable (-1 = blanket)
            - is_blanket: Boolean blanket mask
            - threshold: Detection threshold (if applicable)
            - features: Computed geometric features
    """
    features = compute_geometric_features(gradients, covariance_method=covariance_method)
    # Store sparsify so detection methods can use it if needed
    features['_sparsify'] = sparsify
    features['_gradients'] = gradients

    if method == 'gradient':
        is_blanket, tau = detect_blankets_otsu(features)
    elif method == 'spectral':
        result = detect_blankets_spectral(features['hessian_est'])
        is_blanket = result['is_blanket']
        tau = result['eigengap']
    elif method == 'hybrid':
        result = detect_blankets_hybrid(gradients, features['hessian_est'])
        is_blanket = result['is_blanket']
        tau = result['eigengap']
    elif method == 'coupling':
        is_blanket = detect_blankets_coupling(
            features['hessian_est'], features['coupling'], n_objects)
        tau = 0.0
    elif method == 'persistence':
        result = detect_blankets_persistence(
            features, gradients=None, n_bootstrap=0)
        is_blanket = result['is_blanket']
        tau = 0.0
    elif method == 'pcca':
        result = detect_blankets_pcca(
            features['hessian_est'],
            n_clusters=n_objects + 1,
            ambiguity_threshold=0.6,
            normalized_laplacian=True)
        is_blanket = result['is_blanket']
        tau = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    object_assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)

    ret = {
        'assignment': object_assignment,
        'is_blanket': is_blanket,
        'threshold': tau,
        'features': features
    }

    # For PCCA, also return fuzzy memberships
    if method == 'pcca':
        ret['memberships'] = result['memberships']
        ret['max_membership'] = result['max_membership']
        ret['membership_entropy'] = result['membership_entropy']
        ret['pcca_result'] = result

    return ret


class TopologicalBlankets:
    """
    Topological Blankets: Extract discrete Markov blanket structure from
    continuous energy landscape geometry.

    This is the main public API for the topological_blankets package.

    Args:
        method: Detection method. One of:
            - 'gradient': Otsu thresholding on gradient magnitude
            - 'spectral': Friston eigenvector variance heuristic
            - 'hybrid': Spectral with gradient fallback (recommended)
            - 'coupling': Cross-cluster coupling strength (for asymmetric objects)
            - 'persistence': H0 persistence on coupling graph (multi-scale, adaptive)
            - 'pcca': PCCA+ fuzzy partition (soft blanket membership)
        n_objects: Expected number of objects (None for automatic detection).
        clustering_method: Method for object clustering ('spectral', 'kmeans', 'agglomerative').
        eigengap_threshold: Minimum eigengap for spectral method in hybrid mode.
        covariance_method: Method for Hessian estimation via gradient covariance.
            - 'pearson' (default): Standard np.cov(gradients.T).
            - 'rank': Spearman rank-based covariance for nonparanormal
              robustness (Liu, Lafferty, Wasserman 2009).
        sparsify: Sparsification method for adjacency construction. One of:
            - 'threshold' (default): Hard threshold at 0.01.
            - 'l1': L1 soft-thresholding with data-adaptive lambda (BIC or CV).
            - 'stability': Stability selection via L1 on bootstrap subsamples.

    Example:
        >>> from topological_blankets import TopologicalBlankets
        >>> tb = TopologicalBlankets(method='hybrid', n_objects=2)
        >>> result = tb.fit(gradients)
        >>> objects = tb.get_objects()
        >>> blankets = tb.get_blankets()
        >>> coupling = tb.get_coupling_matrix()
    """

    def __init__(self,
                 method: str = 'hybrid',
                 n_objects: Optional[int] = None,
                 clustering_method: str = 'spectral',
                 eigengap_threshold: float = 0.5,
                 covariance_method: str = 'pearson',
                 sparsify: str = 'threshold'):
        self.method = method
        self.n_objects = n_objects
        self.clustering_method = clustering_method
        self.eigengap_threshold = eigengap_threshold
        self.covariance_method = covariance_method
        self.sparsify = sparsify

        # Results (populated after fit)
        self._features = None
        self._is_blanket = None
        self._assignment = None
        self._threshold = None
        self._detection_info = None
        self._gradients = None
        self._n_samples = None

    def fit(self, gradients: np.ndarray) -> 'TopologicalBlankets':
        """
        Run the full TB pipeline on gradient samples.

        Args:
            gradients: Array of shape (N, n_vars) with gradient samples.

        Returns:
            self (for method chaining).
        """
        n_objects = self.n_objects if self.n_objects is not None else 2

        self._gradients = gradients
        self._n_samples = gradients.shape[0]

        self._features = compute_geometric_features(
            gradients, covariance_method=self.covariance_method)

        if self.method == 'gradient':
            self._is_blanket, self._threshold = detect_blankets_otsu(self._features)
            self._detection_info = {'method_used': 'gradient'}
        elif self.method == 'spectral':
            result = detect_blankets_spectral(self._features['hessian_est'])
            self._is_blanket = result['is_blanket']
            self._threshold = result['eigengap']
            self._detection_info = result
        elif self.method == 'hybrid':
            result = detect_blankets_hybrid(gradients, self._features['hessian_est'],
                                            self.eigengap_threshold)
            self._is_blanket = result['is_blanket']
            self._threshold = result['eigengap']
            self._detection_info = result
        elif self.method == 'coupling':
            self._is_blanket = detect_blankets_coupling(
                self._features['hessian_est'],
                self._features['coupling'], n_objects)
            self._threshold = 0.0
            self._detection_info = {'method_used': 'coupling'}
        elif self.method == 'persistence':
            result = detect_blankets_persistence(
                self._features, gradients=gradients,
                n_bootstrap=200,
                covariance_method=self.covariance_method)
            self._is_blanket = result['is_blanket']
            self._threshold = 0.0
            self._detection_info = {
                'method_used': 'persistence',
                'persistence_diagram': result['persistence_diagram'],
                'significant_features': result['significant_features'],
                'persistence_values': result['persistence_values'],
                'bootstrap': result['bootstrap'],
            }
        elif self.method == 'pcca':
            result = detect_blankets_pcca(
                self._features['hessian_est'],
                n_clusters=n_objects + 1,
                ambiguity_threshold=0.6,
                normalized_laplacian=True)
            self._is_blanket = result['is_blanket']
            self._threshold = 0.0
            self._detection_info = {
                'method_used': 'pcca',
                'memberships': result['memberships'],
                'hard_labels': result['hard_labels'],
                'max_membership': result['max_membership'],
                'membership_entropy': result['membership_entropy'],
                'eigvals': result['eigvals'],
                'eigvecs': result['eigvecs'],
                'simplex_vertices': result['simplex_vertices'],
                'ambiguity_threshold': result['ambiguity_threshold'],
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Auto-detect n_objects from eigengap if not specified
        if self.n_objects is None and self._features is not None:
            from .spectral import compute_eigengap, build_adjacency_from_hessian, build_graph_laplacian
            from scipy.linalg import eigh
            A = build_adjacency_from_hessian(
                self._features['hessian_est'],
                sparsify=self.sparsify,
                n_samples=self._n_samples,
                gradients=gradients,
            )
            L = build_graph_laplacian(A)
            eigvals, _ = eigh(L)
            auto_n, _ = compute_eigengap(eigvals[:min(10, len(eigvals))])
            n_objects = max(2, auto_n - 1)  # Subtract 1 for blanket cluster

        self._assignment = cluster_internals(
            self._features, self._is_blanket,
            n_clusters=n_objects, method=self.clustering_method
        )

        return self

    def fit_energy(self, energy_fn, samples: np.ndarray,
                   method: str = 'autograd') -> 'TopologicalBlankets':
        """
        Fit from an energy function by computing gradients automatically.

        Args:
            energy_fn: Callable energy function E(x) -> scalar.
            samples: Array of shape (N, n_vars) of sample points.
            method: Gradient computation method ('autograd' for finite differences).

        Returns:
            self (for method chaining).
        """
        from .extractors import extract_gradients_callable
        gradients = extract_gradients_callable(energy_fn, samples, method=method)
        return self.fit(gradients)

    def fit_hierarchical(self, gradients: np.ndarray,
                          max_levels: int = 3) -> List[Dict]:
        """
        Run recursive hierarchical detection.

        Args:
            gradients: Array of shape (N, n_vars) with gradient samples.
            max_levels: Maximum recursion depth.

        Returns:
            List of dictionaries, one per hierarchy level.
        """
        self._features = compute_geometric_features(
            gradients, covariance_method=self.covariance_method)
        H_est = self._features['hessian_est']
        return recursive_spectral_detection(H_est, max_levels=max_levels)

    def get_objects(self) -> Dict[int, np.ndarray]:
        """
        Get detected objects as a dict mapping object ID to variable indices.

        Returns:
            Dictionary {object_id: array of variable indices}.
        """
        self._check_fitted()
        objects = {}
        for label in np.unique(self._assignment):
            if label >= 0:
                objects[int(label)] = np.where(self._assignment == label)[0]
        return objects

    def get_blankets(self) -> np.ndarray:
        """
        Get indices of blanket variables.

        Returns:
            Array of blanket variable indices.
        """
        self._check_fitted()
        return np.where(self._is_blanket)[0]

    def get_graph(self) -> Dict:
        """
        Get the coupling graph structure.

        Returns:
            Dictionary with adjacency matrix, Laplacian, and eigenvalues.
        """
        self._check_fitted()
        from .spectral import build_adjacency_from_hessian, build_graph_laplacian
        from scipy.linalg import eigh

        A = build_adjacency_from_hessian(
            self._features['hessian_est'],
            sparsify=self.sparsify,
            n_samples=self._n_samples,
            gradients=self._gradients,
        )
        L = build_graph_laplacian(A)
        eigvals, eigvecs = eigh(L)

        return {
            'adjacency': A,
            'laplacian': L,
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs
        }

    def get_coupling_matrix(self) -> np.ndarray:
        """
        Get the normalized coupling matrix.

        Returns:
            Coupling matrix of shape (n_vars, n_vars).
        """
        self._check_fitted()
        return self._features['coupling']

    def get_assignment(self) -> np.ndarray:
        """
        Get the full variable assignment array.

        Returns:
            Array of labels: object index for internal variables, -1 for blanket.
        """
        self._check_fitted()
        return self._assignment

    def get_features(self) -> Dict:
        """
        Get all computed geometric features.

        Returns:
            Features dictionary from compute_geometric_features().
        """
        self._check_fitted()
        return self._features

    def get_memberships(self) -> Optional[np.ndarray]:
        """
        Get PCCA+ fuzzy membership vectors (only available when method='pcca').

        Returns:
            Membership matrix of shape (n_vars, n_clusters) if method='pcca',
            or None if a different detection method was used.
        """
        self._check_fitted()
        if self._detection_info and 'memberships' in self._detection_info:
            return self._detection_info['memberships']
        return None

    def get_detection_info(self) -> Optional[Dict]:
        """
        Get detailed detection method results.

        Returns:
            Dictionary of detection-specific results, or None if not fitted.
        """
        self._check_fitted()
        return self._detection_info

    def _check_fitted(self):
        if self._features is None:
            raise RuntimeError("TopologicalBlankets has not been fitted yet. Call .fit() first.")
