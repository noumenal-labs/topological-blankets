"""
Object clustering for Topological Blankets.

After blanket detection separates variables into blanket vs internal,
this module clusters internal variables into distinct objects using
the coupling matrix structure.
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from typing import Dict, Optional


def cluster_internals(features: Dict,
                      is_blanket: np.ndarray,
                      n_clusters: int = 2,
                      method: str = 'spectral') -> np.ndarray:
    """
    Cluster internal (non-blanket) variables by coupling structure.

    Uses the coupling submatrix restricted to internal variables to
    identify distinct objects via spectral clustering.

    Args:
        features: Dictionary from compute_geometric_features() (must contain 'coupling').
        is_blanket: Boolean mask of blanket variables.
        n_clusters: Number of object clusters to find.
        method: Clustering method ('spectral', 'kmeans', or 'agglomerative').

    Returns:
        Array of labels: object index for internal variables, -1 for blanket.
    """
    coupling = features['coupling']
    internal = ~is_blanket
    C_int = coupling[np.ix_(internal, internal)]

    if C_int.shape[0] < 2:
        full_labels = np.full(len(is_blanket), -1, dtype=int)
        if np.sum(internal) == 1:
            full_labels[internal] = 0
        return full_labels

    # Clamp n_clusters to available variables
    n_clusters = min(n_clusters, C_int.shape[0])

    if method == 'spectral':
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                        random_state=42, assign_labels='kmeans')
        try:
            labels = clusterer.fit_predict(C_int + 1e-6)
        except Exception:
            labels = np.zeros(C_int.shape[0], dtype=int)
    elif method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        try:
            labels = clusterer.fit_predict(C_int)
        except Exception:
            labels = np.zeros(C_int.shape[0], dtype=int)
    elif method == 'agglomerative':
        # Convert coupling to distance
        distance = 1.0 / (C_int + 1e-6)
        np.fill_diagonal(distance, 0)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters,
                                             metric='precomputed',
                                             linkage='average')
        try:
            labels = clusterer.fit_predict(distance)
        except Exception:
            labels = np.zeros(C_int.shape[0], dtype=int)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Map back to full variable set
    full_labels = np.full(len(is_blanket), -1, dtype=int)
    full_labels[internal] = labels

    return full_labels
