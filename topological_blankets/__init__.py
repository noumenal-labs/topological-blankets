"""
Topological Blankets: Extracting Discrete Markov Blanket Structure
from Continuous Energy Landscape Geometry.

This package implements the Topological Blankets (TB) method for
discovering Markov blanket structure in energy-based models.

Usage:
    from topological_blankets import TopologicalBlankets

    tb = TopologicalBlankets(method='hybrid', n_objects=2)
    result = tb.fit(gradients)

    objects = tb.get_objects()
    blankets = tb.get_blankets()
    coupling = tb.get_coupling_matrix()
"""

from .core import TopologicalBlankets, topological_blankets
from .features import compute_geometric_features
from .detection import (
    detect_blankets_otsu,
    detect_blankets_gradient,
    detect_blankets_spectral,
    detect_blankets_hybrid,
    detect_blankets_coupling,
    detect_blankets_persistence,
    detect_blankets_pcca,
    compute_persistence_diagram,
    compute_persistence_bootstrap,
)
from .pcca import pcca_plus, pcca_blanket_detection
from .clustering import cluster_internals
from .spectral import (
    build_adjacency_from_hessian,
    build_graph_laplacian,
    spectral_partition,
    identify_blanket_from_spectrum,
    compute_eigengap,
    recursive_spectral_detection,
    schur_complement_reduction,
    l1_sparsify_hessian,
    select_lambda_bic,
    select_lambda_cv,
    stability_selection,
)

__all__ = [
    'TopologicalBlankets',
    'topological_blankets',
    'compute_geometric_features',
    'detect_blankets_otsu',
    'detect_blankets_gradient',
    'detect_blankets_spectral',
    'detect_blankets_hybrid',
    'detect_blankets_coupling',
    'detect_blankets_persistence',
    'detect_blankets_pcca',
    'pcca_plus',
    'pcca_blanket_detection',
    'compute_persistence_diagram',
    'compute_persistence_bootstrap',
    'cluster_internals',
    'build_adjacency_from_hessian',
    'build_graph_laplacian',
    'spectral_partition',
    'identify_blanket_from_spectrum',
    'compute_eigengap',
    'recursive_spectral_detection',
    'schur_complement_reduction',
    'l1_sparsify_hessian',
    'select_lambda_bic',
    'select_lambda_cv',
    'stability_selection',
]

__version__ = '0.2.0'
