"""
Spectral Blanket Detection: Friston (2025) Methods
===================================================

Implements Friston's spectral approach to Markov blanket detection from
"A Free Energy Principle: On the Nature of Things" (2025).

Key methods:
1. Graph Laplacian from Hessian/Jacobian
2. Eigenmodes for partition (slow=internal, mid=blanket, fast=external)
3. Recursive hierarchical detection
4. Comparison with gradient-based thresholding

CONSTRAINT: No VERSES AI code (github.com/VersesTech) is used.
All implementations are original.

Based on Friston (2025) pp. 48-51, 58-64, 67-70, 213-217.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh, schur
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =============================================================================
# Core Spectral Methods (Friston 2025)
# =============================================================================

def build_adjacency_from_hessian(H: np.ndarray,
                                  threshold: float = 0.01,
                                  normalize: bool = True) -> np.ndarray:
    """
    Construct adjacency matrix from Hessian (Friston pp. 48-51).

    A_ij = 1 if |H_ij| > threshold (coupling exists)
           0 otherwise

    In Friston's formulation:
    - H ≈ -Γ^{-1} J where J is Jacobian of flow
    - Non-zero entries indicate direct influence
    """
    if normalize:
        # Normalize by diagonal (self-coupling strength)
        D = np.sqrt(np.abs(np.diag(H)) + 1e-8)
        H_norm = np.abs(H) / np.outer(D, D)
    else:
        H_norm = np.abs(H)

    # Threshold to binary adjacency
    A = (H_norm > threshold).astype(float)
    np.fill_diagonal(A, 0)  # No self-loops

    return A


def build_graph_laplacian(A: np.ndarray,
                          normalized: bool = False) -> np.ndarray:
    """
    Build graph Laplacian from adjacency (Friston pp. 48-51).

    L = D - A  (unnormalized)
    L_norm = I - D^{-1/2} A D^{-1/2}  (normalized)

    Properties:
    - L is positive semi-definite
    - Smallest eigenvalue is 0 (constant eigenvector)
    - Number of zero eigenvalues = number of connected components
    - Eigengap indicates partition strength
    """
    D = np.diag(A.sum(axis=1))

    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
        L = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = D - A

    return L


def spectral_partition(L: np.ndarray,
                       n_partitions: int = 3,
                       method: str = 'kmeans') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition variables using spectral embedding (Friston pp. 58-61).

    Process:
    1. Compute k smallest eigenvalues/vectors
    2. Embed points in eigenvector space
    3. Cluster to identify: internal, blanket, external

    Returns:
        labels: Partition assignment per variable
        eigvals: Eigenvalues (sorted ascending)
        eigvecs: Corresponding eigenvectors
    """
    k = min(n_partitions + 2, len(L) - 1)

    # Eigen-decomposition (smallest eigenvalues)
    try:
        if len(L) > 100:
            # Sparse solver for large matrices
            eigvals, eigvecs = eigsh(L, k=k, which='SM')
        else:
            eigvals, eigvecs = eigh(L)
            eigvals = eigvals[:k]
            eigvecs = eigvecs[:, :k]
    except Exception as e:
        print(f"Eigendecomposition failed: {e}")
        return np.zeros(len(L), dtype=int), np.array([0]), np.zeros((len(L), 1))

    # Sort by eigenvalue
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Cluster on eigenvector embedding (skip constant mode)
    embedding = eigvecs[:, 1:n_partitions+1]

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_partitions, random_state=42, n_init=10)
    else:
        # Spectral with precomputed affinity
        clusterer = SpectralClustering(n_clusters=n_partitions,
                                        affinity='nearest_neighbors',
                                        random_state=42)

    try:
        labels = clusterer.fit_predict(embedding)
    except Exception:
        labels = np.zeros(len(L), dtype=int)

    return labels, eigvals, eigvecs


def identify_blanket_from_spectrum(eigvals: np.ndarray,
                                    eigvecs: np.ndarray,
                                    labels: np.ndarray) -> np.ndarray:
    """
    Identify blanket variables from spectral analysis (Friston pp. 67-70).

    Friston's interpretation:
    - Slow modes (small eigenvalues): Stable internal states
    - Fast modes (large eigenvalues): Rapidly mixing external
    - Intermediate: Blanket (mediating structure)

    Heuristic: Blanket = cluster with highest eigenvector variance
    (connects to multiple regions, so varied eigenvector values).

    Note: On block-structured quadratic EBMs, this heuristic tends to
    over-assign blanket (high recall, low precision). The gradient-based
    method with minority-group detection is more reliable for these systems.
    """
    n_clusters = len(np.unique(labels))

    # Compute eigenvector variance per cluster
    cluster_variance = []
    for c in range(n_clusters):
        mask = labels == c
        if np.sum(mask) > 0:
            # Variance across first few non-trivial eigenvectors
            var = np.var(eigvecs[mask, 1:min(4, eigvecs.shape[1])], axis=0).mean()
            cluster_variance.append((c, var))
        else:
            cluster_variance.append((c, 0))

    # Blanket = highest variance cluster (most "connecting")
    cluster_variance.sort(key=lambda x: x[1], reverse=True)
    blanket_cluster = cluster_variance[0][0]

    is_blanket = labels == blanket_cluster

    return is_blanket


def compute_eigengap(eigvals: np.ndarray) -> Tuple[int, float]:
    """
    Find eigengap indicating natural partition (Friston pp. 58-61).

    The eigengap heuristic: Largest gap in eigenvalue spectrum
    indicates number of well-separated clusters.
    """
    if len(eigvals) < 2:
        return 1, 0.0

    gaps = np.diff(eigvals)
    best_gap_idx = np.argmax(gaps)
    best_gap = gaps[best_gap_idx]

    # Number of clusters = position of gap + 1
    n_clusters = best_gap_idx + 1

    return n_clusters, best_gap


# =============================================================================
# Recursive Hierarchical Detection (Friston pp. 53-64)
# =============================================================================

def schur_complement_reduction(H: np.ndarray,
                                keep_idx: np.ndarray,
                                eliminate_idx: np.ndarray) -> np.ndarray:
    """
    Adiabatic elimination via Schur complement (Friston pp. 58-64).

    When eliminating fast (external) modes, effective Hessian for
    slow (internal+blanket) modes is:

    H_eff = H_slow - H_cross^T @ H_fast^{-1} @ H_cross

    This "integrates out" the fast variables.
    """
    if len(eliminate_idx) == 0:
        return H[np.ix_(keep_idx, keep_idx)]

    H_slow = H[np.ix_(keep_idx, keep_idx)]
    H_fast = H[np.ix_(eliminate_idx, eliminate_idx)]
    H_cross = H[np.ix_(eliminate_idx, keep_idx)]

    try:
        H_fast_inv = np.linalg.inv(H_fast + 1e-6 * np.eye(len(H_fast)))
        H_eff = H_slow - H_cross.T @ H_fast_inv @ H_cross
    except np.linalg.LinAlgError:
        # If inversion fails, just use submatrix
        H_eff = H_slow

    return H_eff


def recursive_spectral_detection(H: np.ndarray,
                                  max_levels: int = 3,
                                  min_vars: int = 3,
                                  adjacency_threshold: float = 0.01) -> List[Dict]:
    """
    Friston-style recursive blanket detection (pp. 53-64).

    Algorithm:
    1. Detect blankets via spectral method at current scale
    2. Identify fast (external) modes
    3. Adiabatically eliminate fast modes (Schur complement)
    4. Remaining slow+blanket = new particle
    5. Repeat at coarser scale

    Returns hierarchy of partitions at each scale.
    """
    hierarchy = []
    current_H = H.copy()
    current_vars = list(range(H.shape[0]))

    for level in range(max_levels):
        if len(current_vars) < min_vars:
            break

        # Build Laplacian
        A = build_adjacency_from_hessian(current_H, threshold=adjacency_threshold)
        L = build_graph_laplacian(A)

        # Find natural partition size from eigengap
        eigvals_all, _ = eigh(L)
        n_clusters, eigengap = compute_eigengap(eigvals_all[:min(10, len(eigvals_all))])
        n_clusters = max(2, min(3, n_clusters))  # Clamp to 2-3

        # Spectral partition
        labels, eigvals, eigvecs = spectral_partition(L, n_partitions=n_clusters)

        # Identify blanket
        is_blanket = identify_blanket_from_spectrum(eigvals, eigvecs, labels)

        # Classify: internal (slow), blanket (mid), external (fast)
        # Use eigenvalue contribution per cluster
        cluster_eigval_score = []
        for c in range(n_clusters):
            mask = labels == c
            if np.sum(mask) > 0:
                # Average projection onto slow vs fast modes
                slow_proj = np.mean(np.abs(eigvecs[mask, 1:3]))
                cluster_eigval_score.append((c, slow_proj))

        cluster_eigval_score.sort(key=lambda x: x[1])

        if len(cluster_eigval_score) >= 3:
            internal_cluster = cluster_eigval_score[0][0]  # Slowest
            blanket_cluster = cluster_eigval_score[1][0]   # Mid
            external_cluster = cluster_eigval_score[2][0]  # Fastest
        elif len(cluster_eigval_score) == 2:
            internal_cluster = cluster_eigval_score[0][0]
            blanket_cluster = cluster_eigval_score[1][0]
            external_cluster = None
        else:
            internal_cluster = 0
            blanket_cluster = 0
            external_cluster = None

        # Map back to original variable indices
        internals = [current_vars[i] for i in range(len(labels))
                     if labels[i] == internal_cluster]
        blanket = [current_vars[i] for i in range(len(labels))
                   if labels[i] == blanket_cluster]
        external = [current_vars[i] for i in range(len(labels))
                    if external_cluster is not None and labels[i] == external_cluster]

        hierarchy.append({
            'level': level,
            'internals': internals,
            'blanket': blanket,
            'external': external,
            'eigengap': eigengap,
            'n_vars': len(current_vars),
            'eigvals': eigvals[:6] if len(eigvals) > 6 else eigvals
        })

        # Prepare for next level: Keep internals + blanket, eliminate external
        if external_cluster is not None:
            keep_local = [i for i in range(len(labels))
                          if labels[i] != external_cluster]
            elim_local = [i for i in range(len(labels))
                          if labels[i] == external_cluster]

            if len(keep_local) < min_vars:
                break

            # Schur complement reduction
            current_H = schur_complement_reduction(current_H,
                                                    np.array(keep_local),
                                                    np.array(elim_local))
            current_vars = [current_vars[i] for i in keep_local]
        else:
            # No external identified; try splitting internals further
            keep_local = [i for i in range(len(labels))
                          if labels[i] == internal_cluster]
            if len(keep_local) < min_vars:
                break
            current_H = current_H[np.ix_(keep_local, keep_local)]
            current_vars = [current_vars[i] for i in keep_local]

    return hierarchy


# =============================================================================
# Comparison: Spectral vs Gradient-Based Detection
# =============================================================================

def gradient_based_detection(gradients: np.ndarray,
                              method: str = 'otsu') -> Tuple[np.ndarray, float]:
    """
    Gradient-magnitude based blanket detection.

    Uses Otsu to separate variables into two groups by gradient magnitude,
    then assigns the minority group as blanket (fewer mediating variables
    than internal variables in typical configurations).
    """
    grad_magnitude = np.mean(np.abs(gradients), axis=0)

    if method == 'otsu':
        from skimage.filters import threshold_otsu
        try:
            tau = threshold_otsu(grad_magnitude)
        except ValueError:
            tau = np.percentile(grad_magnitude, 80)
    elif method == 'percentile':
        tau = np.percentile(grad_magnitude, 80)
    else:
        tau = np.median(grad_magnitude) * 1.5

    high_group = grad_magnitude > tau
    low_group = ~high_group
    # Blanket = minority group
    if np.sum(low_group) <= np.sum(high_group):
        is_blanket = low_group
    else:
        is_blanket = high_group

    return is_blanket, tau


def hybrid_detection(gradients: np.ndarray,
                      H_est: np.ndarray,
                      eigengap_threshold: float = 0.5) -> Dict:
    """
    Hybrid: Use spectral if eigengap strong, else fall back to gradient.

    This combines Friston's rigorous method with practical gradient heuristic.
    """
    # Spectral analysis
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    labels, eigvals, eigvecs = spectral_partition(L, n_partitions=3)

    _, eigengap = compute_eigengap(eigvals)

    if eigengap > eigengap_threshold:
        # Strong spectral structure, try spectral partition
        is_blanket = identify_blanket_from_spectrum(eigvals, eigvecs, labels)
        # Sanity check: blanket should be a minority of variables.
        # If spectral assigns > 50% as blanket, the heuristic failed.
        if np.sum(is_blanket) > len(is_blanket) / 2:
            is_blanket, _ = gradient_based_detection(gradients)
            method_used = 'gradient_fallback'
        else:
            method_used = 'spectral'
    else:
        # Weak spectral structure, fall back to gradient
        is_blanket, _ = gradient_based_detection(gradients)
        method_used = 'gradient'

    return {
        'is_blanket': is_blanket,
        'method_used': method_used,
        'eigengap': eigengap,
        'spectral_labels': labels,
        'eigvals': eigvals
    }


# =============================================================================
# Experiment: Compare Methods on Quadratic Toy
# =============================================================================

def build_precision_matrix_hierarchical(n_levels: int = 2,
                                         vars_per_object: int = 3,
                                         vars_per_blanket: int = 2,
                                         intra_strength: float = 6.0,
                                         blanket_strength: float = 0.8,
                                         inter_level_strength: float = 0.3) -> Tuple[np.ndarray, Dict]:
    """
    Build hierarchical precision matrix for testing recursive detection.

    Structure:
    - Level 0: Fine objects with local blankets
    - Level 1: Coarse objects (aggregates of level 0) with global blanket
    """
    # Level 0: Two objects with blanket
    n_obj_0 = 2
    n_vars_0 = n_obj_0 * vars_per_object + vars_per_blanket

    Theta = np.zeros((n_vars_0, n_vars_0))

    # Object blocks
    for i in range(n_obj_0):
        start = i * vars_per_object
        end = start + vars_per_object
        Theta[start:end, start:end] = intra_strength
        np.fill_diagonal(Theta[start:end, start:end],
                        intra_strength * vars_per_object)

    # Blanket block
    blanket_start = n_obj_0 * vars_per_object
    Theta[blanket_start:, blanket_start:] = 1.0
    np.fill_diagonal(Theta[blanket_start:, blanket_start:], vars_per_blanket)

    # Cross-couplings via blanket
    for i in range(n_obj_0):
        obj_start = i * vars_per_object
        obj_end = obj_start + vars_per_object
        Theta[obj_start:obj_end, blanket_start:] = blanket_strength
        Theta[blanket_start:, obj_start:obj_end] = blanket_strength

    # Ensure positive definite
    Theta = (Theta + Theta.T) / 2.0
    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals.min() < 0.1:
        Theta += np.eye(n_vars_0) * (0.1 - eigvals.min() + 0.1)

    # Ground truth
    ground_truth = {
        'level_0': {
            'object_0': list(range(vars_per_object)),
            'object_1': list(range(vars_per_object, 2*vars_per_object)),
            'blanket': list(range(2*vars_per_object, n_vars_0))
        }
    }

    return Theta, ground_truth


def run_spectral_experiment():
    """
    Compare spectral vs gradient detection on quadratic toy.
    """
    print("=" * 70)
    print("Spectral Blanket Detection: Friston (2025) Methods")
    print("=" * 70)
    print("\nComparing: Spectral (Friston) vs Gradient (Original) vs Hybrid\n")

    # Build toy system
    Theta, ground_truth = build_precision_matrix_hierarchical(
        vars_per_object=4,
        vars_per_blanket=3,
        intra_strength=8.0,
        blanket_strength=1.0
    )
    n_vars = Theta.shape[0]

    print(f"System: {n_vars} variables")
    print(f"Ground truth blanket: {ground_truth['level_0']['blanket']}")

    # Langevin sampling
    print("\nSampling via Langevin dynamics...")
    np.random.seed(42)

    n_samples = 5000
    n_steps = 50
    step_size = 0.005
    temp = 0.1

    samples = []
    gradients = []
    x = np.random.randn(n_vars)

    for i in range(n_samples * n_steps):
        grad = Theta @ x
        noise = np.sqrt(2 * step_size * temp) * np.random.randn(n_vars)
        x = x - step_size * grad + noise

        if i % n_steps == 0:
            samples.append(x.copy())
            gradients.append(grad.copy())

    samples = np.array(samples)
    gradients = np.array(gradients)

    # Estimate Hessian via gradient covariance
    H_est = np.cov(gradients.T)

    print(f"Collected {len(samples)} samples")

    # Method 1: Gradient-based (original)
    print("\n" + "-" * 40)
    print("Method 1: Gradient Magnitude (Original)")
    is_blanket_grad, tau = gradient_based_detection(gradients)
    print(f"  Threshold tau = {tau:.3f}")
    print(f"  Detected blanket: {list(np.where(is_blanket_grad)[0])}")

    # Method 2: Spectral (Friston)
    print("\n" + "-" * 40)
    print("Method 2: Spectral Laplacian (Friston 2025)")
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    labels, eigvals, eigvecs = spectral_partition(L, n_partitions=3)
    is_blanket_spectral = identify_blanket_from_spectrum(eigvals, eigvecs, labels)

    print(f"  Eigenvalues: {eigvals[:6]}")
    _, eigengap = compute_eigengap(eigvals)
    print(f"  Eigengap: {eigengap:.3f}")
    print(f"  Spectral labels: {labels}")
    print(f"  Detected blanket: {list(np.where(is_blanket_spectral)[0])}")

    # Method 3: Hybrid
    print("\n" + "-" * 40)
    print("Method 3: Hybrid (Spectral + Gradient Fallback)")
    hybrid_result = hybrid_detection(gradients, H_est, eigengap_threshold=0.3)
    print(f"  Method used: {hybrid_result['method_used']}")
    print(f"  Eigengap: {hybrid_result['eigengap']:.3f}")
    print(f"  Detected blanket: {list(np.where(hybrid_result['is_blanket'])[0])}")

    # Method 4: Recursive hierarchical
    print("\n" + "-" * 40)
    print("Method 4: Recursive Hierarchical (Friston pp. 53-64)")
    hierarchy = recursive_spectral_detection(H_est, max_levels=3)
    for level_info in hierarchy:
        print(f"  Level {level_info['level']}:")
        print(f"    Internals: {level_info['internals']}")
        print(f"    Blanket: {level_info['blanket']}")
        print(f"    External: {level_info['external']}")
        print(f"    Eigengap: {level_info['eigengap']:.3f}")

    # Accuracy comparison
    print("\n" + "=" * 40)
    print("Accuracy vs Ground Truth")
    print("=" * 40)

    true_blanket = set(ground_truth['level_0']['blanket'])
    methods = [
        ('Gradient', is_blanket_grad),
        ('Spectral', is_blanket_spectral),
        ('Hybrid', hybrid_result['is_blanket']),
    ]

    for name, pred in methods:
        pred_blanket = set(np.where(pred)[0])
        precision = len(pred_blanket & true_blanket) / len(pred_blanket) if pred_blanket else 0
        recall = len(pred_blanket & true_blanket) / len(true_blanket) if true_blanket else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name:10s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # Visualization
    visualize_spectral_analysis(H_est, gradients, ground_truth, labels, eigvecs, eigvals)

    # Save structured results
    method_results = {}
    for name, pred in methods:
        pred_blanket = set(np.where(pred)[0])
        precision = len(pred_blanket & true_blanket) / len(pred_blanket) if pred_blanket else 0
        recall = len(pred_blanket & true_blanket) / len(true_blanket) if true_blanket else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        method_results[name.lower()] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detected_blanket': sorted(list(pred_blanket)),
        }

    method_results['hybrid']['eigengap'] = float(hybrid_result['eigengap'])
    method_results['hybrid']['method_used'] = hybrid_result['method_used']

    hierarchy_results = []
    for level_info in hierarchy:
        hierarchy_results.append({
            'level': level_info['level'],
            'internals': level_info['internals'],
            'blanket': level_info['blanket'],
            'external': level_info['external'],
            'eigengap': float(level_info['eigengap']),
            'n_vars': level_info['n_vars'],
        })

    all_metrics = {
        'methods': method_results,
        'hierarchy': hierarchy_results,
        'ground_truth_blanket': sorted(list(true_blanket)),
        'eigvals': eigvals[:6].tolist() if len(eigvals) > 6 else eigvals.tolist(),
    }

    save_results('spectral_friston_comparison', all_metrics,
                 {'vars_per_object': 4, 'vars_per_blanket': 3,
                  'intra_strength': 8.0, 'blanket_strength': 1.0},
                 notes='Spectral vs gradient vs hybrid blanket detection')

    return {
        'gradient': is_blanket_grad,
        'spectral': is_blanket_spectral,
        'hybrid': hybrid_result,
        'hierarchy': hierarchy,
        'ground_truth': ground_truth
    }


def visualize_spectral_analysis(H_est: np.ndarray,
                                 gradients: np.ndarray,
                                 ground_truth: Dict,
                                 labels: np.ndarray,
                                 eigvecs: np.ndarray,
                                 eigvals: np.ndarray):
    """
    Visualize spectral analysis results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Hessian estimate
    ax = axes[0, 0]
    im = ax.imshow(np.abs(H_est), cmap='hot')
    ax.set_title('|Hessian Estimate|')
    ax.set_xlabel('Variable j')
    ax.set_ylabel('Variable i')
    plt.colorbar(im, ax=ax)

    # Mark ground truth boundaries
    blanket_start = min(ground_truth['level_0']['blanket'])
    ax.axhline(y=blanket_start - 0.5, color='cyan', linestyle='--', linewidth=2)
    ax.axvline(x=blanket_start - 0.5, color='cyan', linestyle='--', linewidth=2)

    # 2. Graph Laplacian
    ax = axes[0, 1]
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    im = ax.imshow(L, cmap='coolwarm')
    ax.set_title('Graph Laplacian')
    plt.colorbar(im, ax=ax)

    # 3. Eigenvalue spectrum
    ax = axes[0, 2]
    ax.bar(range(len(eigvals)), eigvals)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Laplacian Spectrum\n(Gaps indicate partitions)')

    # Highlight eigengap
    if len(eigvals) > 1:
        gaps = np.diff(eigvals)
        max_gap_idx = np.argmax(gaps)
        ax.axvline(x=max_gap_idx + 0.5, color='red', linestyle='--',
                  label=f'Max gap at {max_gap_idx}')
        ax.legend()

    # 4. Eigenvector embedding (first 2 non-trivial)
    ax = axes[1, 0]
    if eigvecs.shape[1] >= 3:
        colors = labels
        scatter = ax.scatter(eigvecs[:, 1], eigvecs[:, 2], c=colors, cmap='tab10', s=100)
        for i in range(len(labels)):
            ax.annotate(str(i), (eigvecs[i, 1], eigvecs[i, 2]), fontsize=8)
        ax.set_xlabel('Fiedler Vector (v₁)')
        ax.set_ylabel('Second Vector (v₂)')
        ax.set_title('Spectral Embedding')
        plt.colorbar(scatter, ax=ax, label='Cluster')

    # 5. Gradient magnitude per variable
    ax = axes[1, 1]
    grad_mag = np.mean(np.abs(gradients), axis=0)
    colors = ['#3498db' if i in ground_truth['level_0']['object_0']
              else '#e74c3c' if i in ground_truth['level_0']['object_1']
              else '#2ecc71' for i in range(len(grad_mag))]
    ax.bar(range(len(grad_mag)), grad_mag, color=colors)
    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Mean |∇E|')
    ax.set_title('Gradient Magnitude\n(Blue/Red=Objects, Green=Blanket)')

    # 6. Spectral partition result
    ax = axes[1, 2]
    partition_colors = plt.cm.tab10(labels)
    ax.bar(range(len(labels)), grad_mag, color=partition_colors)
    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Mean |∇E|')
    ax.set_title('Spectral Partition Result\n(Colors = clusters)')

    plt.tight_layout()
    save_figure(fig, 'spectral_analysis', 'spectral_friston')


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_spectral_experiment()
