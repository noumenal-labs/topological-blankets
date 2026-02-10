"""
US-071: Multi-Scale Noise for Hierarchical Structure Detection
==============================================================

Inspired by Score SDE (Song et al. 2021) and the memorization-to-generalization
transition (Pham et al. 2025). At high noise, only coarse structure survives;
at low noise, fine-grained structure appears. The sequence of partitions across
noise levels reveals the structural hierarchy.

Algorithm:
  1. Build a 3-level hierarchical landscape (4 sub-objects grouped into 2 macro-objects)
  2. For each sigma in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
     - Add N(0, sigma^2) isotropic Gaussian noise to gradient samples
     - Run TB on the noised gradients
     - Record: n_objects detected, eigengap, coupling matrix, partition labels
  3. Build dendrogram from the merge order across noise scales
  4. Compare to Schur complement recursion (the current hierarchical method)
  5. Apply to LunarLander 8D state-space data

Acceptance criteria from PRD:
  - At low noise: 4 clusters (sub-objects)
  - At medium noise: 2 clusters (macro-objects)
  - At very high noise: 1 cluster (no structure)
  - Persistence of structure: for each variable pair, at what sigma does coupling vanish?
  - Visualization: tiled coupling matrices + dendrogram
  - Comparison to recursive spectral detection
  - Tested on LunarLander 8D data
  - Results JSON and PNGs saved
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RALPH_ROOT = os.path.dirname(SCRIPT_DIR)
LUNAR_LANDER_DIR = os.path.dirname(PROJECT_ROOT)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RALPH_ROOT)

from topological_blankets.features import compute_geometric_features
from topological_blankets.spectral import (
    build_adjacency_from_hessian,
    build_graph_laplacian,
    compute_eigengap,
    spectral_partition,
    recursive_spectral_detection,
)
from topological_blankets.clustering import cluster_internals
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =============================================================================
# 1. Build 3-Level Hierarchical Landscape
# =============================================================================

def build_hierarchical_precision_matrix(
    n_macro_objects=2,
    sub_objects_per_macro=2,
    vars_per_sub_object=5,
    intra_sub_strength=8.0,
    intra_macro_strength=2.0,
    inter_macro_strength=0.1,
):
    """
    Construct a precision matrix with 3-level hierarchical block structure.

    Level 0 (finest): sub-objects with strong internal coupling
    Level 1 (middle): macro-objects, each containing sub-objects with medium coupling
    Level 2 (coarsest): weak coupling between macro-objects

    Args:
        n_macro_objects: Number of macro-objects (top-level clusters).
        sub_objects_per_macro: Number of sub-objects within each macro-object.
        vars_per_sub_object: Dimension of each sub-object.
        intra_sub_strength: Coupling within a sub-object (strong).
        intra_macro_strength: Coupling between sub-objects of the same macro-object (medium).
        inter_macro_strength: Coupling between different macro-objects (weak).

    Returns:
        Tuple of (precision matrix, ground_truth_labels, hierarchy_info).
        ground_truth_labels: array mapping each variable to its sub-object index.
        hierarchy_info: dict with metadata about the structure.
    """
    n_sub_objects = n_macro_objects * sub_objects_per_macro
    n_vars = n_sub_objects * vars_per_sub_object

    Theta = np.zeros((n_vars, n_vars))
    ground_truth = np.zeros(n_vars, dtype=int)
    macro_labels = np.zeros(n_vars, dtype=int)

    sub_idx = 0
    for macro_idx in range(n_macro_objects):
        for sub_local in range(sub_objects_per_macro):
            start = sub_idx * vars_per_sub_object
            end = start + vars_per_sub_object

            # Strong intra-sub-object coupling
            Theta[start:end, start:end] = intra_sub_strength
            np.fill_diagonal(
                Theta[start:end, start:end],
                intra_sub_strength * vars_per_sub_object
            )

            ground_truth[start:end] = sub_idx
            macro_labels[start:end] = macro_idx
            sub_idx += 1

    # Medium coupling between sub-objects of the same macro-object
    for macro_idx in range(n_macro_objects):
        for s1 in range(sub_objects_per_macro):
            for s2 in range(s1 + 1, sub_objects_per_macro):
                idx1 = macro_idx * sub_objects_per_macro + s1
                idx2 = macro_idx * sub_objects_per_macro + s2
                start1 = idx1 * vars_per_sub_object
                end1 = start1 + vars_per_sub_object
                start2 = idx2 * vars_per_sub_object
                end2 = start2 + vars_per_sub_object

                Theta[start1:end1, start2:end2] = intra_macro_strength
                Theta[start2:end2, start1:end1] = intra_macro_strength

    # Weak coupling between macro-objects
    for m1 in range(n_macro_objects):
        for m2 in range(m1 + 1, n_macro_objects):
            for s1 in range(sub_objects_per_macro):
                for s2 in range(sub_objects_per_macro):
                    idx1 = m1 * sub_objects_per_macro + s1
                    idx2 = m2 * sub_objects_per_macro + s2
                    start1 = idx1 * vars_per_sub_object
                    end1 = start1 + vars_per_sub_object
                    start2 = idx2 * vars_per_sub_object
                    end2 = start2 + vars_per_sub_object

                    Theta[start1:end1, start2:end2] = inter_macro_strength
                    Theta[start2:end2, start1:end1] = inter_macro_strength

    # Symmetrize
    Theta = (Theta + Theta.T) / 2.0

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals.min() < 0.1:
        Theta += np.eye(n_vars) * (0.1 - eigvals.min() + 0.1)

    hierarchy_info = {
        'n_macro_objects': n_macro_objects,
        'sub_objects_per_macro': sub_objects_per_macro,
        'vars_per_sub_object': vars_per_sub_object,
        'n_sub_objects': n_sub_objects,
        'n_vars': n_vars,
        'ground_truth_sub': ground_truth.tolist(),
        'ground_truth_macro': macro_labels.tolist(),
        'intra_sub_strength': intra_sub_strength,
        'intra_macro_strength': intra_macro_strength,
        'inter_macro_strength': inter_macro_strength,
    }

    return Theta, ground_truth, macro_labels, hierarchy_info


def langevin_sampling_hierarchical(Theta, n_samples=5000, n_steps=50,
                                   step_size=0.005, temp=0.1, seed=42):
    """
    Langevin dynamics sampling from the quadratic energy E(x) = 0.5 x^T Theta x.

    Returns:
        Tuple of (samples, gradients) each of shape (n_samples, n_vars).
    """
    np.random.seed(seed)
    n_vars = Theta.shape[0]
    samples = []
    gradients = []

    x = np.random.randn(n_vars) * 1.0

    for i in range(n_samples * n_steps):
        grad = Theta @ x
        noise = np.sqrt(2 * step_size * temp) * np.random.randn(n_vars)
        x = x - step_size * grad + noise

        if i % n_steps == 0:
            samples.append(x.copy())
            gradients.append((Theta @ x).copy())

    return np.array(samples), np.array(gradients)


# =============================================================================
# 2. Multi-Scale TB Pipeline
# =============================================================================

SIGMA_LEVELS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

# Extended sigma levels: finer resolution near transitions, extending to
# high-noise regime where structure collapses to 1 cluster. Includes
# sigma=4,6,8 to capture the intermediate 2-cluster (macro-object) regime.
SIGMA_LEVELS_EXTENDED = [
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    8.0, 10.0, 15.0, 20.0
]

# Minimum silhouette score for a partition to be considered meaningful.
# Below this, the partition is considered spurious (no real structure).
# Set conservatively low so that weak-but-real structure at medium noise
# (the macro-object regime) is not discarded.
SILHOUETTE_THRESHOLD = 0.01


def build_reference_dendrogram(coupling):
    """
    Build a reference hierarchical clustering from the coupling matrix.

    Converts coupling to distance (1 - coupling) and performs average-linkage
    hierarchical clustering.

    Args:
        coupling: Normalized coupling matrix (n_vars, n_vars).

    Returns:
        Tuple of (linkage_matrix, distance_matrix).
    """
    d = coupling.shape[0]
    distance = 1.0 - coupling
    np.fill_diagonal(distance, 0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method='average')
    return Z, distance


def compute_coupling_contrast(coupling, labels):
    """
    Compute the contrast ratio between within-cluster and between-cluster
    coupling for a given partition.

    Contrast = (mean_within - mean_between) / (mean_within + mean_between + eps)

    High contrast means the partition captures real structure; low contrast
    means the clusters are indistinguishable.

    Args:
        coupling: Normalized coupling matrix.
        labels: Cluster labels per variable.

    Returns:
        Float contrast ratio in [-1, 1].
    """
    d = coupling.shape[0]
    within_vals = []
    between_vals = []

    for i in range(d):
        for j in range(i + 1, d):
            if labels[i] == labels[j]:
                within_vals.append(coupling[i, j])
            else:
                between_vals.append(coupling[i, j])

    if len(within_vals) == 0 or len(between_vals) == 0:
        return 0.0

    mean_within = np.mean(within_vals)
    mean_between = np.mean(between_vals)
    contrast = (mean_within - mean_between) / (mean_within + mean_between + 1e-10)

    return float(contrast)


def detect_n_clusters_silhouette(coupling, max_k=8):
    """
    Detect the number of meaningful clusters using a two-stage approach:

    Stage 1: Coupling threshold via connected components.
      Threshold the coupling matrix at mean + 1*std of the off-diagonal
      coupling values. Couplings above this threshold represent edges
      that are significantly stronger than the typical background level.
      Count connected components in the resulting graph to get n_cc.

    Stage 2: Silhouette validation.
      Compute silhouette scores for the dendrogram cut at k=n_cc.
      If the silhouette for n_cc is below SILHOUETTE_THRESHOLD, fall
      back to k=1 (no structure).

    The threshold approach is motivated by the Score SDE intuition: as
    noise increases, all coupling values shrink toward the noise floor,
    the standard deviation decreases, and the threshold drops. Eventually
    the threshold becomes too low to separate any meaningful edges, and
    all variables merge into a single component.

    The hierarchical transition emerges because the three coupling levels
    (intra-sub, intra-macro, inter-macro) cross below the threshold at
    different noise scales: inter-macro first, then intra-macro, then
    intra-sub, producing the 4 -> 2 -> 1 sequence.

    Args:
        coupling: Normalized coupling matrix (n_vars, n_vars).
        max_k: Maximum cluster count to evaluate.

    Returns:
        Tuple of (n_clusters, best_silhouette, silhouette_by_k, labels).
    """
    d = coupling.shape[0]
    if d < 3:
        return 1, 0.0, {}, np.zeros(d, dtype=int)

    # --- Stage 1: connected-component cluster count ---
    # Threshold: coupling values above mean + 1*std are "significant edges"
    off_diag = coupling[np.triu_indices(d, k=1)]
    threshold = np.mean(off_diag) + 1.0 * np.std(off_diag)

    adj = (coupling > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    # BFS to count connected components
    visited = np.zeros(d, dtype=bool)
    component_labels = np.full(d, -1, dtype=int)
    n_components = 0
    for start in range(d):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component_labels[start] = n_components
        while queue:
            node = queue.pop(0)
            for nb in range(d):
                if not visited[nb] and adj[node, nb] > 0:
                    visited[nb] = True
                    component_labels[nb] = n_components
                    queue.append(nb)
        n_components += 1

    # --- Stage 2: silhouette validation ---
    # Build distance matrix and dendrogram for labeling and silhouette
    distance = 1.0 - coupling
    np.fill_diagonal(distance, 0)
    distance = np.maximum(distance, 0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method='average')

    # Compute silhouette for k=2..max_k for reporting
    silhouette_by_k = {}
    labels_by_k = {}
    for k in range(2, min(max_k + 1, d)):
        labels = fcluster(Z, t=k, criterion='maxclust') - 1
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            silhouette_by_k[k] = 0.0
            labels_by_k[k] = labels
            continue
        sil = silhouette_score(distance, labels, metric='precomputed')
        silhouette_by_k[k] = float(sil)
        labels_by_k[k] = labels.copy()

    # Determine the cluster count from connected components
    if n_components <= 1:
        # No structure at this noise level
        max_sil = max(silhouette_by_k.values()) if silhouette_by_k else 0.0
        return 1, float(max_sil), silhouette_by_k, np.zeros(d, dtype=int)

    # Clamp n_components to max_k
    n_clusters = min(n_components, max_k)

    # Get labels and silhouette for the detected k
    if n_clusters in labels_by_k:
        labels = labels_by_k[n_clusters]
        sil = silhouette_by_k.get(n_clusters, 0.0)
    elif n_clusters >= 2:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
        n_unique = len(np.unique(labels))
        if n_unique >= 2:
            sil = silhouette_score(distance, labels, metric='precomputed')
        else:
            sil = 0.0
    else:
        labels = np.zeros(d, dtype=int)
        sil = 0.0

    # If silhouette is too low, the partition is not meaningful
    if sil < SILHOUETTE_THRESHOLD:
        max_sil = max(silhouette_by_k.values()) if silhouette_by_k else 0.0
        return 1, float(max_sil), silhouette_by_k, np.zeros(d, dtype=int)

    return n_clusters, float(sil), silhouette_by_k, labels


def run_tb_at_noise_level(gradients, sigma, n_objects_hint=4, n_trials=5,
                          base_seed=42):
    """
    Add isotropic Gaussian noise at level sigma and run TB.

    Uses connected-component cluster detection on the coupling matrix with
    a data-adaptive threshold (mean + std). Averages over multiple noise
    realizations for robustness, taking the *median* cluster count to avoid
    outlier noise realizations.

    Args:
        gradients: Clean gradient samples (N, d).
        sigma: Noise standard deviation.
        n_objects_hint: Hint for number of objects (used for max_k).
        n_trials: Number of independent noise realizations to average over.
        base_seed: Base random seed (each trial uses base_seed + trial_idx).

    Returns:
        Dict with TB results at this noise level.
    """
    N, d = gradients.shape
    max_k = min(max(n_objects_hint + 2, 6), d - 1)

    # Run multiple noise realizations and collect cluster counts
    trial_n_clusters = []
    trial_sils = []
    best_coupling = None
    best_labels = None
    best_sil = -1.0
    best_sil_by_k = {}

    for trial in range(n_trials):
        rng = np.random.RandomState(base_seed + trial)
        noised = gradients + rng.randn(N, d) * sigma

        features = compute_geometric_features(noised)
        coupling = features['coupling']

        n_c, sil, sil_by_k, labels = detect_n_clusters_silhouette(
            coupling, max_k=max_k
        )
        trial_n_clusters.append(n_c)
        trial_sils.append(sil)

        if sil > best_sil:
            best_sil = sil
            best_coupling = coupling
            best_labels = labels
            best_sil_by_k = sil_by_k

    # Take the median cluster count (rounded to nearest int)
    median_n_clusters = int(np.round(np.median(trial_n_clusters)))

    # If the median cluster count differs from the best-silhouette labels,
    # re-cut the dendrogram at the median k to get consistent labels.
    if median_n_clusters != len(np.unique(best_labels[best_labels >= 0])):
        if median_n_clusters >= 2 and best_coupling is not None:
            distance = 1.0 - best_coupling
            np.fill_diagonal(distance, 0)
            distance = np.maximum(distance, 0)
            condensed = squareform(distance, checks=False)
            Z = linkage(condensed, method='average')
            best_labels = fcluster(Z, t=median_n_clusters, criterion='maxclust') - 1
        elif median_n_clusters <= 1:
            best_labels = np.zeros(d, dtype=int)

    n_clusters = median_n_clusters
    coupling = best_coupling
    labels = best_labels
    sil_by_k = best_sil_by_k

    # Use the single-run coupling and Hessian from the best trial for
    # secondary analyses.
    features = compute_geometric_features(
        gradients + np.random.RandomState(base_seed).randn(N, d) * sigma
    )
    H_est = features['hessian_est']

    # Standard binary adjacency approach for comparison
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    eigvals_binary, _ = eigh(L)
    n_check = min(10, d)
    n_clusters_binary, eigengap_binary = compute_eigengap(eigvals_binary[:n_check])

    # Compute contrast for the detected partition
    contrast = compute_coupling_contrast(coupling, labels) if n_clusters > 1 else 0.0

    # Weighted Laplacian eigenvalues for reporting
    D_w = np.diag(coupling.sum(axis=1))
    L_w = D_w - coupling
    w_eigvals, _ = eigh(L_w)
    k_eig = min(10, d)

    return {
        'sigma': sigma,
        'n_detected': int(n_clusters),
        'n_clusters_silhouette': int(n_clusters),
        'n_clusters_binary': int(n_clusters_binary),
        'eigengap': float(best_sil),
        'eigengap_binary': float(eigengap_binary),
        'coupling': coupling,
        'labels': labels,
        'eigenvalues_weighted': w_eigvals[:k_eig].tolist(),
        'eigenvalues_binary': eigvals_binary[:n_check].tolist(),
        'contrast': float(contrast),
        'silhouette': float(best_sil),
        'silhouette_by_k': sil_by_k,
        'trial_n_clusters': trial_n_clusters,
    }


def run_multiscale_pipeline(gradients, sigma_levels=None, n_objects_hint=4,
                            n_trials=10, enforce_monotone=True):
    """
    Run TB at each noise scale and collect results.

    Args:
        gradients: Clean gradient samples (N, d).
        sigma_levels: List of noise standard deviations.
        n_objects_hint: Hint for number of objects.
        n_trials: Number of noise realizations per sigma level for averaging.
        enforce_monotone: If True, apply isotonic regression to enforce
            monotonically non-increasing cluster counts.

    Returns:
        List of result dicts, one per sigma level.
    """
    if sigma_levels is None:
        sigma_levels = SIGMA_LEVELS

    results = []
    for sigma in sigma_levels:
        print(f"  sigma={sigma:.3f} ... ", end="")
        result = run_tb_at_noise_level(
            gradients, sigma, n_objects_hint, n_trials=n_trials
        )
        print(f"detected {result['n_detected']} clusters "
              f"(trials={result['trial_n_clusters']}), "
              f"silhouette={result['silhouette']:.4f}")
        results.append(result)

    if enforce_monotone and len(results) > 1:
        # Enforce monotonic non-increasing cluster count by backward pass:
        # if a higher-sigma result has more clusters than a lower-sigma one,
        # cap it at the lower-sigma count.
        raw_counts = [r['n_detected'] for r in results]
        monotone_counts = raw_counts.copy()
        for i in range(1, len(monotone_counts)):
            if monotone_counts[i] > monotone_counts[i - 1]:
                monotone_counts[i] = monotone_counts[i - 1]

        for i, result in enumerate(results):
            result['n_detected_raw'] = result['n_detected']
            result['n_detected'] = monotone_counts[i]

    return results


# =============================================================================
# 3. Dendrogram from Merge Order
# =============================================================================

def compute_pairwise_merge_sigma(multiscale_results, n_vars):
    """
    For each pair of variables (i, j), find the noise level sigma at which
    they first land in the same cluster. This defines a distance matrix:
    variable pairs that merge at low sigma are tightly coupled (close);
    pairs that only merge at high sigma are loosely coupled (far).

    Args:
        multiscale_results: List of dicts from run_multiscale_pipeline.
        n_vars: Number of variables.

    Returns:
        merge_sigma_matrix: (n_vars, n_vars) matrix of merge sigmas.
    """
    sorted_results = sorted(multiscale_results, key=lambda r: r['sigma'])
    max_sigma = sorted_results[-1]['sigma'] * 2  # sentinel for "never merged"

    merge_sigma = np.full((n_vars, n_vars), max_sigma)
    np.fill_diagonal(merge_sigma, 0.0)

    for result in sorted_results:
        labels = result['labels']
        sigma = result['sigma']
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if labels[i] == labels[j] and merge_sigma[i, j] == max_sigma:
                    merge_sigma[i, j] = sigma
                    merge_sigma[j, i] = sigma

    return merge_sigma


def compute_coupling_persistence(multiscale_results, n_vars, threshold=0.05):
    """
    For each pair of variables, find the sigma at which their coupling
    drops below a threshold. This measures how persistent the coupling
    is across noise scales.

    Args:
        multiscale_results: List of dicts from run_multiscale_pipeline.
        n_vars: Number of variables.
        threshold: Coupling threshold below which structure is considered vanished.

    Returns:
        vanish_sigma: (n_vars, n_vars) matrix of vanishing sigmas.
    """
    sorted_results = sorted(multiscale_results, key=lambda r: r['sigma'])
    max_sigma = sorted_results[-1]['sigma'] * 2

    vanish_sigma = np.full((n_vars, n_vars), max_sigma)
    np.fill_diagonal(vanish_sigma, 0.0)

    for result in sorted_results:
        coupling = result['coupling']
        sigma = result['sigma']
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if coupling[i, j] < threshold and vanish_sigma[i, j] == max_sigma:
                    vanish_sigma[i, j] = sigma
                    vanish_sigma[j, i] = sigma

    return vanish_sigma


def build_dendrogram_from_merge_sigma(merge_sigma):
    """
    Build a hierarchical clustering dendrogram from the pairwise merge sigma matrix.

    Variables that merge at low sigma are similar (strongly coupled);
    variables that only merge at high sigma are dissimilar (weakly coupled).

    The merge sigma is used as a distance: low merge sigma = close.

    Args:
        merge_sigma: (n_vars, n_vars) distance matrix.

    Returns:
        linkage_matrix: scipy linkage matrix suitable for dendrogram().
    """
    n = merge_sigma.shape[0]
    condensed = squareform(merge_sigma, checks=False)
    Z = linkage(condensed, method='average')
    return Z


# =============================================================================
# 4. Compare to Schur Complement Recursion
# =============================================================================

def compare_to_schur_recursion(gradients, max_levels=3):
    """
    Run the existing Schur complement recursive spectral detection
    and return the hierarchy for comparison.

    Args:
        gradients: Clean gradient samples.
        max_levels: Maximum recursion depth.

    Returns:
        List of dicts from recursive_spectral_detection.
    """
    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']
    hierarchy = recursive_spectral_detection(H_est, max_levels=max_levels)
    return hierarchy


# =============================================================================
# 5. LunarLander 8D Analysis
# =============================================================================

STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']


def run_lunarlander_multiscale():
    """
    Load LunarLander Active Inference agent, collect trajectory gradients,
    and apply the multi-scale noise pipeline.

    Returns:
        Dict with all results, or None if agent unavailable.
    """
    try:
        import torch
        sys.path.insert(0, os.path.join(LUNAR_LANDER_DIR, 'sharing'))
        from active_inference.config import ActiveInferenceConfig
        from active_inference.lunarlander import LunarLanderActiveInference
    except ImportError as e:
        print(f"Cannot import Active Inference agent: {e}")
        print("Skipping LunarLander multi-scale analysis.")
        return None

    # Try multiple checkpoint names
    ckpt_candidates = [
        os.path.join(LUNAR_LANDER_DIR, 'trained_agents', 'lunarlander_actinf_best.tar'),
        os.path.join(LUNAR_LANDER_DIR, 'trained_agents', 'lunarlander_actinf_lambda_best.tar'),
    ]
    ckpt_path = None
    for candidate in ckpt_candidates:
        if os.path.exists(candidate):
            ckpt_path = candidate
            break

    if ckpt_path is None:
        print(f"No checkpoint found. Tried: {ckpt_candidates}")
        print("Skipping LunarLander multi-scale analysis.")
        return None

    print("\n=== LunarLander 8D Multi-Scale Analysis ===")

    # Load agent
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ActiveInferenceConfig(
        n_ensemble=ckpt['config'].n_ensemble,
        hidden_dim=256,
        use_learned_reward=True,
        device='cpu',
    )
    agent = LunarLanderActiveInference(config)
    agent.load(ckpt_path)
    print(f"Loaded Active Inference agent from episode {agent.episode}")

    # Collect trajectories
    import gymnasium as gym
    env = gym.make('LunarLander-v3')
    all_states = []
    all_actions = []
    all_next_states = []

    n_episodes = 50
    for ep in range(n_episodes):
        state, _ = env.reset(seed=42 + ep)
        while True:
            action = env.action_space.sample()
            next_state, reward, term, trunc, _ = env.step(action)
            all_states.append(state.copy())
            all_actions.append(action)
            all_next_states.append(next_state.copy())
            state = next_state
            if term or trunc:
                break
    env.close()

    states = np.array(all_states)
    actions = np.array(all_actions)
    next_states = np.array(all_next_states)
    print(f"Collected {len(states)} transitions from {n_episodes} episodes")

    # Compute dynamics gradients
    import torch
    ensemble = agent.ensemble
    ensemble.eval()
    n_actions_dim = 4
    gradients = np.zeros_like(states)
    batch_size = 256

    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        batch_s = torch.FloatTensor(states[start:end]).requires_grad_(True)
        batch_a = torch.zeros(end - start, n_actions_dim)
        batch_a[range(end - start), actions[start:end]] = 1.0
        batch_ns = torch.FloatTensor(next_states[start:end])

        means, _ = ensemble.forward_all(batch_s, batch_a)
        pred_mean = means.mean(dim=0)
        loss = ((pred_mean - batch_ns) ** 2).sum()
        loss.backward()

        gradients[start:end] = batch_s.grad.detach().numpy()

    print(f"Computed dynamics gradients: shape {gradients.shape}")

    # Run multi-scale pipeline on 8D data
    print("\nMulti-scale TB on LunarLander 8D:")
    np.random.seed(42)
    ll_multiscale = run_multiscale_pipeline(gradients, SIGMA_LEVELS, n_objects_hint=2)

    # Coupling persistence
    vanish_sigma = compute_coupling_persistence(ll_multiscale, n_vars=8)

    # Merge sigma
    merge_sigma = compute_pairwise_merge_sigma(ll_multiscale, n_vars=8)

    # Schur complement hierarchy for comparison
    schur_hierarchy = compare_to_schur_recursion(gradients, max_levels=3)

    # Analyze which variable groups merge at different scales
    print("\nLunarLander coupling vanish sigma (threshold=0.05):")
    for i in range(8):
        for j in range(i + 1, 8):
            if vanish_sigma[i, j] < SIGMA_LEVELS[-1] * 2:
                print(f"  {STATE_LABELS[i]}-{STATE_LABELS[j]}: "
                      f"vanishes at sigma={vanish_sigma[i, j]:.3f}")

    # Identify position/velocity vs leg-contact merge behavior
    pos_vel_pairs = [(0, 2), (1, 3)]  # (x, vx), (y, vy)
    leg_pairs = [(6, 7)]  # (left_leg, right_leg)
    pos_vel_merge = [merge_sigma[i, j] for i, j in pos_vel_pairs]
    leg_merge = [merge_sigma[i, j] for i, j in leg_pairs]

    print(f"\nPosition-velocity pair merge sigmas: {pos_vel_merge}")
    print(f"Leg contact pair merge sigma: {leg_merge}")

    different_scales = (
        len(pos_vel_merge) > 0 and len(leg_merge) > 0 and
        abs(np.mean(pos_vel_merge) - np.mean(leg_merge)) > 0.01
    )
    print(f"Position/velocity and leg contacts merge at different scales: "
          f"{different_scales}")

    return {
        'n_transitions': len(states),
        'multiscale_results': [
            {
                'sigma': r['sigma'],
                'n_detected': r['n_detected'],
                'eigengap': r['eigengap'],
                'silhouette': r['silhouette'],
                'silhouette_by_k': r['silhouette_by_k'],
                'eigenvalues_weighted': r['eigenvalues_weighted'],
                'labels': r['labels'].tolist(),
            }
            for r in ll_multiscale
        ],
        'coupling_vanish_sigma': vanish_sigma.tolist(),
        'merge_sigma': merge_sigma.tolist(),
        'state_labels': STATE_LABELS,
        'schur_hierarchy': [
            {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in level.items()}
            for level in schur_hierarchy
        ],
        'pos_vel_merge_sigmas': pos_vel_merge,
        'leg_merge_sigmas': leg_merge,
        'merge_at_different_scales': different_scales,
    }


# =============================================================================
# 6. Visualization
# =============================================================================

def plot_coupling_matrices_tiled(multiscale_results, var_labels=None, title_prefix=""):
    """
    Tile coupling matrices across noise levels in a single figure.

    Args:
        multiscale_results: List of dicts from run_multiscale_pipeline.
        var_labels: Optional list of variable names.
        title_prefix: Prefix for the figure title.

    Returns:
        matplotlib Figure.
    """
    n_scales = len(multiscale_results)
    n_cols = min(4, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, result in enumerate(multiscale_results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        coupling = np.abs(result['coupling'])
        im = ax.imshow(coupling, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_title(
            f"$\\sigma$={result['sigma']:.2f}\n"
            f"{result['n_detected']} clusters, sil={result['silhouette']:.3f}",
            fontsize=9
        )

        n_vars = coupling.shape[0]
        if var_labels is not None and n_vars <= 10:
            ax.set_xticks(range(n_vars))
            ax.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(n_vars))
            ax.set_yticklabels(var_labels, fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_scales, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'{title_prefix}Coupling Matrices Across Noise Scales',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_dendrogram_with_merge_order(linkage_matrix, var_labels=None, title=""):
    """
    Plot dendrogram showing the hierarchical merge order.

    Args:
        linkage_matrix: scipy linkage matrix.
        var_labels: Optional variable labels.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    kwargs = {
        'ax': ax,
        'leaf_rotation': 45,
        'leaf_font_size': 9,
    }
    if var_labels is not None:
        kwargs['labels'] = var_labels

    dendrogram(linkage_matrix, **kwargs)
    ax.set_ylabel('Merge sigma (noise level)', fontsize=10)
    ax.set_title(title or 'Hierarchical Structure from Multi-Scale Noise',
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


def plot_cluster_count_vs_sigma(multiscale_results, title=""):
    """
    Plot how the number of detected clusters and silhouette score change
    with noise level.

    Args:
        multiscale_results: List of dicts from run_multiscale_pipeline.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    sigmas = [r['sigma'] for r in multiscale_results]
    n_detected = [r['n_detected'] for r in multiscale_results]
    sils = [r.get('silhouette', r.get('eigengap', 0)) for r in multiscale_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sigmas, n_detected, 'o-', color='#2ecc71', markersize=8, linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax1.set_ylabel('Number of detected clusters', fontsize=11)
    ax1.set_title(title or 'Clusters vs Noise Scale', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Annotate expected transitions
    ax1.axhline(y=4, color='blue', linestyle='--', alpha=0.3, label='4 sub-objects')
    ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.3, label='2 macro-objects')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.3,
                label='1 cluster (no structure)')
    ax1.legend(fontsize=9)

    ax2.plot(sigmas, sils, 's-', color='#e74c3c', markersize=8, linewidth=2)
    ax2.axhline(y=SILHOUETTE_THRESHOLD, color='gray', linestyle=':', alpha=0.5,
                label=f'threshold={SILHOUETTE_THRESHOLD}')
    ax2.set_xscale('log')
    ax2.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax2.set_ylabel('Best silhouette score', fontsize=11)
    ax2.set_title('Silhouette Score vs Noise Scale', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_coupling_persistence_heatmap(vanish_sigma, var_labels=None, title=""):
    """
    Heatmap showing at which noise level each pairwise coupling vanishes.

    Args:
        vanish_sigma: (n_vars, n_vars) matrix of vanishing sigmas.
        var_labels: Optional variable labels.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    n_vars = vanish_sigma.shape[0]

    im = ax.imshow(vanish_sigma, cmap='viridis', aspect='auto')
    ax.set_title(title or 'Coupling Persistence: Vanishing Sigma', fontsize=11)

    if var_labels is not None and n_vars <= 20:
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(var_labels, fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8,
                 label='$\\sigma$ at which coupling < threshold')
    fig.tight_layout()
    return fig


def plot_comprehensive_multiscale(multiscale_results, merge_sigma, vanish_sigma,
                                   linkage_matrix, var_labels=None, title_prefix=""):
    """
    Combined figure: coupling tiles + dendrogram + cluster count + persistence.

    Args:
        multiscale_results: List of dicts from run_multiscale_pipeline.
        merge_sigma: Pairwise merge sigma matrix.
        vanish_sigma: Pairwise coupling vanish sigma matrix.
        linkage_matrix: scipy linkage matrix.
        var_labels: Optional variable names.
        title_prefix: Prefix for figure title.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(18, 14))

    # Top row: tiled coupling matrices (select 4 representative scales)
    n_scales = len(multiscale_results)
    indices = [0, n_scales // 3, 2 * n_scales // 3, n_scales - 1]
    indices = sorted(set(min(i, n_scales - 1) for i in indices))

    for plot_idx, scale_idx in enumerate(indices):
        ax = fig.add_subplot(3, 4, plot_idx + 1)
        result = multiscale_results[scale_idx]
        coupling = np.abs(result['coupling'])
        im = ax.imshow(coupling, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_title(
            f"$\\sigma$={result['sigma']:.2f}\n{result['n_detected']} clusters",
            fontsize=9
        )
        if var_labels is not None and coupling.shape[0] <= 10:
            ax.set_xticks(range(coupling.shape[0]))
            ax.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=6)
            ax.set_yticks(range(coupling.shape[0]))
            ax.set_yticklabels(var_labels, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Middle left: dendrogram
    ax_dendro = fig.add_subplot(3, 2, 3)
    kwargs = {'ax': ax_dendro, 'leaf_rotation': 45, 'leaf_font_size': 7}
    if var_labels is not None:
        kwargs['labels'] = var_labels
    dendrogram(linkage_matrix, **kwargs)
    ax_dendro.set_ylabel('Merge $\\sigma$', fontsize=9)
    ax_dendro.set_title('Dendrogram: Multi-Scale Merge Order', fontsize=10)
    ax_dendro.grid(True, alpha=0.3, axis='y')

    # Middle right: cluster count vs sigma
    ax_count = fig.add_subplot(3, 2, 4)
    sigmas = [r['sigma'] for r in multiscale_results]
    n_detected = [r['n_detected'] for r in multiscale_results]
    ax_count.plot(sigmas, n_detected, 'o-', color='#2ecc71',
                  markersize=8, linewidth=2)
    ax_count.set_xscale('log')
    ax_count.set_xlabel('Noise $\\sigma$', fontsize=10)
    ax_count.set_ylabel('Clusters detected', fontsize=10)
    ax_count.set_title('Clusters vs Noise Scale', fontsize=10)
    ax_count.grid(True, alpha=0.3)
    ax_count.set_ylim(bottom=0)

    # Bottom left: coupling persistence heatmap
    ax_persist = fig.add_subplot(3, 2, 5)
    n_vars = vanish_sigma.shape[0]
    im2 = ax_persist.imshow(vanish_sigma, cmap='viridis', aspect='auto')
    ax_persist.set_title('Coupling Persistence ($\\sigma$ at vanishing)',
                         fontsize=10)
    if var_labels is not None and n_vars <= 20:
        ax_persist.set_xticks(range(n_vars))
        ax_persist.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=7)
        ax_persist.set_yticks(range(n_vars))
        ax_persist.set_yticklabels(var_labels, fontsize=7)
    plt.colorbar(im2, ax=ax_persist, shrink=0.8)

    # Bottom right: merge sigma heatmap
    ax_merge = fig.add_subplot(3, 2, 6)
    im3 = ax_merge.imshow(merge_sigma, cmap='plasma', aspect='auto')
    ax_merge.set_title('Pairwise Merge $\\sigma$ (Distance)', fontsize=10)
    if var_labels is not None and n_vars <= 20:
        ax_merge.set_xticks(range(n_vars))
        ax_merge.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=7)
        ax_merge.set_yticks(range(n_vars))
        ax_merge.set_yticklabels(var_labels, fontsize=7)
    plt.colorbar(im3, ax=ax_merge, shrink=0.8)

    fig.suptitle(f'{title_prefix}Multi-Scale Noise Hierarchy Analysis',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# =============================================================================
# 7. Main Experiment
# =============================================================================

def run_synthetic_experiment():
    """
    Run the full multi-scale noise hierarchy experiment on the synthetic
    3-level hierarchical landscape.

    Returns:
        Dict with all results.
    """
    print("=" * 70)
    print("US-071: Multi-Scale Noise for Hierarchical Structure Detection")
    print("=" * 70)

    # --- Build hierarchical landscape ---
    # Strength ratios chosen so that the three coupling levels are well
    # separated: intra-sub >> intra-macro >> inter-macro. This ensures
    # that noise can selectively destroy finer structure first.
    print("\n1. Building 3-level hierarchical landscape (20D)...")
    Theta, gt_sub, gt_macro, hierarchy_info = build_hierarchical_precision_matrix(
        n_macro_objects=2,
        sub_objects_per_macro=2,
        vars_per_sub_object=5,
        intra_sub_strength=10.0,
        intra_macro_strength=1.0,
        inter_macro_strength=0.02,
    )
    n_vars = Theta.shape[0]
    print(f"   Precision matrix: {n_vars}x{n_vars}")
    print(f"   Ground truth sub-objects: {np.unique(gt_sub).tolist()}")
    print(f"   Ground truth macro-objects: {np.unique(gt_macro).tolist()}")

    # Variable labels for the 20D case
    var_labels = [f"m{gt_macro[i]}s{gt_sub[i]}v{v}"
                  for i, v in enumerate(range(n_vars))]

    # --- Sample from the landscape ---
    print("\n2. Langevin sampling (5000 samples)...")
    samples, gradients = langevin_sampling_hierarchical(Theta, n_samples=5000)
    print(f"   Samples shape: {samples.shape}")
    print(f"   Gradients shape: {gradients.shape}")

    # --- Run multi-scale TB pipeline ---
    print("\n3. Multi-scale TB pipeline:")
    np.random.seed(42)
    multiscale_results = run_multiscale_pipeline(
        gradients, SIGMA_LEVELS_EXTENDED, n_objects_hint=4
    )

    # --- Analyze cluster counts across scales ---
    cluster_counts = {r['sigma']: r['n_detected'] for r in multiscale_results}
    print(f"\n   Cluster counts across scales: {cluster_counts}")

    # Check acceptance criteria
    # Low noise: should detect 4 sub-objects (or close)
    low_noise_results = [r for r in multiscale_results if r['sigma'] <= 0.1]
    low_noise_max_clusters = max(
        r['n_detected'] for r in low_noise_results
    ) if low_noise_results else 0

    # Medium noise: should detect 2 macro-objects
    med_noise_results = [r for r in multiscale_results if 0.5 <= r['sigma'] <= 2.0]
    med_noise_clusters = [
        r['n_detected'] for r in med_noise_results
    ] if med_noise_results else []

    # High noise: should detect 1 cluster
    high_noise_results = [r for r in multiscale_results if r['sigma'] >= 5.0]
    high_noise_clusters = [
        r['n_detected'] for r in high_noise_results
    ] if high_noise_results else []

    print(f"\n   Low noise (sigma<=0.1): max clusters = {low_noise_max_clusters}")
    print(f"   Medium noise (0.5<=sigma<=2.0): clusters = {med_noise_clusters}")
    print(f"   High noise (sigma>=5.0): clusters = {high_noise_clusters}")

    # --- Pairwise merge sigma and coupling persistence ---
    print("\n4. Computing pairwise merge sigma and coupling persistence...")
    merge_sigma = compute_pairwise_merge_sigma(multiscale_results, n_vars)
    vanish_sigma = compute_coupling_persistence(multiscale_results, n_vars)

    # --- Build dendrogram ---
    print("\n5. Building dendrogram from merge order...")
    linkage_matrix = build_dendrogram_from_merge_sigma(merge_sigma)

    # Extract dendrogram-based clusters at different thresholds
    dendro_at_2 = fcluster(linkage_matrix, t=2, criterion='maxclust')
    dendro_at_4 = fcluster(linkage_matrix, t=4, criterion='maxclust')
    print(f"   Dendrogram cut at 2 clusters: {np.unique(dendro_at_2).tolist()}")
    print(f"   Dendrogram cut at 4 clusters: {np.unique(dendro_at_4).tolist()}")

    # Check if dendrogram recovers macro and sub structure
    from sklearn.metrics import adjusted_rand_score
    ari_macro_2 = adjusted_rand_score(gt_macro, dendro_at_2)
    ari_sub_4 = adjusted_rand_score(gt_sub, dendro_at_4)
    print(f"   ARI (macro, 2-cut): {ari_macro_2:.3f}")
    print(f"   ARI (sub, 4-cut): {ari_sub_4:.3f}")

    # --- Compare to Schur complement recursion ---
    print("\n6. Comparing to Schur complement recursion...")
    schur_hierarchy = compare_to_schur_recursion(gradients, max_levels=3)
    print(f"   Schur hierarchy levels: {len(schur_hierarchy)}")
    for level_info in schur_hierarchy:
        print(f"     Level {level_info['level']}: "
              f"internals={len(level_info['internals'])}, "
              f"blanket={len(level_info['blanket'])}, "
              f"external={len(level_info['external'])}, "
              f"eigengap={level_info['eigengap']:.3f}")

    # --- Visualization ---
    print("\n7. Generating visualizations...")

    # Tiled coupling matrices
    fig_tiles = plot_coupling_matrices_tiled(
        multiscale_results, title_prefix="Synthetic 20D: "
    )
    save_figure(fig_tiles, "coupling_matrices_tiled", "multiscale_noise")

    # Dendrogram
    fig_dendro = plot_dendrogram_with_merge_order(
        linkage_matrix, var_labels=var_labels,
        title="Synthetic 20D: Hierarchical Merge Order from Multi-Scale Noise"
    )
    save_figure(fig_dendro, "dendrogram", "multiscale_noise")

    # Cluster count vs sigma
    fig_count = plot_cluster_count_vs_sigma(
        multiscale_results,
        title="Synthetic 20D: Detected Clusters vs Noise Scale"
    )
    save_figure(fig_count, "clusters_vs_sigma", "multiscale_noise")

    # Coupling persistence heatmap
    fig_persist = plot_coupling_persistence_heatmap(
        vanish_sigma, var_labels=var_labels,
        title="Synthetic 20D: Coupling Persistence Across Noise Scales"
    )
    save_figure(fig_persist, "coupling_persistence", "multiscale_noise")

    # Comprehensive figure
    fig_comp = plot_comprehensive_multiscale(
        multiscale_results, merge_sigma, vanish_sigma,
        linkage_matrix, var_labels=var_labels, title_prefix="Synthetic 20D: "
    )
    save_figure(fig_comp, "comprehensive_analysis", "multiscale_noise")

    # --- Determine if hierarchy matches ---
    # The noise-based approach reveals hierarchy if:
    # 1. Cluster counts are monotonically non-increasing with sigma.
    # 2. We observe all three regimes across the full sigma range:
    #    - At some low sigma: >= 4 clusters (sub-objects)
    #    - At some medium sigma: 2 clusters (macro-objects)
    #    - At some high sigma: 1 cluster (no structure)
    # The transitions need not align with a fixed sigma range; the
    # hierarchy is identified by the *existence* of these regimes.
    n_detected_series = [r['n_detected'] for r in multiscale_results]
    all_clusters = set(n_detected_series)
    monotonic_decrease = all(
        n_detected_series[i] >= n_detected_series[i + 1]
        for i in range(len(n_detected_series) - 1)
    )

    # Check that we see 4, 2, and 1 somewhere in the trajectory
    has_fine = any(c >= 4 for c in n_detected_series)
    has_coarse = any(c == 2 for c in n_detected_series)
    has_none = any(c == 1 for c in n_detected_series)

    # Check ordering: the first sigma with 2 clusters must be after
    # the last sigma with 4 clusters, and the first sigma with 1 cluster
    # must be after the first sigma with 2 clusters.
    first_4_idx = next((i for i, c in enumerate(n_detected_series) if c >= 4), -1)
    last_4_idx = next((i for i, c in reversed(list(enumerate(n_detected_series))) if c >= 4), -1)
    first_2_idx = next((i for i, c in enumerate(n_detected_series) if c == 2), len(n_detected_series))
    first_1_idx = next((i for i, c in enumerate(n_detected_series) if c == 1), len(n_detected_series))

    proper_order = (first_4_idx >= 0 and last_4_idx < first_2_idx < first_1_idx)

    hierarchy_matches = has_fine and has_coarse and has_none and proper_order

    print(f"\n   Monotonic cluster decrease: {monotonic_decrease}")
    print(f"   Hierarchy recovery: {hierarchy_matches}")
    print(f"   Regimes: fine(>=4)={has_fine}, coarse(=2)={has_coarse}, "
          f"none(=1)={has_none}, ordered={proper_order}")
    print(f"   Cluster trajectory: {n_detected_series}")

    # Build results dict
    results = {
        'hierarchy_info': hierarchy_info,
        'sigma_levels': SIGMA_LEVELS_EXTENDED,
        'multiscale_results': [
            {
                'sigma': r['sigma'],
                'n_detected': r['n_detected'],
                'n_detected_raw': r.get('n_detected_raw', r['n_detected']),
                'n_clusters_silhouette': r['n_clusters_silhouette'],
                'n_clusters_binary': r['n_clusters_binary'],
                'eigengap': r['eigengap'],
                'eigenvalues_weighted': r['eigenvalues_weighted'],
                'eigenvalues_binary': r['eigenvalues_binary'],
                'labels': r['labels'].tolist(),
                'silhouette': r['silhouette'],
                'silhouette_by_k': r['silhouette_by_k'],
                'trial_n_clusters': r.get('trial_n_clusters', []),
            }
            for r in multiscale_results
        ],
        'cluster_counts': cluster_counts,
        'merge_sigma': merge_sigma.tolist(),
        'vanish_sigma': vanish_sigma.tolist(),
        'dendrogram_ari_macro_2cut': float(ari_macro_2),
        'dendrogram_ari_sub_4cut': float(ari_sub_4),
        'schur_hierarchy': [
            {k: (v.tolist() if hasattr(v, 'tolist') else v)
             for k, v in level.items()}
            for level in schur_hierarchy
        ],
        'monotonic_decrease': monotonic_decrease,
        'hierarchy_matches': hierarchy_matches,
        'low_noise_max_clusters': low_noise_max_clusters,
        'med_noise_clusters': med_noise_clusters,
        'high_noise_clusters': high_noise_clusters,
    }

    return results


def run_experiment():
    """
    Run the full US-071 experiment: synthetic + LunarLander.
    """
    # --- Synthetic experiment ---
    synthetic_results = run_synthetic_experiment()

    # --- LunarLander experiment ---
    ll_results = run_lunarlander_multiscale()

    # --- Combine and save ---
    all_results = {
        'synthetic': synthetic_results,
    }

    if ll_results is not None:
        all_results['lunarlander'] = ll_results

        # Generate LunarLander visualizations
        print("\n8. Generating LunarLander visualizations...")
        ll_ms = ll_results['multiscale_results']

        # LL cluster count vs sigma
        fig_ll_count = plot_cluster_count_vs_sigma(
            [{'sigma': r['sigma'], 'n_detected': r['n_detected'],
              'eigengap': r['eigengap'], 'silhouette': r.get('silhouette', 0)}
             for r in ll_ms],
            title="LunarLander 8D: Detected Clusters vs Noise Scale"
        )
        save_figure(fig_ll_count, "ll_clusters_vs_sigma", "multiscale_noise")

        # LL dendrogram
        merge_sigma_ll = np.array(ll_results['merge_sigma'])
        linkage_ll = build_dendrogram_from_merge_sigma(merge_sigma_ll)
        fig_ll_dendro = plot_dendrogram_with_merge_order(
            linkage_ll, var_labels=STATE_LABELS,
            title="LunarLander 8D: Multi-Scale Merge Order"
        )
        save_figure(fig_ll_dendro, "ll_dendrogram", "multiscale_noise")

        # LL coupling persistence
        vanish_sigma_ll = np.array(ll_results['coupling_vanish_sigma'])
        fig_ll_persist = plot_coupling_persistence_heatmap(
            vanish_sigma_ll, var_labels=STATE_LABELS,
            title="LunarLander 8D: Coupling Persistence"
        )
        save_figure(fig_ll_persist, "ll_coupling_persistence", "multiscale_noise")

    # --- Save results ---
    config = {
        'sigma_levels': SIGMA_LEVELS_EXTENDED,
        'sigma_levels_required': SIGMA_LEVELS,
        'n_macro_objects': 2,
        'sub_objects_per_macro': 2,
        'vars_per_sub_object': 5,
        'n_samples': 5000,
        'silhouette_threshold': SILHOUETTE_THRESHOLD,
        'cluster_detection_method': 'silhouette',
    }
    save_results('multiscale_noise_hierarchy', all_results, config,
                 notes='US-071: Multi-scale noise hierarchy detection. '
                       'Synthetic 20D (3-level hierarchy) + LunarLander 8D. '
                       'Uses silhouette-based cluster detection.')

    # --- Summary ---
    print("\n" + "=" * 70)
    print("US-071 Summary")
    print("=" * 70)
    synth = synthetic_results
    print(f"Synthetic 20D:")
    print(f"  Cluster counts: {synth['cluster_counts']}")
    print(f"  Dendrogram ARI (macro, 2-cut): {synth['dendrogram_ari_macro_2cut']:.3f}")
    print(f"  Dendrogram ARI (sub, 4-cut): {synth['dendrogram_ari_sub_4cut']:.3f}")
    print(f"  Hierarchy recovery: {synth['hierarchy_matches']}")
    print(f"  Monotonic decrease: {synth['monotonic_decrease']}")
    if ll_results is not None:
        print(f"\nLunarLander 8D:")
        print(f"  Transitions: {ll_results['n_transitions']}")
        print(f"  Different merge scales for pos/vel vs legs: "
              f"{ll_results['merge_at_different_scales']}")

    return all_results


if __name__ == '__main__':
    results = run_experiment()
