"""
US-092: Benchmark Topological Blankets vs Mutual Information Clustering
========================================================================

Compares two structure discovery approaches on the standardized benchmark
datasets (US-089):

  1. **TB (hybrid)**: Topological Blankets with spectral + gradient
     fallback blanket detection. Uses energy-landscape geometry (Hessian
     eigengaps, Otsu thresholding on gradient norms) to partition
     variables into objects and blankets.

  2. **MI clustering**: A purely information-theoretic baseline that
     builds a pairwise mutual information affinity matrix from the raw
     samples, applies spectral clustering to partition variables into
     groups, and then labels the group whose variables have the highest
     average inter-cluster MI as the blanket.

The comparison is run on all five registered benchmark datasets with 10
seeds each. For every (method, dataset, seed) triple the suite records
ARI, blanket_F1, NMI, and wall-clock runtime. Paired statistical tests
(Wilcoxon signed-rank or paired t, depending on sample size) quantify
significance, and Cohen's d gives practical effect size.

Outputs (saved to ralph/results/):
  - JSON with full raw + summary + statistical comparison data
  - Bar chart comparing mean metrics per dataset
  - Radar chart summarizing cross-dataset profiles
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Path setup (mirrors benchmark_suite.py) ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)
RESULTS_DIR = os.path.join(RALPH_DIR, 'results')

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from experiments.benchmark_suite import (
    BenchmarkSuite,
    build_default_suite,
    _tb_hybrid_method,
    compute_benchmark_metrics,
    paired_statistical_test,
    _json_default,
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

from sklearn.metrics import mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors


# =====================================================================
# MI Clustering Method
# =====================================================================

def _knn_mi_estimator(x, y, k=5):
    """
    KNN-based mutual information estimator (KSG-style, simplified).

    Uses the Kraskov-Stoegbauer-Grassberger approach: for each point,
    find the k-th nearest neighbor distance in the joint space, then
    count neighbors within that distance in each marginal. The MI
    estimate follows from digamma functions.

    This is more robust than histogram-based MI for continuous variables
    and does not require binning choices.

    Args:
        x: 1D array, first variable (n_samples,).
        y: 1D array, second variable (n_samples,).
        k: Number of nearest neighbors.

    Returns:
        Estimated mutual information (non-negative float).
    """
    from scipy.special import digamma

    n = len(x)
    if n < k + 1:
        return 0.0

    # Standardize to avoid scale effects
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    # Joint space
    xy = np.column_stack([x, y])

    # k-th neighbor distance in joint (Chebyshev / max-norm)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
    nn.fit(xy)
    dists, _ = nn.kneighbors(xy)
    eps = dists[:, k]  # distance to k-th neighbor for each point

    # Count neighbors within eps in each marginal
    nx = np.zeros(n, dtype=int)
    ny = np.zeros(n, dtype=int)
    for i in range(n):
        nx[i] = np.sum(np.abs(x - x[i]) < eps[i]) - 1  # exclude self
        ny[i] = np.sum(np.abs(y - y[i]) < eps[i]) - 1

    # Clip to avoid log(0)
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    mi = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)
    return max(0.0, float(mi))


def _build_mi_affinity_matrix(samples, k=5, subsample=None):
    """
    Build a symmetric pairwise MI affinity matrix over variables.

    Each entry A[i, j] = MI(X_i, X_j) estimated via KNN.

    Args:
        samples: (n_samples, n_vars) array.
        k: KNN parameter for MI estimation.
        subsample: If not None, subsample this many rows for speed.

    Returns:
        (n_vars, n_vars) symmetric non-negative affinity matrix.
    """
    n_samples, n_vars = samples.shape

    # Subsample for speed on large datasets
    if subsample is not None and n_samples > subsample:
        idx = np.random.choice(n_samples, subsample, replace=False)
        samples = samples[idx]

    A = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = _knn_mi_estimator(samples[:, i], samples[:, j], k=k)
            A[i, j] = mi
            A[j, i] = mi

    return A


def _mi_clustering_method(samples, gradients, n_clusters=None, k_mi=5):
    """
    MI clustering baseline for structure discovery.

    Steps:
      1. Build pairwise MI affinity matrix from samples.
      2. Spectral-cluster variables using the MI affinity.
      3. Identify the blanket cluster: the cluster whose variables have
         the highest mean inter-cluster MI (i.e., they couple strongly
         to variables in other clusters).
      4. Return assignment array with blanket variables labeled -1.

    Args:
        samples: (n_samples, n_vars) array.
        gradients: (n_samples, n_vars) array (unused, for API compat).
        n_clusters: Number of clusters. If None, auto-detect via
                    eigengap of the MI-derived graph Laplacian.
        k_mi: KNN parameter for MI estimation.

    Returns:
        Label array: label[i] = cluster index, or -1 for blanket.
    """
    n_vars = samples.shape[1]

    # Cap subsample for tractability
    subsample = min(800, samples.shape[0])

    # Build MI affinity
    A = _build_mi_affinity_matrix(samples, k=k_mi, subsample=subsample)

    # Auto-detect n_clusters from eigengap of graph Laplacian
    if n_clusters is None:
        D = np.diag(A.sum(axis=1) + 1e-12)
        L = D - A
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        eigvals = np.sort(np.linalg.eigvalsh(L_norm))
        # Eigengap heuristic: look at gaps in the first 10 eigenvalues
        max_k = min(10, len(eigvals) - 1)
        gaps = np.diff(eigvals[:max_k + 1])
        # Skip the first eigenvalue (always ~0)
        if len(gaps) > 1:
            n_clusters = int(np.argmax(gaps[1:]) + 2)
            n_clusters = max(2, min(n_clusters, n_vars // 2))
        else:
            n_clusters = 2

    # Ensure affinity is suitable for spectral clustering
    # Add small diagonal to avoid singularities
    A_sc = A + 1e-8 * np.eye(n_vars)

    # Spectral clustering on the MI affinity
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
    )
    labels = sc.fit_predict(A_sc)

    # Identify blanket cluster: the cluster whose variables have
    # the highest average MI with variables in other clusters
    cluster_ids = np.unique(labels)
    inter_cluster_mi = {}
    for c in cluster_ids:
        members = np.where(labels == c)[0]
        others = np.where(labels != c)[0]
        if len(others) == 0:
            inter_cluster_mi[c] = 0.0
            continue
        # Mean MI between this cluster's variables and all other variables
        cross_mi = A[np.ix_(members, others)]
        inter_cluster_mi[c] = float(np.mean(cross_mi))

    # The blanket cluster is the one with highest inter-cluster MI
    blanket_cluster = max(inter_cluster_mi, key=inter_cluster_mi.get)

    # Build assignment: remap cluster labels, blanket = -1
    assignment = np.zeros(n_vars, dtype=int)
    obj_counter = 0
    for c in cluster_ids:
        if c == blanket_cluster:
            assignment[labels == c] = -1
        else:
            assignment[labels == c] = obj_counter
            obj_counter += 1

    return assignment


# =====================================================================
# Build the comparison suite
# =====================================================================

def build_comparison_suite():
    """
    Build a BenchmarkSuite with TB_hybrid and MI_clustering registered
    on all five standard datasets.
    """
    # Start from the default suite (has datasets + TB methods)
    suite = build_default_suite()

    # Remove TB_gradient since this experiment focuses on TB_hybrid vs MI
    if 'TB_gradient' in suite._methods:
        del suite._methods['TB_gradient']

    # Register MI clustering
    suite.register_method('MI_clustering', _mi_clustering_method)

    return suite


# =====================================================================
# Visualization: grouped bar chart
# =====================================================================

def plot_comparison_bars(summary, methods, datasets, save_path=None):
    """
    Create a grouped bar chart comparing methods across datasets
    for ARI, blanket_F1, and NMI.
    """
    metrics = ['ARI', 'blanket_F1', 'NMI']
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = {'TB_hybrid': '#2ecc71', 'MI_clustering': '#e74c3c'}
    bar_width = 0.35

    for ax, metric in zip(axes, metrics):
        x = np.arange(n_datasets)
        for i, method in enumerate(methods):
            means = []
            stds = []
            for ds in datasets:
                key = f"{method}|{ds}"
                if key in summary:
                    means.append(summary[key][metric + '_mean'])
                    stds.append(summary[key][metric + '_std'])
                else:
                    means.append(0.0)
                    stds.append(0.0)

            offset = (i - (n_methods - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds,
                   label=method, color=colors.get(method, '#3498db'),
                   alpha=0.85, capsize=3)

        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle('US-092: TB hybrid vs MI Clustering', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison bar chart saved to {save_path}")
    else:
        save_figure(fig, 'tb_vs_mi_bars', 'benchmark_vs_mi')
    plt.close(fig)

    return fig


def plot_runtime_comparison(summary, methods, datasets, save_path=None):
    """Bar chart comparing runtime across datasets."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'TB_hybrid': '#2ecc71', 'MI_clustering': '#e74c3c'}
    bar_width = 0.35
    n_datasets = len(datasets)
    x = np.arange(n_datasets)

    for i, method in enumerate(methods):
        means = []
        stds = []
        for ds in datasets:
            key = f"{method}|{ds}"
            if key in summary:
                means.append(summary[key]['wall_clock_seconds_mean'])
                stds.append(summary[key]['wall_clock_seconds_std'])
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=method, color=colors.get(method, '#3498db'),
               alpha=0.85, capsize=3)

    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Wall-clock time (seconds)', fontsize=10)
    ax.set_title('US-092: Runtime Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Runtime chart saved to {save_path}")
    else:
        save_figure(fig, 'tb_vs_mi_runtime', 'benchmark_vs_mi')
    plt.close(fig)

    return fig


# =====================================================================
# Main entry point
# =====================================================================

def run_benchmark_vs_mi(n_seeds=10, verbose=True):
    """
    Execute the full TB vs MI clustering benchmark.

    Returns:
        (suite, results) tuple.
    """
    suite = build_comparison_suite()

    if verbose:
        print("=" * 70)
        print("US-092: Topological Blankets vs Mutual Information Clustering")
        print("=" * 70)
        print(f"  Methods:  {list(suite._methods.keys())}")
        print(f"  Datasets: {list(suite._datasets.keys())}")
        print(f"  Seeds:    {n_seeds}")
        print("=" * 70)
        print()

    results = suite.run(n_seeds=n_seeds, verbose=verbose)

    # ── Summary table ────────────────────────────────────────────────
    print("\n")
    suite.print_summary_table()

    # ── Statistical comparisons ──────────────────────────────────────
    print("\n")
    suite.print_statistical_comparisons()

    # ── Save results JSON ────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'benchmark_vs_mi_clustering.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults JSON saved to {json_path}")

    # Also save via the standard results utility
    save_results(
        'benchmark_vs_mi_clustering',
        results,
        results.get('config', {}),
        notes='US-092: TB hybrid vs MI clustering comparison, '
              f'{n_seeds} seeds across 5 datasets'
    )

    # ── Plots ────────────────────────────────────────────────────────
    methods = list(results['config']['methods'])
    datasets = list(results['config']['datasets'])
    summary = results['summary']

    bar_path = os.path.join(RESULTS_DIR, 'benchmark_vs_mi_bars.png')
    plot_comparison_bars(summary, methods, datasets, save_path=bar_path)

    runtime_path = os.path.join(RESULTS_DIR, 'benchmark_vs_mi_runtime.png')
    plot_runtime_comparison(summary, methods, datasets, save_path=runtime_path)

    radar_path = os.path.join(RESULTS_DIR, 'benchmark_vs_mi_radar.png')
    suite.plot_radar(save_path=radar_path)

    # ── Final summary printout ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("US-092 RESULTS SUMMARY")
    print("=" * 70)

    # Aggregate wins
    tb_wins = 0
    mi_wins = 0
    ties = 0
    sig_results = []

    comps = results.get('statistical_comparisons', {})
    for comp_key, metrics_dict in sorted(comps.items()):
        parts = comp_key.split('|')
        pair = parts[0]
        ds = parts[1] if len(parts) > 1 else ''
        for metric_name, stat_result in metrics_dict.items():
            p = stat_result.get('p_value', 1.0)
            diff = stat_result.get('mean_diff', 0.0)
            sig = 'significant' if p < 0.05 else 'not significant'
            direction = 'TB > MI' if diff > 0 else ('MI > TB' if diff < 0 else 'tied')

            if p < 0.05:
                if diff > 0:
                    tb_wins += 1
                elif diff < 0:
                    mi_wins += 1
                else:
                    ties += 1
            else:
                ties += 1

            sig_results.append({
                'dataset': ds,
                'metric': metric_name,
                'mean_diff': diff,
                'p_value': p,
                'cohens_d': stat_result.get('cohens_d', 0.0),
                'effect_size': stat_result.get('effect_size', 'N/A'),
                'direction': direction,
                'significant': sig,
            })

    print(f"\nSignificant wins: TB_hybrid={tb_wins}, MI_clustering={mi_wins}, ties/ns={ties}")
    print(f"\nDetailed statistical results:")
    print(f"{'Dataset':<20s} {'Metric':<12s} {'Diff':>8s} {'p-value':>10s} {'d':>8s} {'Effect':>12s} {'Winner':>10s}")
    print("-" * 82)
    for r in sig_results:
        star = '*' if r['p_value'] < 0.05 else ' '
        print(f"{r['dataset']:<20s} {r['metric']:<12s} {r['mean_diff']:>+8.3f} "
              f"{r['p_value']:>10.4f}{star} {r['cohens_d']:>+8.3f} "
              f"{r['effect_size']:>12s} {r['direction']:>10s}")

    print("\n" + "=" * 70)
    print("Benchmark complete. Outputs saved to ralph/results/")
    print("=" * 70)

    return suite, results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='US-092: Benchmark TB vs MI Clustering')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds (default: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 3 seeds')
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.n_seeds
    suite, results = run_benchmark_vs_mi(n_seeds=n_seeds)
