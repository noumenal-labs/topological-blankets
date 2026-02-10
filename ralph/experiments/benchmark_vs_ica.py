"""
US-093: Benchmark TB vs ICA-based Structure Discovery
======================================================

Compares Topological Blankets (hybrid method) against ICA-based structure
discovery on the standardized benchmark datasets from BenchmarkSuite (US-089).

ICA-based structure discovery pipeline:
  1. Run FastICA on the sample matrix to extract independent components.
  2. Extract the mixing matrix A (n_features x n_components).
  3. Threshold the absolute mixing matrix to build a dependency graph
     (edge where |A[i,j]| > threshold).
  4. Spectral cluster the dependency graph to find object partitions.
  5. Assign blanket labels to variables with high cross-cluster mixing
     weights (variables that load strongly onto components associated
     with multiple clusters).

Metrics reported per (method, dataset):
  - ARI (Adjusted Rand Index) for partition recovery
  - blanket_F1 for blanket variable detection
  - NMI (Normalized Mutual Information)
  - wall_clock_seconds for runtime

Statistical significance: paired t-test or Wilcoxon signed-rank across
10 seeds, with Cohen's d effect size.

Outputs:
  - Console summary table
  - Statistical comparison printout
  - results/  JSON with full metrics
  - results/  comparison bar plots and radar chart
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

# ── Path setup ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from experiments.benchmark_suite import (
    BenchmarkSuite, build_default_suite,
    _tb_hybrid_method,
    _json_default,
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

from sklearn.decomposition import FastICA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from scipy import stats


# =========================================================================
# ICA-based structure discovery method
# =========================================================================

def ica_structure_discovery(samples, gradients,
                            n_components=None,
                            mixing_threshold_percentile=75,
                            cross_cluster_percentile=80):
    """
    ICA-based structure discovery: extract independent components from the
    sample matrix, build a dependency graph from the mixing matrix, cluster
    the graph, and identify blanket variables from cross-cluster mixing.

    Args:
        samples: (n_samples, n_vars) array of observations.
        gradients: (n_samples, n_vars) array of gradients (unused by ICA,
                   but accepted to match the benchmark interface).
        n_components: Number of ICA components. If None, set to n_vars.
        mixing_threshold_percentile: Percentile of |mixing_matrix| values
            above which an edge is placed in the dependency graph.
        cross_cluster_percentile: Percentile of cross-cluster mixing weight
            above which a variable is labeled as blanket.

    Returns:
        labels: (n_vars,) array where labels[i] = cluster index, or -1
                for blanket variables.
    """
    n_samples, n_vars = samples.shape

    if n_components is None:
        n_components = min(n_vars, n_samples - 1)

    # Step 1: Run FastICA to find independent components
    ica = FastICA(
        n_components=n_components,
        max_iter=500,
        tol=1e-4,
        random_state=42,
        whiten='unit-variance',
    )
    try:
        S = ica.fit_transform(samples)  # (n_samples, n_components)
        A = ica.mixing_                  # (n_vars, n_components)
    except Exception:
        # If ICA fails to converge, return all-zero labels (no structure)
        return np.zeros(n_vars, dtype=int)

    # Step 2: Build dependency graph from the mixing matrix.
    # The absolute mixing matrix captures how strongly each variable
    # depends on each independent source. Variables that load onto the
    # same sources are structurally coupled.
    abs_A = np.abs(A)

    # Affinity between variables i and j: cosine similarity of their
    # mixing-matrix rows, so variables loading on the same components
    # are adjacent.
    norms = np.linalg.norm(abs_A, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    A_normed = abs_A / norms
    affinity = A_normed @ A_normed.T  # (n_vars, n_vars), values in [0,1]
    np.fill_diagonal(affinity, 0.0)

    # Threshold: zero out weak connections
    threshold = np.percentile(affinity[affinity > 0], mixing_threshold_percentile)
    affinity[affinity < threshold] = 0.0

    # Step 3: Spectral clustering on the affinity graph.
    # Auto-detect number of clusters from the eigengap of the graph Laplacian.
    degree = np.sum(affinity, axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    L_sym = np.eye(n_vars) - D_inv_sqrt @ affinity @ D_inv_sqrt

    eigvals = np.sort(np.linalg.eigvalsh(L_sym))
    # Eigengap heuristic: look for the largest gap in the first ~10 eigenvalues
    max_k = min(10, n_vars - 1)
    gaps = np.diff(eigvals[:max_k + 1])
    # Skip the first gap (between 0 and the second eigenvalue); start at index 1
    if len(gaps) > 2:
        n_clusters = int(np.argmax(gaps[1:]) + 2)
    else:
        n_clusters = 2
    n_clusters = max(2, min(n_clusters, n_vars // 2))

    # Ensure the affinity matrix is suitable for SpectralClustering
    affinity_for_sc = affinity.copy()
    # Add small epsilon to diagonal so every node has nonzero degree
    np.fill_diagonal(affinity_for_sc, 1e-6)

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=10,
        )
        cluster_labels = sc.fit_predict(affinity_for_sc)
    except Exception:
        # Fallback: simple k-means on mixing matrix rows
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(abs_A)

    # Step 4: Identify blanket variables.
    # A blanket variable has high mixing weights to components that belong
    # to *different* clusters. Compute cross-cluster mixing weight for each
    # variable.
    #
    # For each variable i in cluster c_i, measure how much its mixing row
    # overlaps with the centroid mixing patterns of other clusters.
    cluster_centroids = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if np.sum(mask) > 0:
            cluster_centroids[c] = np.mean(abs_A[mask], axis=0)

    cross_cluster_weight = np.zeros(n_vars)
    for i in range(n_vars):
        own_cluster = cluster_labels[i]
        own_centroid = cluster_centroids[own_cluster]
        other_sims = []
        for c, centroid in cluster_centroids.items():
            if c != own_cluster:
                # Cosine similarity between variable's mixing row and other-cluster centroid
                dot = np.dot(abs_A[i], centroid)
                norm_prod = (np.linalg.norm(abs_A[i]) * np.linalg.norm(centroid))
                if norm_prod > 1e-10:
                    other_sims.append(dot / norm_prod)
        if other_sims:
            cross_cluster_weight[i] = np.max(other_sims)

    # Variables with cross-cluster weight above the threshold are blanket
    if np.any(cross_cluster_weight > 0):
        blanket_threshold = np.percentile(
            cross_cluster_weight[cross_cluster_weight > 0],
            cross_cluster_percentile
        )
        blanket_mask = cross_cluster_weight >= blanket_threshold
    else:
        blanket_mask = np.zeros(n_vars, dtype=bool)

    # Ensure at least some blanket variables if there are enough variables
    # (this avoids the degenerate case where the method finds no blanket at all)
    # but cap at 40% of variables
    n_blanket = np.sum(blanket_mask)
    max_blanket = max(1, int(0.4 * n_vars))
    if n_blanket > max_blanket:
        # Keep only the top max_blanket by cross_cluster_weight
        sorted_idx = np.argsort(cross_cluster_weight)[::-1]
        blanket_mask = np.zeros(n_vars, dtype=bool)
        blanket_mask[sorted_idx[:max_blanket]] = True

    # Build final labels: cluster index for internal variables, -1 for blanket
    labels = cluster_labels.copy()
    labels[blanket_mask] = -1

    return labels


# =========================================================================
# Plotting: comparison bar chart and radar
# =========================================================================

def plot_comparison_bars(summary, methods, datasets, save_path=None):
    """
    Create grouped bar charts comparing methods across datasets for each metric.
    """
    metrics = ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds']
    metric_labels = ['ARI', 'Blanket F1', 'NMI', 'Runtime (s)']

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    colors = {'TB_hybrid': '#2ecc71', 'ICA': '#e74c3c'}
    bar_width = 0.35

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))

        for m_idx, method in enumerate(methods):
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

            offset = (m_idx - 0.5) * bar_width
            ax.bar(x + offset + bar_width / 2, means, bar_width,
                   yerr=stds, label=method,
                   color=colors.get(method, '#3498db'),
                   alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('TB vs ICA: Structure Discovery Benchmark (US-093)', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison bar chart saved to {save_path}")
    else:
        save_figure(fig, 'tb_vs_ica_bars', 'benchmark_vs_ica')
    plt.close(fig)

    return fig


def plot_comparison_radar(summary, methods, datasets, save_path=None):
    """
    Radar chart: aggregate method profiles across all datasets.
    """
    metrics = ['ARI', 'blanket_F1', 'NMI']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = {'TB_hybrid': '#2ecc71', 'ICA': '#e74c3c'}

    for method in methods:
        values = []
        for metric in metrics:
            vals = []
            for ds in datasets:
                key = f"{method}|{ds}"
                if key in summary:
                    vals.append(summary[key][metric + '_mean'])
            values.append(np.mean(vals) if vals else 0.0)

        norm_values = [max(0.0, min(1.0, v)) for v in values]
        norm_values += norm_values[:1]

        color = colors.get(method, '#3498db')
        ax.plot(angles, norm_values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, norm_values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('TB vs ICA: Quality Profile (US-093)', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Radar chart saved to {save_path}")
    else:
        save_figure(fig, 'tb_vs_ica_radar', 'benchmark_vs_ica')
    plt.close(fig)

    return fig


# =========================================================================
# Main: build suite, register both methods, run, report
# =========================================================================

def run_tb_vs_ica_benchmark(n_seeds=10, verbose=True):
    """
    Build BenchmarkSuite with TB (hybrid) and ICA methods registered,
    run on all 5 standard datasets with n_seeds seeds, produce results.
    """
    # Build the default suite (has all 5 datasets + TB methods)
    base_suite = build_default_suite()

    # Create a fresh suite with only TB_hybrid and ICA
    suite = BenchmarkSuite()

    # Register TB hybrid
    suite.register_method('TB_hybrid', _tb_hybrid_method)

    # Register ICA
    suite.register_method('ICA', ica_structure_discovery)

    # Copy over all datasets from the default suite
    for ds_name, ds_config in base_suite._datasets.items():
        suite.register_dataset(ds_name, ds_config['generator'], ds_config['ground_truth'])

    if verbose:
        print("=" * 72)
        print("US-093: TB vs ICA Benchmark")
        print("=" * 72)
        print(f"  Methods:  {list(suite._methods.keys())}")
        print(f"  Datasets: {list(suite._datasets.keys())}")
        print(f"  Seeds:    {n_seeds}")
        print("=" * 72)
        print()

    # Run the benchmark
    results = suite.run(n_seeds=n_seeds, verbose=verbose)

    # Print summary table
    print("\n")
    print("=" * 72)
    print("RESULTS TABLE: TB vs ICA")
    print("=" * 72)
    suite.print_summary_table()

    # Print statistical comparisons
    print("\n")
    print("=" * 72)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 72)
    suite.print_statistical_comparisons()

    # Save results JSON
    results_dir = Path(TB_PACKAGE_DIR) / 'ralph' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = str(results_dir / 'benchmark_tb_vs_ica.json')
    suite.save_results_json(json_path)

    # Generate comparison plots
    summary = results['summary']
    methods = ['TB_hybrid', 'ICA']
    datasets = list(suite._datasets.keys())

    bar_path = str(results_dir / 'benchmark_tb_vs_ica_bars.png')
    plot_comparison_bars(summary, methods, datasets, save_path=bar_path)

    radar_path = str(results_dir / 'benchmark_tb_vs_ica_radar.png')
    plot_comparison_radar(summary, methods, datasets, save_path=radar_path)

    # Also generate the built-in radar
    suite.plot_radar(save_path=str(results_dir / 'benchmark_tb_vs_ica_suite_radar.png'))

    # ── Compact results summary ──────────────────────────────────────────
    print("\n")
    print("=" * 72)
    print("COMPACT SUMMARY: TB_hybrid vs ICA (mean +/- std across seeds)")
    print("=" * 72)

    header = f"{'Dataset':<20s} | {'Metric':<12s} | {'TB_hybrid':>18s} | {'ICA':>18s} | {'Winner':>10s}"
    print(header)
    print("-" * len(header))

    for ds in datasets:
        for metric in ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds']:
            key_tb = f"TB_hybrid|{ds}"
            key_ica = f"ICA|{ds}"

            tb_mean = summary[key_tb][metric + '_mean']
            tb_std = summary[key_tb][metric + '_std']
            ica_mean = summary[key_ica][metric + '_mean']
            ica_std = summary[key_ica][metric + '_std']

            if metric == 'wall_clock_seconds':
                winner = 'TB' if tb_mean < ica_mean else 'ICA'
            else:
                winner = 'TB' if tb_mean > ica_mean else ('ICA' if ica_mean > tb_mean else 'TIE')

            short_metric = metric.replace('wall_clock_seconds', 'runtime(s)')
            print(f"{ds:<20s} | {short_metric:<12s} | "
                  f"{tb_mean:>7.4f}+/-{tb_std:>6.4f} | "
                  f"{ica_mean:>7.4f}+/-{ica_std:>6.4f} | "
                  f"{winner:>10s}")
        print("-" * len(header))

    # ── Win counts ───────────────────────────────────────────────────────
    print("\nOverall wins (quality metrics: ARI, blanket_F1, NMI):")
    tb_wins = 0
    ica_wins = 0
    ties = 0
    for ds in datasets:
        for metric in ['ARI', 'blanket_F1', 'NMI']:
            key_tb = f"TB_hybrid|{ds}"
            key_ica = f"ICA|{ds}"
            tb_val = summary[key_tb][metric + '_mean']
            ica_val = summary[key_ica][metric + '_mean']
            if tb_val > ica_val + 1e-6:
                tb_wins += 1
            elif ica_val > tb_val + 1e-6:
                ica_wins += 1
            else:
                ties += 1

    print(f"  TB_hybrid wins: {tb_wins}/{tb_wins + ica_wins + ties}")
    print(f"  ICA wins:       {ica_wins}/{tb_wins + ica_wins + ties}")
    print(f"  Ties:           {ties}/{tb_wins + ica_wins + ties}")

    print(f"\nResults saved to: {json_path}")
    print(f"Bar chart: {bar_path}")
    print(f"Radar chart: {radar_path}")
    print("\nUS-093 complete.")

    return suite, results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='US-093: Benchmark TB vs ICA-based Structure Discovery'
    )
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds (default: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 3 seeds')
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.n_seeds
    suite, results = run_tb_vs_ica_benchmark(n_seeds=n_seeds)
