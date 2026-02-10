"""
US-094: Benchmark TB vs Spectral Clustering on Raw Covariance
==============================================================

Compares Topological Blankets (hybrid method) against a simple baseline:
spectral clustering applied to the absolute sample covariance matrix.

The baseline works as follows:
  1. Compute the sample covariance matrix from the raw data.
  2. Take absolute values to form a non-negative affinity matrix.
  3. Apply sklearn SpectralClustering with affinity='precomputed'.
  4. Identify blanket variables as those with high inter-cluster
     covariance (i.e., variables that couple distinct clusters).

This represents the simplest possible structure-discovery approach:
cluster variables by how correlated they are, with no gradient or
energy landscape information at all. If TB outperforms this baseline,
it validates that the geometric/topological features TB extracts from
gradients carry genuine structural signal beyond raw correlation.

Outputs:
  - Summary table (ARI, blanket_F1, NMI, runtime) per dataset
  - Paired statistical significance tests (Wilcoxon / paired-t)
  - JSON results and comparison plots saved to ralph/results/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from sklearn.cluster import SpectralClustering
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

RESULTS_DIR = Path(RALPH_DIR) / "results"


# =========================================================================
# Spectral Covariance Baseline
# =========================================================================

def spectral_covariance_method(samples, gradients, n_clusters=None):
    """
    Baseline: spectral clustering on the absolute sample covariance matrix.

    Steps:
      1. Compute sample covariance from raw samples.
      2. Take |cov| as the affinity (symmetric, non-negative).
      3. Run SpectralClustering with affinity='precomputed'.
      4. Identify blanket variables: those whose mean absolute covariance
         with variables in *other* clusters exceeds a threshold.

    Args:
        samples: (n_samples, n_vars) array of observations.
        gradients: (n_samples, n_vars) array; ignored by this method.
        n_clusters: Number of clusters. If None, auto-detect via
                    eigengap of the covariance-based Laplacian.

    Returns:
        assignment: (n_vars,) int array; label per variable, -1 = blanket.
    """
    n_vars = samples.shape[1]

    # Step 1: Sample covariance
    cov = np.cov(samples, rowvar=False)  # (n_vars, n_vars)

    # Step 2: Absolute covariance as affinity
    affinity = np.abs(cov)

    # Ensure diagonal is zero for Laplacian-based eigengap detection,
    # but SpectralClustering handles diagonal internally.
    # Zero out the diagonal for cleaner clustering signal.
    np.fill_diagonal(affinity, 0.0)

    # Auto-detect n_clusters from eigengap if not specified
    if n_clusters is None:
        n_clusters = _auto_detect_clusters(affinity, max_k=10)

    # Step 3: Spectral clustering with precomputed affinity
    # Add small epsilon to diagonal for numerical stability
    affinity_stable = affinity.copy()
    np.fill_diagonal(affinity_stable, np.max(affinity) * 0.01)

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=10,
        )
        labels = sc.fit_predict(affinity_stable)
    except Exception:
        # Fallback: simple k-means on the covariance rows
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(affinity)

    # Step 4: Identify blanket variables via inter-cluster covariance.
    # For each variable, compute the ratio of its mean absolute covariance
    # with variables in other clusters vs. variables in its own cluster.
    # Variables with high cross-cluster coupling are blanket candidates.
    assignment = labels.copy()
    cross_coupling = np.zeros(n_vars)

    for i in range(n_vars):
        own_cluster = labels[i]
        same = [j for j in range(n_vars) if j != i and labels[j] == own_cluster]
        diff = [j for j in range(n_vars) if labels[j] != own_cluster]

        intra = np.mean(affinity[i, same]) if same else 0.0
        inter = np.mean(affinity[i, diff]) if diff else 0.0

        # Cross-coupling ratio: how much this variable talks to other clusters
        if intra + inter > 1e-12:
            cross_coupling[i] = inter / (intra + inter)
        else:
            cross_coupling[i] = 0.0

    # Variables in the top quartile of cross-coupling are blanket candidates.
    # Use a threshold: if cross_coupling > median + 0.5 * IQR, mark as blanket.
    q25, q50, q75 = np.percentile(cross_coupling, [25, 50, 75])
    iqr = q75 - q25
    blanket_threshold = q50 + 0.5 * iqr

    # Also require minimum cross-coupling > 0.3 to avoid labeling
    # everything as blanket in well-separated cases.
    blanket_threshold = max(blanket_threshold, 0.3)

    blanket_mask = cross_coupling > blanket_threshold
    assignment[blanket_mask] = -1

    return assignment


def _auto_detect_clusters(affinity, max_k=10):
    """
    Estimate the number of clusters from the eigengap of the
    graph Laplacian derived from the affinity matrix.
    """
    n = affinity.shape[0]
    max_k = min(max_k, n - 1)

    # Degree matrix and normalized Laplacian
    D = np.diag(np.sum(affinity, axis=1))
    L = D - affinity

    # Avoid singular D
    d_inv_sqrt = np.zeros(n)
    diag_D = np.diag(D)
    nonzero = diag_D > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(diag_D[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    from scipy.linalg import eigh
    eigvals, _ = eigh(L_norm)
    eigvals = np.sort(np.real(eigvals))

    # Eigengap: largest gap in the first max_k eigenvalues
    gaps = np.diff(eigvals[1:max_k + 1])  # skip the first (near-zero) eigenvalue
    if len(gaps) == 0:
        return 2

    best_k = np.argmax(gaps) + 2  # +2 because: +1 for diff offset, +1 because we skipped eigval 0
    return max(2, best_k)


# =========================================================================
# Convenience wrapper matching the BenchmarkSuite method signature
# =========================================================================

def _spectral_cov_method(samples, gradients):
    """Spectral covariance wrapper for the BenchmarkSuite interface."""
    return spectral_covariance_method(samples, gradients)


# =========================================================================
# Comparison bar chart
# =========================================================================

def plot_comparison_bars(summary, methods, datasets, save_path=None):
    """
    Create grouped bar charts comparing methods across datasets
    for ARI, blanket_F1, and NMI.
    """
    metrics = ['ARI', 'blanket_F1', 'NMI']
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    bar_width = 0.8 / n_methods

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(n_datasets)

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

            offset = (m_idx - n_methods / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds,
                   label=method, color=colors[m_idx % len(colors)],
                   alpha=0.85, capsize=3)

        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)

    fig.suptitle('US-094: TB (hybrid) vs Spectral Covariance Baseline', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    else:
        save_figure(fig, 'tb_vs_spectral_cov_comparison', 'us094_benchmark')
    plt.close(fig)

    return fig


def plot_runtime_comparison(summary, methods, datasets, save_path=None):
    """Bar chart comparing runtime across datasets."""
    n_datasets = len(datasets)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71', '#e74c3c']
    bar_width = 0.8 / n_methods
    x = np.arange(n_datasets)

    for m_idx, method in enumerate(methods):
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

        offset = (m_idx - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=method, color=colors[m_idx % len(colors)],
               alpha=0.85, capsize=3)

    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Runtime (seconds)', fontsize=10)
    ax.set_title('Runtime Comparison: TB (hybrid) vs Spectral Covariance', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Runtime chart saved to {save_path}")
    else:
        save_figure(fig, 'tb_vs_spectral_cov_runtime', 'us094_benchmark')
    plt.close(fig)

    return fig


# =========================================================================
# Main
# =========================================================================

def run_us094_benchmark(n_seeds=10, verbose=True):
    """
    Execute US-094: Benchmark TB (hybrid) vs Spectral Covariance baseline
    across all 5 standardized datasets with paired statistical testing.
    """
    # Build the default suite (which already has datasets registered)
    suite = build_default_suite()

    # Clear the default methods and register only what we need
    suite._methods.clear()
    suite.register_method('TB_hybrid', _tb_hybrid_method)
    suite.register_method('Spectral_Cov', _spectral_cov_method)

    if verbose:
        print("=" * 70)
        print("US-094: TB (hybrid) vs Spectral Covariance Baseline")
        print("=" * 70)
        print(f"  Methods:  {list(suite._methods.keys())}")
        print(f"  Datasets: {list(suite._datasets.keys())}")
        print(f"  Seeds:    {n_seeds}")
        print("=" * 70)
        print()

    # Run the benchmark
    results = suite.run(n_seeds=n_seeds, verbose=verbose)

    # Print summary
    print("\n")
    print("=" * 70)
    print("RESULTS TABLE: TB (hybrid) vs Spectral Covariance")
    print("=" * 70)
    suite.print_summary_table()

    # Print statistical comparisons
    print("\n")
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    suite.print_statistical_comparisons()

    # Compute and display a condensed comparison
    summary = results['summary']
    methods = results['config']['methods']
    datasets = results['config']['datasets']

    print("\n")
    print("=" * 70)
    print("CONDENSED COMPARISON (TB_hybrid advantage)")
    print("=" * 70)
    print(f"{'Dataset':<20s} | {'Metric':<12s} | {'TB_hybrid':>10s} | {'Spec_Cov':>10s} | {'Delta':>8s} | {'p-value':>10s} | {'Sig':>5s}")
    print("-" * 90)

    for ds in datasets:
        for metric in ['ARI', 'blanket_F1', 'NMI']:
            tb_key = f"TB_hybrid|{ds}"
            sc_key = f"Spectral_Cov|{ds}"
            comp_key = f"TB_hybrid_vs_Spectral_Cov|{ds}"

            tb_val = summary[tb_key][metric + '_mean'] if tb_key in summary else 0.0
            sc_val = summary[sc_key][metric + '_mean'] if sc_key in summary else 0.0
            delta = tb_val - sc_val

            # Get p-value from statistical comparisons
            p_val = 1.0
            comps = results.get('statistical_comparisons', {})
            if comp_key in comps and metric in comps[comp_key]:
                p_val = comps[comp_key][metric].get('p_value', 1.0)

            sig = '*' if p_val < 0.05 else ''
            if p_val < 0.01:
                sig = '**'
            if p_val < 0.001:
                sig = '***'

            print(f"{ds:<20s} | {metric:<12s} | {tb_val:>10.3f} | {sc_val:>10.3f} | {delta:>+8.3f} | {p_val:>10.4f} | {sig:>5s}")
        print("-" * 90)

    # Overall summary
    print("\nOverall mean across datasets:")
    for metric in ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds']:
        tb_vals = [summary[f"TB_hybrid|{ds}"][metric + '_mean']
                   for ds in datasets if f"TB_hybrid|{ds}" in summary]
        sc_vals = [summary[f"Spectral_Cov|{ds}"][metric + '_mean']
                   for ds in datasets if f"Spectral_Cov|{ds}" in summary]
        tb_mean = np.mean(tb_vals) if tb_vals else 0.0
        sc_mean = np.mean(sc_vals) if sc_vals else 0.0
        label = metric.replace('wall_clock_seconds', 'runtime(s)')
        print(f"  {label:<15s}: TB_hybrid={tb_mean:.3f}, Spectral_Cov={sc_mean:.3f}, "
              f"delta={tb_mean - sc_mean:+.3f}")

    # Save results JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "us094_tb_vs_spectral_covariance.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults JSON saved to {json_path}")

    # Also save through the standard results pipeline
    save_results(
        'us094_tb_vs_spectral_covariance',
        results,
        results.get('config', {}),
        notes='US-094: Benchmark TB (hybrid) vs spectral clustering on raw covariance'
    )

    # Generate plots
    plot_comparison_bars(
        summary, methods, datasets,
        save_path=str(RESULTS_DIR / "us094_comparison_bars.png")
    )
    plot_runtime_comparison(
        summary, methods, datasets,
        save_path=str(RESULTS_DIR / "us094_runtime_comparison.png")
    )

    # Radar chart
    suite.plot_radar(save_path=str(RESULTS_DIR / "us094_radar_chart.png"))

    print("\nUS-094 benchmark complete. All outputs saved to ralph/results/")
    return suite, results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='US-094: Benchmark TB vs Spectral Covariance'
    )
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds (default: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 3 seeds')
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.n_seeds
    suite, results = run_us094_benchmark(n_seeds=n_seeds)
