"""
US-090: Benchmark TB vs Graphical Lasso on Standardized Datasets
================================================================

This script runs a head-to-head comparison of Topological Blankets (TB,
hybrid method) against Graphical Lasso (sklearn GraphicalLassoCV) on all
five standardized benchmark datasets from US-089.

For each dataset and seed the Graphical Lasso method:
  1. Fits GraphicalLassoCV to the sample matrix.
  2. Extracts the precision matrix.
  3. Thresholds the absolute off-diagonal entries (Otsu or median).
  4. Builds a binary adjacency graph from the thresholded precision.
  5. Applies spectral clustering to the adjacency graph.
  6. Assigns blanket labels to the highest-degree nodes (those whose
     degree exceeds the 75th percentile in the thresholded graph).

Outputs:
  - Comparison table (ARI, blanket_F1, NMI, runtime) for TB vs GL per dataset.
  - Paired t-test / Wilcoxon significance for each metric per dataset.
  - Results JSON saved to ralph/results/.
  - Bar-chart comparison plots saved to ralph/results/.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Path setup (mirrors benchmark_suite.py) ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from experiments.benchmark_suite import (
    BenchmarkSuite,
    build_default_suite,
    _tb_hybrid_method,
    compute_benchmark_metrics,
    paired_statistical_test,
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import SpectralClustering
from scipy import stats


# =========================================================================
# Graphical Lasso method wrapper
# =========================================================================

def _graphical_lasso_method(samples, gradients):
    """
    Graphical Lasso baseline conforming to the BenchmarkSuite interface.

    Steps:
      1. Fit GraphicalLassoCV to the sample covariance.
      2. Extract the precision matrix.
      3. Threshold absolute off-diagonal entries to get a binary adjacency.
      4. Spectral-cluster the thresholded adjacency.
      5. Mark high-degree nodes as blanket variables (label = -1).

    Parameters:
        samples:   (n_samples, n_vars) array of observations.
        gradients: (n_samples, n_vars) array (unused by GL, included for API).

    Returns:
        labels: (n_vars,) array with object indices 0..k-1 and -1 for blanket.
    """
    n_vars = samples.shape[1]

    # ── Step 1: Fit GL ────────────────────────────────────────────────
    try:
        gl = GraphicalLassoCV(cv=3, max_iter=500, assume_centered=False)
        gl.fit(samples)
        precision = gl.precision_
    except Exception:
        # If GL fails to converge, fall back to empirical inverse
        cov = np.cov(samples.T) + 1e-4 * np.eye(n_vars)
        precision = np.linalg.inv(cov)

    # ── Step 2-3: Threshold to binary adjacency ──────────────────────
    abs_prec = np.abs(precision.copy())
    np.fill_diagonal(abs_prec, 0.0)
    upper_vals = abs_prec[np.triu_indices(n_vars, k=1)]
    nonzero_vals = upper_vals[upper_vals > 1e-10]

    if len(nonzero_vals) > 2:
        # Otsu-style threshold: split nonzero values at the point that
        # minimizes intra-class variance (binary histogram version).
        sorted_vals = np.sort(nonzero_vals)
        best_thresh = np.median(nonzero_vals)
        best_var = np.inf
        for candidate in np.percentile(sorted_vals, np.arange(10, 91, 5)):
            lo = sorted_vals[sorted_vals <= candidate]
            hi = sorted_vals[sorted_vals > candidate]
            if len(lo) == 0 or len(hi) == 0:
                continue
            w0 = len(lo) / len(sorted_vals)
            w1 = len(hi) / len(sorted_vals)
            inter_var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
            # Maximize inter-class variance (equivalent to minimizing intra)
            if inter_var > 0 and (1.0 / inter_var) < best_var:
                best_var = 1.0 / inter_var
                best_thresh = candidate
        threshold = best_thresh
    else:
        threshold = 0.0

    adjacency = (abs_prec > threshold).astype(float)

    # ── Step 4: Spectral clustering ──────────────────────────────────
    # Determine number of clusters from the graph Laplacian eigengap
    degree = adjacency.sum(axis=1)
    D = np.diag(degree + 1e-10)
    L = D - adjacency
    try:
        eigvals = np.sort(np.linalg.eigvalsh(L))
        gaps = np.diff(eigvals[:min(10, len(eigvals))])
        if len(gaps) > 1:
            # Skip the first eigenvalue (always ~0 for connected graph)
            n_clusters = int(np.argmax(gaps[1:]) + 2)
            n_clusters = max(2, min(n_clusters, n_vars // 2))
        else:
            n_clusters = 2
    except Exception:
        n_clusters = 2

    # Add a small identity to adjacency to ensure connectivity for spectral clustering
    adj_for_clustering = adjacency + 0.01 * np.eye(n_vars)

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=10,
        )
        labels = sc.fit_predict(adj_for_clustering)
    except Exception:
        # Fallback: assign all to one cluster
        labels = np.zeros(n_vars, dtype=int)

    # ── Step 5: Mark high-degree nodes as blanket ────────────────────
    if len(degree) > 0 and degree.max() > 0:
        degree_threshold = np.percentile(degree, 75)
        blanket_mask = degree >= degree_threshold
        # Require at least 1 blanket variable
        if blanket_mask.sum() == 0:
            blanket_mask[np.argmax(degree)] = True
        labels[blanket_mask] = -1

    return labels


# =========================================================================
# Comparison runner
# =========================================================================

def run_comparison(n_seeds=10, verbose=True):
    """
    Build the default benchmark suite, register GL alongside TB hybrid,
    run all datasets for n_seeds, then produce results and plots.
    """
    # Build the default suite (which has all 5 datasets + TB methods)
    suite = build_default_suite()

    # Remove TB_gradient so we only compare TB_hybrid vs GL
    if 'TB_gradient' in suite._methods:
        del suite._methods['TB_gradient']

    # Register graphical lasso
    suite.register_method('GraphicalLasso', _graphical_lasso_method)

    if verbose:
        print("=" * 70)
        print("US-090: TB vs Graphical Lasso Benchmark")
        print("=" * 70)
        print(f"  Methods:  {list(suite._methods.keys())}")
        print(f"  Datasets: {list(suite._datasets.keys())}")
        print(f"  Seeds:    {n_seeds}")
        print("=" * 70)

    # Run the benchmark
    results = suite.run(n_seeds=n_seeds, verbose=verbose)

    # ── Print detailed comparison table ───────────────────────────────
    print("\n")
    print("=" * 90)
    print("RESULTS TABLE: TB_hybrid vs GraphicalLasso")
    print("=" * 90)
    header = (f"{'Dataset':<22s} | {'Method':<16s} | "
              f"{'ARI':>10s} | {'blanket_F1':>12s} | "
              f"{'NMI':>10s} | {'time(s)':>10s}")
    print(header)
    print("-" * len(header))

    summary = results['summary']
    datasets = results['config']['datasets']
    methods = results['config']['methods']

    for ds_name in datasets:
        for method_name in methods:
            key = f"{method_name}|{ds_name}"
            if key not in summary:
                continue
            row = summary[key]
            ari_str = f"{row['ARI_mean']:.3f}+/-{row['ARI_std']:.3f}"
            f1_str = f"{row['blanket_F1_mean']:.3f}+/-{row['blanket_F1_std']:.3f}"
            nmi_str = f"{row['NMI_mean']:.3f}+/-{row['NMI_std']:.3f}"
            time_str = f"{row['wall_clock_seconds_mean']:.3f}"
            print(f"{ds_name:<22s} | {method_name:<16s} | "
                  f"{ari_str:>10s} | {f1_str:>12s} | "
                  f"{nmi_str:>10s} | {time_str:>10s}")
        print("-" * len(header))

    # ── Statistical significance ──────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("STATISTICAL SIGNIFICANCE (paired test across seeds)")
    print("=" * 90)
    suite.print_statistical_comparisons()

    # ── Win/loss tally ────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("WIN/LOSS TALLY (TB_hybrid vs GraphicalLasso)")
    print("=" * 70)
    tb_wins_ari = 0
    tb_wins_f1 = 0
    tb_wins_nmi = 0
    n_datasets = 0
    for ds_name in datasets:
        tb_key = f"TB_hybrid|{ds_name}"
        gl_key = f"GraphicalLasso|{ds_name}"
        if tb_key not in summary or gl_key not in summary:
            continue
        n_datasets += 1
        tb_row = summary[tb_key]
        gl_row = summary[gl_key]

        ari_winner = "TB" if tb_row['ARI_mean'] > gl_row['ARI_mean'] else "GL"
        f1_winner = "TB" if tb_row['blanket_F1_mean'] > gl_row['blanket_F1_mean'] else "GL"
        nmi_winner = "TB" if tb_row['NMI_mean'] > gl_row['NMI_mean'] else "GL"

        if ari_winner == "TB":
            tb_wins_ari += 1
        if f1_winner == "TB":
            tb_wins_f1 += 1
        if nmi_winner == "TB":
            tb_wins_nmi += 1

        print(f"  {ds_name:<22s}  ARI: {ari_winner:<4s}  F1: {f1_winner:<4s}  NMI: {nmi_winner:<4s}")

    print(f"\n  TB wins ARI:        {tb_wins_ari}/{n_datasets}")
    print(f"  TB wins blanket_F1: {tb_wins_f1}/{n_datasets}")
    print(f"  TB wins NMI:        {tb_wins_nmi}/{n_datasets}")

    # ── Save results ──────────────────────────────────────────────────
    config = {
        'n_seeds': n_seeds,
        'methods': methods,
        'datasets': datasets,
        'experiment': 'US-090',
        'description': 'TB_hybrid vs GraphicalLasso on standardized benchmarks',
    }

    save_results(
        'benchmark_vs_graphical_lasso',
        results,
        config,
        notes='US-090: TB vs Graphical Lasso. Paired comparison across 5 datasets, '
              f'{n_seeds} seeds per dataset.'
    )

    # ── Plots ─────────────────────────────────────────────────────────
    _plot_comparison_bars(summary, datasets, methods)
    _plot_runtime_comparison(summary, datasets, methods)

    # Also generate the radar chart from the suite
    suite.plot_radar()

    print("\nUS-090 complete.")
    return results


# =========================================================================
# Plotting helpers
# =========================================================================

def _plot_comparison_bars(summary, datasets, methods):
    """Grouped bar chart: ARI, blanket_F1, NMI for each dataset, by method."""
    metrics = ['ARI', 'blanket_F1', 'NMI']
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = {'TB_hybrid': '#2ecc71', 'GraphicalLasso': '#3498db'}
    default_colors = ['#e74c3c', '#9b59b6', '#f39c12']

    for ax, metric in zip(axes, metrics):
        x = np.arange(n_datasets)
        width = 0.8 / n_methods

        for m_idx, method_name in enumerate(methods):
            means = []
            stds = []
            for ds_name in datasets:
                key = f"{method_name}|{ds_name}"
                if key in summary:
                    means.append(summary[key][f'{metric}_mean'])
                    stds.append(summary[key][f'{metric}_std'])
                else:
                    means.append(0.0)
                    stds.append(0.0)

            color = colors.get(method_name, default_colors[m_idx % len(default_colors)])
            offset = (m_idx - (n_methods - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, label=method_name,
                   color=color, alpha=0.85, capsize=3)

        ax.set_xticks(x)
        short_names = [d.replace('quadratic_ebm_', 'Q').replace('lunarlander_', 'LL')
                       .replace('fetchpush_', 'FP').replace('ising_', 'Ising') for d in datasets]
        ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('US-090: TB_hybrid vs GraphicalLasso', fontsize=13)
    plt.tight_layout()
    save_figure(fig, 'tb_vs_glasso_metrics', 'benchmark_vs_graphical_lasso')


def _plot_runtime_comparison(summary, datasets, methods):
    """Runtime comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_datasets = len(datasets)
    n_methods = len(methods)
    x = np.arange(n_datasets)
    width = 0.8 / n_methods

    colors = {'TB_hybrid': '#2ecc71', 'GraphicalLasso': '#3498db'}
    default_colors = ['#e74c3c', '#9b59b6']

    for m_idx, method_name in enumerate(methods):
        means = []
        stds = []
        for ds_name in datasets:
            key = f"{method_name}|{ds_name}"
            if key in summary:
                means.append(summary[key]['wall_clock_seconds_mean'])
                stds.append(summary[key]['wall_clock_seconds_std'])
            else:
                means.append(0.0)
                stds.append(0.0)

        color = colors.get(method_name, default_colors[m_idx % len(default_colors)])
        offset = (m_idx - (n_methods - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method_name,
               color=color, alpha=0.85, capsize=3)

    short_names = [d.replace('quadratic_ebm_', 'Q').replace('lunarlander_', 'LL')
                   .replace('fetchpush_', 'FP').replace('ising_', 'Ising') for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Wall Clock (seconds)', fontsize=11)
    ax.set_title('US-090: Runtime Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, 'tb_vs_glasso_runtime', 'benchmark_vs_graphical_lasso')


# =========================================================================
# Main entry point
# =========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='US-090: Benchmark TB vs Graphical Lasso')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds per (method, dataset) pair')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with n_seeds=3')
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.n_seeds
    results = run_comparison(n_seeds=n_seeds)
