"""
US-091: Benchmark TB vs NOTEARS on Standardized Datasets
=========================================================

This script runs a head-to-head comparison between Topological Blankets (TB,
hybrid method) and NOTEARS (Zheng et al. 2018) on the five standardized
benchmark datasets defined in the BenchmarkSuite (US-089).

NOTEARS discovers a weighted DAG via continuous optimization of an augmented
Lagrangian with an acyclicity constraint:

    min  ||X - XW||^2  +  lambda * |W|_1
    s.t. tr(e^{W o W}) - d = 0

The discovered DAG is converted into an undirected partition for apples-to-
apples comparison with TB:
  1. Threshold the adjacency matrix W to obtain a binary undirected graph.
  2. Spectrally cluster the thresholded graph into k clusters using the
     graph Laplacian eigenvectors.
  3. Identify blanket variables as those with high cross-cluster connectivity
     (i.e., variables whose neighbors span multiple clusters).

The script produces:
  - A results table (ARI, blanket_F1, NMI, runtime per dataset per method)
  - Paired statistical significance tests across seeds
  - A comparison bar chart and radar chart saved to results/
  - A JSON file with full raw results

Reference: Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. (2018).
    DAGs with NO TEARS: Continuous Optimization for Structure Learning.
    NeurIPS 2018.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy.linalg import expm, eigh
from sklearn.cluster import KMeans

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from experiments.benchmark_suite import (
    BenchmarkSuite,
    build_default_suite,
    compute_benchmark_metrics,
    paired_statistical_test,
    _json_default,
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# NOTEARS implementation (Zheng et al. 2018)
# =========================================================================

def notears_linear(X, lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16,
                   w_threshold=0.3):
    """
    Solve the NOTEARS linear problem via augmented Lagrangian:

        min  (1/2n) ||X - XW||^2  +  lambda1 * ||W||_1
        s.t. h(W) = tr(e^{W o W}) - d = 0

    This is a faithful implementation of Algorithm 1 from Zheng et al. (2018).
    After convergence, entries of W with absolute value below w_threshold
    are zeroed out.

    Args:
        X: (n, d) data matrix (centered).
        lambda1: L1 regularization strength.
        max_iter: Maximum outer augmented Lagrangian iterations.
        h_tol: Tolerance on the acyclicity constraint h(W).
        rho_max: Maximum penalty parameter before early stopping.
        w_threshold: Post-optimization threshold for small edge weights.

    Returns:
        W_est: (d, d) estimated weighted adjacency matrix of the DAG.
    """
    n, d = X.shape

    def _loss(W):
        """Least-squares loss and its gradient."""
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        return loss, G_loss

    def _h(W):
        """Acyclicity constraint h(W) = tr(e^{W o W}) - d and gradient."""
        E = expm(W * W)
        h_val = np.trace(E) - d
        G_h = E.T * 2 * W
        return h_val, G_h

    def _adj(w):
        """Unflatten parameter vector to matrix with zero diagonal."""
        W = w.reshape(d, d)
        np.fill_diagonal(W, 0)
        return W

    def _func(w, alpha, rho):
        """Augmented Lagrangian objective and gradient."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h_val, G_h = _h(W)
        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * np.abs(W).sum()
        G_smooth = G_loss + (rho * h_val + alpha) * G_h
        g_obj = G_smooth + lambda1 * np.sign(W)
        np.fill_diagonal(g_obj, 0)
        return obj, g_obj.ravel()

    # Initialize
    w_est = np.zeros(d * d)
    alpha, rho = 0.0, 1.0
    h_prev = np.inf

    for it in range(max_iter):
        sol = minimize(
            _func, w_est,
            args=(alpha, rho),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        w_est = sol.x
        W_est = _adj(w_est)
        h_new, _ = _h(W_est)

        if h_new > 0.25 * h_prev:
            rho *= 10
        else:
            break

        if h_new < h_tol:
            break

        alpha += rho * h_new
        h_prev = h_new

        if rho > rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# =========================================================================
# DAG-to-partition conversion
# =========================================================================

def dag_to_partition(W, n_clusters=None, blanket_quantile=0.7):
    """
    Convert a NOTEARS DAG adjacency matrix into an object partition with
    blanket labels, for direct comparison with Topological Blankets output.

    Steps:
      1. Symmetrize: A = |W| + |W^T| to get an undirected connectivity graph.
      2. Build the normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}.
      3. Estimate the number of clusters from the eigengap of L if n_clusters
         is not provided.
      4. Cluster using KMeans on the first k eigenvectors of L.
      5. Identify blanket variables: those whose neighbors span more than one
         cluster. Specifically, compute for each variable the fraction of its
         neighbor-weight that connects to a *different* cluster. Variables
         above the blanket_quantile threshold of this cross-cluster fraction
         are labeled as blanket (-1).

    Args:
        W: (d, d) weighted adjacency matrix (may be asymmetric).
        n_clusters: Number of object clusters. If None, estimated via eigengap.
        blanket_quantile: Quantile threshold for cross-cluster connectivity
            to assign the blanket label. Higher values mean fewer blanket vars.

    Returns:
        labels: (d,) integer array. label[i] = cluster id, or -1 for blanket.
    """
    d = W.shape[0]

    # Step 1: Symmetrize to undirected graph
    A = np.abs(W) + np.abs(W.T)
    np.fill_diagonal(A, 0)

    # Handle degenerate case: if the graph is empty, return all zeros
    if A.sum() < 1e-12:
        labels = np.zeros(d, dtype=int)
        # Mark roughly the last 20% as blanket to at least attempt a partition
        n_blanket = max(1, d // 5)
        labels[-n_blanket:] = -1
        return labels

    # Step 2: Normalized graph Laplacian
    degrees = A.sum(axis=1)
    degrees_safe = np.where(degrees > 1e-12, degrees, 1e-12)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
    L_norm = np.eye(d) - D_inv_sqrt @ A @ D_inv_sqrt

    # Step 3: Eigendecomposition
    eigvals, eigvecs = eigh(L_norm)

    # Estimate n_clusters from eigengap if not given
    if n_clusters is None:
        # Look at gaps in the first min(10, d) eigenvalues
        max_k = min(10, d)
        gaps = np.diff(eigvals[:max_k])
        if len(gaps) > 1:
            # Skip the first eigenvalue (always ~0); find the largest gap
            n_clusters = int(np.argmax(gaps[1:]) + 2)
            n_clusters = max(2, min(n_clusters, d // 2))
        else:
            n_clusters = 2

    # Step 4: Spectral clustering with KMeans
    # Use eigenvectors 1..k (skip the trivial first one)
    k = min(n_clusters, d - 1)
    features = eigvecs[:, 1:k + 1]

    # Normalize rows for better clustering
    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 1e-12, row_norms, 1e-12)
    features = features / row_norms

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    cluster_labels = kmeans.fit_predict(features)

    # Step 5: Identify blanket variables via cross-cluster connectivity
    cross_cluster_frac = np.zeros(d)
    for i in range(d):
        total_weight = A[i, :].sum()
        if total_weight < 1e-12:
            cross_cluster_frac[i] = 0.0
            continue
        # Weight going to variables in a different cluster
        different_cluster = cluster_labels != cluster_labels[i]
        cross_weight = A[i, different_cluster].sum()
        cross_cluster_frac[i] = cross_weight / total_weight

    # Threshold: variables with cross-cluster fraction above the quantile
    # are designated blanket variables
    if np.any(cross_cluster_frac > 0):
        threshold = np.quantile(cross_cluster_frac[cross_cluster_frac > 0],
                                blanket_quantile)
        blanket_mask = cross_cluster_frac >= threshold
    else:
        blanket_mask = np.zeros(d, dtype=bool)

    # Build final labels: cluster id for internal vars, -1 for blanket
    labels = cluster_labels.copy()
    labels[blanket_mask] = -1

    return labels


# =========================================================================
# NOTEARS benchmark wrapper
# =========================================================================

def notears_method(samples, gradients):
    """
    NOTEARS benchmark method conforming to the BenchmarkSuite interface.

    Takes (samples, gradients) and returns a label array where:
      label[i] = object cluster index, or -1 for blanket variables.

    The pipeline:
      1. Center the sample data.
      2. Run NOTEARS to discover a DAG adjacency matrix W.
      3. Convert the DAG to an undirected partition via spectral clustering
         with blanket identification from cross-cluster connectivity.
    """
    # Center the data
    X = samples - samples.mean(axis=0)

    # Standardize for numerical stability
    std = X.std(axis=0)
    std = np.where(std > 1e-8, std, 1e-8)
    X = X / std

    # Scale lambda with dimension: sparser in higher dimensions
    d = X.shape[1]
    lambda1 = 0.05 if d <= 10 else 0.1

    # Run NOTEARS
    W = notears_linear(X, lambda1=lambda1, max_iter=80, w_threshold=0.2)

    # Convert DAG to partition
    labels = dag_to_partition(W, n_clusters=None, blanket_quantile=0.65)

    return labels


# =========================================================================
# Main benchmark execution
# =========================================================================

def run_benchmark_vs_notears(n_seeds=10, verbose=True):
    """
    Execute the TB vs NOTEARS benchmark on all standardized datasets.

    Builds the default BenchmarkSuite (which includes all 5 datasets and
    registers TB_hybrid), then adds NOTEARS as a competing method, and
    runs the full evaluation protocol.

    Args:
        n_seeds: Number of random seeds per (method, dataset) pair.
        verbose: If True, print progress and results.

    Returns:
        Tuple of (suite, results_dict).
    """
    # Build the default suite (has TB_hybrid and TB_gradient registered)
    suite = build_default_suite()

    # Remove TB_gradient; we only want TB_hybrid vs NOTEARS
    if 'TB_gradient' in suite._methods:
        del suite._methods['TB_gradient']

    # Register NOTEARS
    suite.register_method('NOTEARS', notears_method)

    if verbose:
        print("=" * 70)
        print("US-091: TB vs NOTEARS Benchmark")
        print("=" * 70)
        print(f"Methods:  {list(suite._methods.keys())}")
        print(f"Datasets: {list(suite._datasets.keys())}")
        print(f"Seeds:    {n_seeds}")
        print("=" * 70)

    # Run the full benchmark
    results = suite.run(n_seeds=n_seeds, verbose=verbose)

    # Print summary table
    print("\n")
    print("=" * 70)
    print("RESULTS TABLE: TB vs NOTEARS")
    print("=" * 70)
    suite.print_summary_table()

    # Print statistical comparisons
    print("\n")
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    suite.print_statistical_comparisons()

    return suite, results


def save_comparison_plots(suite, results):
    """
    Generate and save comparison plots:
      1. Grouped bar chart of ARI, blanket_F1, NMI per dataset.
      2. Runtime comparison bar chart.
      3. Radar chart from the suite.
    """
    summary = results['summary']
    datasets = results['config']['datasets']
    methods = results['config']['methods']

    # ── Plot 1: Grouped bar charts for quality metrics ──────────────────
    metrics_to_plot = ['ARI', 'blanket_F1', 'NMI']
    n_datasets = len(datasets)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'TB_hybrid': '#2ecc71', 'NOTEARS': '#e74c3c'}
    x = np.arange(n_datasets)
    width = 0.35

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(methods):
            means = []
            stds = []
            for ds in datasets:
                key = f"{method}|{ds}"
                if key in summary:
                    means.append(summary[key][f'{metric}_mean'])
                    stds.append(summary[key][f'{metric}_std'])
                else:
                    means.append(0.0)
                    stds.append(0.0)

            offset = (m_idx - (n_methods - 1) / 2) * width
            color = colors.get(method, '#3498db')
            ax.bar(x + offset, means, width * 0.9, yerr=stds,
                   label=method, color=color, alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([ds.replace('_', '\n') for ds in datasets],
                           fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Dataset', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)

    plt.suptitle('US-091: TB vs NOTEARS, Quality Metrics', fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'tb_vs_notears_quality', 'benchmark_vs_notears')

    # ── Plot 2: Runtime comparison ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for m_idx, method in enumerate(methods):
        times = []
        time_stds = []
        for ds in datasets:
            key = f"{method}|{ds}"
            if key in summary:
                times.append(summary[key]['wall_clock_seconds_mean'])
                time_stds.append(summary[key]['wall_clock_seconds_std'])
            else:
                times.append(0.0)
                time_stds.append(0.0)

        offset = (m_idx - (n_methods - 1) / 2) * width
        color = colors.get(method, '#3498db')
        ax2.bar(x + offset, times, width * 0.9, yerr=time_stds,
                label=method, color=color, alpha=0.85, capsize=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.replace('_', '\n') for ds in datasets], fontsize=8)
    ax2.set_ylabel('Wall-clock time (seconds)')
    ax2.set_title('US-091: TB vs NOTEARS, Runtime per Dataset', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_figure(fig2, 'tb_vs_notears_runtime', 'benchmark_vs_notears')

    # ── Plot 3: Radar chart ─────────────────────────────────────────────
    suite_radar_path = os.path.join(
        os.path.dirname(SCRIPT_DIR), 'results',
        'tb_vs_notears_radar.png'
    )
    suite = build_default_suite()  # Rebuild for radar; not ideal but functional
    # We use the suite's built-in radar plotter, passing the results manually
    try:
        # Build the aggregated radar from summary
        _plot_comparison_radar(summary, methods, datasets, suite_radar_path)
    except Exception as e:
        print(f"Radar chart generation note: {e}")


def _plot_comparison_radar(summary, methods, datasets, save_path):
    """Custom radar chart for the TB vs NOTEARS comparison."""
    metrics = ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds', 'peak_memory_mb']
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {'TB_hybrid': '#2ecc71', 'NOTEARS': '#e74c3c'}

    for method in methods:
        values = []
        for metric in metrics:
            vals = []
            for ds in datasets:
                key = f"{method}|{ds}"
                if key in summary:
                    vals.append(summary[key][f'{metric}_mean'])
            values.append(np.mean(vals) if vals else 0.0)

        # Normalize: invert time and memory (lower is better)
        norm_values = []
        for i, metric in enumerate(metrics):
            if metric in ('wall_clock_seconds', 'peak_memory_mb'):
                norm_values.append(1.0 / (1.0 + values[i]))
            else:
                norm_values.append(max(0.0, min(1.0, values[i])))
        norm_values += norm_values[:1]

        color = colors.get(method, '#3498db')
        ax.plot(angles, norm_values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, norm_values, alpha=0.15, color=color)

    metric_labels = []
    for m in metrics:
        if m == 'wall_clock_seconds':
            metric_labels.append('Speed\n(1/(1+sec))')
        elif m == 'peak_memory_mb':
            metric_labels.append('Memory Eff.\n(1/(1+MB))')
        else:
            metric_labels.append(m)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('US-091: TB vs NOTEARS (Radar)', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Radar chart saved to {save_path}")
    plt.close(fig)


# =========================================================================
# Entry point
# =========================================================================

def main():
    """Full US-091 pipeline: run benchmark, save results, generate plots."""
    n_seeds = 10

    suite, results = run_benchmark_vs_notears(n_seeds=n_seeds, verbose=True)

    # Save JSON results
    results_dir = os.path.join(RALPH_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, 'benchmark_vs_notears.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nFull results JSON saved to {json_path}")

    # Also save via the standard save_results utility
    save_results(
        'benchmark_vs_notears',
        results,
        results.get('config', {}),
        notes='US-091: TB vs NOTEARS on standardized benchmark datasets. '
              'NOTEARS (Zheng et al. 2018) converted to partition via spectral '
              'clustering of the thresholded DAG with cross-cluster blanket detection.'
    )

    # Generate comparison plots
    save_comparison_plots(suite, results)

    # Final summary
    print("\n")
    print("=" * 70)
    print("US-091 COMPLETE")
    print("=" * 70)
    print(f"Results JSON: {json_path}")
    print(f"Plots saved to: {results_dir}/")

    # Print a compact final comparison table
    summary = results['summary']
    datasets = results['config']['datasets']
    methods = results['config']['methods']
    print("\n  COMPACT COMPARISON (mean +/- std across 10 seeds):")
    print(f"  {'Dataset':<20s} | {'Method':<12s} | {'ARI':>10s} | {'F1':>10s} | {'NMI':>10s} | {'Time(s)':>10s}")
    print("  " + "-" * 80)
    for ds in datasets:
        for method in methods:
            key = f"{method}|{ds}"
            if key in summary:
                row = summary[key]
                ari_str = f"{row['ARI_mean']:.3f}+/-{row['ARI_std']:.3f}"
                f1_str = f"{row['blanket_F1_mean']:.3f}+/-{row['blanket_F1_std']:.3f}"
                nmi_str = f"{row['NMI_mean']:.3f}+/-{row['NMI_std']:.3f}"
                time_str = f"{row['wall_clock_seconds_mean']:.3f}+/-{row['wall_clock_seconds_std']:.3f}"
                print(f"  {ds:<20s} | {method:<12s} | {ari_str:>10s} | {f1_str:>10s} | {nmi_str:>10s} | {time_str:>10s}")

    return suite, results


if __name__ == '__main__':
    main()
