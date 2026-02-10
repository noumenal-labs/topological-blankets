"""
US-022: Scaling Benchmark to 100+ Dimensions
=============================================

Systematic scaling test: 10, 20, 50, 100, 200 dimensions.
Measures ARI, F1, wall-clock time, memory usage.
Tests sparse Hessian approximation (diagonal + low-rank).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import tracemalloc
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import detect_blankets_otsu
from topological_blankets.clustering import cluster_internals
from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, compute_metrics
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# Sparse Hessian approximation
# =========================================================================

def compute_features_sparse(gradients, rank=None):
    """
    Compute features using sparse Hessian approximation.

    If rank is None: diagonal only.
    If rank > 0: diagonal + low-rank correction via truncated SVD.
    """
    n_samples, n_vars = gradients.shape

    grad_magnitude = np.mean(np.abs(gradients), axis=0)
    grad_variance = np.var(gradients, axis=0)

    if rank is None or rank == 0:
        # Diagonal-only Hessian
        H_est = np.diag(grad_variance)
    else:
        # Diagonal + low-rank: H ≈ diag(var) + U S U^T
        grad_centered = gradients - gradients.mean(axis=0)
        k = min(rank, min(n_samples, n_vars) - 2)
        if k >= 1:
            try:
                # Try dense truncated SVD (more robust than sparse for moderate dims)
                from numpy.linalg import svd
                U_full, S_full, Vt_full = svd(
                    grad_centered / np.sqrt(n_samples - 1), full_matrices=False)
                Vt = Vt_full[:k]
                S = S_full[:k]
                H_est = np.diag(grad_variance) + Vt.T @ np.diag(S ** 2) @ Vt
            except Exception:
                H_est = np.diag(grad_variance)
        else:
            H_est = np.diag(grad_variance)

    # Coupling matrix
    D = np.sqrt(np.abs(np.diag(H_est))) + 1e-8
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': H_est,
        'coupling': coupling,
    }


# =========================================================================
# Scaling experiment
# =========================================================================

def run_scaling_at_dim(total_dim, n_objects, vars_per_blanket=3, n_trials=5):
    """Run TB pipeline at a given total dimension."""
    vars_per_object = (total_dim - vars_per_blanket) // n_objects

    if vars_per_object < 2:
        return None

    results = {'full': [], 'diag': [], 'rank5': [], 'rank10': []}
    times = {'full': [], 'diag': [], 'rank5': [], 'rank10': []}
    memory_peaks = {'full': [], 'diag': [], 'rank5': [], 'rank10': []}

    for trial in range(n_trials):
        cfg = QuadraticEBMConfig(
            n_objects=n_objects,
            vars_per_object=vars_per_object,
            vars_per_blanket=vars_per_blanket,
            intra_strength=6.0,
            blanket_strength=0.8,
        )

        actual_dim = n_objects * vars_per_object + vars_per_blanket
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)

        np.random.seed(42 + trial)
        n_samples = max(3000, actual_dim * 30)
        samples, gradients = langevin_sampling(
            Theta, n_samples=n_samples, n_steps=30,
            step_size=0.005, temp=0.1
        )

        # Full Hessian
        tracemalloc.start()
        t0 = time.time()
        tc_result = tb_pipeline(gradients, n_objects=n_objects, method='gradient')
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        m = compute_metrics(tc_result, truth)
        results['full'].append(m)
        times['full'].append(elapsed)
        memory_peaks['full'].append(peak / (1024 * 1024))  # MB

        # Sparse approximations
        for label, rank in [('diag', None), ('rank5', 5), ('rank10', 10)]:
            tracemalloc.start()
            t0 = time.time()
            features = compute_features_sparse(gradients, rank=rank if label != 'diag' else 0)
            is_blanket, _ = detect_blankets_otsu(features)
            assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)
            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            m = compute_metrics({'assignment': assignment, 'is_blanket': is_blanket}, truth)
            results[label].append(m)
            times[label].append(elapsed)
            memory_peaks[label].append(peak / (1024 * 1024))

    # Aggregate
    summary = {}
    for label in results:
        aris = [r['object_ari'] for r in results[label]]
        f1s = [r['blanket_f1'] for r in results[label]]
        summary[label] = {
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'mean_time_s': float(np.mean(times[label])),
            'mean_memory_mb': float(np.mean(memory_peaks[label])),
            'per_trial': results[label],
        }

    return summary


def run_scaling_benchmark():
    """Run the full scaling benchmark."""
    print("=" * 70)
    print("US-022: Scaling Benchmark to 100+ Dimensions")
    print("=" * 70)

    dimensions = [10, 20, 50, 100, 200]
    n_objects = 2
    n_trials = 5

    all_metrics = {}

    for total_dim in dimensions:
        print(f"\n--- Dimension: {total_dim} ---")
        summary = run_scaling_at_dim(total_dim, n_objects, n_trials=n_trials)

        if summary is None:
            print(f"  Skipped (too few vars for {n_objects} objects)")
            continue

        all_metrics[str(total_dim)] = summary

        for label in ['full', 'diag', 'rank5', 'rank10']:
            s = summary[label]
            print(f"  {label:8s}: ARI={s['mean_ari']:.3f}+/-{s['std_ari']:.3f}, "
                  f"F1={s['mean_f1']:.3f}, time={s['mean_time_s']:.2f}s, "
                  f"mem={s['mean_memory_mb']:.1f}MB")

    config = {
        'dimensions': dimensions,
        'n_objects': n_objects,
        'n_trials': n_trials,
        'methods': ['full', 'diag', 'rank5', 'rank10'],
    }

    save_results('scaling_benchmark', all_metrics, config,
                 notes='US-022: Scaling to 200D. Full vs sparse Hessian comparison.')

    _plot_scaling_curves(all_metrics, dimensions)

    print("\nUS-022 complete.")
    return all_metrics


def _plot_scaling_curves(all_metrics, dimensions):
    """Plot ARI and time vs dimension for each method."""
    dims = [d for d in dimensions if str(d) in all_metrics]
    methods = ['full', 'diag', 'rank5', 'rank10']
    colors = {'full': '#2ecc71', 'diag': '#e74c3c', 'rank5': '#3498db', 'rank10': '#f39c12'}
    labels = {'full': 'Full Hessian', 'diag': 'Diagonal only',
              'rank5': 'Diag + rank-5', 'rank10': 'Diag + rank-10'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ARI vs dim
    ax = axes[0]
    for m in methods:
        aris = [all_metrics[str(d)][m]['mean_ari'] for d in dims]
        stds = [all_metrics[str(d)][m]['std_ari'] for d in dims]
        ax.errorbar(dims, aris, yerr=stds, label=labels[m], color=colors[m],
                    marker='o', capsize=3, linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('ARI')
    ax.set_title('Object Recovery vs Dimension')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Time vs dim
    ax = axes[1]
    for m in methods:
        ts = [all_metrics[str(d)][m]['mean_time_s'] for d in dims]
        ax.plot(dims, ts, label=labels[m], color=colors[m], marker='s', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Wall-clock Time (s)')
    ax.set_title('Runtime vs Dimension')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Memory vs dim
    ax = axes[2]
    for m in methods:
        mems = [all_metrics[str(d)][m]['mean_memory_mb'] for d in dims]
        ax.plot(dims, mems, label=labels[m], color=colors[m], marker='D', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage vs Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'scaling_curves', 'scaling_benchmark')


if __name__ == '__main__':
    run_scaling_benchmark()
