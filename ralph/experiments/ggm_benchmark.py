"""
US-018: Gaussian Graphical Model Benchmark
==========================================

Compares TB graph recovery to graphical lasso on sparse precision matrices.
Graph types: chain, grid, random sparse (Erdos-Renyi), scale-free (Barabasi-Albert).
Each tested at 3 sparsity/density levels.

Metrics: Structural Hamming Distance (SHD), edge F1, precision, recall.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets.features import compute_geometric_features
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# Graph generators
# =========================================================================

def make_chain_graph(p, strength=0.5):
    """Chain (tridiagonal) precision matrix."""
    Theta = np.eye(p)
    for i in range(p - 1):
        Theta[i, i + 1] = Theta[i + 1, i] = strength
    return Theta


def make_grid_graph(side, strength=0.4):
    """2D grid precision matrix."""
    p = side * side
    Theta = np.eye(p)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if j + 1 < side:
                Theta[idx, idx + 1] = Theta[idx + 1, idx] = strength
            if i + 1 < side:
                Theta[idx, idx + side] = Theta[idx + side, idx] = strength
    return Theta


def make_random_sparse_graph(p, edge_prob=0.2, strength=0.3):
    """Erdos-Renyi random sparse precision matrix."""
    Theta = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < edge_prob:
                val = strength * (0.5 + np.random.rand())
                Theta[i, j] = Theta[j, i] = val
    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals[0] < 0.1:
        Theta += (0.1 - eigvals[0] + 0.1) * np.eye(p)
    return Theta


def make_scale_free_graph(p, m=2, strength=0.4):
    """Barabasi-Albert scale-free precision matrix via preferential attachment."""
    Theta = np.eye(p)
    degrees = np.zeros(p)

    # Start with a small clique of size m+1
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            Theta[i, j] = Theta[j, i] = strength
            degrees[i] += 1
            degrees[j] += 1

    # Preferential attachment for remaining nodes
    for new_node in range(m + 1, p):
        probs = degrees[:new_node] / (degrees[:new_node].sum() + 1e-10)
        targets = np.random.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            val = strength * (0.5 + np.random.rand())
            Theta[new_node, t] = Theta[t, new_node] = val
            degrees[new_node] += 1
            degrees[t] += 1

    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals[0] < 0.1:
        Theta += (0.1 - eigvals[0] + 0.1) * np.eye(p)
    return Theta


# =========================================================================
# Edge recovery metrics
# =========================================================================

def extract_edges(matrix, threshold=None):
    """Extract edge set from a matrix (above threshold or nonzero off-diagonal)."""
    p = matrix.shape[0]
    A = np.abs(matrix.copy())
    np.fill_diagonal(A, 0)

    if threshold is None:
        # Adaptive: Otsu on nonzero values
        vals = A[np.triu_indices(p, k=1)]
        nonzero = vals[vals > 1e-10]
        if len(nonzero) > 0:
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(nonzero)
            except ValueError:
                threshold = np.median(nonzero)
        else:
            threshold = 0.0

    edges = set()
    for i in range(p):
        for j in range(i + 1, p):
            if A[i, j] > threshold:
                edges.add((i, j))
    return edges, threshold


def compute_graph_metrics(true_edges, pred_edges, p):
    """Compute SHD, edge F1, precision, recall."""
    total_possible = p * (p - 1) // 2

    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    tn = total_possible - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd = fp + fn

    return {
        'shd': int(shd),
        'edge_f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'true_edges': len(true_edges),
        'pred_edges': len(pred_edges),
    }


# =========================================================================
# Main experiment
# =========================================================================

def run_ggm_benchmark():
    """Run the full GGM benchmark."""
    print("=" * 70)
    print("US-018: Gaussian Graphical Model Benchmark")
    print("=" * 70)

    p = 16  # dimension
    n_samples = 5000
    n_trials = 3

    # Graph configs: (name, generator_fn, density_params)
    # density_params are strengths/probs that roughly give 10%, 20%, 40% edge density
    graph_configs = [
        ('chain', lambda s: make_chain_graph(p, strength=s), [0.3, 0.5, 0.8]),
        ('grid', lambda s: make_grid_graph(4, strength=s), [0.3, 0.5, 0.8]),
        ('random_sparse', lambda ep: make_random_sparse_graph(p, edge_prob=ep), [0.10, 0.20, 0.40]),
        ('scale_free', lambda s: make_scale_free_graph(p, m=2, strength=s), [0.3, 0.5, 0.8]),
    ]

    all_metrics = {}

    for graph_name, gen_fn, params in graph_configs:
        print(f"\n--- Graph type: {graph_name} ---")
        all_metrics[graph_name] = {}

        for param_idx, param in enumerate(params):
            density_label = ['low', 'medium', 'high'][param_idx]
            key = f"{density_label}"
            print(f"  Density level: {density_label} (param={param})")

            tb_trials = []
            glasso_trials = []

            for trial in range(n_trials):
                np.random.seed(42 + trial + param_idx * 100)

                # Generate precision matrix and sample from N(0, Theta^{-1})
                Theta = gen_fn(param)
                Sigma = np.linalg.inv(Theta)
                samples = np.random.multivariate_normal(np.zeros(p), Sigma, size=n_samples)

                # True edges
                true_edges, _ = extract_edges(Theta, threshold=1e-6)

                # TB approach: compute gradients, estimate Hessian, threshold
                # For Gaussian, grad E(x) = Theta @ x, so gradients = samples @ Theta.T
                # But to test the full pipeline, use finite-difference-like approach:
                # gradients ≈ samples @ Theta (since E = 0.5 x^T Theta x, grad = Theta x)
                gradients = samples @ Theta
                features = compute_geometric_features(gradients)
                H_est = features['hessian_est']

                tb_edges, tb_thresh = extract_edges(H_est)
                tb_m = compute_graph_metrics(true_edges, tb_edges, p)
                tb_trials.append(tb_m)

                # Graphical Lasso
                from sklearn.covariance import GraphicalLassoCV
                try:
                    gl = GraphicalLassoCV(cv=3, max_iter=500)
                    gl.fit(samples)
                    gl_precision = gl.precision_
                    gl_edges, _ = extract_edges(gl_precision, threshold=1e-4)
                    gl_m = compute_graph_metrics(true_edges, gl_edges, p)
                except Exception:
                    gl_m = {'shd': p * (p - 1) // 2, 'edge_f1': 0.0,
                            'precision': 0.0, 'recall': 0.0,
                            'tp': 0, 'fp': 0, 'fn': len(true_edges),
                            'true_edges': len(true_edges), 'pred_edges': 0}
                glasso_trials.append(gl_m)

            # Average metrics
            for method_name, trials in [('tb', tb_trials), ('glasso', glasso_trials)]:
                avg = {}
                for metric_key in ['shd', 'edge_f1', 'precision', 'recall']:
                    vals = [t[metric_key] for t in trials]
                    avg[f'mean_{metric_key}'] = float(np.mean(vals))
                    avg[f'std_{metric_key}'] = float(np.std(vals))
                avg['per_trial'] = trials

                if method_name not in all_metrics[graph_name]:
                    all_metrics[graph_name][method_name] = {}
                all_metrics[graph_name][method_name][key] = avg

                print(f"    {method_name:8s}: F1={avg['mean_edge_f1']:.3f}, "
                      f"SHD={avg['mean_shd']:.1f}, "
                      f"Prec={avg['mean_precision']:.3f}, Rec={avg['mean_recall']:.3f}")

    config = {
        'dimension': p,
        'n_samples': n_samples,
        'n_trials': n_trials,
        'graph_types': [g[0] for g in graph_configs],
    }

    save_results('ggm_benchmark', all_metrics, config,
                 notes='US-018: GGM benchmark. TB vs graphical lasso on 4 graph types x 3 density levels.')

    _plot_ggm_comparison(all_metrics)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: TB vs Graphical Lasso (Edge F1)")
    print(f"{'Graph':<16s} {'Density':<10s} {'TB F1':>8s} {'GLasso F1':>10s} {'Winner':>8s}")
    print("-" * 55)
    tb_wins = 0
    total = 0
    for graph_name in all_metrics:
        for density in all_metrics[graph_name].get('tb', {}):
            tb_f1 = all_metrics[graph_name]['tb'][density]['mean_edge_f1']
            gl_f1 = all_metrics[graph_name]['glasso'][density]['mean_edge_f1']
            winner = 'TB' if tb_f1 > gl_f1 else 'GLasso' if gl_f1 > tb_f1 else 'Tie'
            if winner == 'TB':
                tb_wins += 1
            total += 1
            print(f"{graph_name:<16s} {density:<10s} {tb_f1:>8.3f} {gl_f1:>10.3f} {winner:>8s}")

    print(f"\nTB wins {tb_wins}/{total} comparisons")
    print("\nUS-018 complete.")
    return all_metrics


def _plot_ggm_comparison(all_metrics):
    """Bar chart comparing TB vs glasso across graph types."""
    graph_names = list(all_metrics.keys())
    densities = ['low', 'medium', 'high']

    fig, axes = plt.subplots(1, len(graph_names), figsize=(4 * len(graph_names), 5))
    if len(graph_names) == 1:
        axes = [axes]

    for ax, graph_name in zip(axes, graph_names):
        tb_f1s = []
        gl_f1s = []
        for d in densities:
            tb_f1s.append(all_metrics[graph_name]['tb'].get(d, {}).get('mean_edge_f1', 0))
            gl_f1s.append(all_metrics[graph_name]['glasso'].get(d, {}).get('mean_edge_f1', 0))

        x = np.arange(len(densities))
        width = 0.35
        ax.bar(x - width / 2, tb_f1s, width, label='TB', color='#2ecc71', alpha=0.8)
        ax.bar(x + width / 2, gl_f1s, width, label='GLasso', color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(densities)
        ax.set_ylabel('Edge F1')
        ax.set_title(graph_name, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, 'ggm_comparison', 'ggm_benchmark')


if __name__ == '__main__':
    run_ggm_benchmark()
