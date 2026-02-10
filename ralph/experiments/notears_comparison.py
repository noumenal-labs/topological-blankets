"""
US-036: Comparative Analysis with NOTEARS
==========================================

Compare TB with NOTEARS (DAG structure learning with continuous optimization).
Apply to GGM data from US-018 and LunarLander trajectory data from US-025.
Compare SHD, F1, and runtime across TB, NOTEARS, and graphical lasso.

NOTEARS minimizes: ||X - XW||^2 + lambda*||W||_1
subject to: h(W) = trace(e^{W circ W}) - d = 0 (DAG constraint)

Reference: Zheng et al. (2018) "DAGs with NO TEARS"
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from scipy.linalg import expm

NOUMENAL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, NOUMENAL_DIR)

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']


# =========================================================================
# NOTEARS implementation (Zheng et al., 2018)
# =========================================================================

def notears_linear(X, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16):
    """
    Solve min ||X - XW||^2 + lambda1 * ||W||_1
    s.t. h(W) = trace(e^{W circ W}) - d = 0

    Augmented Lagrangian approach.

    Args:
        X: (n, d) data matrix
        lambda1: L1 penalty
        loss_type: 'l2' (least squares)
        max_iter: max outer iterations
        h_tol: tolerance for acyclicity constraint
        rho_max: max penalty parameter

    Returns:
        W_est: (d, d) estimated weighted adjacency matrix
    """
    n, d = X.shape

    def _loss(W):
        """Least squares loss."""
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        return loss, G_loss

    def _h(W):
        """Acyclicity constraint: trace(e^{W circ W}) - d."""
        E = expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * 2 * W
        return h, G_h

    def _adj(w):
        """Unflatten and zero diagonal."""
        W = w.reshape(d, d)
        np.fill_diagonal(W, 0)
        return W

    def _func(w, alpha, rho):
        """Augmented Lagrangian objective."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(W).sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = G_smooth + lambda1 * np.sign(W)
        np.fill_diagonal(g_obj, 0)
        return obj, g_obj.ravel()

    # Initialize
    w_est = np.zeros(d * d)
    alpha, rho = 0.0, 1.0
    h_prev = np.inf

    for it in range(max_iter):
        # Solve subproblem
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
    W_est[np.abs(W_est) < 0.3] = 0  # threshold small values
    return W_est


# =========================================================================
# Graph comparison metrics
# =========================================================================

def structural_hamming_distance(W_true, W_est):
    """Compute SHD between two adjacency matrices."""
    # Convert to binary
    B_true = (np.abs(W_true) > 0).astype(int)
    B_est = (np.abs(W_est) > 0).astype(int)
    np.fill_diagonal(B_true, 0)
    np.fill_diagonal(B_est, 0)
    return int(np.sum(B_true != B_est))


def edge_metrics(W_true, W_est):
    """Compute precision, recall, F1 for edge recovery."""
    B_true = (np.abs(W_true) > 0).astype(int)
    B_est = (np.abs(W_est) > 0).astype(int)
    np.fill_diagonal(B_true, 0)
    np.fill_diagonal(B_est, 0)

    tp = np.sum((B_true == 1) & (B_est == 1))
    fp = np.sum((B_true == 0) & (B_est == 1))
    fn = np.sum((B_true == 1) & (B_est == 0))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return float(precision), float(recall), float(f1)


# =========================================================================
# GGM data generation (from US-018)
# =========================================================================

def generate_ggm_data(graph_type='chain', d=10, n_samples=2000, seed=42):
    """Generate data from a Gaussian Graphical Model."""
    rng = np.random.RandomState(seed)

    if graph_type == 'chain':
        W = np.zeros((d, d))
        for i in range(d - 1):
            w = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
            W[i, i+1] = w
    elif graph_type == 'grid':
        side = int(np.ceil(np.sqrt(d)))
        W = np.zeros((d, d))
        for i in range(d):
            r, c = i // side, i % side
            if c + 1 < side and i + 1 < d:
                W[i, i+1] = rng.uniform(0.3, 0.8)
            if r + 1 < side and i + side < d:
                W[i, i+side] = rng.uniform(0.3, 0.8)
    elif graph_type == 'random_sparse':
        W = np.zeros((d, d))
        n_edges = int(0.15 * d * (d - 1))
        edges_added = 0
        while edges_added < n_edges:
            i, j = rng.randint(0, d, 2)
            if i != j and W[i, j] == 0:
                W[i, j] = rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
                edges_added += 1
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Make precision matrix PD
    Theta = W + W.T
    Theta = Theta + (np.abs(np.min(np.linalg.eigvalsh(Theta))) + 1.0) * np.eye(d)
    Sigma = np.linalg.inv(Theta)

    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n_samples)
    return X, W, Theta


# =========================================================================
# TB graph extraction
# =========================================================================

def tb_extract_graph(X, n_objects=None):
    """Apply TB to data and extract adjacency from coupling matrix."""
    from topological_blankets.features import compute_geometric_features

    # Compute gradients (for GGM, gradient of log-density is -Theta @ x)
    # We approximate via sample-level score
    mean = X.mean(axis=0)
    gradients = -(X - mean)  # approximate score for Gaussian

    features = compute_geometric_features(gradients)
    coupling = features['coupling']

    # Threshold coupling matrix to get adjacency
    coupling_abs = np.abs(coupling)
    np.fill_diagonal(coupling_abs, 0)

    # Adaptive threshold: use Otsu on nonzero values
    nonzero = coupling_abs[coupling_abs > 0].ravel()
    if len(nonzero) == 0:
        return np.zeros_like(coupling), coupling

    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(nonzero)
    except ValueError:
        thresh = np.median(nonzero)

    W_tb = coupling_abs.copy()
    W_tb[W_tb < thresh] = 0
    return W_tb, coupling


def glasso_extract_graph(X):
    """Apply graphical lasso and extract adjacency."""
    from sklearn.covariance import GraphicalLassoCV
    try:
        model = GraphicalLassoCV(cv=3, max_iter=500)
        model.fit(X)
        precision = model.precision_
        W_glasso = np.abs(precision.copy())
        np.fill_diagonal(W_glasso, 0)
        # Threshold small values
        thresh = 0.01 * np.max(W_glasso)
        W_glasso[W_glasso < thresh] = 0
        return W_glasso
    except Exception as e:
        print(f"  GraphicalLasso failed: {e}")
        return np.zeros((X.shape[1], X.shape[1]))


# =========================================================================
# Experiments
# =========================================================================

def run_ggm_comparison(graph_types=None, d=10, n_samples=2000):
    """Compare TB, NOTEARS, and glasso on GGM data."""
    if graph_types is None:
        graph_types = ['chain', 'random_sparse']

    print("\n--- GGM Comparison ---")
    results = []

    for graph_type in graph_types:
        print(f"\n  Graph: {graph_type}, d={d}")
        X, W_true, Theta = generate_ggm_data(graph_type, d, n_samples)

        # True adjacency (undirected)
        W_true_sym = (np.abs(W_true) + np.abs(W_true.T)) > 0

        # TB
        t0 = time.time()
        W_tb, coupling = tb_extract_graph(X)
        tb_time = time.time() - t0
        tb_shd = structural_hamming_distance(W_true_sym, W_tb > 0)
        tb_p, tb_r, tb_f1 = edge_metrics(W_true_sym, W_tb > 0)
        print(f"  TB:      SHD={tb_shd}, F1={tb_f1:.3f}, time={tb_time:.2f}s")

        # NOTEARS
        t0 = time.time()
        W_notears = notears_linear(X, lambda1=0.1)
        notears_time = time.time() - t0
        W_notears_sym = (np.abs(W_notears) + np.abs(W_notears.T)) > 0
        nt_shd = structural_hamming_distance(W_true_sym, W_notears_sym)
        nt_p, nt_r, nt_f1 = edge_metrics(W_true_sym, W_notears_sym)
        print(f"  NOTEARS: SHD={nt_shd}, F1={nt_f1:.3f}, time={notears_time:.2f}s")

        # Graphical Lasso
        t0 = time.time()
        W_glasso = glasso_extract_graph(X)
        glasso_time = time.time() - t0
        gl_shd = structural_hamming_distance(W_true_sym, W_glasso > 0)
        gl_p, gl_r, gl_f1 = edge_metrics(W_true_sym, W_glasso > 0)
        print(f"  GLasso:  SHD={gl_shd}, F1={gl_f1:.3f}, time={glasso_time:.2f}s")

        results.append({
            'graph_type': graph_type,
            'd': d,
            'n_samples': n_samples,
            'tb': {'shd': tb_shd, 'precision': tb_p, 'recall': tb_r, 'f1': tb_f1, 'time': tb_time},
            'notears': {'shd': nt_shd, 'precision': nt_p, 'recall': nt_r, 'f1': nt_f1, 'time': notears_time},
            'glasso': {'shd': gl_shd, 'precision': gl_p, 'recall': gl_r, 'f1': gl_f1, 'time': glasso_time},
        })

    return results


def run_lunarlander_comparison():
    """Compare TB, NOTEARS, and glasso on LunarLander trajectory data."""
    print("\n--- LunarLander Trajectory Comparison ---")

    data_dir = os.path.join(NOUMENAL_DIR, 'results', 'trajectory_data')
    states = np.load(os.path.join(data_dir, 'states.npy'))

    # Subsample for speed (NOTEARS can be slow on large datasets)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(states), min(2000, len(states)), replace=False)
    X = states[idx]

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # TB
    t0 = time.time()
    W_tb, coupling_tb = tb_extract_graph(X)
    tb_time = time.time() - t0

    # NOTEARS
    t0 = time.time()
    W_notears = notears_linear(X, lambda1=0.05)
    notears_time = time.time() - t0

    # GLasso
    t0 = time.time()
    W_glasso = glasso_extract_graph(X)
    glasso_time = time.time() - t0

    # Compare graphs
    print(f"  TB edges:      {int(np.sum(np.abs(W_tb) > 0))}, time={tb_time:.2f}s")
    print(f"  NOTEARS edges: {int(np.sum(np.abs(W_notears) > 0))}, time={notears_time:.2f}s")
    print(f"  GLasso edges:  {int(np.sum(np.abs(W_glasso) > 0))}, time={glasso_time:.2f}s")

    # Mutual agreement between methods
    tb_bin = (np.abs(W_tb) > 0).astype(int)
    nt_bin = ((np.abs(W_notears) + np.abs(W_notears.T)) > 0).astype(int)
    gl_bin = (np.abs(W_glasso) > 0).astype(int)
    np.fill_diagonal(tb_bin, 0)
    np.fill_diagonal(nt_bin, 0)
    np.fill_diagonal(gl_bin, 0)

    tb_nt_agree = np.sum(tb_bin == nt_bin) / tb_bin.size
    tb_gl_agree = np.sum(tb_bin == gl_bin) / tb_bin.size
    nt_gl_agree = np.sum(nt_bin == gl_bin) / nt_bin.size

    print(f"  Agreement: TB-NOTEARS={tb_nt_agree:.3f}, TB-GLasso={tb_gl_agree:.3f}, NOTEARS-GLasso={nt_gl_agree:.3f}")

    return {
        'n_samples': len(X),
        'state_labels': STATE_LABELS,
        'tb': {
            'n_edges': int(np.sum(np.abs(W_tb) > 0)),
            'coupling_matrix': coupling_tb.tolist(),
            'adjacency': (np.abs(W_tb) > 0).astype(int).tolist(),
            'time': tb_time,
        },
        'notears': {
            'n_edges': int(np.sum(np.abs(W_notears) > 0)),
            'adjacency': W_notears.tolist(),
            'time': notears_time,
        },
        'glasso': {
            'n_edges': int(np.sum(np.abs(W_glasso) > 0)),
            'adjacency': (np.abs(W_glasso) > 0).astype(int).tolist(),
            'time': glasso_time,
        },
        'agreement': {
            'tb_notears': tb_nt_agree,
            'tb_glasso': tb_gl_agree,
            'notears_glasso': nt_gl_agree,
        },
    }


def run_runtime_comparison():
    """Runtime comparison at various dimensions."""
    print("\n--- Runtime Comparison ---")
    dims = [8, 10, 15, 20]
    results = []

    for d in dims:
        X, W_true, _ = generate_ggm_data('chain', d, 2000)

        t0 = time.time()
        tb_extract_graph(X)
        tb_time = time.time() - t0

        t0 = time.time()
        notears_linear(X, lambda1=0.1)
        notears_time = time.time() - t0

        t0 = time.time()
        glasso_extract_graph(X)
        glasso_time = time.time() - t0

        print(f"  d={d:3d}: TB={tb_time:.3f}s, NOTEARS={notears_time:.3f}s, GLasso={glasso_time:.3f}s")
        results.append({
            'd': d,
            'tb_time': tb_time,
            'notears_time': notears_time,
            'glasso_time': glasso_time,
        })

    return results


# =========================================================================
# Visualization
# =========================================================================

def plot_ggm_comparison(ggm_results):
    """Bar chart comparing methods on GGM data."""
    n_graphs = len(ggm_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # F1 comparison
    ax = axes[0]
    x = np.arange(n_graphs)
    width = 0.25
    labels = [r['graph_type'] for r in ggm_results]

    tb_f1 = [r['tb']['f1'] for r in ggm_results]
    nt_f1 = [r['notears']['f1'] for r in ggm_results]
    gl_f1 = [r['glasso']['f1'] for r in ggm_results]

    ax.bar(x - width, tb_f1, width, label='TB', color='#3498db')
    ax.bar(x, nt_f1, width, label='NOTEARS', color='#e74c3c')
    ax.bar(x + width, gl_f1, width, label='GLasso', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Edge F1')
    ax.set_title('Structure Recovery: F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SHD comparison
    ax = axes[1]
    tb_shd = [r['tb']['shd'] for r in ggm_results]
    nt_shd = [r['notears']['shd'] for r in ggm_results]
    gl_shd = [r['glasso']['shd'] for r in ggm_results]

    ax.bar(x - width, tb_shd, width, label='TB', color='#3498db')
    ax.bar(x, nt_shd, width, label='NOTEARS', color='#e74c3c')
    ax.bar(x + width, gl_shd, width, label='GLasso', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('SHD (lower is better)')
    ax.set_title('Structure Recovery: SHD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lunarlander_graphs(ll_results):
    """Side-by-side discovered graphs for LunarLander."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (method, key) in zip(axes, [('TB', 'tb'), ('NOTEARS', 'notears'), ('GLasso', 'glasso')]):
        adj = np.array(ll_results[key]['adjacency'])
        adj_abs = np.abs(adj)
        np.fill_diagonal(adj_abs, 0)

        im = ax.imshow(adj_abs, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(8))
        ax.set_xticklabels(STATE_LABELS, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(8))
        ax.set_yticklabels(STATE_LABELS, fontsize=8)
        n_edges = int(np.sum(adj_abs > 0))
        ax.set_title(f'{method} ({n_edges} edges, {ll_results[key]["time"]:.2f}s)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig


def plot_runtime(runtime_results):
    """Runtime vs dimension."""
    fig, ax = plt.subplots(figsize=(8, 5))
    dims = [r['d'] for r in runtime_results]

    ax.plot(dims, [r['tb_time'] for r in runtime_results], 'o-', label='TB', color='#3498db', linewidth=2)
    ax.plot(dims, [r['notears_time'] for r in runtime_results], 's-', label='NOTEARS', color='#e74c3c', linewidth=2)
    ax.plot(dims, [r['glasso_time'] for r in runtime_results], '^-', label='GLasso', color='#2ecc71', linewidth=2)

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Time (s)')
    ax.set_title('Runtime vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =========================================================================
# Main
# =========================================================================

def run_us036():
    """US-036: NOTEARS comparison."""
    print("=" * 70)
    print("US-036: Comparative Analysis with NOTEARS")
    print("=" * 70)

    # GGM comparison
    ggm_results = run_ggm_comparison(graph_types=['chain', 'random_sparse'], d=10)

    fig_ggm = plot_ggm_comparison(ggm_results)
    save_figure(fig_ggm, 'notears_ggm_comparison', 'notears_comparison')

    # LunarLander comparison
    ll_results = run_lunarlander_comparison()

    fig_ll = plot_lunarlander_graphs(ll_results)
    save_figure(fig_ll, 'notears_lunarlander_comparison', 'notears_comparison')

    # Runtime comparison
    runtime_results = run_runtime_comparison()

    fig_runtime = plot_runtime(runtime_results)
    save_figure(fig_runtime, 'notears_runtime_comparison', 'notears_comparison')

    # Save results
    all_results = {
        'ggm_comparison': ggm_results,
        'lunarlander_comparison': ll_results,
        'runtime_comparison': runtime_results,
    }

    save_results('notears_comparison', all_results, {},
                 notes='US-036: TB vs NOTEARS vs graphical lasso on GGM and LunarLander data. '
                       'NOTEARS implemented from Zheng et al. (2018).')

    print("\nUS-036 complete.")
    return all_results


if __name__ == '__main__':
    run_us036()
