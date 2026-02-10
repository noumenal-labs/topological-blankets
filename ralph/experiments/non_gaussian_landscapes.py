"""
US-021: Non-Gaussian Energy Landscapes
=======================================

Test TB on multi-modal energy functions where the quadratic assumption breaks:
- Double well: E(x) = (x^2 - 1)^2 extended to nD
- Mexican hat: E(x,y) = (x^2 + y^2 - 1)^2
- Mixture of Gaussians EBM: E(x) = -log sum_k pi_k N(x; mu_k, Sigma_k)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# Energy functions and Langevin sampling
# =========================================================================

def langevin_sample(energy_grad_fn, dim, n_samples=3000, n_steps=50,
                    step_size=0.01, temp=0.1, x0_range=2.0):
    """General Langevin dynamics sampler."""
    samples = []
    gradients = []
    x = np.random.randn(dim) * x0_range

    for _ in range(n_samples + 500):  # burn-in 500
        grad = energy_grad_fn(x)
        noise = np.random.randn(dim) * np.sqrt(2 * step_size * temp)
        x = x - step_size * grad + noise
        if len(samples) >= 500 or _ >= 500:
            pass
        if _ >= 500:
            samples.append(x.copy())
            gradients.append(grad.copy())

    return np.array(samples[:n_samples]), np.array(gradients[:n_samples])


# --- Double Well ---

def double_well_energy(x):
    """E(x) = sum_i (x_i^2 - 1)^2. Two minima at x_i = +/-1."""
    return np.sum((x ** 2 - 1) ** 2)


def double_well_grad(x):
    """Gradient of double well: 4*x_i*(x_i^2 - 1)."""
    return 4 * x * (x ** 2 - 1)


# --- Mexican Hat ---

def mexican_hat_energy(x):
    """E(x) = (|x|^2 - 1)^2. Ring minimum at |x| = 1."""
    r2 = np.sum(x ** 2)
    return (r2 - 1) ** 2


def mexican_hat_grad(x):
    """Gradient: 4*x*(|x|^2 - 1)."""
    r2 = np.sum(x ** 2)
    return 4 * x * (r2 - 1)


# --- Mixture of Gaussians EBM ---

def make_gmm_energy(means, covs, weights=None):
    """
    Create GMM energy function: E(x) = -log sum_k pi_k N(x; mu_k, Sigma_k).
    """
    K = len(means)
    if weights is None:
        weights = np.ones(K) / K

    def energy(x):
        log_probs = []
        for k in range(K):
            lp = np.log(weights[k] + 1e-30) + multivariate_normal.logpdf(x, means[k], covs[k])
            log_probs.append(lp)
        return -np.logaddexp.reduce(log_probs)

    def grad(x):
        # Numerically: finite differences
        dim = len(x)
        g = np.zeros(dim)
        eps = 1e-5
        e0 = energy(x)
        for d in range(dim):
            x_p = x.copy()
            x_p[d] += eps
            g[d] = (energy(x_p) - e0) / eps
        return g

    return energy, grad


# =========================================================================
# Run experiments
# =========================================================================

def run_double_well(dim=6):
    """Double well in nD. Expected: 2 objects (+ and - basins) + blanket at x=0."""
    print(f"\n--- Double Well ({dim}D) ---")

    samples, gradients = langevin_sample(double_well_grad, dim,
                                          n_samples=5000, step_size=0.005, temp=0.3)

    result = tb_pipeline(gradients, n_objects=2, method='gradient')
    assignment = result['assignment']
    is_blanket = result['is_blanket']

    n_obj0 = int(np.sum(assignment == 0))
    n_obj1 = int(np.sum(assignment == 1))
    n_blanket = int(np.sum(is_blanket))

    print(f"  Objects: {n_obj0} + {n_obj1} vars, Blanket: {n_blanket} vars")
    print(f"  Assignment: {assignment}")

    # For double well, all dimensions are equivalent, so TB can't distinguish
    # objects by dimension. The structure is in sample space, not variable space.
    # This tests whether TB gracefully handles this case.

    return {
        'dim': dim,
        'n_objects_found': len(set(assignment[assignment >= 0])),
        'n_blanket': n_blanket,
        'assignment': assignment.tolist(),
        'is_blanket': is_blanket.tolist(),
        'hessian_est': result['features']['hessian_est'].tolist(),
    }


def run_mexican_hat(dim=4):
    """Mexican hat. Expected: circular blanket structure."""
    print(f"\n--- Mexican Hat ({dim}D) ---")

    samples, gradients = langevin_sample(mexican_hat_grad, dim,
                                          n_samples=5000, step_size=0.003, temp=0.2)

    result = tb_pipeline(gradients, n_objects=2, method='gradient')
    assignment = result['assignment']
    is_blanket = result['is_blanket']

    n_blanket = int(np.sum(is_blanket))
    print(f"  Objects: assignment={assignment}")
    print(f"  Blanket: {n_blanket} vars")

    return {
        'dim': dim,
        'n_blanket': n_blanket,
        'assignment': assignment.tolist(),
        'is_blanket': is_blanket.tolist(),
        'hessian_est': result['features']['hessian_est'].tolist(),
    }


def run_gmm(dim=6, K=3, separation='well_separated'):
    """Mixture of Gaussians EBM."""
    print(f"\n--- GMM ({dim}D, K={K}, {separation}) ---")

    if separation == 'well_separated':
        spread = 4.0
    else:
        spread = 1.5

    # Generate means along random directions
    np.random.seed(42)
    means = []
    for k in range(K):
        angle = 2 * np.pi * k / K
        mu = np.zeros(dim)
        mu[0] = spread * np.cos(angle)
        mu[1] = spread * np.sin(angle)
        means.append(mu)

    covs = [np.eye(dim) * 0.5 for _ in range(K)]
    energy_fn, grad_fn = make_gmm_energy(means, covs)

    samples, gradients = langevin_sample(grad_fn, dim,
                                          n_samples=5000, step_size=0.005, temp=0.5,
                                          x0_range=spread)

    result = tb_pipeline(gradients, n_objects=K, method='gradient')
    assignment = result['assignment']
    is_blanket = result['is_blanket']

    # Also try coupling method
    result_coupling = tb_pipeline(gradients, n_objects=K, method='coupling')

    n_blanket = int(np.sum(is_blanket))
    print(f"  Gradient method: assignment={assignment}, blanket={n_blanket} vars")
    print(f"  Coupling method: assignment={result_coupling['assignment']}, "
          f"blanket={int(np.sum(result_coupling['is_blanket']))} vars")

    return {
        'dim': dim,
        'K': K,
        'separation': separation,
        'gradient': {
            'assignment': assignment.tolist(),
            'is_blanket': is_blanket.tolist(),
            'n_blanket': n_blanket,
        },
        'coupling': {
            'assignment': result_coupling['assignment'].tolist(),
            'is_blanket': result_coupling['is_blanket'].tolist(),
            'n_blanket': int(np.sum(result_coupling['is_blanket'])),
        },
        'hessian_est': result['features']['hessian_est'].tolist(),
    }


# =========================================================================
# Visualization
# =========================================================================

def plot_landscape_hessians(results):
    """Plot estimated Hessians for each landscape type."""
    landscape_names = list(results.keys())
    n = len(landscape_names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, landscape_names):
        H = np.array(results[name]['hessian_est'])
        im = ax.imshow(np.abs(H), cmap='YlOrRd', aspect='auto')
        ax.set_title(f'{name}\n(blanket={results[name].get("n_blanket", "?")} vars)',
                     fontsize=9)
        dim = H.shape[0]
        if dim <= 10:
            ax.set_xticks(range(dim))
            ax.set_yticks(range(dim))
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig


# =========================================================================
# Main
# =========================================================================

def run_non_gaussian_experiment():
    """Run all non-Gaussian landscape experiments."""
    print("=" * 70)
    print("US-021: Non-Gaussian Energy Landscapes")
    print("=" * 70)

    results = {}

    # Double well
    results['double_well_6D'] = run_double_well(dim=6)

    # Mexican hat
    results['mexican_hat_4D'] = run_mexican_hat(dim=4)

    # GMM well-separated
    results['gmm_K3_separated'] = run_gmm(dim=6, K=3, separation='well_separated')

    # GMM overlapping
    results['gmm_K3_overlapping'] = run_gmm(dim=6, K=3, separation='overlapping')

    # Compare gradient vs spectral on non-Gaussian
    print("\n--- Method comparison on non-Gaussian ---")
    np.random.seed(42)
    samples_dw, grads_dw = langevin_sample(double_well_grad, 6,
                                            n_samples=5000, step_size=0.005, temp=0.3)
    for method in ['gradient', 'spectral', 'coupling']:
        r = tb_pipeline(grads_dw, n_objects=2, method=method)
        print(f"  Double well, {method}: assignment={r['assignment']}, "
              f"blanket={int(np.sum(r['is_blanket']))}")

    # Visualization
    fig = plot_landscape_hessians(results)
    save_figure(fig, 'non_gaussian_hessians', 'non_gaussian')

    config = {
        'landscapes': list(results.keys()),
        'langevin_samples': 5000,
    }

    save_results('non_gaussian_landscapes', results, config,
                 notes='US-021: TB on double well, Mexican hat, GMM. Tests beyond quadratic assumption.')

    print("\nUS-021 complete.")
    return results


if __name__ == '__main__':
    run_non_gaussian_experiment()
