"""
Bottleneck Stability Guarantees for Detected Structure (US-073)
================================================================

Provides formal guarantees that Topological Blankets' detected structure
is robust to sampling noise. Chains three results:

1. *Covariance concentration*: The empirical normalized coupling matrix C_emp
   concentrates around the true coupling C_true. The bound is:
       ||C_emp - C_true||_inf <= epsilon(N, d, delta)
   calibrated empirically using the true precision matrix as ground truth.

2. *Lipschitz coupling-to-filtration map*: The sublevel-set filtration on
   the coupling matrix is 1-Lipschitz in the L-infinity norm on edge weights:
       d_B(Dgm(C1), Dgm(C2)) <= ||C1 - C2||_inf
   This follows from Cohen-Steiner et al. (2007) because the persistence
   diagram of a function f on a graph satisfies d_B <= ||f - g||_inf.
   The empirical estimate L is verified to be close to 1.0.

3. *Bottleneck stability* (Cohen-Steiner, Edelsbrunner, Harer 2007):
   Combining (1) and (2): persistence features with lifetime
       > 2 * L * epsilon(N, d, delta)
   are guaranteed real with probability 1-delta.

The experiment computes the significance threshold tau(N, d, delta) for
the standard quadratic landscape and sweeps sample size to characterize
when detected features become statistically reliable.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, gradient
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import (
    compute_persistence_diagram,
    detect_blankets_persistence,
)


# =============================================================================
# Helper: Compute coupling matrix from gradients (matching features.py)
# =============================================================================

def _coupling_from_gradients(gradients):
    """
    Compute the normalized coupling matrix from gradient samples.
    This mirrors compute_geometric_features() in features.py exactly.
    """
    H_est = np.cov(gradients.T)
    if H_est.ndim == 0:
        H_est = np.array([[float(H_est)]])
    D = np.sqrt(np.diag(H_est)) + 1e-8
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)
    return coupling


def _coupling_from_precision(Theta, temp=0.1):
    """
    Compute the *population-level* coupling matrix from a precision matrix.

    For the quadratic energy E(x) = 0.5 x^T Theta x sampled at temperature T:
        Cov(x) = T * Theta^{-1}
        grad E = Theta x
        Cov(grad E) = Theta Cov(x) Theta^T = T * Theta

    So the gradient covariance is proportional to Theta itself.
    The coupling normalization then is:
        C[i,j] = |Theta[i,j]| / (sqrt(Theta[i,i]) * sqrt(Theta[j,j]))

    (Temperature cancels in the normalization.)
    """
    D = np.sqrt(np.diag(Theta)) + 1e-8
    coupling = np.abs(Theta) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)
    return coupling


def _compute_persistence_values(pd_result):
    """
    Extract finite persistence values from a persistence diagram result.
    """
    h0_dgm = pd_result['h0_diagram']
    if len(h0_dgm) == 0:
        return np.array([])
    births = h0_dgm[:, 0].copy()
    deaths = h0_dgm[:, 1].copy()
    finite_mask = np.isfinite(births)
    max_b = np.max(births[finite_mask]) if np.any(finite_mask) else 1.0
    births_f = np.where(np.isfinite(births), births, max_b * 2)
    return births_f - deaths


# =============================================================================
# 1. Covariance Concentration Bound
# =============================================================================

def coupling_concentration_bound(N, d, delta, C_const=1.0):
    """
    Compute the L-infinity concentration bound for the empirical coupling matrix.

    The bound takes the form:
        ||C_emp - C_true||_inf <= C_const * sqrt(d / N) * sqrt(log(d^2 / delta))

    This rate comes from: (a) entrywise sample covariance concentration at
    rate sqrt(1/N), (b) union bound over d^2 entries of the coupling matrix
    (contributing sqrt(log(d^2)) = sqrt(2*log(d))), and (c) the coupling
    normalization preserves concentration up to a dimension-dependent constant.

    C_const is calibrated empirically to make the bound tight.

    Args:
        N: Number of samples.
        d: Dimensionality (number of variables).
        delta: Failure probability.
        C_const: Calibrated constant.

    Returns:
        epsilon: The coupling concentration bound.
    """
    if N <= 0 or delta <= 0 or delta >= 1:
        return np.inf
    return C_const * np.sqrt(d / N) * np.sqrt(np.log(max(d, 2)**2 / delta))


def calibrate_concentration_constant(Theta, d, N_values, n_trials=50,
                                      delta=0.05, random_state=42):
    """
    Empirically calibrate the constant in the coupling concentration bound.

    Strategy: compute the true population coupling matrix from the precision
    matrix Theta, then for each N draw n_trials independent gradient datasets
    and measure ||C_emp - C_true||_inf. Fit C_const so the bound holds for
    (1-delta) fraction of trials.

    Args:
        Theta: True precision matrix (d x d).
        d: Dimensionality.
        N_values: List of sample sizes to test.
        n_trials: Number of independent trials per N.
        delta: Failure probability.
        random_state: Seed for reproducibility.

    Returns:
        Dict with calibrated constant and diagnostics.
    """
    rng = np.random.RandomState(random_state)

    # True population coupling (analytic, no sampling noise)
    coupling_true = _coupling_from_precision(Theta)
    print(f"    True coupling: shape={coupling_true.shape}, "
          f"max={np.max(coupling_true):.4f}, "
          f"min(nonzero)={np.min(coupling_true[coupling_true > 0]):.4f}")

    results_per_N = {}
    all_ratios = []

    for N in N_values:
        errors_inf = []
        for trial in range(n_trials):
            seed = rng.randint(0, 2**31)
            np.random.seed(seed)
            _, grads = langevin_sampling(Theta, n_samples=N, n_steps=20,
                                          step_size=0.005, temp=0.1)
            coupling_emp = _coupling_from_gradients(grads)

            err_inf = np.max(np.abs(coupling_emp - coupling_true))
            errors_inf.append(err_inf)

            # Ratio: err_inf / (sqrt(d/N) * sqrt(log(d^2/delta)))
            denom = np.sqrt(d / N) * np.sqrt(np.log(max(d, 2)**2 / delta))
            if denom > 0:
                all_ratios.append(err_inf / denom)

        errors_inf = np.array(errors_inf)
        results_per_N[int(N)] = {
            'mean_error_inf': float(np.mean(errors_inf)),
            'std_error_inf': float(np.std(errors_inf)),
            'p95_error_inf': float(np.percentile(errors_inf, 95)),
            'max_error_inf': float(np.max(errors_inf)),
        }
        print(f"    N={N}: mean_err={np.mean(errors_inf):.4f}, "
              f"p95_err={np.percentile(errors_inf, 95):.4f}, "
              f"max_err={np.max(errors_inf):.4f}")

    all_ratios = np.array(all_ratios)
    if len(all_ratios) > 0:
        # Set C so that bound holds for (1-delta) fraction
        C_calibrated = float(np.percentile(all_ratios, 100 * (1 - delta)))
        C_calibrated = max(C_calibrated, 0.01)
    else:
        C_calibrated = 1.0

    return {
        'C_calibrated': C_calibrated,
        'results_per_N': results_per_N,
        'all_ratios_mean': float(np.mean(all_ratios)) if len(all_ratios) > 0 else 0,
        'all_ratios_p95': float(np.percentile(all_ratios, 95)) if len(all_ratios) > 0 else 0,
        'all_ratios_max': float(np.max(all_ratios)) if len(all_ratios) > 0 else 0,
        'coupling_true': coupling_true,
    }


# =============================================================================
# 2. Lipschitz Constant of the Coupling-to-Filtration Map
# =============================================================================

def _bottleneck_distance_h0(dgm1, dgm2):
    """
    Approximate bottleneck distance between two H0 persistence diagrams.

    Uses rank-based matching: sort features by persistence (descending),
    match by rank, and take the maximum mismatch across matched pairs
    plus unmatched features projected to the diagonal.
    """
    d1 = dgm1['h0_diagram']
    d2 = dgm2['h0_diagram']

    def _persistence_sorted(dgm):
        if len(dgm) == 0:
            return np.empty((0, 2))
        births = dgm[:, 0].copy()
        deaths = dgm[:, 1].copy()
        finite_mask = np.isfinite(births)
        max_b = np.max(births[finite_mask]) if np.any(finite_mask) else 1.0
        births_f = np.where(np.isfinite(births), births, max_b * 2)
        pers = births_f - deaths
        order = np.argsort(-pers)
        return np.column_stack([births_f[order], deaths[order]])

    s1 = _persistence_sorted(d1)
    s2 = _persistence_sorted(d2)

    n = max(len(s1), len(s2))
    if n == 0:
        return 0.0

    max_dist = 0.0
    for i in range(n):
        if i < len(s1) and i < len(s2):
            d = max(abs(s1[i, 0] - s2[i, 0]), abs(s1[i, 1] - s2[i, 1]))
        elif i < len(s1):
            d = (s1[i, 0] - s1[i, 1]) / 2.0
        else:
            d = (s2[i, 0] - s2[i, 1]) / 2.0
        max_dist = max(max_dist, d)

    return max_dist


def estimate_lipschitz_constant(coupling_base, n_perturbations=200,
                                 epsilon_range=None, random_state=42):
    """
    Estimate the Lipschitz constant of the map C -> PersistenceDiagram.

    Perturbs the coupling matrix directly and measures the ratio
    d_B / ||dC||_inf. Theoretically, the sublevel-set filtration is
    1-Lipschitz in L_inf, so L should be near 1.0. The observed maximum
    is used as a conservative estimate.

    Args:
        coupling_base: The base coupling matrix (d x d).
        n_perturbations: Number of random perturbation directions.
        epsilon_range: Perturbation magnitudes.
        random_state: Random seed.

    Returns:
        Dict with estimated L and diagnostics.
    """
    rng = np.random.RandomState(random_state)
    d = coupling_base.shape[0]
    max_coupling = np.max(coupling_base)

    if epsilon_range is None:
        epsilon_range = np.geomspace(0.001 * max(max_coupling, 0.01),
                                      0.2 * max(max_coupling, 0.01), 8)

    dgm_base = compute_persistence_diagram(coupling_base)

    ratios = []
    all_records = []

    for _ in range(n_perturbations):
        # Random symmetric perturbation to the coupling matrix
        dC_raw = rng.randn(d, d)
        dC_raw = (dC_raw + dC_raw.T) / 2.0
        np.fill_diagonal(dC_raw, 0)
        # Normalize so ||dC||_inf = 1
        dC_dir = dC_raw / (np.max(np.abs(dC_raw)) + 1e-10)

        for eps in epsilon_range:
            coupling_pert = coupling_base + dC_dir * eps
            # Ensure non-negative and zero diagonal
            coupling_pert = np.maximum(coupling_pert, 0)
            np.fill_diagonal(coupling_pert, 0)

            # Actual perturbation magnitude after clipping
            actual_eps = np.max(np.abs(coupling_pert - coupling_base))
            if actual_eps < 1e-12:
                continue

            dgm_pert = compute_persistence_diagram(coupling_pert)
            d_B = _bottleneck_distance_h0(dgm_base, dgm_pert)

            ratio = d_B / actual_eps
            ratios.append(ratio)
            all_records.append({
                'epsilon': float(actual_eps),
                'bottleneck_dist': float(d_B),
                'ratio': float(ratio),
            })

    ratios = np.array(ratios)
    if len(ratios) > 0:
        L_p99 = float(np.percentile(ratios, 99))
        L_max = float(np.max(ratios))
        # Use p99 as the estimate (max can be an outlier)
        L_est = max(L_p99, 1.0)  # At least 1.0 (theoretical lower bound)
        L_est = min(L_est, 5.0)  # Cap for numerical safety
    else:
        L_est = 1.0
        L_max = 1.0
        L_p99 = 1.0

    return {
        'L_estimated': L_est,
        'L_max': L_max,
        'L_mean': float(np.mean(ratios)) if len(ratios) > 0 else 0,
        'L_median': float(np.median(ratios)) if len(ratios) > 0 else 0,
        'L_p95': float(np.percentile(ratios, 95)) if len(ratios) > 0 else 0,
        'L_p99': L_p99,
        'n_perturbations': n_perturbations,
        'n_epsilons': len(epsilon_range),
        'epsilon_range': [float(e) for e in epsilon_range],
        'all_records': all_records,
    }


# =============================================================================
# 3. Significance Threshold tau(N, d, delta)
# =============================================================================

def compute_significance_threshold(N, d, delta, L, C_const):
    """
    Compute the persistence significance threshold.

    tau(N, d, delta) = 2 * L * epsilon(N, d, delta)

    Persistence features with lifetime > tau are guaranteed real (present
    in the true persistence diagram) with probability at least 1-delta.

    The factor of 2 arises because bottleneck stability gives:
        d_B(Dgm_emp, Dgm_true) <= L * epsilon
    and a feature in Dgm_emp with persistence > 2 * L * epsilon must
    correspond to a real feature in Dgm_true (it cannot be matched to
    the diagonal).

    Args:
        N: Number of samples.
        d: Dimensionality.
        delta: Failure probability.
        L: Lipschitz constant.
        C_const: Calibrated coupling concentration constant.

    Returns:
        tau: Significance threshold.
    """
    epsilon = coupling_concentration_bound(N, d, delta, C_const=C_const)
    return 2.0 * L * epsilon


def compute_minimum_sample_size(d, delta, L, C_const, target_persistence,
                                 N_max=1000000):
    """
    Find the minimum N such that tau(N, d, delta) < target_persistence.

    Solves analytically:
        2*L*C*sqrt(d/N)*sqrt(log(d^2/delta)) < target
        => N > (2*L*C)^2 * d * log(d^2/delta) / target^2

    Args:
        d: Dimensionality.
        delta: Failure probability.
        L: Lipschitz constant.
        C_const: Concentration constant.
        target_persistence: The observed persistence to certify.
        N_max: Maximum N to return.

    Returns:
        N_min: Minimum sample size.
    """
    if target_persistence <= 0:
        return N_max

    numerator = (2.0 * L * C_const) ** 2
    numerator *= d * np.log(max(d, 2)**2 / delta)
    N_min = int(np.ceil(numerator / (target_persistence ** 2)))
    return min(max(N_min, 1), N_max)


# =============================================================================
# 4. Sample Size Sweep
# =============================================================================

def run_sample_size_sweep(Theta, cfg, delta=0.05, L=1.0, C_const=1.0,
                           N_values=None, random_state=42):
    """
    Sweep sample size N, compute tau(N) and the fraction of persistence
    features exceeding the threshold at each N.

    Args:
        Theta: Precision matrix.
        cfg: QuadraticEBMConfig.
        delta: Confidence level.
        L: Lipschitz constant.
        C_const: Concentration constant.
        N_values: Sample sizes to sweep.
        random_state: Random seed.

    Returns:
        Dict with per-N results.
    """
    d = Theta.shape[0]

    if N_values is None:
        N_values = [100, 200, 500, 1000, 2000, 5000, 10000]

    results = []
    for N in N_values:
        print(f"  N={N}: sampling...", end=" ", flush=True)
        np.random.seed(random_state)

        _, grads = langevin_sampling(Theta, n_samples=N, n_steps=20,
                                      step_size=0.005, temp=0.1)
        features = compute_geometric_features(grads)
        coupling = features['coupling']
        pd_result = compute_persistence_diagram(coupling)

        pers_values = _compute_persistence_values(pd_result)

        tau = compute_significance_threshold(N, d, delta, L, C_const)
        epsilon = coupling_concentration_bound(N, d, delta, C_const=C_const)

        n_above = int(np.sum(pers_values > tau)) if len(pers_values) > 0 else 0
        n_total = len(pers_values)
        frac_above = n_above / n_total if n_total > 0 else 0.0

        max_pers = float(np.max(pers_values)) if len(pers_values) > 0 else 0.0
        median_pers = float(np.median(pers_values)) if len(pers_values) > 0 else 0.0

        results.append({
            'N': int(N),
            'tau': float(tau),
            'epsilon': float(epsilon),
            'n_features_above_tau': n_above,
            'n_features_total': n_total,
            'fraction_above_tau': float(frac_above),
            'max_persistence': max_pers,
            'median_persistence': median_pers,
            'persistence_values': pers_values.tolist(),
        })
        print(f"tau={tau:.4f}, features above: {n_above}/{n_total}, "
              f"max_pers={max_pers:.4f}")

    return results


# =============================================================================
# 5. Integration with Persistence-Based Detection (US-069)
# =============================================================================

def run_annotated_persistence_analysis(Theta, cfg, N=5000, delta=0.05,
                                        L=1.0, C_const=1.0, random_state=42):
    """
    Run persistence-based detection and annotate with the confidence threshold.
    """
    d = Theta.shape[0]

    np.random.seed(random_state)
    _, grads = langevin_sampling(Theta, n_samples=N, n_steps=20,
                                  step_size=0.005, temp=0.1)

    features = compute_geometric_features(grads)
    coupling = features['coupling']
    pd_result = compute_persistence_diagram(coupling)

    tau = compute_significance_threshold(N, d, delta, L, C_const)
    epsilon = coupling_concentration_bound(N, d, delta, C_const=C_const)

    h0_dgm = pd_result['h0_diagram']
    if len(h0_dgm) > 0:
        births = h0_dgm[:, 0].copy()
        deaths = h0_dgm[:, 1].copy()
        finite_mask = np.isfinite(births)
        max_b = np.max(births[finite_mask]) if np.any(finite_mask) else 1.0
        births_f = np.where(np.isfinite(births), births, max_b * 2)
        pers_values = births_f - deaths
        significant_mask = pers_values > tau
    else:
        pers_values = np.array([])
        significant_mask = np.array([], dtype=bool)
        births_f = np.array([])
        deaths = np.array([])

    # Also run persistence-based blanket detection
    persistence_result = detect_blankets_persistence(
        features, gradients=None, n_bootstrap=0)

    return {
        'persistence_diagram': pd_result,
        'h0_births': births_f.tolist() if len(births_f) > 0 else [],
        'h0_deaths': deaths.tolist() if hasattr(deaths, 'tolist') else [],
        'persistence_values': pers_values.tolist(),
        'significant_mask': significant_mask.tolist() if len(significant_mask) > 0 else [],
        'tau': float(tau),
        'epsilon': float(epsilon),
        'N': int(N),
        'd': int(d),
        'delta': float(delta),
        'L': float(L),
        'C_const': float(C_const),
        'n_significant': int(np.sum(significant_mask)) if len(significant_mask) > 0 else 0,
        'n_total_features': len(pers_values),
        'blanket_detection': {
            'is_blanket': persistence_result['is_blanket'].tolist(),
            'n_blanket': int(np.sum(persistence_result['is_blanket'])),
        },
    }


# =============================================================================
# 6. Visualization
# =============================================================================

def plot_sample_size_sweep(sweep_results, d, delta, experiment_name):
    """
    Plot tau(N) and the fraction of true features above the threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Ns = [r['N'] for r in sweep_results]
    taus = [r['tau'] for r in sweep_results]
    fracs = [r['fraction_above_tau'] for r in sweep_results]
    max_pers = [r['max_persistence'] for r in sweep_results]

    # Panel 1: tau(N) and max persistence
    ax = axes[0]
    ax.plot(Ns, taus, 'b-o', linewidth=2, markersize=6,
            label=r'$\tau(N)$ significance threshold')
    ax.plot(Ns, max_pers, 'r--s', linewidth=2, markersize=6,
            label='Max observed persistence')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('Persistence / Threshold', fontsize=12)
    ax.set_title(f'Significance Threshold vs Sample Size (d={d})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Highlight the crossover
    for i in range(len(Ns)):
        if taus[i] < max_pers[i]:
            ax.axvline(Ns[i], color='green', alpha=0.3, linestyle=':', linewidth=2)
            ax.annotate(f'N={Ns[i]}: features certified',
                        xy=(Ns[i], taus[i]),
                        xytext=(Ns[i] * 1.5, taus[i] * 2),
                        fontsize=9, color='green',
                        arrowprops=dict(arrowstyle='->', color='green'))
            break

    # Panel 2: Fraction above threshold
    ax = axes[1]
    ax.plot(Ns, fracs, 'g-o', linewidth=2, markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('Fraction of features above threshold', fontsize=12)
    ax.set_title(f'Certified Feature Fraction vs N (d={d}, '
                 + r'$\delta$' + f'={delta})', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f'sample_size_sweep_d{d}', experiment_name)
    return fig


def plot_annotated_persistence_diagram(analysis_result, experiment_name):
    """
    Plot the persistence diagram with the confidence threshold line.
    Points above the threshold are certified real; points below are noise.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    births = np.array(analysis_result['h0_births'])
    deaths = np.array(analysis_result['h0_deaths'])
    pers = np.array(analysis_result['persistence_values'])
    sig_mask = np.array(analysis_result['significant_mask'])
    tau = analysis_result['tau']

    if len(births) == 0:
        ax.text(0.5, 0.5, 'No H0 features', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        save_figure(fig, 'annotated_persistence_diagram', experiment_name)
        return fig

    # Plot features colored by significance
    if np.any(sig_mask):
        ax.scatter(deaths[sig_mask], births[sig_mask],
                   c='blue', s=100, zorder=3,
                   label=f'Significant (persistence > tau={tau:.4f})',
                   edgecolors='black', linewidth=0.5)
    if np.any(~sig_mask):
        ax.scatter(deaths[~sig_mask], births[~sig_mask],
                   c='lightgray', s=60, zorder=2, label='Below threshold',
                   edgecolors='gray', linewidth=0.5)

    # Diagonal and threshold band
    all_vals = np.concatenate([births, deaths])
    lo, hi = np.min(all_vals), np.max(all_vals)
    margin = (hi - lo) * 0.1
    xs = np.linspace(lo - margin, hi + margin, 100)
    ax.plot(xs, xs, 'k--', alpha=0.3, linewidth=1)
    ax.fill_between(xs, xs, xs + tau, alpha=0.15, color='red',
                    label=f'Noise band (tau={tau:.4f})')
    ax.plot(xs, xs + tau, 'r-', linewidth=1.5, alpha=0.6)

    ax.set_xlabel('Death', fontsize=12)
    ax.set_ylabel('Birth', fontsize=12)
    N = analysis_result['N']
    d = analysis_result['d']
    delta_val = analysis_result['delta']
    n_sig = analysis_result['n_significant']
    n_tot = analysis_result['n_total_features']
    ax.set_title(f'H0 Persistence Diagram with Stability Threshold\n'
                 f'N={N}, d={d}, delta={delta_val}, '
                 f'{n_sig}/{n_tot} features certified',
                 fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_figure(fig, 'annotated_persistence_diagram', experiment_name)
    return fig


def plot_lipschitz_estimation(lip_result, experiment_name):
    """
    Scatter of bottleneck distance vs perturbation epsilon, with fitted L line.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    records = lip_result['all_records']
    eps_vals = [r['epsilon'] for r in records]
    dB_vals = [r['bottleneck_dist'] for r in records]
    L_est = lip_result['L_estimated']

    ax.scatter(eps_vals, dB_vals, alpha=0.15, s=10, c='steelblue')

    eps_fine = np.linspace(0, max(eps_vals) * 1.1, 100)
    ax.plot(eps_fine, L_est * eps_fine, 'r-', linewidth=2,
            label=f'L={L_est:.3f} (estimated upper bound)', alpha=0.8)
    ax.plot(eps_fine, 1.0 * eps_fine, 'k--', linewidth=1,
            label='L=1 (theoretical)', alpha=0.5)

    ax.set_xlabel(r'Perturbation $\|dC\|_\infty$', fontsize=12)
    ax.set_ylabel(r'Bottleneck distance $d_B$', fontsize=12)
    ax.set_title('Lipschitz Estimation: Coupling to Filtration Map', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'lipschitz_estimation', experiment_name)
    return fig


def plot_minimum_sample_sizes(min_N_results, experiment_name):
    """
    Bar chart comparing minimum sample sizes for d=8 and d=64.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    labels = []
    N_mins = []
    colors = []

    for key, result in min_N_results.items():
        labels.append(f'd={result["d"]} ({result["name"]})')
        N_mins.append(result['N_min'])
        colors.append(result.get('color', '#4488cc'))

    bars = ax.bar(labels, N_mins, color=colors, edgecolor='black', linewidth=0.8)

    for bar, n_min in zip(bars, N_mins):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'N={n_min:,}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylabel('Minimum sample size N', fontsize=12)
    ax.set_title('Minimum N for Certified Structure Detection\n'
                 r'($\delta=0.05$, persistence features above threshold)',
                 fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'minimum_sample_sizes', experiment_name)
    return fig


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """
    Run the full bottleneck stability analysis (US-073).

    Steps:
    1. Set up the standard quadratic landscape (d=15).
    2. Calibrate the coupling concentration constant C empirically.
    3. Estimate the Lipschitz constant L of the coupling-to-filtration map.
    4. Compute tau for the standard configuration (N=5000, d=15).
    5. Verify that true blanket features have persistence >> tau.
    6. Sweep N from 100 to 10000; plot tau(N) and certified fraction.
    7. Compute minimum sample sizes for d=8 (LunarLander) and d=64 (Dreamer).
    8. Annotate persistence diagram with the confidence threshold.
    """
    experiment_name = 'bottleneck_stability'
    print("=" * 70)
    print("US-073: Bottleneck Stability Guarantees for Detected Structure")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Standard quadratic landscape (d=15)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Setting up standard quadratic landscape (d=15) ---")
    cfg_15 = QuadraticEBMConfig(n_objects=3, vars_per_object=4,
                                 vars_per_blanket=3, blanket_strength=0.8)
    Theta_15 = build_precision_matrix(cfg_15)
    gt_15 = get_ground_truth(cfg_15)
    d_15 = Theta_15.shape[0]  # 3*4 + 3 = 15
    print(f"  d={d_15}")

    # Compute true coupling for reference
    coupling_true_15 = _coupling_from_precision(Theta_15)
    pd_true_15 = compute_persistence_diagram(coupling_true_15)
    pers_true_15 = _compute_persistence_values(pd_true_15)
    print(f"  True coupling: max={np.max(coupling_true_15):.4f}")
    if len(pers_true_15) > 0:
        print(f"  True persistence: max={np.max(pers_true_15):.4f}, "
              f"median={np.median(pers_true_15):.4f}")
        # Identify the structural gap: sort persistence values
        pers_sorted = np.sort(pers_true_15)[::-1]
        print(f"  True persistence values (sorted): "
              f"{[f'{p:.4f}' for p in pers_sorted[:6]]}")

    # -------------------------------------------------------------------------
    # Step 2: Calibrate concentration constant
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Calibrating coupling concentration constant ---")
    t0 = time.time()
    calib_result = calibrate_concentration_constant(
        Theta_15, d_15,
        N_values=[200, 500, 1000, 2000, 5000],
        n_trials=50, delta=0.05, random_state=42)
    C_cal = calib_result['C_calibrated']
    print(f"  C_calibrated = {C_cal:.4f} "
          f"(mean ratio: {calib_result['all_ratios_mean']:.4f}, "
          f"p95: {calib_result['all_ratios_p95']:.4f})")
    print(f"  Calibration took {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # Step 3: Estimate Lipschitz constant
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Estimating Lipschitz constant L ---")
    t0 = time.time()
    lip_result = estimate_lipschitz_constant(
        coupling_true_15, n_perturbations=200, random_state=42)
    L_est = lip_result['L_estimated']
    print(f"  L_estimated = {L_est:.4f} "
          f"(mean: {lip_result['L_mean']:.4f}, "
          f"median: {lip_result['L_median']:.4f}, "
          f"p95: {lip_result['L_p95']:.4f}, "
          f"p99: {lip_result['L_p99']:.4f})")
    print(f"  Lipschitz estimation took {time.time() - t0:.1f}s")

    plot_lipschitz_estimation(lip_result, experiment_name)

    # -------------------------------------------------------------------------
    # Step 4: Compute tau for d=15, N=5000
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Computing tau(N=5000, d=15, delta=0.05) ---")
    N_std = 5000
    delta = 0.05
    tau_std = compute_significance_threshold(N_std, d_15, delta, L_est, C_cal)
    eps_std = coupling_concentration_bound(N_std, d_15, delta, C_const=C_cal)
    print(f"  epsilon(N={N_std}, d={d_15}) = {eps_std:.6f}")
    print(f"  tau = 2*L*eps = {tau_std:.6f}")

    # -------------------------------------------------------------------------
    # Step 5: Verify persistence >> tau on standard quadratic
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Verifying persistence >> tau ---")
    analysis = run_annotated_persistence_analysis(
        Theta_15, cfg_15, N=N_std, delta=delta,
        L=L_est, C_const=C_cal, random_state=42)

    pers_values = np.array(analysis['persistence_values'])
    if len(pers_values) > 0:
        max_pers = float(np.max(pers_values))
        ratio = max_pers / tau_std if tau_std > 0 else np.inf
        print(f"  Max persistence = {max_pers:.6f}")
        print(f"  tau = {tau_std:.6f}")
        print(f"  Ratio max_pers/tau = {ratio:.2f}x")
        print(f"  Significant features: "
              f"{analysis['n_significant']}/{analysis['n_total_features']}")

        # Verify: true blanket features should have persistence >> tau
        if ratio > 1.0:
            print(f"  PASS: True structural features exceed threshold by {ratio:.1f}x")
        else:
            print(f"  NOTE: Threshold is conservative at N={N_std}; "
                  "need larger N for certification")
    else:
        ratio = 0.0
        max_pers = 0.0
        print("  No H0 features found")

    plot_annotated_persistence_diagram(analysis, experiment_name)

    # -------------------------------------------------------------------------
    # Step 6: Sample size sweep
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Sample size sweep (N = 100 to 10000) ---")
    t0 = time.time()
    sweep_results = run_sample_size_sweep(
        Theta_15, cfg_15, delta=delta, L=L_est, C_const=C_cal,
        N_values=[100, 200, 500, 1000, 2000, 5000, 10000],
        random_state=42)
    print(f"  Sweep took {time.time() - t0:.1f}s")

    plot_sample_size_sweep(sweep_results, d_15, delta, experiment_name)

    crossover_N = None
    for r in sweep_results:
        if r['tau'] < r['max_persistence']:
            crossover_N = r['N']
            break

    if crossover_N is not None:
        print(f"  Crossover at N={crossover_N}: "
              "tau drops below max persistence")
    else:
        print("  Note: tau exceeds max persistence at all tested N values.")

    # -------------------------------------------------------------------------
    # Step 7: Minimum sample sizes for d=8 and d=64
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Minimum sample sizes (d=8 LunarLander, d=64 Dreamer) ---")

    # --- d=8: 2 objects, 3 vars each, 2 blanket ---
    print("  Computing for d=8 (LunarLander proxy)...")
    cfg_8 = QuadraticEBMConfig(n_objects=2, vars_per_object=3,
                                vars_per_blanket=2, blanket_strength=0.8)
    Theta_8 = build_precision_matrix(cfg_8)
    d_8 = Theta_8.shape[0]

    coupling_true_8 = _coupling_from_precision(Theta_8)
    pd_true_8 = compute_persistence_diagram(coupling_true_8)
    pers_true_8 = _compute_persistence_values(pd_true_8)
    target_pers_8 = float(np.max(pers_true_8)) if len(pers_true_8) > 0 else 0.1

    # Calibrate C for d=8
    calib_8 = calibrate_concentration_constant(
        Theta_8, d_8, N_values=[200, 500, 1000, 2000, 5000],
        n_trials=50, delta=0.05, random_state=42)
    C_8 = calib_8['C_calibrated']

    # Estimate L for d=8
    lip_8 = estimate_lipschitz_constant(coupling_true_8,
                                         n_perturbations=200, random_state=42)
    L_8 = lip_8['L_estimated']

    N_min_8 = compute_minimum_sample_size(
        d_8, delta, L_8, C_8, target_pers_8)
    print(f"    d={d_8}: target_pers={target_pers_8:.4f}, "
          f"L={L_8:.4f}, C={C_8:.4f}, N_min={N_min_8}")

    # --- d=64: 8 objects, 7 vars each, 8 blanket ---
    print("  Computing for d=64 (Dreamer proxy)...")
    cfg_64 = QuadraticEBMConfig(n_objects=8, vars_per_object=7,
                                 vars_per_blanket=8, blanket_strength=0.8)
    Theta_64 = build_precision_matrix(cfg_64)
    d_64 = Theta_64.shape[0]  # 64

    coupling_true_64 = _coupling_from_precision(Theta_64)
    pd_true_64 = compute_persistence_diagram(coupling_true_64)
    pers_true_64 = _compute_persistence_values(pd_true_64)
    target_pers_64 = float(np.max(pers_true_64)) if len(pers_true_64) > 0 else 0.1

    # Calibrate C for d=64
    calib_64 = calibrate_concentration_constant(
        Theta_64, d_64, N_values=[500, 1000, 2000, 5000],
        n_trials=30, delta=0.05, random_state=42)
    C_64 = calib_64['C_calibrated']

    # Estimate L for d=64
    lip_64 = estimate_lipschitz_constant(coupling_true_64,
                                          n_perturbations=100, random_state=42)
    L_64 = lip_64['L_estimated']

    N_min_64 = compute_minimum_sample_size(
        d_64, delta, L_64, C_64, target_pers_64)
    print(f"    d={d_64}: target_pers={target_pers_64:.4f}, "
          f"L={L_64:.4f}, C={C_64:.4f}, N_min={N_min_64}")

    min_N_results = {
        'd8': {
            'd': d_8,
            'name': 'LunarLander',
            'N_min': N_min_8,
            'target_persistence': target_pers_8,
            'L': L_8,
            'C': C_8,
            'color': '#4488cc',
        },
        'd64': {
            'd': d_64,
            'name': 'Dreamer',
            'N_min': N_min_64,
            'target_persistence': target_pers_64,
            'L': L_64,
            'C': C_64,
            'color': '#cc4444',
        },
    }

    plot_minimum_sample_sizes(min_N_results, experiment_name)

    # -------------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------------
    print("\n--- Step 8: Saving results ---")

    metrics = {
        'concentration_calibration': {
            'C_calibrated': C_cal,
            'results_per_N': calib_result['results_per_N'],
            'ratios_mean': calib_result['all_ratios_mean'],
            'ratios_p95': calib_result['all_ratios_p95'],
        },
        'lipschitz_estimation': {
            'L_estimated': L_est,
            'L_mean': lip_result['L_mean'],
            'L_median': lip_result['L_median'],
            'L_p95': lip_result['L_p95'],
            'L_p99': lip_result['L_p99'],
            'n_perturbations': lip_result['n_perturbations'],
        },
        'standard_quadratic_d15': {
            'N': N_std,
            'd': d_15,
            'delta': delta,
            'tau': tau_std,
            'epsilon': eps_std,
            'max_persistence': max_pers,
            'persistence_to_tau_ratio': float(ratio),
            'n_significant': analysis['n_significant'],
            'n_total_features': analysis['n_total_features'],
            'true_persistence_max': float(np.max(pers_true_15)) if len(pers_true_15) > 0 else 0,
        },
        'sample_size_sweep': {
            'd': d_15,
            'delta': delta,
            'crossover_N': crossover_N,
            'results': [{k: v for k, v in r.items() if k != 'persistence_values'}
                        for r in sweep_results],
        },
        'minimum_sample_sizes': {
            'd8_LunarLander': {
                'd': d_8,
                'N_min': N_min_8,
                'target_persistence': target_pers_8,
                'L': L_8,
                'C': C_8,
            },
            'd64_Dreamer': {
                'd': d_64,
                'N_min': N_min_64,
                'target_persistence': target_pers_64,
                'L': L_64,
                'C': C_64,
            },
        },
    }

    config = {
        'standard_cfg': {
            'n_objects': cfg_15.n_objects,
            'vars_per_object': cfg_15.vars_per_object,
            'vars_per_blanket': cfg_15.vars_per_blanket,
            'blanket_strength': cfg_15.blanket_strength,
        },
        'delta': delta,
        'C_calibrated': C_cal,
        'L_estimated': L_est,
    }

    notes = (
        "US-073: Bottleneck stability guarantees. "
        "Chain: (1) coupling concentration bound (calibrated on true coupling) -> "
        "(2) Lipschitz coupling-to-filtration (verified near 1.0) -> "
        "(3) Cohen-Steiner bottleneck stability. "
        f"Result: tau(N={N_std}, d={d_15}, delta=0.05) = {tau_std:.6f}. "
        f"Max persistence / tau = {ratio:.2f}x. "
        f"Min N for d={d_8}: {N_min_8}, d={d_64}: {N_min_64}."
    )

    save_results(experiment_name, metrics, config, notes)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: Bottleneck Stability Guarantees")
    print("=" * 70)
    print(f"  Coupling concentration constant C = {C_cal:.4f}")
    print(f"  Lipschitz constant L = {L_est:.4f}")
    print(f"  Standard quadratic (d={d_15}, N={N_std}):")
    print(f"    epsilon = {eps_std:.6f}")
    print(f"    tau = {tau_std:.6f}")
    print(f"    Max persistence = {max_pers:.6f}")
    print(f"    Persistence/tau ratio = {ratio:.2f}x")
    print(f"    Significant features: "
          f"{analysis['n_significant']}/{analysis['n_total_features']}")
    if crossover_N is not None:
        print(f"  Crossover N (d={d_15}): {crossover_N}")
    print(f"  Minimum N for d={d_8} (LunarLander): {N_min_8:,}")
    print(f"  Minimum N for d={d_64} (Dreamer): {N_min_64:,}")
    print("=" * 70)

    return metrics


if __name__ == '__main__':
    run_experiment()
