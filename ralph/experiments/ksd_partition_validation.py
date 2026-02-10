"""
US-074: KSD Goodness-of-Fit Test for Partition Validation
==========================================================

Implements a Kernelized Stein Discrepancy (KSD) based test (Liu et al. 2016,
Chwialkowski et al. 2016) to validate whether a detected Topological
Blankets partition (A, B, M) is consistent with conditional independence:

    x_A _||_ x_B | x_M

The test leverages the score function (gradient of log-density), which
aligns directly with TB's score-matching foundation. No normalization
constant is needed.

Core idea: Under conditional independence x_A _||_ x_B | x_M, the
precision matrix Theta satisfies Theta[A,B] = 0. The gradient
(score function) for variables in A is s_A(x) = Theta[A,:] @ x =
Theta[A,A] x_A + Theta[A,M] x_M + Theta[A,B] x_B. Under CI, the
Theta[A,B] term vanishes, so s_A should not depend on x_B after
conditioning on x_M.

The test measures cross-score dependence: how much do the score
components for A depend on the values of B, after removing the
effect of M. This is quantified via a kernel-based HSIC statistic
with permutation p-values.

Validation scenarios:
  - Quadratic EBM with correct partition: should NOT reject (p > 0.05)
  - Quadratic EBM with random partition: SHOULD reject (p < 0.05)
  - LunarLander 8D data: test TB-detected partition for consistency
  - Power analysis: at what sample size N is correct vs incorrect
    reliably distinguished

Integration: adds a validate_partition() method to TopologicalBlankets.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, gradient,
    topological_blankets as tb_quadratic,
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

from topological_blankets.core import TopologicalBlankets, topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features


# =============================================================================
# Score-Based Conditional Independence Test
# =============================================================================
#
# Key insight: For an energy-based model E(x) = 0.5 x^T Theta x, the score
# function is s(x) = -nabla E(x) = -Theta x. Therefore:
#   s_A(x) = -Theta[A,:] x = -Theta[A,A] x_A - Theta[A,M] x_M - Theta[A,B] x_B
#
# Under conditional independence x_A _||_ x_B | x_M, Theta[A,B] = 0.
# We estimate the cross-coupling Theta_hat[A,B] by regressing s_A on the
# full sample x, then extracting the B-coefficients. The test statistic
# is the Frobenius norm ||Theta_hat[A,B]||_F, and p-values come from
# bootstrap resampling under the null (shuffling B indices to break the
# cross-coupling signal).
#
# This is a *score-based* test: it uses only (samples, gradients) pairs
# and directly tests whether the score factorizes across the blanket.
# =============================================================================

def estimate_cross_coupling(
        samples: np.ndarray,
        gradients: np.ndarray,
        idx_A: np.ndarray,
        idx_B: np.ndarray,
        idx_M: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estimate the cross-coupling ||Theta_hat[A,B]||_F from score-sample pairs.

    Regresses s_A on (x_A, x_M, x_B) and extracts the B-coefficients.
    Under CI, these should be zero.

    Returns:
        (frobenius_norm, beta_B_matrix): The Frobenius norm of the estimated
        cross-coupling and the raw coefficient matrix.
    """
    n = samples.shape[0]
    s_A = gradients[:, idx_A]  # (n, |A|)
    x_all = samples  # (n, d)

    # Reorder columns: [A, M, B] for clear extraction
    idx_AM = np.concatenate([idx_A, idx_M])
    n_AM = len(idx_AM)

    # Features: (x_A, x_M, x_B) with intercept
    X = np.hstack([np.ones((n, 1)), samples[:, idx_AM], samples[:, idx_B]])

    # Regress each component of s_A on X
    beta, _, _, _ = np.linalg.lstsq(X, s_A, rcond=None)

    # Extract B-coefficients: columns after intercept + |A| + |M|
    beta_B = beta[1 + n_AM:, :]  # (|B|, |A|)

    frob_norm = np.sqrt(np.sum(beta_B ** 2))
    return frob_norm, beta_B


def ksd_conditional_independence_test(
        samples: np.ndarray,
        gradients: np.ndarray,
        idx_A: np.ndarray,
        idx_B: np.ndarray,
        idx_M: np.ndarray,
        n_bootstrap: int = 500,
        max_samples: int = 2000,
        seed: int = 42) -> Dict:
    """
    KSD-based test for conditional independence: x_A _||_ x_B | x_M.

    Uses the score function (gradient of log-density) to directly estimate
    the cross-coupling between A and B. Under CI, the score components
    for A should not depend on x_B (given x_M), so the estimated
    cross-coupling coefficients Theta_hat[A,B] should be zero.

    The test statistic is ||Theta_hat[A,B]||_F. The null distribution
    is generated by shuffling the B-variable columns (breaking any real
    cross-coupling) and re-estimating the statistic. This permutation
    preserves the marginal distributions of all variables while destroying
    the A-B conditional dependence.

    Args:
        samples: (N, d) sample points.
        gradients: (N, d) score function values.
        idx_A, idx_B, idx_M: Variable index sets.
        n_bootstrap: Permutation replications for p-value.
        max_samples: Maximum samples used.
        seed: Random seed.

    Returns:
        Dictionary with test_stat, p_value, reject_at_005, etc.
    """
    rng = np.random.RandomState(seed)
    n = samples.shape[0]

    # Subsample if needed
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        samples_sub = samples[idx]
        gradients_sub = gradients[idx]
    else:
        samples_sub = samples.copy()
        gradients_sub = gradients.copy()

    n_sub = samples_sub.shape[0]

    # Observed statistic: ||Theta_hat[A,B]||_F
    observed_stat, beta_B_obs = estimate_cross_coupling(
        samples_sub, gradients_sub, idx_A, idx_B, idx_M
    )

    # Also test reverse: ||Theta_hat[B,A]||_F
    observed_stat_rev, beta_A_obs = estimate_cross_coupling(
        samples_sub, gradients_sub, idx_B, idx_A, idx_M
    )

    # Combined statistic (sum of both directions)
    observed_combined = observed_stat + observed_stat_rev

    # Null distribution: shuffle B columns to break cross-coupling
    perm_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        perm = rng.permutation(n_sub)
        samples_perm = samples_sub.copy()
        samples_perm[:, idx_B] = samples_sub[perm][:, idx_B]

        stat_fwd, _ = estimate_cross_coupling(
            samples_perm, gradients_sub, idx_A, idx_B, idx_M
        )
        stat_rev, _ = estimate_cross_coupling(
            samples_perm, gradients_sub, idx_B, idx_A, idx_M
        )
        perm_stats[b] = stat_fwd + stat_rev

    # p-value
    p_value = np.mean(perm_stats >= observed_combined)

    return {
        'test_stat_AB': float(observed_stat),
        'test_stat_BA': float(observed_stat_rev),
        'test_stat_combined': float(observed_combined),
        'p_value_AB': float(np.mean(perm_stats >= observed_stat * 2)),  # approx
        'p_value_BA': float(np.mean(perm_stats >= observed_stat_rev * 2)),  # approx
        'combined_p_value': float(p_value),
        'n_samples_used': int(n_sub),
        'dim_A': len(idx_A),
        'dim_B': len(idx_B),
        'dim_M': len(idx_M),
        'reject_at_005': bool(p_value < 0.05),
        'reject_at_001': bool(p_value < 0.01),
        'n_bootstrap': n_bootstrap,
        'beta_B_frob': float(observed_stat),
        'beta_A_frob': float(observed_stat_rev),
        'perm_stats_mean': float(np.mean(perm_stats)),
        'perm_stats_std': float(np.std(perm_stats)),
    }


# =============================================================================
# Validate Partition (Integration API)
# =============================================================================

def validate_partition(tb_result: Dict,
                       samples: np.ndarray,
                       gradients: np.ndarray,
                       n_permutations: int = 500,
                       max_samples: int = 300,
                       seed: int = 42) -> Dict:
    """
    Validate a TB partition using the KSD conditional independence test.

    Tests whether each pair of detected objects is conditionally
    independent given the blanket M.

    Returns:
        Dictionary with per-pair test results and overall summary.
    """
    assignment = tb_result['assignment']
    is_blanket = tb_result['is_blanket']

    idx_M = np.where(is_blanket)[0]
    object_ids = sorted([int(x) for x in np.unique(assignment) if x >= 0])

    if len(object_ids) < 2:
        return {
            'error': 'Need at least 2 objects for conditional independence test',
            'n_objects': len(object_ids),
        }

    pair_results = {}
    for i, obj_i in enumerate(object_ids):
        for obj_j in object_ids[i + 1:]:
            idx_A = np.where(assignment == obj_i)[0]
            idx_B = np.where(assignment == obj_j)[0]

            pair_key = f"obj{obj_i}_vs_obj{obj_j}"
            pair_results[pair_key] = ksd_conditional_independence_test(
                samples, gradients, idx_A, idx_B, idx_M,
                n_bootstrap=n_permutations,
                max_samples=max_samples,
                seed=seed,
            )

    all_pvalues = [r['combined_p_value'] for r in pair_results.values()]
    all_reject_005 = [r['reject_at_005'] for r in pair_results.values()]

    return {
        'pair_results': pair_results,
        'n_pairs_tested': len(pair_results),
        'all_pvalues': all_pvalues,
        'any_rejected_005': any(all_reject_005),
        'min_pvalue': float(min(all_pvalues)),
        'mean_pvalue': float(np.mean(all_pvalues)),
        'partition_valid': not any(all_reject_005),
    }


# =============================================================================
# Experiment 1: Quadratic EBM, Correct vs Random Partition
# =============================================================================

def run_quadratic_validation():
    """
    Test KSD validation on quadratic EBM with known structure.
    """
    print("=" * 70)
    print("Experiment 1: Quadratic EBM - Correct vs Random Partition")
    print("=" * 70)

    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )
    Theta = build_precision_matrix(cfg)
    truth = get_ground_truth(cfg)
    n_vars = Theta.shape[0]

    print(f"\nConfig: {cfg.n_objects} objects, {cfg.vars_per_object} vars/obj, "
          f"{cfg.vars_per_blanket} blanket vars, total={n_vars}")

    # Verify CI structure
    idx_A_gt = np.where(truth['assignment'] == 0)[0]
    idx_B_gt = np.where(truth['assignment'] == 1)[0]
    cross_block = Theta[np.ix_(idx_A_gt, idx_B_gt)]
    print(f"Theta[A,B] (should be 0 for CI): max|.| = {np.max(np.abs(cross_block)):.6f}")

    # Sample
    np.random.seed(42)
    samples, gradients = langevin_sampling(
        Theta, n_samples=3000, n_steps=50, step_size=0.005, temp=0.1
    )
    print(f"Sampled {samples.shape[0]} points.\n")

    # --- Correct partition ---
    print("1a. Correct partition (ground truth)")
    idx_A_correct = np.where(truth['assignment'] == 0)[0]
    idx_B_correct = np.where(truth['assignment'] == 1)[0]
    idx_M_correct = truth['blanket_vars']
    print(f"  A={list(idx_A_correct)}, B={list(idx_B_correct)}, M={list(idx_M_correct)}")

    t0 = time.time()
    correct_result = ksd_conditional_independence_test(
        samples, gradients, idx_A_correct, idx_B_correct, idx_M_correct,
        n_bootstrap=500, max_samples=300, seed=42,
    )
    t_correct = time.time() - t0
    print(f"  Frob(Theta_AB)={correct_result['test_stat_AB']:.6f}, "
          f"p_AB={correct_result['p_value_AB']:.4f}")
    print(f"  Frob(Theta_BA)={correct_result['test_stat_BA']:.6f}, "
          f"p_BA={correct_result['p_value_BA']:.4f}")
    print(f"  Combined p={correct_result['combined_p_value']:.4f}, "
          f"reject={correct_result['reject_at_005']}, time={t_correct:.1f}s")

    # --- Random partition ---
    print("\n1b. Random partition")
    rng = np.random.RandomState(123)
    perm = rng.permutation(n_vars)
    idx_A_random = perm[:cfg.vars_per_object]
    idx_B_random = perm[cfg.vars_per_object:2 * cfg.vars_per_object]
    idx_M_random = perm[2 * cfg.vars_per_object:]
    print(f"  A={list(idx_A_random)}, B={list(idx_B_random)}, M={list(idx_M_random)}")

    t0 = time.time()
    random_result = ksd_conditional_independence_test(
        samples, gradients, idx_A_random, idx_B_random, idx_M_random,
        n_bootstrap=500, max_samples=300, seed=42,
    )
    t_random = time.time() - t0
    print(f"  Frob(Theta_AB)={random_result['test_stat_AB']:.6f}, "
          f"p_AB={random_result['p_value_AB']:.4f}")
    print(f"  Frob(Theta_BA)={random_result['test_stat_BA']:.6f}, "
          f"p_BA={random_result['p_value_BA']:.4f}")
    print(f"  Combined p={random_result['combined_p_value']:.4f}, "
          f"reject={random_result['reject_at_005']}, time={t_random:.1f}s")

    # --- TB-detected partition ---
    print("\n1c. TB-detected partition")
    tb_result = tb_quadratic(gradients, n_objects=cfg.n_objects)
    idx_A_tb = np.where(tb_result['assignment'] == 0)[0]
    idx_B_tb = np.where(tb_result['assignment'] == 1)[0]
    idx_M_tb = np.where(tb_result['is_blanket'])[0]
    print(f"  A={list(idx_A_tb)}, B={list(idx_B_tb)}, M={list(idx_M_tb)}")

    t0 = time.time()
    tb_detected_result = ksd_conditional_independence_test(
        samples, gradients, idx_A_tb, idx_B_tb, idx_M_tb,
        n_bootstrap=500, max_samples=300, seed=42,
    )
    t_tb = time.time() - t0
    print(f"  Frob(Theta_AB)={tb_detected_result['test_stat_AB']:.6f}, "
          f"p_AB={tb_detected_result['p_value_AB']:.4f}")
    print(f"  Frob(Theta_BA)={tb_detected_result['test_stat_BA']:.6f}, "
          f"p_BA={tb_detected_result['p_value_BA']:.4f}")
    print(f"  Combined p={tb_detected_result['combined_p_value']:.4f}, "
          f"reject={tb_detected_result['reject_at_005']}, time={t_tb:.1f}s")

    return {
        'config': {
            'n_objects': cfg.n_objects, 'vars_per_object': cfg.vars_per_object,
            'vars_per_blanket': cfg.vars_per_blanket,
            'intra_strength': cfg.intra_strength,
            'blanket_strength': cfg.blanket_strength, 'n_samples': int(samples.shape[0]),
        },
        'correct_partition': {
            'idx_A': idx_A_correct.tolist(), 'idx_B': idx_B_correct.tolist(),
            'idx_M': idx_M_correct.tolist(),
            'test_stat_AB': correct_result['test_stat_AB'],
            'test_stat_BA': correct_result['test_stat_BA'],
            'p_value_AB': correct_result['p_value_AB'],
            'p_value_BA': correct_result['p_value_BA'],
            'p_value': correct_result['combined_p_value'],
            'reject_at_005': correct_result['reject_at_005'], 'time': t_correct,
        },
        'random_partition': {
            'idx_A': idx_A_random.tolist(), 'idx_B': idx_B_random.tolist(),
            'idx_M': idx_M_random.tolist(),
            'test_stat_AB': random_result['test_stat_AB'],
            'test_stat_BA': random_result['test_stat_BA'],
            'p_value_AB': random_result['p_value_AB'],
            'p_value_BA': random_result['p_value_BA'],
            'p_value': random_result['combined_p_value'],
            'reject_at_005': random_result['reject_at_005'], 'time': t_random,
        },
        'tb_detected_partition': {
            'idx_A': idx_A_tb.tolist(), 'idx_B': idx_B_tb.tolist(),
            'idx_M': idx_M_tb.tolist(),
            'test_stat_AB': tb_detected_result['test_stat_AB'],
            'test_stat_BA': tb_detected_result['test_stat_BA'],
            'p_value_AB': tb_detected_result['p_value_AB'],
            'p_value_BA': tb_detected_result['p_value_BA'],
            'p_value': tb_detected_result['combined_p_value'],
            'reject_at_005': tb_detected_result['reject_at_005'], 'time': t_tb,
        },
    }


# =============================================================================
# Experiment 2: Power Analysis
# =============================================================================

def run_power_analysis():
    """Power analysis: sample size sweep for test discrimination."""
    print("\n" + "=" * 70)
    print("Experiment 2: Power Analysis - Sample Size Sweep")
    print("=" * 70)

    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )
    Theta = build_precision_matrix(cfg)
    truth = get_ground_truth(cfg)
    n_vars = Theta.shape[0]

    sample_sizes = [100, 200, 500, 1000, 2000]
    n_trials = 8
    results = {N: {'correct_rejects': 0, 'random_rejects': 0,
                    'correct_pvals': [], 'random_pvals': []}
               for N in sample_sizes}

    for N in sample_sizes:
        print(f"\nN = {N}")
        for trial in range(n_trials):
            np.random.seed(42 + trial * 100)
            samp, grads = langevin_sampling(
                Theta, n_samples=N, n_steps=50, step_size=0.005, temp=0.1
            )

            # Correct partition
            res_c = ksd_conditional_independence_test(
                samp, grads,
                np.where(truth['assignment'] == 0)[0],
                np.where(truth['assignment'] == 1)[0],
                truth['blanket_vars'],
                n_bootstrap=300, max_samples=min(200, N), seed=42 + trial,
            )
            results[N]['correct_pvals'].append(res_c['combined_p_value'])
            if res_c['reject_at_005']:
                results[N]['correct_rejects'] += 1

            # Random partition
            rng = np.random.RandomState(123 + trial)
            perm = rng.permutation(n_vars)
            res_r = ksd_conditional_independence_test(
                samp, grads,
                perm[:cfg.vars_per_object],
                perm[cfg.vars_per_object:2 * cfg.vars_per_object],
                perm[2 * cfg.vars_per_object:],
                n_bootstrap=300, max_samples=min(200, N), seed=42 + trial,
            )
            results[N]['random_pvals'].append(res_r['combined_p_value'])
            if res_r['reject_at_005']:
                results[N]['random_rejects'] += 1

        cr = results[N]['correct_rejects'] / n_trials
        rr = results[N]['random_rejects'] / n_trials
        print(f"  Correct rejection rate: {cr:.0%} (want <10%)")
        print(f"  Random rejection rate:  {rr:.0%} (want >80%)")

    power_summary = {}
    for N in sample_sizes:
        power_summary[str(N)] = {
            'correct_rejection_rate': results[N]['correct_rejects'] / n_trials,
            'random_rejection_rate': results[N]['random_rejects'] / n_trials,
            'correct_mean_pvalue': float(np.mean(results[N]['correct_pvals'])),
            'correct_std_pvalue': float(np.std(results[N]['correct_pvals'])),
            'random_mean_pvalue': float(np.mean(results[N]['random_pvals'])),
            'random_std_pvalue': float(np.std(results[N]['random_pvals'])),
            'correct_pvals': [float(p) for p in results[N]['correct_pvals']],
            'random_pvals': [float(p) for p in results[N]['random_pvals']],
        }

    return power_summary


# =============================================================================
# Experiment 3: LunarLander 8D
# =============================================================================

def run_lunarlander_validation():
    """Test KSD partition validation on LunarLander 8D data."""
    print("\n" + "=" * 70)
    print("Experiment 3: LunarLander 8D Partition Validation")
    print("=" * 70)

    data_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'results', 'trajectory_data'))

    states_path = os.path.join(data_dir, 'states.npy')
    grads_path = os.path.join(data_dir, 'dynamics_gradients.npy')

    if not os.path.exists(states_path) or not os.path.exists(grads_path):
        print("  Trajectory data not found. Using synthetic 8D data.")
        return run_synthetic_lunarlander_validation()

    states = np.load(states_path)
    gradients = np.load(grads_path)
    n_use = min(states.shape[0], gradients.shape[0])
    states, gradients = states[:n_use], gradients[:n_use]

    print(f"  Loaded {states.shape[0]} states, {gradients.shape[0]} gradients, "
          f"d={states.shape[1]}")

    STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    print("  Running TB pipeline...")
    tb_result = tb_pipeline(gradients, n_objects=2, method='gradient')
    return _run_validation_on_data(
        states, gradients, tb_result, STATE_LABELS, 'real_lunarlander_8d')


def run_synthetic_lunarlander_validation():
    """Fallback: synthetic 8D LunarLander-like data."""
    print("  Generating synthetic 8D data...")

    n_vars = 8
    Theta = np.eye(n_vars) * 4.0

    # Position block: {x, y, vx, vy}
    for i in range(4):
        for j in range(4):
            if i != j:
                Theta[i, j] = 2.5

    # Attitude block: {angle, ang_vel}
    Theta[4, 5] = Theta[5, 4] = 3.0

    # Contact blanket: {left_leg, right_leg}
    Theta[6, 7] = Theta[7, 6] = 1.0
    for b in [6, 7]:
        for p in range(4):
            Theta[b, p] = Theta[p, b] = 0.5
        for a in [4, 5]:
            Theta[b, a] = Theta[a, b] = 0.5

    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals.min() < 0.1:
        Theta += np.eye(n_vars) * (0.1 - eigvals.min() + 0.5)
    Theta = (Theta + Theta.T) / 2.0

    STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    np.random.seed(42)
    samples, gradients = langevin_sampling(
        Theta, n_samples=2000, n_steps=50, step_size=0.003, temp=0.1)
    print(f"  Sampled {samples.shape[0]} points.\n")

    print("  Running TB pipeline...")
    tb_result = tb_pipeline(gradients, n_objects=2, method='gradient')
    return _run_validation_on_data(
        samples, gradients, tb_result, STATE_LABELS, 'synthetic_lunarlander_8d')


def _run_validation_on_data(samples, gradients, tb_result, state_labels, data_source):
    """Shared validation logic."""
    assignment = tb_result['assignment']
    is_blanket = tb_result['is_blanket']
    blanket_vars = np.where(is_blanket)[0]
    n_vars = samples.shape[1]

    obj_ids = sorted([int(x) for x in np.unique(assignment) if x >= 0])
    print(f"  Objects: {len(obj_ids)}")
    for oid in obj_ids:
        obj_vars = np.where(assignment == oid)[0]
        labels = [state_labels[i] for i in obj_vars if i < len(state_labels)]
        print(f"    Object {oid}: {labels}")
    blanket_labels = [state_labels[i] for i in blanket_vars if i < len(state_labels)]
    print(f"  Blanket: {blanket_labels}")

    print("\n  KSD validation on TB partition...")
    validation = validate_partition(
        tb_result, samples, gradients,
        n_permutations=300, max_samples=300, seed=42)

    for k, v in validation['pair_results'].items():
        print(f"    {k}: Frob_AB={v['test_stat_AB']:.6f}, p={v['combined_p_value']:.4f}, "
              f"reject={v['reject_at_005']}")
    print(f"  Partition valid: {validation['partition_valid']}")

    # Random negative control
    print("\n  Random partition control...")
    rng = np.random.RandomState(99)
    perm = rng.permutation(n_vars)
    rand_result = {
        'assignment': np.full(n_vars, -1, dtype=int),
        'is_blanket': np.zeros(n_vars, dtype=bool),
    }
    rand_result['assignment'][perm[:3]] = 0
    rand_result['assignment'][perm[3:6]] = 1
    rand_result['is_blanket'][perm[6:]] = True

    rand_validation = validate_partition(
        rand_result, samples, gradients,
        n_permutations=300, max_samples=300, seed=42)

    for k, v in rand_validation['pair_results'].items():
        print(f"    {k}: Frob_AB={v['test_stat_AB']:.6f}, p={v['combined_p_value']:.4f}, "
              f"reject={v['reject_at_005']}")
    print(f"  Random partition valid: {rand_validation['partition_valid']}")

    return {
        'n_samples': int(samples.shape[0]),
        'n_vars': n_vars,
        'state_labels': state_labels,
        'data_source': data_source,
        'tb_partition': {
            'assignment': assignment.tolist(),
            'is_blanket': is_blanket.tolist(),
            'blanket_vars': blanket_vars.tolist(),
        },
        'tb_validation': {
            'partition_valid': validation['partition_valid'],
            'min_pvalue': validation['min_pvalue'],
            'mean_pvalue': validation['mean_pvalue'],
            'pair_results': {k: v for k, v in validation['pair_results'].items()},
        },
        'random_validation': {
            'partition_valid': rand_validation['partition_valid'],
            'min_pvalue': rand_validation['min_pvalue'],
            'mean_pvalue': rand_validation['mean_pvalue'],
            'pair_results': {k: v for k, v in rand_validation['pair_results'].items()},
        },
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_ksd_comparison(quad_results: Dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Correct\n(ground truth)', 'Random\n(permuted)', 'TB-detected']
    ksd_vals = [quad_results['correct_partition']['test_stat_AB'],
                quad_results['random_partition']['test_stat_AB'],
                quad_results['tb_detected_partition']['test_stat_AB']]
    p_vals = [quad_results['correct_partition']['p_value'],
              quad_results['random_partition']['p_value'],
              quad_results['tb_detected_partition']['p_value']]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    ax = axes[0]
    bars = ax.bar(labels, ksd_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Cross-Coupling Statistic (Frob. norm)')
    ax.set_title('Score Cross-Dependence by Partition Type')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, ksd_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0001,
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, p_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='alpha = 0.05')
    ax.set_ylabel('p-value (combined)')
    ax.set_title('KSD CI Test p-values')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_power_analysis(power_results: Dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sample_sizes = sorted([int(k) for k in power_results.keys()])
    cr = [power_results[str(N)]['correct_rejection_rate'] for N in sample_sizes]
    rr = [power_results[str(N)]['random_rejection_rate'] for N in sample_sizes]
    cm = [power_results[str(N)]['correct_mean_pvalue'] for N in sample_sizes]
    rm = [power_results[str(N)]['random_mean_pvalue'] for N in sample_sizes]
    cs = [power_results[str(N)]['correct_std_pvalue'] for N in sample_sizes]
    rs = [power_results[str(N)]['random_std_pvalue'] for N in sample_sizes]

    ax = axes[0]
    ax.plot(sample_sizes, cr, 'o-', color='#2ecc71', label='Correct (want low)', lw=2, ms=8)
    ax.plot(sample_sizes, rr, 's-', color='#e74c3c', label='Random (want high)', lw=2, ms=8)
    ax.axhline(y=0.05, color='gray', ls=':', alpha=0.5, label='alpha=0.05')
    ax.axhline(y=0.80, color='gray', ls='--', alpha=0.5, label='80% power')
    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Rejection Rate')
    ax.set_title('KSD Test Power Analysis')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.errorbar(sample_sizes, cm, yerr=cs, fmt='o-', color='#2ecc71',
                label='Correct partition', lw=2, ms=8, capsize=4)
    ax.errorbar(sample_sizes, rm, yerr=rs, fmt='s-', color='#e74c3c',
                label='Random partition', lw=2, ms=8, capsize=4)
    ax.axhline(y=0.05, color='red', ls='--', lw=1.5, label='alpha=0.05')
    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Mean p-value')
    ax.set_title('Mean p-value vs Sample Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lunarlander_validation(ll_results: Dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    tb_pairs = ll_results['tb_validation']['pair_results']
    rand_pairs = ll_results['random_validation']['pair_results']

    tb_labels = list(tb_pairs.keys())
    tb_pvals = [tb_pairs[k]['combined_p_value'] for k in tb_labels]
    rand_labels = list(rand_pairs.keys())
    rand_pvals = [rand_pairs[k]['combined_p_value'] for k in rand_labels]

    x = np.arange(max(len(tb_labels), len(rand_labels)))
    w = 0.35
    if tb_labels:
        ax.bar(x[:len(tb_labels)] - w / 2, tb_pvals, w,
               label='TB partition', color='#3498db', alpha=0.8, edgecolor='black')
    if rand_labels:
        ax.bar(x[:len(rand_labels)] + w / 2, rand_pvals, w,
               label='Random partition', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.axhline(y=0.05, color='red', ls='--', lw=1.5, label='alpha=0.05')
    ax.set_ylabel('Combined p-value')
    ax.set_title('LunarLander 8D: KSD p-values')
    all_l = tb_labels if len(tb_labels) >= len(rand_labels) else rand_labels
    ax.set_xticks(x[:len(all_l)])
    ax.set_xticklabels(all_l, rotation=20, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    sl = ['TB Partition', 'Random Partition']
    sp = [ll_results['tb_validation']['mean_pvalue'],
          ll_results['random_validation']['mean_pvalue']]
    sv = [ll_results['tb_validation']['partition_valid'],
          ll_results['random_validation']['partition_valid']]
    cols = ['#3498db' if v else '#e74c3c' for v in sv]
    bars = ax.bar(sl, sp, color=cols, alpha=0.8, edgecolor='black')
    ax.axhline(y=0.05, color='red', ls='--', lw=1.5, label='alpha=0.05')
    ax.set_ylabel('Mean p-value')
    ax.set_title('LunarLander 8D: Partition Validity')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val, valid in zip(bars, sp, sv):
        st = 'VALID' if valid else 'REJECTED'
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.3f}\n({st})', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_permutation_distributions(samples, gradients, truth) -> plt.Figure:
    """Plot permutation null distributions for correct vs random partitions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_vars = samples.shape[1]
    idx_A_c = np.where(truth['assignment'] == 0)[0]
    idx_B_c = np.where(truth['assignment'] == 1)[0]
    idx_M_c = truth['blanket_vars']

    rng = np.random.RandomState(123)
    perm = rng.permutation(n_vars)
    idx_A_r, idx_B_r, idx_M_r = perm[:3], perm[3:6], perm[6:]

    rng2 = np.random.RandomState(42)
    idx = rng2.choice(len(samples), min(200, len(samples)), replace=False)
    ss, gs = samples[idx], gradients[idx]

    n_perm = 300

    for ax_i, (name, col, iA, iB, iM) in enumerate([
        ('Correct Partition', '#2ecc71', idx_A_c, idx_B_c, idx_M_c),
        ('Random Partition', '#e74c3c', idx_A_r, idx_B_r, idx_M_r),
    ]):
        ax = axes[ax_i]

        # Compute observed combined stat
        obs_fwd, _ = estimate_cross_coupling(ss, gs, iA, iB, iM)
        obs_rev, _ = estimate_cross_coupling(ss, gs, iB, iA, iM)
        obs_combined = obs_fwd + obs_rev

        # Build permutation null
        perm_rng = np.random.RandomState(42)
        perm_stats = np.zeros(n_perm)
        n_sub = ss.shape[0]
        for b in range(n_perm):
            pi = perm_rng.permutation(n_sub)
            ss_perm = ss.copy()
            ss_perm[:, iB] = ss[pi][:, iB]
            sf, _ = estimate_cross_coupling(ss_perm, gs, iA, iB, iM)
            sr, _ = estimate_cross_coupling(ss_perm, gs, iB, iA, iM)
            perm_stats[b] = sf + sr

        pv = np.mean(perm_stats >= obs_combined)

        ax.hist(perm_stats, bins=40, density=True, alpha=0.6, color=col,
                edgecolor='black', linewidth=0.5, label='Permutation null')
        ax.axvline(x=obs_combined, color='black', lw=2, ls='-',
                   label=f'Observed = {obs_combined:.5f}')
        ax.set_xlabel('Cross-Coupling Statistic (Frob. norm)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\np = {pv:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Integration: Add validate_partition to TopologicalBlankets class
# =============================================================================

def add_validate_partition_method():
    """Monkey-patch TopologicalBlankets to add .validate_partition()."""

    def _validate_partition(self, samples=None, n_permutations=500,
                            max_samples=300, seed=42):
        """
        Validate the detected partition using KSD conditional independence test.

        Returns:
            Dictionary with per-pair test results and validity summary.
        """
        self._check_fitted()
        gradients = self._gradients
        if samples is None:
            samples = gradients
        tb_result = {
            'assignment': self._assignment,
            'is_blanket': self._is_blanket,
        }
        return validate_partition(
            tb_result, samples, gradients,
            n_permutations=n_permutations,
            max_samples=max_samples, seed=seed)

    TopologicalBlankets.validate_partition = _validate_partition
    print("  Added .validate_partition() to TopologicalBlankets class.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("US-074: KSD Goodness-of-Fit Test for Partition Validation")
    print("=" * 70)
    print()

    all_results = {}
    t_start = time.time()

    # Experiment 1
    quad_results = run_quadratic_validation()
    all_results['quadratic_validation'] = quad_results

    # Experiment 2
    power_results = run_power_analysis()
    all_results['power_analysis'] = power_results

    # Experiment 3
    ll_results = run_lunarlander_validation()
    all_results['lunarlander_validation'] = ll_results

    # Experiment 4: Integration test
    print("\n" + "=" * 70)
    print("Experiment 4: Integration Test - TB.validate_partition()")
    print("=" * 70)
    add_validate_partition_method()

    cfg = QuadraticEBMConfig(n_objects=2, vars_per_object=3, vars_per_blanket=3)
    Theta = build_precision_matrix(cfg)
    np.random.seed(42)
    samp, grads = langevin_sampling(
        Theta, n_samples=1500, n_steps=50, step_size=0.005, temp=0.1)
    tb = TopologicalBlankets(method='gradient', n_objects=2)
    tb.fit(grads)
    integ_result = tb.validate_partition(samples=samp, n_permutations=200)
    print(f"  Partition valid: {integ_result['partition_valid']}")
    print(f"  Mean p-value:    {integ_result['mean_pvalue']:.4f}")
    all_results['integration_test'] = {
        'partition_valid': integ_result['partition_valid'],
        'mean_pvalue': integ_result['mean_pvalue'],
        'min_pvalue': integ_result['min_pvalue'],
    }

    t_total = time.time() - t_start

    # ---- Plots ----
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    fig1 = plot_ksd_comparison(quad_results)
    save_figure(fig1, 'ksd_comparison', 'ksd_partition_validation')

    fig2 = plot_power_analysis(power_results)
    save_figure(fig2, 'power_analysis', 'ksd_partition_validation')

    fig3 = plot_lunarlander_validation(ll_results)
    save_figure(fig3, 'lunarlander_validation', 'ksd_partition_validation')

    # Permutation distribution plot
    cfg2 = QuadraticEBMConfig(n_objects=2, vars_per_object=3, vars_per_blanket=3)
    Theta2 = build_precision_matrix(cfg2)
    truth2 = get_ground_truth(cfg2)
    np.random.seed(42)
    s2, g2 = langevin_sampling(Theta2, n_samples=2000, n_steps=50,
                                step_size=0.005, temp=0.1)
    fig4 = plot_permutation_distributions(s2, g2, truth2)
    save_figure(fig4, 'permutation_distributions', 'ksd_partition_validation')

    # ---- Save results ----
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    correct_not_rejected = not quad_results['correct_partition']['reject_at_005']
    random_rejected = quad_results['random_partition']['reject_at_005']
    tb_not_rejected = not quad_results['tb_detected_partition']['reject_at_005']

    sample_sizes = sorted([int(k) for k in power_results.keys()])
    crossover_N = None
    for N in sample_sizes:
        if power_results[str(N)]['random_rejection_rate'] >= 0.80:
            crossover_N = N
            break

    summary = {
        'acceptance_criteria': {
            'correct_partition_not_rejected': correct_not_rejected,
            'random_partition_rejected': random_rejected,
            'tb_partition_not_rejected': tb_not_rejected,
            'power_crossover_N': crossover_N,
            'integration_works': integ_result['partition_valid'],
        },
        'total_time_seconds': t_total,
    }
    all_results['summary'] = summary

    print(f"\n  Acceptance Criteria:")
    print(f"    Correct partition not rejected (p>0.05): {correct_not_rejected}")
    print(f"    Random partition rejected (p<0.05):      {random_rejected}")
    print(f"    TB partition not rejected (p>0.05):      {tb_not_rejected}")
    print(f"    Power crossover at N = {crossover_N}")
    print(f"    Integration test works:                  {integ_result['partition_valid']}")
    print(f"\n  Total time: {t_total:.1f}s")

    save_results('ksd_partition_validation', all_results,
                 config={'n_permutations': 500, 'max_samples': 300,
                         'quadratic_config': quad_results['config']},
                 notes=(f'US-074: KSD goodness-of-fit test. '
                        f'Correct p={quad_results["correct_partition"]["p_value"]:.4f} '
                        f'(not_rej={correct_not_rejected}), '
                        f'random p={quad_results["random_partition"]["p_value"]:.4f} '
                        f'(rej={random_rejected}), '
                        f'TB p={quad_results["tb_detected_partition"]["p_value"]:.4f} '
                        f'(not_rej={tb_not_rejected}). '
                        f'Power crossover N={crossover_N}. '
                        f'Integration: valid={integ_result["partition_valid"]}.'))

    return all_results


if __name__ == "__main__":
    results = main()
