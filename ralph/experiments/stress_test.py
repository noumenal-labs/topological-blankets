"""
Stress Test: Topological Blankets on Large/Difficult Configurations
===================================================================

Tests the method under conditions beyond the standard Level 1 sweep:
1. Large scale: up to 10 objects, 20 vars/object (200+ dimensions)
2. Hard regime: very weak blanket coupling, high noise
3. Asymmetric: unequal object sizes
4. Many blanket variables: blanket size comparable to object size
5. Hessian estimation validation: compare estimated vs true Hessian
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, topological_blankets, compute_metrics,
    compute_geometric_features, gradient
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


def run_large_scale_test():
    """Test 1: Scale to large systems (up to 10 objects, 20 vars/object)."""
    print("\n" + "=" * 70)
    print("STRESS TEST 1: Large-Scale Systems")
    print("=" * 70)

    configs = [
        # (n_objects, vars_per_object, vars_per_blanket, label)
        (5, 5, 5, "5obj x 5var"),
        (5, 10, 5, "5obj x 10var"),
        (6, 8, 6, "6obj x 8var"),
        (8, 5, 8, "8obj x 5var"),
        (8, 10, 8, "8obj x 10var"),
        (10, 5, 10, "10obj x 5var"),
        (10, 10, 10, "10obj x 10var"),
        (10, 15, 10, "10obj x 15var"),
        (10, 20, 10, "10obj x 20var"),
    ]

    results = {}
    n_trials = 3

    for n_obj, vpo, vpb, label in configs:
        dim = n_obj * vpo + vpb
        print(f"\n--- {label} (dim={dim}) ---")

        trial_aris = []
        trial_f1s = []
        total_time = 0

        for trial in range(n_trials):
            cfg = QuadraticEBMConfig(
                n_objects=n_obj,
                vars_per_object=vpo,
                vars_per_blanket=vpb,
                intra_strength=6.0,
                blanket_strength=0.8,
            )
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            t0 = time.time()

            # Scale samples with dimension
            n_samples = max(3000, dim * 100)
            samples, gradients = langevin_sampling(
                Theta, n_samples=n_samples, n_steps=50,
                step_size=0.003, temp=0.1
            )

            tc_result = topological_blankets(gradients, n_objects=n_obj)
            elapsed = time.time() - t0
            total_time += elapsed

            m = compute_metrics(tc_result, truth)
            trial_aris.append(m['object_ari'])
            trial_f1s.append(m['blanket_f1'])

        mean_ari = float(np.mean(trial_aris))
        mean_f1 = float(np.mean(trial_f1s))
        mean_time = total_time / n_trials

        results[label] = {
            'n_objects': n_obj,
            'vars_per_object': vpo,
            'vars_per_blanket': vpb,
            'total_dimension': dim,
            'mean_ari': mean_ari,
            'std_ari': float(np.std(trial_aris)),
            'mean_f1': mean_f1,
            'std_f1': float(np.std(trial_f1s)),
            'mean_time_s': round(mean_time, 2),
            'per_trial_ari': [float(a) for a in trial_aris],
            'per_trial_f1': [float(f) for f in trial_f1s],
        }

        status = "OK" if mean_ari >= 0.9 else ("WARN" if mean_ari >= 0.7 else "FAIL")
        print(f"  ARI={mean_ari:.3f} +/- {np.std(trial_aris):.3f}, "
              f"F1={mean_f1:.3f}, time={mean_time:.1f}s  [{status}]")

    save_results('stress_large_scale', results, {
        'n_trials': n_trials,
        'step_size': 0.003,
        'temp': 0.1,
    }, notes='Stress test: scaling to 10 objects, 20 vars/object, 200+ dims')

    return results


def run_hard_regime_test():
    """Test 2: Difficult detection regimes."""
    print("\n" + "=" * 70)
    print("STRESS TEST 2: Hard Detection Regimes")
    print("=" * 70)

    # Test: very weak coupling, coupling approaching intra-strength, high temp
    scenarios = [
        # (blanket_str, intra_str, temp, label)
        (0.05, 6.0, 0.1, "Very weak coupling (0.05)"),
        (0.01, 6.0, 0.1, "Extremely weak coupling (0.01)"),
        (3.0, 6.0, 0.1, "Strong coupling (3.0, ratio 2:1)"),
        (5.0, 6.0, 0.1, "Near-equal coupling (5.0, ratio 1.2:1)"),
        (0.8, 6.0, 5.0, "High temperature (T=5.0)"),
        (0.8, 6.0, 10.0, "Very high temperature (T=10.0)"),
        (0.8, 2.0, 0.1, "Weak intra-coupling (2.0)"),
        (0.8, 1.5, 0.1, "Very weak intra (1.5, ratio 1.9:1)"),
        (0.05, 6.0, 5.0, "Worst case: weak coupling + high T"),
    ]

    results = {}
    n_trials = 5

    for b_str, i_str, temp, label in scenarios:
        print(f"\n--- {label} ---")
        print(f"    blanket_str={b_str}, intra_str={i_str}, temp={temp}, "
              f"ratio={i_str/b_str:.1f}:1")

        trial_aris = []
        trial_f1s = []

        for trial in range(n_trials):
            cfg = QuadraticEBMConfig(
                n_objects=2,
                vars_per_object=3,
                vars_per_blanket=3,
                intra_strength=i_str,
                blanket_strength=b_str,
            )
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            samples, gradients = langevin_sampling(
                Theta, n_samples=5000, n_steps=50,
                step_size=0.003, temp=temp
            )

            tc_result = topological_blankets(gradients, n_objects=2)
            m = compute_metrics(tc_result, truth)
            trial_aris.append(m['object_ari'])
            trial_f1s.append(m['blanket_f1'])

        mean_ari = float(np.mean(trial_aris))
        mean_f1 = float(np.mean(trial_f1s))

        results[label] = {
            'blanket_strength': b_str,
            'intra_strength': i_str,
            'temperature': temp,
            'coupling_ratio': round(i_str / b_str, 2),
            'mean_ari': mean_ari,
            'std_ari': float(np.std(trial_aris)),
            'mean_f1': mean_f1,
            'std_f1': float(np.std(trial_f1s)),
            'per_trial_ari': [float(a) for a in trial_aris],
            'per_trial_f1': [float(f) for f in trial_f1s],
        }

        status = "OK" if mean_ari >= 0.9 else ("WARN" if mean_ari >= 0.7 else "FAIL")
        print(f"  ARI={mean_ari:.3f} +/- {np.std(trial_aris):.3f}, "
              f"F1={mean_f1:.3f}  [{status}]")

    save_results('stress_hard_regime', results, {
        'n_trials': n_trials, 'n_objects': 2, 'vars_per_object': 3,
    }, notes='Stress test: extreme coupling ratios, temperatures, weak intra')

    return results


def run_asymmetric_test():
    """Test 3: Asymmetric object sizes (non-uniform vars_per_object)."""
    print("\n" + "=" * 70)
    print("STRESS TEST 3: Asymmetric Object Sizes")
    print("=" * 70)

    # Custom precision matrices with unequal object sizes
    scenarios = [
        # (object_sizes, blanket_size, label)
        ([3, 8], 3, "2 objects: 3+8 vars"),
        ([2, 2, 10], 3, "3 objects: 2+2+10 vars"),
        ([3, 5, 7], 4, "3 objects: 3+5+7 vars"),
        ([2, 3, 4, 5], 4, "4 objects: 2+3+4+5 vars"),
        ([5, 5, 5, 5, 5], 5, "5 equal objects: 5x5 vars"),
        ([2, 2, 2, 2, 2, 2, 2, 2], 4, "8 small objects: 2x8 vars"),
    ]

    results = {}
    n_trials = 5

    for obj_sizes, b_size, label in scenarios:
        n_obj = len(obj_sizes)
        total_internal = sum(obj_sizes)
        dim = total_internal + b_size
        print(f"\n--- {label} (dim={dim}) ---")

        trial_aris = []
        trial_f1s = []

        for trial in range(n_trials):
            # Build custom precision matrix for asymmetric case
            n = dim
            Theta = np.zeros((n, n))
            intra_str = 6.0
            blanket_str = 0.8

            # Object blocks (variable sizes)
            start = 0
            gt_assignment = np.full(n, -1)
            for obj_idx, obj_size in enumerate(obj_sizes):
                end = start + obj_size
                Theta[start:end, start:end] = intra_str
                np.fill_diagonal(Theta[start:end, start:end],
                                intra_str * obj_size)
                gt_assignment[start:end] = obj_idx
                start = end

            # Blanket block
            blanket_start = total_internal
            Theta[blanket_start:, blanket_start:] = 1.0
            np.fill_diagonal(Theta[blanket_start:, blanket_start:], b_size)

            # Cross couplings
            start = 0
            for obj_idx, obj_size in enumerate(obj_sizes):
                end = start + obj_size
                Theta[start:end, blanket_start:] = blanket_str
                Theta[blanket_start:, start:end] = blanket_str
                start = end

            Theta = (Theta + Theta.T) / 2.0
            eigvals = np.linalg.eigvalsh(Theta)
            if eigvals.min() < 0.1:
                Theta += np.eye(n) * (0.1 - eigvals.min() + 0.1)

            truth = {
                'assignment': gt_assignment,
                'blanket_vars': np.arange(blanket_start, n),
                'internal_vars': np.arange(blanket_start),
                'is_blanket': gt_assignment == -1,
                'n_objects': n_obj,
            }

            np.random.seed(42 + trial)
            n_samples = max(3000, dim * 80)
            samples, gradients = langevin_sampling(
                Theta, n_samples=n_samples, n_steps=50,
                step_size=0.003, temp=0.1
            )

            tc_result = topological_blankets(gradients, n_objects=n_obj)
            m = compute_metrics(tc_result, truth)
            trial_aris.append(m['object_ari'])
            trial_f1s.append(m['blanket_f1'])

        mean_ari = float(np.mean(trial_aris))
        mean_f1 = float(np.mean(trial_f1s))

        results[label] = {
            'object_sizes': obj_sizes,
            'blanket_size': b_size,
            'total_dimension': dim,
            'n_objects': n_obj,
            'mean_ari': mean_ari,
            'std_ari': float(np.std(trial_aris)),
            'mean_f1': mean_f1,
            'std_f1': float(np.std(trial_f1s)),
            'per_trial_ari': [float(a) for a in trial_aris],
            'per_trial_f1': [float(f) for f in trial_f1s],
        }

        status = "OK" if mean_ari >= 0.9 else ("WARN" if mean_ari >= 0.7 else "FAIL")
        print(f"  ARI={mean_ari:.3f} +/- {np.std(trial_aris):.3f}, "
              f"F1={mean_f1:.3f}  [{status}]")

    save_results('stress_asymmetric', results, {
        'n_trials': n_trials, 'intra_strength': 6.0, 'blanket_strength': 0.8,
    }, notes='Stress test: asymmetric object sizes')

    return results


def run_blanket_ratio_test():
    """Test 4: Vary the ratio of blanket to internal variables."""
    print("\n" + "=" * 70)
    print("STRESS TEST 4: Blanket Size Ratio")
    print("=" * 70)

    # What happens when blanket is very large or very small relative to internals?
    configs = [
        # (n_objects, vars_per_object, vars_per_blanket, label)
        (2, 5, 1, "Tiny blanket: 1 var"),
        (2, 5, 2, "Small blanket: 2 vars"),
        (2, 5, 5, "Equal blanket: 5 vars (=vpo)"),
        (2, 5, 8, "Large blanket: 8 vars"),
        (2, 5, 10, "Huge blanket: 10 vars (=total internal)"),
        (2, 5, 15, "Oversized blanket: 15 vars (> internal)"),
        (3, 4, 1, "3obj, tiny blanket: 1 var"),
        (3, 4, 6, "3obj, equal blanket: 6 vars (=vpo*1.5)"),
        (3, 4, 12, "3obj, huge blanket: 12 vars (=total internal)"),
    ]

    results = {}
    n_trials = 5

    for n_obj, vpo, vpb, label in configs:
        dim = n_obj * vpo + vpb
        blanket_ratio = vpb / (n_obj * vpo)
        print(f"\n--- {label} (dim={dim}, blanket_ratio={blanket_ratio:.2f}) ---")

        trial_aris = []
        trial_f1s = []

        for trial in range(n_trials):
            cfg = QuadraticEBMConfig(
                n_objects=n_obj,
                vars_per_object=vpo,
                vars_per_blanket=vpb,
                intra_strength=6.0,
                blanket_strength=0.8,
            )
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            samples, gradients = langevin_sampling(
                Theta, n_samples=5000, n_steps=50,
                step_size=0.003, temp=0.1
            )

            tc_result = topological_blankets(gradients, n_objects=n_obj)
            m = compute_metrics(tc_result, truth)
            trial_aris.append(m['object_ari'])
            trial_f1s.append(m['blanket_f1'])

        mean_ari = float(np.mean(trial_aris))
        mean_f1 = float(np.mean(trial_f1s))

        results[label] = {
            'n_objects': n_obj,
            'vars_per_object': vpo,
            'vars_per_blanket': vpb,
            'total_dimension': dim,
            'blanket_ratio': round(blanket_ratio, 3),
            'mean_ari': mean_ari,
            'std_ari': float(np.std(trial_aris)),
            'mean_f1': mean_f1,
            'std_f1': float(np.std(trial_f1s)),
            'per_trial_ari': [float(a) for a in trial_aris],
            'per_trial_f1': [float(f) for f in trial_f1s],
        }

        # Note: when blanket > internal, minority-group heuristic will
        # assign INTERNAL as blanket, so we expect failure
        status = "OK" if mean_ari >= 0.9 else ("WARN" if mean_ari >= 0.7 else "FAIL")
        print(f"  ARI={mean_ari:.3f} +/- {np.std(trial_aris):.3f}, "
              f"F1={mean_f1:.3f}, blanket_ratio={blanket_ratio:.2f}  [{status}]")

    save_results('stress_blanket_ratio', results, {
        'n_trials': n_trials, 'intra_strength': 6.0, 'blanket_strength': 0.8,
    }, notes='Stress test: blanket size from 1 var to > total internal')

    return results


def run_hessian_validation():
    """Test 5: Compare estimated Hessian (gradient covariance) vs true Hessian."""
    print("\n" + "=" * 70)
    print("STRESS TEST 5: Hessian Estimation Validation")
    print("=" * 70)

    configs = [
        (2, 3, 3, 3000, "Small (dim=9, 3k samples)"),
        (2, 3, 3, 10000, "Small (dim=9, 10k samples)"),
        (2, 3, 3, 50000, "Small (dim=9, 50k samples)"),
        (3, 5, 3, 5000, "Medium (dim=18, 5k samples)"),
        (3, 5, 3, 20000, "Medium (dim=18, 20k samples)"),
        (5, 5, 5, 5000, "Large (dim=30, 5k samples)"),
        (5, 5, 5, 30000, "Large (dim=30, 30k samples)"),
    ]

    results = {}

    for n_obj, vpo, vpb, n_samples, label in configs:
        dim = n_obj * vpo + vpb
        print(f"\n--- {label} ---")

        cfg = QuadraticEBMConfig(
            n_objects=n_obj, vars_per_object=vpo, vars_per_blanket=vpb,
            intra_strength=6.0, blanket_strength=0.8,
        )
        Theta = build_precision_matrix(cfg)

        np.random.seed(42)
        t0 = time.time()
        samples, gradients = langevin_sampling(
            Theta, n_samples=n_samples, n_steps=50,
            step_size=0.003, temp=0.1
        )
        elapsed = time.time() - t0

        # Estimated Hessian from gradient covariance
        H_est = np.cov(gradients.T)

        # True Hessian is Theta itself (for quadratic energy)
        H_true = Theta

        # For quadratic E(x) = 1/2 x^T Theta x:
        # grad = Theta @ x, so Cov(grad) = Theta @ Cov(x) @ Theta^T
        # At equilibrium, Cov(x) = temp * Theta^{-1}
        # So Cov(grad) = temp * Theta @ Theta^{-1} @ Theta^T = temp * Theta
        # Therefore H_est should approximate temp * Theta
        temp = 0.1

        # Normalize both for comparison
        H_est_norm = H_est / (np.max(np.abs(H_est)) + 1e-10)
        H_true_norm = H_true / (np.max(np.abs(H_true)) + 1e-10)

        # Metrics
        frobenius_error = np.linalg.norm(H_est_norm - H_true_norm, 'fro')
        frobenius_rel = frobenius_error / np.linalg.norm(H_true_norm, 'fro')

        # Check if sparsity pattern matches (which entries are large)
        threshold = 0.1
        est_pattern = np.abs(H_est_norm) > threshold
        true_pattern = np.abs(H_true_norm) > threshold
        pattern_match = np.mean(est_pattern == true_pattern)

        # Check if off-diagonal structure preserved
        # The coupling matrix (normalized off-diagonal) is what matters for clustering
        D_est = np.sqrt(np.diag(H_est)) + 1e-8
        coupling_est = np.abs(H_est) / np.outer(D_est, D_est)
        np.fill_diagonal(coupling_est, 0)

        D_true = np.sqrt(np.diag(H_true)) + 1e-8
        coupling_true = np.abs(H_true) / np.outer(D_true, D_true)
        np.fill_diagonal(coupling_true, 0)

        coupling_corr = np.corrcoef(coupling_est.flatten(),
                                     coupling_true.flatten())[0, 1]

        results[label] = {
            'n_objects': n_obj,
            'vars_per_object': vpo,
            'total_dimension': dim,
            'n_samples': n_samples,
            'frobenius_relative_error': round(float(frobenius_rel), 4),
            'sparsity_pattern_match': round(float(pattern_match), 4),
            'coupling_correlation': round(float(coupling_corr), 4),
            'sampling_time_s': round(elapsed, 2),
        }

        print(f"  Frobenius rel error: {frobenius_rel:.4f}")
        print(f"  Sparsity pattern match: {pattern_match:.4f}")
        print(f"  Coupling matrix correlation: {coupling_corr:.4f}")
        print(f"  Sampling time: {elapsed:.1f}s")

    save_results('stress_hessian_validation', results, {
        'intra_strength': 6.0, 'blanket_strength': 0.8, 'temp': 0.1,
    }, notes='Stress test: gradient covariance vs true Hessian')

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("TOPOLOGICAL BLANKETS: COMPREHENSIVE STRESS TEST")
    print("=" * 70)

    all_results = {}

    all_results['large_scale'] = run_large_scale_test()
    all_results['hard_regime'] = run_hard_regime_test()
    all_results['asymmetric'] = run_asymmetric_test()
    all_results['blanket_ratio'] = run_blanket_ratio_test()
    all_results['hessian_validation'] = run_hessian_validation()

    # Final summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    for category, res in all_results.items():
        print(f"\n{category.upper()}:")
        for label, data in res.items():
            if 'mean_ari' in data:
                ari = data['mean_ari']
                f1 = data['mean_f1']
                status = "PASS" if ari >= 0.9 else ("WARN" if ari >= 0.7 else "FAIL")
                dim = data.get('total_dimension', data.get('n_samples', '?'))
                print(f"  [{status}] {label}: ARI={ari:.3f}, F1={f1:.3f}")
            elif 'coupling_correlation' in data:
                corr = data['coupling_correlation']
                status = "PASS" if corr >= 0.95 else ("WARN" if corr >= 0.8 else "FAIL")
                print(f"  [{status}] {label}: coupling_corr={corr:.4f}, "
                      f"frob_err={data['frobenius_relative_error']:.4f}")

    print("\n" + "=" * 70)
    print("ALL STRESS TESTS COMPLETE")
    print("=" * 70)
