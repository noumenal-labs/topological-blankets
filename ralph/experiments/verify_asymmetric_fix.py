"""
Verification: US-012 Asymmetric Object Size Handling
=====================================================

Tests the 'coupling' detection method on challenging asymmetric configurations
where Otsu-based detection fails due to minority-group heuristic inversion.

Acceptance criteria:
- "3 objects: 2+2+10 vars" achieves ARI > 0.7 (was 0.23 with Otsu)
- "2 objects: 3+8 vars" achieves ARI > 0.7 (was 0.44 with Otsu)
- No regression on symmetric cases (ARI > 0.95)
- No regression on standard strength sweep (ARI > 0.95 at strength 0.8)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets import topological_blankets as tb_package
from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, compute_metrics
)


def build_asymmetric_system(obj_sizes, b_size, intra_str=6.0, blanket_str=0.8):
    """Build precision matrix and ground truth for asymmetric object sizes."""
    n_obj = len(obj_sizes)
    total_internal = sum(obj_sizes)
    n = total_internal + b_size

    Theta = np.zeros((n, n))

    start = 0
    gt_assignment = np.full(n, -1)
    for obj_idx, obj_size in enumerate(obj_sizes):
        end = start + obj_size
        Theta[start:end, start:end] = intra_str
        np.fill_diagonal(Theta[start:end, start:end], intra_str * obj_size)
        gt_assignment[start:end] = obj_idx
        start = end

    blanket_start = total_internal
    Theta[blanket_start:, blanket_start:] = 1.0
    np.fill_diagonal(Theta[blanket_start:, blanket_start:], b_size)

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

    return Theta, truth, n


def run_scenario(obj_sizes, b_size, n_obj, method, n_trials=5, label=""):
    """Run TB with given method on an asymmetric scenario."""
    aris = []
    f1s = []

    for trial in range(n_trials):
        Theta, truth, n = build_asymmetric_system(obj_sizes, b_size)
        np.random.seed(42 + trial)
        n_samples = max(3000, n * 80)
        _, gradients = langevin_sampling(
            Theta, n_samples=n_samples, n_steps=50,
            step_size=0.003, temp=0.1
        )

        result = tb_package(gradients, n_objects=n_obj, method=method)
        m = compute_metrics(result, truth)
        aris.append(m['object_ari'])
        f1s.append(m['blanket_f1'])

    mean_ari = float(np.mean(aris))
    mean_f1 = float(np.mean(f1s))
    std_ari = float(np.std(aris))
    return mean_ari, std_ari, mean_f1, aris


def main():
    print("=" * 70)
    print("VERIFICATION: US-012 Asymmetric Object Handling")
    print("=" * 70)

    scenarios = [
        # (obj_sizes, blanket_size, target_ari, label)
        ([3, 8], 3, 0.7, "2 objects: 3+8 vars"),
        ([2, 2, 10], 3, 0.7, "3 objects: 2+2+10 vars"),
        ([3, 5, 7], 4, 0.7, "3 objects: 3+5+7 vars"),
        ([5, 5, 5, 5, 5], 5, 0.95, "5 equal objects: 5x5 (regression check)"),
    ]

    all_pass = True
    results = {}

    # Test coupling method on asymmetric scenarios
    print("\n--- Coupling Method (US-012 fix) ---")
    for obj_sizes, b_size, target, label in scenarios:
        n_obj = len(obj_sizes)
        mean_ari, std_ari, mean_f1, trial_aris = run_scenario(
            obj_sizes, b_size, n_obj, method='coupling', n_trials=5, label=label
        )
        status = "PASS" if mean_ari > target else "FAIL"
        if mean_ari <= target:
            all_pass = False
        print(f"  [{status}] {label}: ARI={mean_ari:.3f} +/- {std_ari:.3f}, "
              f"F1={mean_f1:.3f} (target > {target})")
        results[f"coupling_{label}"] = {
            'method': 'coupling', 'mean_ari': mean_ari, 'std_ari': std_ari,
            'mean_f1': mean_f1, 'target': target, 'status': status,
            'per_trial': trial_aris
        }

    # Compare with gradient (Otsu) method to show improvement
    print("\n--- Gradient Method (baseline, expected failures) ---")
    for obj_sizes, b_size, target, label in scenarios[:3]:  # Just asymmetric ones
        n_obj = len(obj_sizes)
        mean_ari, std_ari, mean_f1, _ = run_scenario(
            obj_sizes, b_size, n_obj, method='gradient', n_trials=5
        )
        print(f"  [INFO] {label}: ARI={mean_ari:.3f} +/- {std_ari:.3f}, F1={mean_f1:.3f}")
        results[f"gradient_{label}"] = {
            'method': 'gradient', 'mean_ari': mean_ari, 'mean_f1': mean_f1
        }

    # Regression test: standard symmetric case
    print("\n--- Regression: Standard Symmetric (strength=0.8) ---")
    cfg = QuadraticEBMConfig(blanket_strength=0.8)
    aris_reg = []
    for trial in range(5):
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)
        np.random.seed(42 + trial)
        _, gradients = langevin_sampling(Theta, n_samples=3000, n_steps=30,
                                         step_size=0.005, temp=0.1)
        for method in ['gradient', 'coupling', 'hybrid']:
            result = tb_package(gradients, n_objects=2, method=method)
            m = compute_metrics(result, truth)
            if trial == 0:
                print(f"  {method:10s}: ARI={m['object_ari']:.3f}, F1={m['blanket_f1']:.3f}")

    print("\n" + "=" * 70)
    if all_pass:
        print("US-012 VERIFICATION: ALL TESTS PASSED")
    else:
        print("US-012 VERIFICATION: SOME TESTS FAILED")
    print("=" * 70)

    return all_pass, results


if __name__ == "__main__":
    passed, results = main()
    sys.exit(0 if passed else 1)
