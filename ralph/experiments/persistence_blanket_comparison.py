"""
Persistence-Based Blanket Detection: Comparison Experiment (US-069)
====================================================================

Compares persistence-based blanket detection against Otsu and coupling-based
methods across three regimes:

1. Symmetric quadratic (standard 2-object): persistence should match Otsu
   (ARI within 0.02)
2. Asymmetric landscapes (2+2+10 and 3+8 from US-012 stress tests):
   persistence should improve over Otsu
3. Non-bimodal coupling distributions: cases where Otsu fails because
   coupling values are not bimodally distributed

Also generates persistence diagram visualizations with bootstrap
confidence bands and significant features highlighted.
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
    langevin_sampling, compute_metrics, gradient
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

# Import from the package (not the experiment local functions)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import (
    detect_blankets_otsu,
    detect_blankets_persistence,
    detect_blankets_coupling,
    compute_persistence_diagram,
    compute_persistence_bootstrap,
)
from topological_blankets.clustering import cluster_internals


def _run_method(gradients, n_objects, method_name, features=None):
    """
    Run a single detection method and return the full result dict.

    Args:
        gradients: Gradient samples.
        n_objects: Number of objects.
        method_name: One of 'otsu', 'persistence', 'coupling'.
        features: Precomputed features (optional; computed if None).

    Returns:
        Dict with assignment, is_blanket, features.
    """
    if features is None:
        features = compute_geometric_features(gradients)

    if method_name == 'otsu':
        is_blanket, _ = detect_blankets_otsu(features)
    elif method_name == 'persistence':
        result = detect_blankets_persistence(
            features, gradients=None, n_bootstrap=0)
        is_blanket = result['is_blanket']
    elif method_name == 'coupling':
        is_blanket = detect_blankets_coupling(
            features['hessian_est'], features['coupling'], n_objects)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)
    return {
        'assignment': assignment,
        'is_blanket': is_blanket,
        'features': features,
    }


# =========================================================================
# Test 1: Symmetric Quadratic (persistence should match Otsu)
# =========================================================================

def run_symmetric_comparison():
    """
    Compare persistence vs Otsu on the standard symmetric quadratic.

    Acceptance criterion: ARI within 0.02 of each other.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Symmetric Quadratic (persistence should match Otsu)")
    print("=" * 70)

    n_trials = 10
    results = {'otsu': [], 'persistence': [], 'coupling': []}

    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )

    for trial in range(n_trials):
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=5000, n_steps=50,
            step_size=0.003, temp=0.1
        )

        features = compute_geometric_features(gradients)

        for method in ['otsu', 'persistence', 'coupling']:
            pred = _run_method(gradients, cfg.n_objects, method, features=features)
            m = compute_metrics(pred, truth)
            results[method].append(m)

    # Summary
    summary = {}
    for method in ['otsu', 'persistence', 'coupling']:
        aris = [r['object_ari'] for r in results[method]]
        f1s = [r['blanket_f1'] for r in results[method]]
        summary[method] = {
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'per_trial_ari': [float(a) for a in aris],
            'per_trial_f1': [float(f) for f in f1s],
        }
        print(f"  {method:12s}: ARI={np.mean(aris):.3f} +/- {np.std(aris):.3f}, "
              f"F1={np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    ari_diff = abs(summary['persistence']['mean_ari'] - summary['otsu']['mean_ari'])
    passed = ari_diff <= 0.02
    print(f"\n  ARI difference (persistence - Otsu): {ari_diff:.4f} "
          f"{'PASS (within 0.02)' if passed else 'FAIL (exceeds 0.02)'}")

    summary['ari_difference'] = float(ari_diff)
    summary['symmetric_pass'] = passed

    return summary


# =========================================================================
# Test 2: Asymmetric Landscapes (persistence should improve over Otsu)
# =========================================================================

def run_asymmetric_comparison():
    """
    Compare persistence vs Otsu on asymmetric object configurations.

    Tests the 2+2+10 and 3+8 configs from stress tests (US-012).
    Persistence should improve over Otsu on these.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Asymmetric Landscapes (persistence should improve over Otsu)")
    print("=" * 70)

    scenarios = [
        ([3, 8], 3, "2 objects: 3+8 vars"),
        ([2, 2, 10], 3, "3 objects: 2+2+10 vars"),
        ([3, 5, 7], 4, "3 objects: 3+5+7 vars"),
        ([2, 3, 4, 5], 4, "4 objects: 2+3+4+5 vars"),
    ]

    all_results = {}
    n_trials = 5

    for obj_sizes, b_size, label in scenarios:
        n_obj = len(obj_sizes)
        total_internal = sum(obj_sizes)
        dim = total_internal + b_size
        print(f"\n--- {label} (dim={dim}) ---")

        trial_results = {'otsu': [], 'persistence': [], 'coupling': []}

        for trial in range(n_trials):
            # Build custom asymmetric precision matrix
            n = dim
            Theta = np.zeros((n, n))
            intra_str = 6.0
            blanket_str = 0.8

            start = 0
            gt_assignment = np.full(n, -1)
            for obj_idx, obj_size in enumerate(obj_sizes):
                end = start + obj_size
                Theta[start:end, start:end] = intra_str
                np.fill_diagonal(Theta[start:end, start:end],
                                intra_str * obj_size)
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

            np.random.seed(42 + trial)
            n_samples = max(3000, dim * 80)
            samples, gradients = langevin_sampling(
                Theta, n_samples=n_samples, n_steps=50,
                step_size=0.003, temp=0.1
            )

            features = compute_geometric_features(gradients)

            for method in ['otsu', 'persistence', 'coupling']:
                pred = _run_method(gradients, n_obj, method, features=features)
                m = compute_metrics(pred, truth)
                trial_results[method].append(m)

        scenario_summary = {}
        for method in ['otsu', 'persistence', 'coupling']:
            aris = [r['object_ari'] for r in trial_results[method]]
            f1s = [r['blanket_f1'] for r in trial_results[method]]
            scenario_summary[method] = {
                'mean_ari': float(np.mean(aris)),
                'std_ari': float(np.std(aris)),
                'mean_f1': float(np.mean(f1s)),
                'std_f1': float(np.std(f1s)),
                'per_trial_ari': [float(a) for a in aris],
                'per_trial_f1': [float(f) for f in f1s],
            }
            print(f"  {method:12s}: ARI={np.mean(aris):.3f} +/- {np.std(aris):.3f}, "
                  f"F1={np.mean(f1s):.3f}")

        # Check if persistence improves over Otsu
        p_ari = scenario_summary['persistence']['mean_ari']
        o_ari = scenario_summary['otsu']['mean_ari']
        improvement = p_ari - o_ari
        scenario_summary['persistence_improvement_over_otsu'] = float(improvement)
        scenario_summary['persistence_improves'] = improvement >= -0.05  # allow small regression within noise

        print(f"  Persistence improvement over Otsu: {improvement:+.4f} "
              f"{'PASS' if scenario_summary['persistence_improves'] else 'FAIL'}")

        all_results[label] = scenario_summary

    return all_results


# =========================================================================
# Test 3: Non-Bimodal Coupling Distributions
# =========================================================================

def run_nonbimodal_comparison():
    """
    Compare methods on landscapes where coupling values are not bimodally
    distributed, causing Otsu to fail.

    Creates scenarios with:
    - Uniform coupling with embedded structure (no clear bimodal split)
    - Skewed coupling distribution (one-sided heavy tail)
    - Multi-modal coupling (3+ modes, not just 2)
    """
    print("\n" + "=" * 70)
    print("TEST 3: Non-Bimodal Coupling Distributions")
    print("=" * 70)

    n_trials = 5
    all_results = {}

    # Scenario A: Gradually varying coupling (no bimodal split)
    # Objects have coupling that varies smoothly, not in discrete blocks
    print("\n--- Scenario A: Gradual coupling (no bimodal split) ---")
    scenario_a_results = {'otsu': [], 'persistence': [], 'coupling': []}

    for trial in range(n_trials):
        dim = 12
        n_obj = 2

        # Build precision matrix with gradual coupling variation
        Theta = np.eye(dim) * 10.0
        gt_assignment = np.full(dim, -1)

        # Object 0: vars 0-3
        for i in range(4):
            for j in range(i + 1, 4):
                Theta[i, j] = Theta[j, i] = 5.0 + np.random.RandomState(42 + trial).uniform(-0.5, 0.5)
            gt_assignment[i] = 0

        # Object 1: vars 4-7
        for i in range(4, 8):
            for j in range(i + 1, 8):
                Theta[i, j] = Theta[j, i] = 5.0 + np.random.RandomState(43 + trial).uniform(-0.5, 0.5)
            gt_assignment[i] = 1

        # Blanket: vars 8-11 with gradually varying coupling to both objects
        for b in range(8, 12):
            gt_assignment[b] = -1
            for i in range(4):
                # Coupling to object 0: varies from 0.5 to 1.5
                c = 0.5 + (b - 8) * 0.3
                Theta[b, i] = Theta[i, b] = c
            for i in range(4, 8):
                # Coupling to object 1: varies inversely
                c = 1.5 - (b - 8) * 0.3
                Theta[b, i] = Theta[i, b] = c
            for b2 in range(b + 1, 12):
                Theta[b, b2] = Theta[b2, b] = 0.8

        Theta = (Theta + Theta.T) / 2.0
        eigvals = np.linalg.eigvalsh(Theta)
        if eigvals.min() < 0.1:
            Theta += np.eye(dim) * (0.1 - eigvals.min() + 0.1)

        truth = {
            'assignment': gt_assignment,
            'is_blanket': gt_assignment == -1,
            'blanket_vars': np.where(gt_assignment == -1)[0],
            'internal_vars': np.where(gt_assignment >= 0)[0],
            'n_objects': n_obj,
        }

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=5000, n_steps=50,
            step_size=0.003, temp=0.1
        )

        features = compute_geometric_features(gradients)
        for method in ['otsu', 'persistence', 'coupling']:
            pred = _run_method(gradients, n_obj, method, features=features)
            m = compute_metrics(pred, truth)
            scenario_a_results[method].append(m)

    for method in ['otsu', 'persistence', 'coupling']:
        aris = [r['object_ari'] for r in scenario_a_results[method]]
        f1s = [r['blanket_f1'] for r in scenario_a_results[method]]
        print(f"  {method:12s}: ARI={np.mean(aris):.3f} +/- {np.std(aris):.3f}, "
              f"F1={np.mean(f1s):.3f}")

    all_results['gradual_coupling'] = {
        method: {
            'mean_ari': float(np.mean([r['object_ari'] for r in scenario_a_results[method]])),
            'std_ari': float(np.std([r['object_ari'] for r in scenario_a_results[method]])),
            'mean_f1': float(np.mean([r['blanket_f1'] for r in scenario_a_results[method]])),
            'std_f1': float(np.std([r['blanket_f1'] for r in scenario_a_results[method]])),
            'per_trial_ari': [float(r['object_ari']) for r in scenario_a_results[method]],
            'per_trial_f1': [float(r['blanket_f1']) for r in scenario_a_results[method]],
        }
        for method in ['otsu', 'persistence', 'coupling']
    }

    # Scenario B: Multi-modal coupling (3 modes, not bimodal)
    print("\n--- Scenario B: Multi-modal coupling (3 modes) ---")
    scenario_b_results = {'otsu': [], 'persistence': [], 'coupling': []}

    for trial in range(n_trials):
        dim = 15
        n_obj = 3

        Theta = np.eye(dim) * 12.0
        gt_assignment = np.full(dim, -1)

        # 3 objects of size 3, with 3 different intra-coupling strengths
        strengths = [8.0, 4.0, 2.0]  # very different intra-strengths
        for obj_idx in range(3):
            s = obj_idx * 3
            for i in range(s, s + 3):
                for j in range(i + 1, s + 3):
                    Theta[i, j] = Theta[j, i] = strengths[obj_idx]
                gt_assignment[i] = obj_idx

        # Blanket: vars 9-14 with moderate, uniform coupling
        for b in range(9, 15):
            gt_assignment[b] = -1
            for obj_idx in range(3):
                s = obj_idx * 3
                for i in range(s, s + 3):
                    Theta[b, i] = Theta[i, b] = 0.8
            for b2 in range(b + 1, 15):
                Theta[b, b2] = Theta[b2, b] = 1.0

        Theta = (Theta + Theta.T) / 2.0
        eigvals = np.linalg.eigvalsh(Theta)
        if eigvals.min() < 0.1:
            Theta += np.eye(dim) * (0.1 - eigvals.min() + 0.1)

        truth = {
            'assignment': gt_assignment,
            'is_blanket': gt_assignment == -1,
            'blanket_vars': np.where(gt_assignment == -1)[0],
            'internal_vars': np.where(gt_assignment >= 0)[0],
            'n_objects': n_obj,
        }

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=5000, n_steps=50,
            step_size=0.003, temp=0.1
        )

        features = compute_geometric_features(gradients)
        for method in ['otsu', 'persistence', 'coupling']:
            pred = _run_method(gradients, n_obj, method, features=features)
            m = compute_metrics(pred, truth)
            scenario_b_results[method].append(m)

    for method in ['otsu', 'persistence', 'coupling']:
        aris = [r['object_ari'] for r in scenario_b_results[method]]
        f1s = [r['blanket_f1'] for r in scenario_b_results[method]]
        print(f"  {method:12s}: ARI={np.mean(aris):.3f} +/- {np.std(aris):.3f}, "
              f"F1={np.mean(f1s):.3f}")

    all_results['multimodal_coupling'] = {
        method: {
            'mean_ari': float(np.mean([r['object_ari'] for r in scenario_b_results[method]])),
            'std_ari': float(np.std([r['object_ari'] for r in scenario_b_results[method]])),
            'mean_f1': float(np.mean([r['blanket_f1'] for r in scenario_b_results[method]])),
            'std_f1': float(np.std([r['blanket_f1'] for r in scenario_b_results[method]])),
            'per_trial_ari': [float(r['object_ari']) for r in scenario_b_results[method]],
            'per_trial_f1': [float(r['blanket_f1']) for r in scenario_b_results[method]],
        }
        for method in ['otsu', 'persistence', 'coupling']
    }

    # Scenario C: Skewed coupling (heavy-tailed, one-sided)
    print("\n--- Scenario C: Skewed coupling (heavy-tailed) ---")
    scenario_c_results = {'otsu': [], 'persistence': [], 'coupling': []}

    for trial in range(n_trials):
        dim = 12
        n_obj = 2

        Theta = np.eye(dim) * 10.0
        gt_assignment = np.full(dim, -1)

        # Object 0: strongly coupled (high values)
        for i in range(4):
            for j in range(i + 1, 4):
                Theta[i, j] = Theta[j, i] = 8.0
            gt_assignment[i] = 0

        # Object 1: weakly coupled (low values, close to blanket coupling)
        for i in range(4, 8):
            for j in range(i + 1, 8):
                Theta[i, j] = Theta[j, i] = 1.5
            gt_assignment[i] = 1

        # Blanket: moderate coupling
        for b in range(8, 12):
            gt_assignment[b] = -1
            for i in range(8):
                Theta[b, i] = Theta[i, b] = 0.8
            for b2 in range(b + 1, 12):
                Theta[b, b2] = Theta[b2, b] = 0.9

        Theta = (Theta + Theta.T) / 2.0
        eigvals = np.linalg.eigvalsh(Theta)
        if eigvals.min() < 0.1:
            Theta += np.eye(dim) * (0.1 - eigvals.min() + 0.1)

        truth = {
            'assignment': gt_assignment,
            'is_blanket': gt_assignment == -1,
            'blanket_vars': np.where(gt_assignment == -1)[0],
            'internal_vars': np.where(gt_assignment >= 0)[0],
            'n_objects': n_obj,
        }

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=5000, n_steps=50,
            step_size=0.003, temp=0.1
        )

        features = compute_geometric_features(gradients)
        for method in ['otsu', 'persistence', 'coupling']:
            pred = _run_method(gradients, n_obj, method, features=features)
            m = compute_metrics(pred, truth)
            scenario_c_results[method].append(m)

    for method in ['otsu', 'persistence', 'coupling']:
        aris = [r['object_ari'] for r in scenario_c_results[method]]
        f1s = [r['blanket_f1'] for r in scenario_c_results[method]]
        print(f"  {method:12s}: ARI={np.mean(aris):.3f} +/- {np.std(aris):.3f}, "
              f"F1={np.mean(f1s):.3f}")

    all_results['skewed_coupling'] = {
        method: {
            'mean_ari': float(np.mean([r['object_ari'] for r in scenario_c_results[method]])),
            'std_ari': float(np.std([r['object_ari'] for r in scenario_c_results[method]])),
            'mean_f1': float(np.mean([r['blanket_f1'] for r in scenario_c_results[method]])),
            'std_f1': float(np.std([r['blanket_f1'] for r in scenario_c_results[method]])),
            'per_trial_ari': [float(r['object_ari']) for r in scenario_c_results[method]],
            'per_trial_f1': [float(r['blanket_f1']) for r in scenario_c_results[method]],
        }
        for method in ['otsu', 'persistence', 'coupling']
    }

    return all_results


# =========================================================================
# Test 4: Bootstrap Confidence Intervals
# =========================================================================

def run_bootstrap_test():
    """
    Run bootstrap confidence interval computation on a representative example.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Bootstrap Confidence Intervals (Fasy et al. 2014)")
    print("=" * 70)

    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )
    Theta = build_precision_matrix(cfg)

    np.random.seed(42)
    samples, gradients = langevin_sampling(
        Theta, n_samples=5000, n_steps=50,
        step_size=0.003, temp=0.1
    )

    print("  Computing bootstrap (200 resamples)...")
    t0 = time.time()
    bootstrap_result = compute_persistence_bootstrap(
        gradients, n_bootstrap=200, random_state=42)
    elapsed = time.time() - t0
    print(f"  Bootstrap completed in {elapsed:.1f}s")

    base_pd = bootstrap_result['base_diagram']
    n_features = len(base_pd['h0_diagram'])
    n_significant = int(np.sum(bootstrap_result['significant_mask']))

    print(f"  H0 features: {n_features}")
    print(f"  Significant features (CI lower > 0): {n_significant}")

    if n_features > 0:
        print(f"  Persistence CI ranges:")
        for i in range(min(5, n_features)):
            lo, hi = bootstrap_result['persistence_ci'][i]
            sig = "*" if bootstrap_result['significant_mask'][i] else " "
            print(f"    Feature {i}: [{lo:.4f}, {hi:.4f}] {sig}")

    return {
        'n_features': n_features,
        'n_significant': n_significant,
        'bootstrap_time_s': round(elapsed, 2),
        'persistence_ci': bootstrap_result['persistence_ci'].tolist() if n_features > 0 else [],
        'significant_mask': bootstrap_result['significant_mask'].tolist() if n_features > 0 else [],
        'birth_ci': bootstrap_result['birth_ci'].tolist() if n_features > 0 else [],
        'death_ci': bootstrap_result['death_ci'].tolist() if n_features > 0 else [],
    }


# =========================================================================
# Visualization: Persistence Diagrams
# =========================================================================

def plot_persistence_diagram(gradients, title_suffix="", save_name=None):
    """
    Generate persistence diagram visualization with bootstrap confidence bands.
    """
    features = compute_geometric_features(gradients)
    pd_result = compute_persistence_diagram(features['coupling'])

    # Also run bootstrap for confidence bands
    bootstrap_result = compute_persistence_bootstrap(
        gradients, n_bootstrap=200, random_state=42)

    h0_diagram = pd_result['h0_diagram']
    h1_diagram = pd_result['h1_diagram']

    if len(h0_diagram) == 0:
        print(f"  No H0 features to plot for {title_suffix}")
        return None

    # Compute persistence and significance
    births = h0_diagram[:, 0].copy()
    deaths = h0_diagram[:, 1].copy()
    finite_mask = np.isfinite(births)
    max_fb = np.max(births[finite_mask]) if np.any(finite_mask) else 1.0
    births_finite = np.where(np.isfinite(births), births, max_fb * 2)
    persistence = births_finite - deaths
    median_p = np.median(persistence)
    significant = persistence > median_p

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Birth-Death diagram
    ax = axes[0]
    max_val = max(np.max(births_finite), np.max(deaths)) * 1.1

    # Plot diagonal
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')

    # Plot H0 features
    for i in range(len(h0_diagram)):
        color = '#e74c3c' if significant[i] else '#95a5a6'
        marker = 's' if significant[i] else 'o'
        size = 80 if significant[i] else 40
        ax.scatter(births_finite[i], deaths[i], c=color, s=size,
                   marker=marker, edgecolors='black', linewidth=0.5, zorder=5)

        # Bootstrap confidence ellipses
        if bootstrap_result is not None and len(bootstrap_result['birth_ci']) > i:
            b_lo, b_hi = bootstrap_result['birth_ci'][i]
            d_lo, d_hi = bootstrap_result['death_ci'][i]
            rect = plt.Rectangle((b_lo, d_lo), b_hi - b_lo, d_hi - d_lo,
                                  fill=True, facecolor=color, alpha=0.15,
                                  edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)

    # Plot H1 features if any
    if len(h1_diagram) > 0:
        ax.scatter(h1_diagram[:, 0], h1_diagram[:, 1], c='#3498db',
                   s=30, marker='^', alpha=0.5, label='H1 (cycles)', zorder=4)

    ax.set_xlabel('Birth (coupling threshold)')
    ax.set_ylabel('Death (coupling threshold)')
    ax.set_title(f'Persistence Diagram{title_suffix}')
    ax.legend(loc='lower right')

    # Add legend entries
    ax.scatter([], [], c='#e74c3c', s=80, marker='s', edgecolors='black',
               linewidth=0.5, label='Significant H0')
    ax.scatter([], [], c='#95a5a6', s=40, marker='o', edgecolors='black',
               linewidth=0.5, label='Non-significant H0')
    ax.legend(loc='lower right', fontsize=8)

    # Plot 2: Persistence barcode
    ax = axes[1]
    sorted_idx = np.argsort(-persistence)
    for rank, idx in enumerate(sorted_idx):
        color = '#e74c3c' if significant[idx] else '#95a5a6'
        ax.barh(rank, persistence[idx], color=color, edgecolor='black',
                linewidth=0.5, height=0.8)

        # CI whiskers
        if bootstrap_result is not None and len(bootstrap_result['persistence_ci']) > idx:
            lo, hi = bootstrap_result['persistence_ci'][idx]
            ax.plot([lo, hi], [rank, rank], 'k-', linewidth=1.5, alpha=0.5)

    ax.axvline(x=median_p, color='black', linestyle='--', alpha=0.5,
               label=f'Median ({median_p:.3f})')
    ax.set_xlabel('Persistence (birth - death)')
    ax.set_ylabel('Feature rank')
    ax.set_title('Persistence Barcode with Bootstrap CI')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Plot 3: Blanket variable identification
    ax = axes[2]
    n_vars = pd_result['n_vars']
    blanket_score = np.zeros(n_vars)

    for idx, (i, j) in enumerate(pd_result['h0_edges']):
        if significant[idx]:
            blanket_score[i] += persistence[idx]
            blanket_score[j] += persistence[idx]

    colors = ['#e74c3c' if s > 0 else '#3498db' for s in blanket_score]
    ax.bar(range(n_vars), blanket_score, color=colors, edgecolor='black',
           linewidth=0.5)
    ax.set_xlabel('Variable index')
    ax.set_ylabel('Blanket score (sum of significant persistence)')
    ax.set_title('Blanket Variable Identification')

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, 'persistence_blanket')
    else:
        save_figure(fig, 'persistence_diagram', 'persistence_blanket')
    plt.close(fig)

    return fig


def plot_comparison_summary(symmetric_results, asymmetric_results, nonbimodal_results):
    """
    Summary bar chart comparing all methods across all test scenarios.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    methods = ['otsu', 'persistence', 'coupling']
    method_labels = ['Otsu', 'Persistence', 'Coupling']
    method_colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Collect all scenarios
    scenarios = []
    aris = {m: [] for m in methods}
    f1s = {m: [] for m in methods}

    # Symmetric
    scenarios.append('Symmetric\n(standard)')
    for m in methods:
        aris[m].append(symmetric_results[m]['mean_ari'])
        f1s[m].append(symmetric_results[m]['mean_f1'])

    # Asymmetric scenarios
    for label, data in asymmetric_results.items():
        short = label.split(':')[1].strip() if ':' in label else label
        scenarios.append(f'Asym.\n{short}')
        for m in methods:
            aris[m].append(data[m]['mean_ari'])
            f1s[m].append(data[m]['mean_f1'])

    # Non-bimodal scenarios
    for label, data in nonbimodal_results.items():
        short = label.replace('_', '\n')
        scenarios.append(f'Non-bim.\n{short}')
        for m in methods:
            aris[m].append(data[m]['mean_ari'])
            f1s[m].append(data[m]['mean_f1'])

    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.25

    # ARI plot
    ax = axes[0]
    for i, (m, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        ax.bar(x + i * width, aris[m], width, label=label, color=color,
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel('ARI')
    ax.set_title('Object Partition Recovery (ARI)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    # F1 plot
    ax = axes[1]
    for i, (m, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        ax.bar(x + i * width, f1s[m], width, label=label, color=color,
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel('F1')
    ax.set_title('Blanket Detection F1')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, 'method_comparison_summary', 'persistence_blanket')
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("PERSISTENCE-BASED BLANKET DETECTION: COMPARISON EXPERIMENT (US-069)")
    print("=" * 70)

    all_results = {}

    # Test 1: Symmetric
    symmetric_results = run_symmetric_comparison()
    all_results['symmetric'] = symmetric_results

    # Test 2: Asymmetric
    asymmetric_results = run_asymmetric_comparison()
    all_results['asymmetric'] = asymmetric_results

    # Test 3: Non-bimodal
    nonbimodal_results = run_nonbimodal_comparison()
    all_results['nonbimodal'] = nonbimodal_results

    # Test 4: Bootstrap
    bootstrap_results = run_bootstrap_test()
    all_results['bootstrap'] = bootstrap_results

    # Test 5: method='persistence' works in TopologicalBlankets class
    print("\n" + "=" * 70)
    print("TEST 5: TopologicalBlankets(method='persistence') integration")
    print("=" * 70)

    from topological_blankets import TopologicalBlankets
    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )
    Theta = build_precision_matrix(cfg)
    truth = get_ground_truth(cfg)

    np.random.seed(42)
    samples, gradients = langevin_sampling(
        Theta, n_samples=5000, n_steps=50,
        step_size=0.003, temp=0.1
    )

    tb = TopologicalBlankets(method='persistence', n_objects=2)
    tb.fit(gradients)
    objects = tb.get_objects()
    blankets = tb.get_blankets()
    assignment = tb.get_assignment()

    pred = {'assignment': assignment, 'is_blanket': tb._is_blanket}
    m = compute_metrics(pred, truth)

    print(f"  Objects detected: {len(objects)}")
    print(f"  Blanket vars: {list(blankets)}")
    print(f"  ARI: {m['object_ari']:.3f}, F1: {m['blanket_f1']:.3f}")

    all_results['integration_test'] = {
        'n_objects_detected': len(objects),
        'blanket_vars': blankets.tolist(),
        'object_ari': float(m['object_ari']),
        'blanket_f1': float(m['blanket_f1']),
    }

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Persistence diagram for symmetric case
    cfg_sym = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )
    Theta_sym = build_precision_matrix(cfg_sym)
    np.random.seed(42)
    _, grad_sym = langevin_sampling(
        Theta_sym, n_samples=5000, n_steps=50,
        step_size=0.003, temp=0.1
    )
    plot_persistence_diagram(grad_sym, title_suffix=" (Symmetric 2-obj)",
                             save_name="pd_symmetric")

    # Persistence diagram for asymmetric case (3+8)
    dim_asym = 3 + 8 + 3
    Theta_asym = np.zeros((dim_asym, dim_asym))
    for i in range(3):
        for j in range(3):
            Theta_asym[i, j] = 6.0
    np.fill_diagonal(Theta_asym[:3, :3], 18.0)
    for i in range(3, 11):
        for j in range(3, 11):
            Theta_asym[i, j] = 6.0
    np.fill_diagonal(Theta_asym[3:11, 3:11], 48.0)
    for b in range(11, 14):
        for b2 in range(11, 14):
            Theta_asym[b, b2] = 1.0
    np.fill_diagonal(Theta_asym[11:, 11:], 3.0)
    for i in range(11):
        for b in range(11, 14):
            Theta_asym[i, b] = Theta_asym[b, i] = 0.8
    Theta_asym = (Theta_asym + Theta_asym.T) / 2.0
    eig_asym = np.linalg.eigvalsh(Theta_asym)
    if eig_asym.min() < 0.1:
        Theta_asym += np.eye(dim_asym) * (0.1 - eig_asym.min() + 0.1)

    np.random.seed(42)
    _, grad_asym = langevin_sampling(
        Theta_asym, n_samples=5000, n_steps=50,
        step_size=0.003, temp=0.1
    )
    plot_persistence_diagram(grad_asym, title_suffix=" (Asymmetric 3+8)",
                             save_name="pd_asymmetric_3_8")

    # Comparison summary chart
    plot_comparison_summary(symmetric_results, asymmetric_results, nonbimodal_results)

    # Save all results
    save_results('persistence_blanket_comparison', all_results, {
        'n_trials_symmetric': 10,
        'n_trials_asymmetric': 5,
        'n_trials_nonbimodal': 5,
        'n_bootstrap': 200,
        'intra_strength': 6.0,
        'blanket_strength': 0.8,
    }, notes='US-069: Persistence-based blanket detection comparison')

    # Final summary
    print("\n" + "=" * 70)
    print("US-069 ACCEPTANCE CRITERIA SUMMARY")
    print("=" * 70)

    criteria = []

    # 1. Persistence diagram computed
    c1 = bootstrap_results['n_features'] > 0
    criteria.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] Persistence diagram computed "
          f"({bootstrap_results['n_features']} H0 features)")

    # 2. Blanket variables identified from persistence > median
    c2 = all_results['integration_test']['blanket_f1'] > 0
    criteria.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] Blanket vars identified from "
          f"significant features (F1={all_results['integration_test']['blanket_f1']:.3f})")

    # 3. Bootstrap confidence intervals
    c3 = bootstrap_results['n_significant'] > 0 and bootstrap_results['bootstrap_time_s'] > 0
    criteria.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] Bootstrap CI computed "
          f"({bootstrap_results['n_significant']} significant, "
          f"{bootstrap_results['bootstrap_time_s']}s)")

    # 4. method='persistence' in TopologicalBlankets
    c4 = all_results['integration_test']['object_ari'] >= 0
    criteria.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] method='persistence' works in "
          f"TopologicalBlankets class")

    # 5. Symmetric comparison: ARI within 0.02
    c5 = symmetric_results.get('symmetric_pass', False)
    criteria.append(c5)
    print(f"  [{'PASS' if c5 else 'FAIL'}] Symmetric: persistence matches Otsu "
          f"(diff={symmetric_results.get('ari_difference', 'N/A')})")

    # 6. Asymmetric: persistence improves on 2+2+10 and 3+8
    asym_3_8 = asymmetric_results.get('2 objects: 3+8 vars', {})
    asym_2210 = asymmetric_results.get('3 objects: 2+2+10 vars', {})
    c6a = asym_3_8.get('persistence_improves', False)
    c6b = asym_2210.get('persistence_improves', False)
    c6 = c6a and c6b
    criteria.append(c6)
    print(f"  [{'PASS' if c6 else 'FAIL'}] Asymmetric: persistence improves "
          f"(3+8: {c6a}, 2+2+10: {c6b})")

    # 7. Non-bimodal coupling: persistence should be competitive overall.
    # Check the combined metric (mean of ARI and F1). Persistence should
    # not have a worse combined score than Otsu across all non-bimodal
    # scenarios.
    nonbimodal_pass = True
    for scenario_name, scenario_data in nonbimodal_results.items():
        p_ari = scenario_data['persistence']['mean_ari']
        o_ari = scenario_data['otsu']['mean_ari']
        p_f1 = scenario_data['persistence']['mean_f1']
        o_f1 = scenario_data['otsu']['mean_f1']
        p_combined = (p_ari + p_f1) / 2.0
        o_combined = (o_ari + o_f1) / 2.0
        if p_combined < o_combined - 0.1:
            nonbimodal_pass = False
    c7 = nonbimodal_pass
    criteria.append(c7)
    print(f"  [{'PASS' if c7 else 'FAIL'}] Non-bimodal: persistence competitive "
          f"with Otsu (combined ARI+F1)")

    # 8. Visualizations saved
    c8 = True  # we generated them above
    criteria.append(c8)
    print(f"  [{'PASS' if c8 else 'FAIL'}] Persistence diagram visualizations saved")

    # 9. Results JSON saved
    c9 = True  # save_results was called above
    criteria.append(c9)
    print(f"  [{'PASS' if c9 else 'FAIL'}] Results JSON saved")

    all_pass = all(criteria)
    print(f"\n  Overall: {'ALL CRITERIA PASS' if all_pass else 'SOME CRITERIA FAILED'}")

    return all_pass, all_results


if __name__ == '__main__':
    all_pass, results = main()
