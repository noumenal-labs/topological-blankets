"""
PCCA+ Fuzzy Partition Comparison (US-072)
==========================================

Compares PCCA+ fuzzy blanket detection against existing crisp methods:
- Otsu (gradient-based)
- Coupling (cross-cluster coupling entropy)
- Persistence (H0 sublevel set filtration)

Also tests PCCA+ on asymmetric landscapes (US-012 configurations) to evaluate
whether fuzzy membership handles unequal object sizes better than crisp methods.

Produces:
- Membership vector stacked bar chart per variable (blanket highlighted)
- 2D PCA embedding colored by max membership (blanket in transition zones)
- Comparison metrics: ARI and blanket F1 across methods
- Asymmetric landscape results
- Results JSON and PNGs saved to results/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Also add the topological_blankets package root (two levels up from experiments/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from topological_blankets import topological_blankets as tb_func
from topological_blankets import (
    TopologicalBlankets,
    detect_blankets_pcca,
    compute_geometric_features,
)
from topological_blankets.pcca import pcca_plus, pcca_blanket_detection
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure
from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, compute_metrics,
)

EXPERIMENT_NAME = "pcca_fuzzy_comparison"


# =============================================================================
# Asymmetric system builder (from verify_asymmetric_fix.py)
# =============================================================================

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


# =============================================================================
# Comparison on standard quadratic
# =============================================================================

def run_standard_comparison(n_trials: int = 10) -> Dict:
    """
    Compare PCCA+ against Otsu, coupling, and persistence on the standard
    symmetric quadratic EBM (2 objects, 3 vars each, 3 blanket vars).
    """
    print("\n" + "=" * 70)
    print("STANDARD QUADRATIC COMPARISON")
    print("=" * 70)

    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8
    )
    Theta = build_precision_matrix(cfg)
    truth = get_ground_truth(cfg)

    methods = ['gradient', 'coupling', 'persistence', 'pcca']
    all_metrics = {m: {'object_ari': [], 'blanket_f1': [], 'full_ari': []}
                   for m in methods}

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=3000, n_steps=30,
            step_size=0.005, temp=0.1
        )

        for method in methods:
            result = tb_func(gradients, n_objects=cfg.n_objects, method=method)
            metrics = compute_metrics(result, truth)
            for key in metrics:
                all_metrics[method][key].append(metrics[key])

    # Summarize
    summary = {}
    for method in methods:
        s = {}
        for key in all_metrics[method]:
            vals = all_metrics[method][key]
            s[f'mean_{key}'] = float(np.mean(vals))
            s[f'std_{key}'] = float(np.std(vals))
        summary[method] = s
        print(f"\n  {method}:")
        print(f"    Object ARI: {s['mean_object_ari']:.3f} +/- {s['std_object_ari']:.3f}")
        print(f"    Blanket F1: {s['mean_blanket_f1']:.3f} +/- {s['std_blanket_f1']:.3f}")
        print(f"    Full ARI:   {s['mean_full_ari']:.3f} +/- {s['std_full_ari']:.3f}")

    return {
        'config': {
            'n_objects': cfg.n_objects,
            'vars_per_object': cfg.vars_per_object,
            'vars_per_blanket': cfg.vars_per_blanket,
            'intra_strength': cfg.intra_strength,
            'blanket_strength': cfg.blanket_strength,
            'n_trials': n_trials,
        },
        'methods': summary,
    }


# =============================================================================
# Asymmetric landscape comparison
# =============================================================================

def run_asymmetric_comparison(n_trials: int = 5) -> Dict:
    """
    Test PCCA+ on asymmetric landscapes (US-012 configurations).
    Evaluates whether fuzzy membership handles unequal object sizes better
    than crisp methods.
    """
    print("\n" + "=" * 70)
    print("ASYMMETRIC LANDSCAPE COMPARISON")
    print("=" * 70)

    scenarios = [
        ([3, 8], 3, "2 objects: 3+8 vars"),
        ([2, 2, 10], 3, "3 objects: 2+2+10 vars"),
        ([3, 5, 7], 4, "3 objects: 3+5+7 vars"),
        ([5, 5, 5], 3, "3 equal objects (regression)"),
    ]

    methods = ['gradient', 'coupling', 'persistence', 'pcca']
    results = {}

    for obj_sizes, b_size, label in scenarios:
        print(f"\n  Scenario: {label}")
        n_obj = len(obj_sizes)
        scenario_metrics = {m: {'object_ari': [], 'blanket_f1': []}
                           for m in methods}

        for trial in range(n_trials):
            Theta, truth, n_vars = build_asymmetric_system(obj_sizes, b_size)
            np.random.seed(42 + trial)
            n_samples = max(3000, n_vars * 80)
            _, gradients = langevin_sampling(
                Theta, n_samples=n_samples, n_steps=50,
                step_size=0.003, temp=0.1
            )

            for method in methods:
                result = tb_func(gradients, n_objects=n_obj, method=method)
                m = compute_metrics(result, truth)
                scenario_metrics[method]['object_ari'].append(m['object_ari'])
                scenario_metrics[method]['blanket_f1'].append(m['blanket_f1'])

        scenario_summary = {}
        for method in methods:
            mean_ari = float(np.mean(scenario_metrics[method]['object_ari']))
            std_ari = float(np.std(scenario_metrics[method]['object_ari']))
            mean_f1 = float(np.mean(scenario_metrics[method]['blanket_f1']))
            scenario_summary[method] = {
                'mean_object_ari': mean_ari,
                'std_object_ari': std_ari,
                'mean_blanket_f1': mean_f1,
            }
            print(f"    {method:12s}: ARI={mean_ari:.3f} +/- {std_ari:.3f}, F1={mean_f1:.3f}")

        results[label] = {
            'obj_sizes': obj_sizes,
            'blanket_size': b_size,
            'n_objects': n_obj,
            'methods': scenario_summary,
        }

    return results


# =============================================================================
# Visualization: Membership stacked bar chart
# =============================================================================

def plot_membership_bar_chart(memberships: np.ndarray,
                              is_blanket: np.ndarray,
                              truth_is_blanket: np.ndarray,
                              variable_labels: List[str] = None) -> plt.Figure:
    """
    Stacked bar chart showing fuzzy membership vectors per variable.
    Blanket variables are highlighted with a marker.
    """
    n_vars, k = memberships.shape

    if variable_labels is None:
        variable_labels = [f'x{i}' for i in range(n_vars)]

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 0.6), 5))

    x = np.arange(n_vars)
    colors = plt.cm.Set2(np.linspace(0, 1, k))
    bottom = np.zeros(n_vars)

    for cluster_idx in range(k):
        ax.bar(x, memberships[:, cluster_idx], bottom=bottom,
               color=colors[cluster_idx], label=f'Cluster {cluster_idx}',
               width=0.7, edgecolor='white', linewidth=0.5)
        bottom += memberships[:, cluster_idx]

    # Mark detected blanket variables
    blanket_idx = np.where(is_blanket)[0]
    for idx in blanket_idx:
        ax.plot(idx, 1.05, 'v', color='red', markersize=8, zorder=5)

    # Mark ground-truth blanket variables
    truth_blanket_idx = np.where(truth_is_blanket)[0]
    for idx in truth_blanket_idx:
        ax.plot(idx, 1.10, '*', color='black', markersize=8, zorder=5)

    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Membership')
    ax.set_title('PCCA+ Fuzzy Membership Vectors per Variable')
    ax.set_xticks(x)
    ax.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.legend(loc='upper right', fontsize=8)

    # Add legend for markers
    ax.plot([], [], 'rv', markersize=8, label='Detected blanket')
    ax.plot([], [], 'k*', markersize=8, label='True blanket')
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    # Add threshold line at 0.6
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(n_vars - 0.5, 0.61, 'threshold=0.6', fontsize=7,
            color='gray', ha='right')

    fig.tight_layout()
    return fig


# =============================================================================
# Visualization: 2D PCA embedding colored by max membership
# =============================================================================

def plot_embedding_by_membership(eigvecs: np.ndarray,
                                  max_membership: np.ndarray,
                                  is_blanket: np.ndarray,
                                  truth_is_blanket: np.ndarray,
                                  n_clusters: int) -> plt.Figure:
    """
    2D PCA embedding of eigenvectors, colored by max membership.
    Blanket variables appear in transition zones (low max membership).
    """
    # Use PCA on eigenvectors (skip constant mode)
    if eigvecs.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(eigvecs[:, 1:min(n_clusters + 1, eigvecs.shape[1])])
    else:
        coords = eigvecs[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Colored by max membership
    ax = axes[0]
    norm = Normalize(vmin=0.3, vmax=1.0)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=max_membership,
                         cmap='RdYlGn', norm=norm, s=80, edgecolors='black',
                         linewidths=0.5, zorder=3)

    # Highlight detected blanket variables
    blanket_mask = is_blanket
    if np.any(blanket_mask):
        ax.scatter(coords[blanket_mask, 0], coords[blanket_mask, 1],
                   facecolors='none', edgecolors='red', linewidths=2,
                   s=150, zorder=4, label='Detected blanket')

    plt.colorbar(scatter, ax=ax, label='Max membership')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Eigenvector Embedding (colored by max membership)')
    ax.legend(fontsize=8)

    # Panel 2: Ground truth coloring
    ax = axes[1]
    truth_colors = np.where(truth_is_blanket, -1, 0)
    # Use distinct colors for truth
    for i in range(coords.shape[0]):
        if truth_is_blanket[i]:
            ax.scatter(coords[i, 0], coords[i, 1], c='orange', s=80,
                       edgecolors='black', linewidths=0.5, marker='s', zorder=3)
        else:
            ax.scatter(coords[i, 0], coords[i, 1], c='steelblue', s=80,
                       edgecolors='black', linewidths=0.5, marker='o', zorder=3)

    # Add variable index labels
    for i in range(coords.shape[0]):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha='center', va='bottom', color='gray')

    ax.scatter([], [], c='orange', marker='s', label='True blanket')
    ax.scatter([], [], c='steelblue', marker='o', label='True internal')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Eigenvector Embedding (ground truth)')
    ax.legend(fontsize=8)

    fig.suptitle('PCCA+ Embedding: Blanket Variables in Transition Zones',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# Visualization: Method comparison bar chart
# =============================================================================

def plot_method_comparison(standard_results: Dict,
                           asymmetric_results: Dict) -> plt.Figure:
    """
    Bar chart comparing ARI and F1 across methods for both standard
    and asymmetric configurations.
    """
    methods = ['gradient', 'coupling', 'persistence', 'pcca']
    method_labels = ['Otsu', 'Coupling', 'Persistence', 'PCCA+']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: Standard quadratic
    ax = axes[0, 0]
    ari_means = [standard_results['methods'][m]['mean_object_ari'] for m in methods]
    ari_stds = [standard_results['methods'][m]['std_object_ari'] for m in methods]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    bars = ax.bar(method_labels, ari_means, yerr=ari_stds, color=colors,
                  edgecolor='black', linewidth=0.5, capsize=4)
    ax.set_ylabel('Object ARI')
    ax.set_title('Standard Quadratic: Object Partition')
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, ari_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)

    ax = axes[0, 1]
    f1_means = [standard_results['methods'][m]['mean_blanket_f1'] for m in methods]
    f1_stds = [standard_results['methods'][m]['std_blanket_f1'] for m in methods]
    bars = ax.bar(method_labels, f1_means, yerr=f1_stds, color=colors,
                  edgecolor='black', linewidth=0.5, capsize=4)
    ax.set_ylabel('Blanket F1')
    ax.set_title('Standard Quadratic: Blanket Detection')
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)

    # Bottom row: Asymmetric average
    asym_scenarios = [k for k in asymmetric_results
                      if 'regression' not in k.lower()]

    ax = axes[1, 0]
    ari_means_asym = []
    for m in methods:
        vals = [asymmetric_results[s]['methods'][m]['mean_object_ari']
                for s in asym_scenarios]
        ari_means_asym.append(float(np.mean(vals)))
    bars = ax.bar(method_labels, ari_means_asym, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Object ARI (avg over asymmetric)')
    ax.set_title('Asymmetric Landscapes: Object Partition')
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, ari_means_asym):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)

    ax = axes[1, 1]
    f1_means_asym = []
    for m in methods:
        vals = [asymmetric_results[s]['methods'][m]['mean_blanket_f1']
                for s in asym_scenarios]
        f1_means_asym.append(float(np.mean(vals)))
    bars = ax.bar(method_labels, f1_means_asym, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Blanket F1 (avg over asymmetric)')
    ax.set_title('Asymmetric Landscapes: Blanket Detection')
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, f1_means_asym):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)

    fig.suptitle('PCCA+ vs Crisp Methods: Blanket Detection Comparison',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# Single-run PCCA+ detailed analysis (for visualizations)
# =============================================================================

def run_pcca_detailed_analysis() -> Dict:
    """
    Run a single PCCA+ analysis with detailed outputs for visualization.
    Uses the standard quadratic EBM.
    """
    print("\n" + "=" * 70)
    print("PCCA+ DETAILED ANALYSIS (single run)")
    print("=" * 70)

    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8
    )
    Theta = build_precision_matrix(cfg)
    truth = get_ground_truth(cfg)

    np.random.seed(42)
    samples, gradients = langevin_sampling(
        Theta, n_samples=5000, n_steps=50,
        step_size=0.005, temp=0.1
    )

    # Run PCCA+ via the class API
    tb = TopologicalBlankets(method='pcca', n_objects=cfg.n_objects)
    tb.fit(gradients)

    detection_info = tb.get_detection_info()
    memberships = detection_info['memberships']
    max_membership = detection_info['max_membership']
    membership_entropy = detection_info['membership_entropy']
    eigvecs = detection_info['eigvecs']
    is_blanket = tb._is_blanket

    n_vars = gradients.shape[1]

    # Build variable labels
    var_labels = []
    for obj_idx in range(cfg.n_objects):
        for v in range(cfg.vars_per_object):
            var_labels.append(f'Obj{obj_idx}_v{v}')
    for v in range(cfg.vars_per_blanket):
        var_labels.append(f'B_v{v}')

    print(f"\n  Variables: {n_vars}")
    print(f"  Detected blanket vars: {np.where(is_blanket)[0].tolist()}")
    print(f"  True blanket vars:     {np.where(truth['is_blanket'])[0].tolist()}")
    print(f"\n  Membership vectors:")
    for i in range(n_vars):
        marker = " <-- BLANKET" if is_blanket[i] else ""
        truth_marker = " [TRUE BLANKET]" if truth['is_blanket'][i] else ""
        print(f"    {var_labels[i]:10s}: {memberships[i]} "
              f"(max={max_membership[i]:.3f}, H={membership_entropy[i]:.3f})"
              f"{marker}{truth_marker}")

    # Validate membership properties
    row_sums = memberships.sum(axis=1)
    non_negative = np.all(memberships >= -1e-10)
    sums_to_one = np.allclose(row_sums, 1.0, atol=1e-6)
    print(f"\n  Membership validation:")
    print(f"    Non-negative: {non_negative}")
    print(f"    Rows sum to 1: {sums_to_one} (range: [{row_sums.min():.6f}, {row_sums.max():.6f}])")

    # Compute metrics
    assignment = tb.get_assignment()
    pred = {
        'assignment': assignment,
        'is_blanket': is_blanket,
    }
    metrics = compute_metrics(pred, truth)
    print(f"\n  Metrics:")
    print(f"    Object ARI:  {metrics['object_ari']:.3f}")
    print(f"    Blanket F1:  {metrics['blanket_f1']:.3f}")
    print(f"    Full ARI:    {metrics['full_ari']:.3f}")

    # Generate visualizations

    # 1. Membership stacked bar chart
    fig_bar = plot_membership_bar_chart(
        memberships, is_blanket, truth['is_blanket'], var_labels)
    save_figure(fig_bar, 'pcca_membership_bar_chart', EXPERIMENT_NAME)

    # 2. PCA embedding colored by max membership
    fig_embed = plot_embedding_by_membership(
        eigvecs, max_membership, is_blanket, truth['is_blanket'],
        n_clusters=cfg.n_objects + 1)
    save_figure(fig_embed, 'pcca_embedding_max_membership', EXPERIMENT_NAME)

    return {
        'n_vars': n_vars,
        'detected_blanket': np.where(is_blanket)[0].tolist(),
        'true_blanket': np.where(truth['is_blanket'])[0].tolist(),
        'max_membership': max_membership.tolist(),
        'membership_entropy': membership_entropy.tolist(),
        'memberships': memberships.tolist(),
        'non_negative': bool(non_negative),
        'sums_to_one': bool(sums_to_one),
        'metrics': metrics,
    }


# =============================================================================
# Verify PCCA+ membership properties
# =============================================================================

def verify_membership_properties(n_trials: int = 10) -> Dict:
    """
    Verify that PCCA+ memberships satisfy the required properties across
    multiple random seeds: non-negativity and row-sum = 1.
    """
    print("\n" + "=" * 70)
    print("PCCA+ MEMBERSHIP PROPERTY VERIFICATION")
    print("=" * 70)

    cfg = QuadraticEBMConfig()
    Theta = build_precision_matrix(cfg)

    all_non_negative = True
    all_sums_to_one = True
    max_violation_neg = 0.0
    max_violation_sum = 0.0

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        _, gradients = langevin_sampling(
            Theta, n_samples=3000, n_steps=30,
            step_size=0.005, temp=0.1
        )

        features = compute_geometric_features(gradients)
        result = detect_blankets_pcca(features['hessian_est'], n_clusters=3)
        memberships = result['memberships']

        min_val = memberships.min()
        row_sums = memberships.sum(axis=1)
        sum_dev = np.max(np.abs(row_sums - 1.0))

        if min_val < -1e-10:
            all_non_negative = False
        if sum_dev > 1e-6:
            all_sums_to_one = False

        max_violation_neg = min(max_violation_neg, min_val)
        max_violation_sum = max(max_violation_sum, sum_dev)

    print(f"  Trials: {n_trials}")
    print(f"  All non-negative: {all_non_negative} (worst: {max_violation_neg:.2e})")
    print(f"  All sum to 1: {all_sums_to_one} (worst deviation: {max_violation_sum:.2e})")

    return {
        'n_trials': n_trials,
        'all_non_negative': all_non_negative,
        'all_sums_to_one': all_sums_to_one,
        'worst_negative': float(max_violation_neg),
        'worst_sum_deviation': float(max_violation_sum),
    }


# =============================================================================
# TopologicalBlankets class API verification
# =============================================================================

def verify_class_api() -> Dict:
    """
    Verify that method='pcca' works through the TopologicalBlankets class API.
    Returns both hard partition and fuzzy memberships.
    """
    print("\n" + "=" * 70)
    print("TOPOLOGICALBLANKETS CLASS API VERIFICATION (method='pcca')")
    print("=" * 70)

    cfg = QuadraticEBMConfig()
    Theta = build_precision_matrix(cfg)

    np.random.seed(42)
    _, gradients = langevin_sampling(
        Theta, n_samples=3000, n_steps=30,
        step_size=0.005, temp=0.1
    )

    tb = TopologicalBlankets(method='pcca', n_objects=cfg.n_objects)
    tb.fit(gradients)

    # Check all outputs
    assignment = tb.get_assignment()
    blankets = tb.get_blankets()
    objects = tb.get_objects()
    memberships = tb.get_memberships()
    detection_info = tb.get_detection_info()

    has_hard_partition = assignment is not None and len(assignment) > 0
    has_memberships = memberships is not None and memberships.shape[0] > 0
    has_blankets = blankets is not None
    has_objects = len(objects) > 0
    has_detection_info = detection_info is not None and 'memberships' in detection_info

    print(f"  Hard partition available: {has_hard_partition}")
    print(f"  Fuzzy memberships available: {has_memberships}")
    if has_memberships:
        print(f"    Shape: {memberships.shape}")
    print(f"  Blanket indices: {blankets.tolist()}")
    print(f"  Objects: {len(objects)}")
    print(f"  Detection info keys: {list(detection_info.keys())}")

    return {
        'has_hard_partition': has_hard_partition,
        'has_memberships': has_memberships,
        'memberships_shape': list(memberships.shape) if has_memberships else None,
        'blanket_indices': blankets.tolist(),
        'n_objects': len(objects),
        'detection_info_keys': list(detection_info.keys()),
        'has_detection_info': has_detection_info,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("US-072: PCCA+ Fuzzy Partition for Soft Blanket Membership")
    print("=" * 70)

    all_results = {}

    # 1. Verify membership properties
    prop_results = verify_membership_properties(n_trials=10)
    all_results['membership_properties'] = prop_results

    # 2. Verify class API
    api_results = verify_class_api()
    all_results['class_api'] = api_results

    # 3. Detailed single-run analysis with visualizations
    detailed = run_pcca_detailed_analysis()
    all_results['detailed_analysis'] = detailed

    # 4. Standard quadratic comparison (10 trials)
    standard = run_standard_comparison(n_trials=10)
    all_results['standard_comparison'] = standard

    # 5. Asymmetric landscape comparison
    asymmetric = run_asymmetric_comparison(n_trials=5)
    all_results['asymmetric_comparison'] = asymmetric

    # 6. Comparison visualization
    fig_comparison = plot_method_comparison(standard, asymmetric)
    save_figure(fig_comparison, 'pcca_method_comparison', EXPERIMENT_NAME)

    # Save all results
    save_results(
        EXPERIMENT_NAME,
        metrics=all_results,
        config={
            'standard_n_trials': 10,
            'asymmetric_n_trials': 5,
            'ambiguity_threshold': 0.6,
            'normalized_laplacian': True,
        },
        notes=(
            "US-072: PCCA+ fuzzy partition for soft blanket membership. "
            "Compares PCCA+ against Otsu, coupling, and persistence methods "
            "on both standard symmetric and asymmetric quadratic landscapes. "
            "PCCA+ produces fuzzy membership vectors where blanket variables "
            "naturally emerge as having high membership in multiple clusters."
        )
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Membership non-negative: {prop_results['all_non_negative']}")
    print(f"  Membership sums to 1:    {prop_results['all_sums_to_one']}")
    print(f"  Class API works:         {api_results['has_hard_partition'] and api_results['has_memberships']}")
    print(f"  Standard PCCA+ ARI:      {standard['methods']['pcca']['mean_object_ari']:.3f}")
    print(f"  Standard PCCA+ F1:       {standard['methods']['pcca']['mean_blanket_f1']:.3f}")

    all_pass = (
        prop_results['all_non_negative'] and
        prop_results['all_sums_to_one'] and
        api_results['has_hard_partition'] and
        api_results['has_memberships'] and
        api_results['has_detection_info']
    )

    print(f"\n  ALL ACCEPTANCE CRITERIA MET: {all_pass}")
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
