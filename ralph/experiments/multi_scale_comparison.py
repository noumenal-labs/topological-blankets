"""
US-030: Multi-Scale Comparison Across Representations
======================================================

Systematic comparison of TB structure across 3 representations of the same
LunarLander agent-environment system:

1. Raw 8D state space (Active Inference dynamics gradients, from US-025)
2. Active Inference ensemble disagreement (from US-026)
3. Dreamer 64D latent space (from US-029)

Key question: does learned representation structure match physical structure?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

NOUMENAL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, NOUMENAL_DIR)

from experiments.utils.results import save_results, load_results
from experiments.utils.plotting import save_figure

STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']


def find_latest_result(pattern):
    """Find the most recent result file matching a pattern."""
    results_dir = os.path.join(NOUMENAL_DIR, 'results')
    matches = []
    for f in os.listdir(results_dir):
        if pattern in f and f.endswith('.json'):
            matches.append(os.path.join(results_dir, f))
    if not matches:
        raise FileNotFoundError(f"No result file matching '{pattern}' in results/")
    matches.sort()
    return matches[-1]


def load_actinf_results():
    """Load Active Inference TB analysis results from US-025."""
    path = find_latest_result('actinf_tb_analysis')
    data = load_results(path)
    print(f"Loaded Active Inference results from {os.path.basename(path)}")
    return data


def load_dreamer_results():
    """Load Dreamer TB analysis results from US-029."""
    path = find_latest_result('dreamer_tb_analysis')
    data = load_results(path)
    print(f"Loaded Dreamer results from {os.path.basename(path)}")
    return data


def extract_representation_summary(results, name, method='gradient'):
    """Extract comparable metrics from a TB analysis result."""
    if 'metrics' in results:
        metrics = results['metrics']
    else:
        metrics = results

    # Handle different result structures
    if 'dynamics' in metrics:
        # Active Inference results have dynamics/reward/disagreement sub-keys
        data = metrics['dynamics']
    else:
        # Dreamer results are flat
        data = metrics

    method_data = data.get(f'{method}_method', data.get('methods', {}).get(method, {}))

    assignment = np.array(method_data.get('assignment', []))
    is_blanket = np.array(method_data.get('is_blanket', []))
    n_blanket = int(np.sum(is_blanket)) if len(is_blanket) > 0 else 0
    n_objects = len(set(assignment[assignment >= 0])) if len(assignment) > 0 else 0

    eigengap = data.get('eigengap', 0)
    eigenvalues = data.get('eigenvalues', [])
    coupling = np.array(data.get('coupling', data.get('coupling_matrix', [])))

    return {
        'name': name,
        'n_dims': len(assignment),
        'n_objects': n_objects,
        'n_blanket': n_blanket,
        'eigengap': float(eigengap),
        'assignment': assignment,
        'is_blanket': is_blanket,
        'coupling': coupling,
        'eigenvalues': eigenvalues[:20] if len(eigenvalues) > 20 else eigenvalues,
    }


def compute_object_correspondence(actinf_results, dreamer_results):
    """
    Analyze whether Dreamer latent-space objects correspond to physical state objects.

    Uses the latent-to-physical correlation from the Dreamer analysis to project
    latent partition labels to physical state space, then compares with the
    state-space partition.
    """
    dreamer_data = dreamer_results if 'metrics' not in dreamer_results else dreamer_results['metrics']

    correlation = np.array(dreamer_data.get('latent_physical_correlation', []))  # (64, 8)
    strongest_correlate = np.array(dreamer_data.get('strongest_correlate_per_latent', []))
    correlate_strength = np.array(dreamer_data.get('correlate_strength_per_latent', []))

    methods_data = dreamer_data.get('methods', {})
    dreamer_assign = np.array(methods_data.get('gradient', {}).get('assignment', []))
    dreamer_blanket = np.array(methods_data.get('gradient', {}).get('is_blanket', []))

    actinf_data = actinf_results if 'metrics' not in actinf_results else actinf_results['metrics']
    if 'dynamics' in actinf_data:
        actinf_data = actinf_data['dynamics']
    actinf_assign = np.array(actinf_data.get('gradient_method', {}).get('assignment', []))

    if len(correlation) == 0 or len(dreamer_assign) == 0:
        return {'error': 'Missing data for correspondence analysis'}

    # For each latent cluster, find which physical variables it maps to
    cluster_to_physical = {}
    for cluster_id in sorted(set(dreamer_assign)):
        mask = dreamer_assign == cluster_id
        latent_dims = np.where(mask)[0]

        # Mean absolute correlation with each physical variable
        if len(latent_dims) > 0:
            mean_corr = np.mean(np.abs(correlation[latent_dims]), axis=0)  # (8,)
            top_physical = np.argsort(mean_corr)[::-1][:3]
            cluster_to_physical[int(cluster_id)] = {
                'n_latent_dims': int(len(latent_dims)),
                'top_physical_vars': [STATE_LABELS[i] for i in top_physical],
                'top_correlations': mean_corr[top_physical].tolist(),
                'all_correlations': {STATE_LABELS[i]: float(mean_corr[i]) for i in range(8)},
            }

    # Blanket analysis
    blanket_dims = np.where(dreamer_blanket)[0] if len(dreamer_blanket) > 0 else np.array([])
    if len(blanket_dims) > 0 and len(correlation) > 0:
        blanket_mean_corr = np.mean(np.abs(correlation[blanket_dims]), axis=0)
        blanket_top = np.argsort(blanket_mean_corr)[::-1][:3]
        blanket_physical = {
            'n_blanket_dims': int(len(blanket_dims)),
            'top_physical_vars': [STATE_LABELS[i] for i in blanket_top],
            'top_correlations': blanket_mean_corr[blanket_top].tolist(),
        }
    else:
        blanket_physical = {'n_blanket_dims': 0}

    # Compute normalized mutual information between projected labels
    # Project latent partition to physical space via strongest correlate
    if len(strongest_correlate) > 0 and len(actinf_assign) > 0:
        # For each physical variable, find the majority latent cluster assignment
        physical_assignment = np.zeros(8, dtype=int)
        for phys_idx in range(8):
            latent_dims_for_phys = np.where(strongest_correlate == phys_idx)[0]
            if len(latent_dims_for_phys) > 0:
                cluster_labels = dreamer_assign[latent_dims_for_phys]
                # Majority vote
                unique, counts = np.unique(cluster_labels, return_counts=True)
                physical_assignment[phys_idx] = unique[np.argmax(counts)]

        # NMI between state-space partition and projected latent partition
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(actinf_assign, physical_assignment)
    else:
        nmi = None
        physical_assignment = None

    return {
        'cluster_to_physical': cluster_to_physical,
        'blanket_physical': blanket_physical,
        'nmi_state_vs_projected_latent': float(nmi) if nmi is not None else None,
        'projected_latent_assignment': physical_assignment.tolist() if physical_assignment is not None else None,
        'state_assignment': actinf_assign.tolist() if len(actinf_assign) > 0 else None,
    }


# =========================================================================
# Visualization
# =========================================================================

def plot_comparison_figure(actinf_summary, dreamer_summary, disagree_summary=None):
    """Side-by-side coupling matrices for all representations."""
    panels = [
        ('8D State Space\n(Active Inference Dynamics)', actinf_summary),
        ('64D Latent Space\n(Dreamer Autoencoder)', dreamer_summary),
    ]
    if disagree_summary is not None:
        panels.insert(1, ('8D State Space\n(Ensemble Disagreement)', disagree_summary))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (title, summary) in zip(axes, panels):
        coupling = summary['coupling']
        if len(coupling) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            continue

        coupling_abs = np.abs(coupling)
        im = ax.imshow(coupling_abs, cmap='YlOrRd', aspect='auto')

        n_dims = coupling.shape[0]
        if n_dims <= 10:
            ax.set_xticks(range(n_dims))
            ax.set_xticklabels(STATE_LABELS[:n_dims], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(n_dims))
            ax.set_yticklabels(STATE_LABELS[:n_dims], fontsize=8)
        else:
            ax.set_xlabel(f'Latent Dimension ({n_dims}D)')
            ax.set_ylabel(f'Latent Dimension')

        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate with key stats
        stats_text = (f"Objects: {summary['n_objects']}\n"
                      f"Blanket: {summary['n_blanket']}\n"
                      f"Eigengap: {summary['eigengap']:.2f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_eigenvalue_comparison(actinf_summary, dreamer_summary):
    """Side-by-side eigenvalue spectra."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    eig1 = actinf_summary['eigenvalues']
    ax1.plot(range(len(eig1)), eig1, 'o-', color='#3498db', markersize=5)
    ax1.set_title(f'8D State Space (eigengap={actinf_summary["eigengap"]:.2f})')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.grid(True, alpha=0.3)

    eig2 = dreamer_summary['eigenvalues']
    ax2.plot(range(len(eig2)), eig2, 'o-', color='#e74c3c', markersize=4)
    ax2.set_title(f'64D Latent Space (eigengap={dreamer_summary["eigengap"]:.2f})')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correspondence_heatmap(correspondence, dreamer_results):
    """Visualize latent-to-physical mapping per cluster."""
    dreamer_data = dreamer_results if 'metrics' not in dreamer_results else dreamer_results['metrics']
    correlation = np.array(dreamer_data.get('latent_physical_correlation', []))

    if len(correlation) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No correlation data available', ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(np.abs(correlation).T, cmap='viridis', aspect='auto')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Physical State Variable')
    ax.set_yticks(range(8))
    ax.set_yticklabels(STATE_LABELS, fontsize=9)
    ax.set_title('Latent-to-Physical Correlation (|r|) with Partition Overlay')

    # Draw partition boundaries from Dreamer TB result
    methods_data = dreamer_data.get('methods', {})
    assignment = np.array(methods_data.get('gradient', {}).get('assignment', []))
    if len(assignment) > 0:
        sorted_idx = np.argsort(assignment)
        unique_labels = sorted(set(assignment))
        boundaries = []
        for label in unique_labels:
            count = int(np.sum(assignment == label))
            if boundaries:
                boundaries.append(boundaries[-1] + count)
            else:
                boundaries.append(count)
        for b in boundaries[:-1]:
            ax.axvline(x=b - 0.5, color='cyan', linewidth=1.5, alpha=0.7, linestyle='--')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


# =========================================================================
# Main experiment
# =========================================================================

def run_multi_scale_comparison():
    """US-030: Multi-scale comparison across representations."""
    print("=" * 70)
    print("US-030: Multi-Scale Comparison Across Representations")
    print("=" * 70)

    # Load results
    actinf = load_actinf_results()
    dreamer = load_dreamer_results()

    # Extract summaries
    actinf_dynamics = extract_representation_summary(actinf, 'Active Inference Dynamics', 'gradient')
    dreamer_latent = extract_representation_summary(dreamer, 'Dreamer 64D Latent', 'gradient')

    # Also extract disagreement if available
    actinf_metrics = actinf.get('metrics', actinf)
    disagree_summary = None
    if 'disagreement' in actinf_metrics:
        disagree_data = actinf_metrics['disagreement']
        disagree_method = disagree_data.get('gradient_method', {})
        disagree_assign = np.array(disagree_method.get('assignment', []))
        disagree_blanket = np.array(disagree_method.get('is_blanket', []))
        disagree_summary = {
            'name': 'Ensemble Disagreement',
            'n_dims': len(disagree_assign),
            'n_objects': len(set(disagree_assign[disagree_assign >= 0])) if len(disagree_assign) > 0 else 0,
            'n_blanket': int(np.sum(disagree_blanket)) if len(disagree_blanket) > 0 else 0,
            'eigengap': float(disagree_data.get('eigengap', 0)),
            'coupling': np.array(disagree_data.get('coupling', [])),
            'eigenvalues': disagree_data.get('eigenvalues', [])[:20],
            'assignment': disagree_assign,
            'is_blanket': disagree_blanket,
        }

    # Print comparison table
    print("\n--- Comparison Table ---")
    print(f"{'Representation':<35} {'Dims':>5} {'Objects':>8} {'Blanket':>8} {'Eigengap':>10}")
    print("-" * 70)
    for s in [actinf_dynamics, disagree_summary, dreamer_latent]:
        if s is not None:
            print(f"{s['name']:<35} {s['n_dims']:>5} {s['n_objects']:>8} {s['n_blanket']:>8} {s['eigengap']:>10.3f}")

    # Object correspondence analysis
    print("\n--- Object Correspondence Analysis ---")
    correspondence = compute_object_correspondence(actinf, dreamer)

    if 'cluster_to_physical' in correspondence:
        for cluster_id, info in correspondence['cluster_to_physical'].items():
            label = 'Blanket' if cluster_id < 0 else f'Object {cluster_id}'
            print(f"  Latent {label} ({info['n_latent_dims']} dims):")
            for var, corr in zip(info['top_physical_vars'], info['top_correlations']):
                print(f"    -> {var}: {corr:.3f}")

    if correspondence.get('nmi_state_vs_projected_latent') is not None:
        nmi = correspondence['nmi_state_vs_projected_latent']
        print(f"\n  NMI (state partition vs projected latent partition): {nmi:.3f}")

    # Key insight
    nmi_val = correspondence.get('nmi_state_vs_projected_latent', 0) or 0
    if nmi_val > 0.3:
        insight = "Learned representations partially preserve the physical Markov blanket structure"
    elif nmi_val > 0.1:
        insight = "Learned representations weakly preserve the physical Markov blanket structure"
    else:
        insight = "Learned representations do not directly preserve the physical Markov blanket structure, though individual latent dimensions show strong correlations with physical variables"
    print(f"\n  Key insight: {insight}")

    # Plots
    fig_compare = plot_comparison_figure(actinf_dynamics, dreamer_latent, disagree_summary)
    save_figure(fig_compare, 'multi_scale_coupling_comparison', 'multi_scale_comparison')

    fig_eig = plot_eigenvalue_comparison(actinf_dynamics, dreamer_latent)
    save_figure(fig_eig, 'multi_scale_eigenvalue_comparison', 'multi_scale_comparison')

    fig_corr = plot_correspondence_heatmap(correspondence, dreamer)
    save_figure(fig_corr, 'multi_scale_correspondence', 'multi_scale_comparison')

    # Build results
    comparison_table = []
    for s in [actinf_dynamics, disagree_summary, dreamer_latent]:
        if s is not None:
            comparison_table.append({
                'name': s['name'],
                'n_dims': s['n_dims'],
                'n_objects': s['n_objects'],
                'n_blanket': s['n_blanket'],
                'eigengap': s['eigengap'],
            })

    all_results = {
        'comparison_table': comparison_table,
        'object_correspondence': correspondence,
        'key_insight': insight,
        'representations': {
            'actinf_dynamics': {
                'n_dims': actinf_dynamics['n_dims'],
                'n_objects': actinf_dynamics['n_objects'],
                'n_blanket': actinf_dynamics['n_blanket'],
                'eigengap': actinf_dynamics['eigengap'],
                'assignment': actinf_dynamics['assignment'].tolist(),
            },
            'dreamer_latent': {
                'n_dims': dreamer_latent['n_dims'],
                'n_objects': dreamer_latent['n_objects'],
                'n_blanket': dreamer_latent['n_blanket'],
                'eigengap': dreamer_latent['eigengap'],
            },
        },
    }

    if disagree_summary is not None:
        all_results['representations']['ensemble_disagreement'] = {
            'n_dims': disagree_summary['n_dims'],
            'n_objects': disagree_summary['n_objects'],
            'n_blanket': disagree_summary['n_blanket'],
            'eigengap': disagree_summary['eigengap'],
        }

    save_results('multi_scale_comparison', all_results, {
        'state_labels': STATE_LABELS,
    }, notes='US-030: Multi-scale comparison across 8D state, 8D disagreement, and 64D Dreamer latent. '
             'Coupling matrices, eigenvalue spectra, and latent-to-physical correspondence.')

    print("\nUS-030 complete.")
    return all_results


if __name__ == '__main__':
    run_multi_scale_comparison()
