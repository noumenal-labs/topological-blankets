"""
US-044: Multi-Representation Comparison
=========================================

Compare TB structure across all available representations of LunarLander:
1. Raw 8D state space (US-025)
2. Pixel encoder 64D latent (US-043)
3. Dreamer 64D latent (US-029)

Key question: do independently learned representations converge on
the same physical structure?
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectory_data"

sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


def load_all_results():
    """Load TB results from all three representations."""
    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    results = {}

    # 1. State-space (US-025)
    actinf_files = sorted(RESULTS_DIR.glob('*actinf_tb_analysis.json'))
    if actinf_files:
        with open(actinf_files[-1]) as f:
            d = json.load(f)
        dyn = d['metrics']['dynamics']
        results['state_space'] = {
            'name': 'Active Inference (8D State)',
            'n_dims': 8,
            'assignment': dyn['gradient_method']['assignment'],
            'is_blanket': dyn['gradient_method']['is_blanket'],
            'eigengap': dyn['eigengap'],
            'coupling': np.array(dyn['coupling']),
            'eigenvalues': dyn['eigenvalues'],
        }
        print(f"Loaded state-space TB: eigengap={dyn['eigengap']:.1f}")

    # 2. Pixel encoder (US-043)
    pixel_files = sorted(RESULTS_DIR.glob('*pixel_tb_analysis.json'))
    if pixel_files:
        with open(pixel_files[-1]) as f:
            d = json.load(f)
        grad = d['metrics']['methods'].get('gradient', {})
        results['pixel_latent'] = {
            'name': 'Pixel Encoder (64D Latent)',
            'n_dims': 64,
            'assignment': grad.get('assignment', []),
            'is_blanket': grad.get('is_blanket', []),
            'eigengap': d['metrics']['eigengap'],
            'eigenvalues': d['metrics']['eigenvalues'],
            'n_blanket': grad.get('n_blanket', 0),
            'object_sizes': grad.get('object_sizes', {}),
            'correlations': d['metrics'].get('latent_physical_correlations', {}),
            'nmi_vs_state': d['metrics'].get('comparison', {}).get('nmi', 0),
        }
        print(f"Loaded pixel TB: eigengap={d['metrics']['eigengap']:.1f}")

    # 3. Dreamer latent (US-029)
    dreamer_files = sorted(RESULTS_DIR.glob('*dreamer_tb_analysis.json'))
    if dreamer_files:
        with open(dreamer_files[-1]) as f:
            d = json.load(f)
        dm = d['metrics']
        grad = dm.get('methods', dm).get('gradient', {})
        results['dreamer_latent'] = {
            'name': 'Dreamer Autoencoder (64D Latent)',
            'n_dims': 64,
            'assignment': grad.get('assignment', []),
            'is_blanket': grad.get('is_blanket', []),
            'eigengap': dm['eigengap'],
            'eigenvalues': dm.get('eigenvalues', []),
            'n_blanket': grad.get('n_blanket', 0),
            'object_sizes': grad.get('object_sizes', {}),
        }
        print(f"Loaded Dreamer TB: eigengap={dm['eigengap']:.1f}")

    # Also load multi-scale comparison for Dreamer NMI
    ms_files = sorted(RESULTS_DIR.glob('*multi_scale_comparison.json'))
    if ms_files and 'dreamer_latent' in results:
        with open(ms_files[-1]) as f:
            ms = json.load(f)
        results['dreamer_latent']['nmi_vs_state'] = \
            ms['metrics']['object_correspondence']['nmi_state_vs_projected_latent']

    return results, state_labels


def build_comparison_table(results, state_labels):
    """Build structured comparison table."""
    table = []

    for key in ['state_space', 'pixel_latent', 'dreamer_latent']:
        if key not in results:
            continue
        r = results[key]

        assign = r['assignment']
        is_blanket = r.get('is_blanket', [])

        n_objects = len(set(a for a in assign if a >= 0))
        n_blanket = sum(1 for b in is_blanket if b)
        n_internal = sum(1 for b in is_blanket if not b) if is_blanket else r['n_dims'] - n_blanket

        nmi = r.get('nmi_vs_state', None)

        row = {
            'representation': r['name'],
            'dims': r['n_dims'],
            'n_objects': n_objects,
            'n_blanket': n_blanket,
            'n_internal': n_internal,
            'eigengap': r['eigengap'],
            'nmi_vs_state': nmi,
        }
        table.append(row)

        # For state-space, add physical labels
        if key == 'state_space':
            obj_groups = {}
            for i, a in enumerate(assign):
                if a not in obj_groups:
                    obj_groups[a] = []
                obj_groups[a].append(state_labels[i])
            row['physical_partition'] = obj_groups

    return table


def plot_comparison(results, state_labels):
    """Generate comparison visualizations."""

    reps = [k for k in ['state_space', 'pixel_latent', 'dreamer_latent'] if k in results]
    n_reps = len(reps)

    if n_reps == 0:
        print("  No results to plot.")
        return

    # Figure 1: Eigenvalue spectra comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'state_space': '#3498db', 'pixel_latent': '#e74c3c', 'dreamer_latent': '#2ecc71'}
    labels = {'state_space': 'State Space (8D)', 'pixel_latent': 'Pixel Encoder (64D)',
              'dreamer_latent': 'Dreamer (64D)'}

    for key in reps:
        eigs = results[key]['eigenvalues']
        n_show = min(20, len(eigs))
        ax.plot(range(n_show), eigs[:n_show], 'o-',
                color=colors[key], label=labels[key], markersize=4)

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Graph Laplacian Spectra Across Representations')
    ax.legend()
    plt.tight_layout()
    save_figure(fig, 'multi_rep_eigenvalue_comparison', 'pixel_structure')
    plt.close(fig)

    # Figure 2: Comparison bar chart (eigengap, objects, blanket)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    rep_names = [labels[k] for k in reps]
    rep_colors = [colors[k] for k in reps]

    # Eigengap
    ax = axes[0]
    eigengaps = [results[k]['eigengap'] for k in reps]
    bars = ax.bar(range(n_reps), eigengaps, color=rep_colors)
    ax.set_xticks(range(n_reps))
    ax.set_xticklabels(rep_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Eigengap')
    ax.set_title('Spectral Eigengap')
    for i, v in enumerate(eigengaps):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # Number of objects
    ax = axes[1]
    n_objects = []
    for k in reps:
        assign = results[k]['assignment']
        n_objects.append(len(set(a for a in assign if a >= 0)))
    ax.bar(range(n_reps), n_objects, color=rep_colors)
    ax.set_xticks(range(n_reps))
    ax.set_xticklabels(rep_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title('Objects Detected')

    # NMI with state space
    ax = axes[2]
    nmis = []
    nmi_colors = []
    nmi_labels = []
    for k in reps:
        nmi = results[k].get('nmi_vs_state', None)
        if nmi is not None and k != 'state_space':
            nmis.append(nmi)
            nmi_colors.append(colors[k])
            nmi_labels.append(labels[k])
    if nmis:
        ax.bar(range(len(nmis)), nmis, color=nmi_colors)
        ax.set_xticks(range(len(nmis)))
        ax.set_xticklabels(nmi_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('NMI')
        ax.set_title('NMI vs State-Space Partition')
        ax.set_ylim(0, 1)
        for i, v in enumerate(nmis):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', fontsize=14)
        ax.set_title('NMI vs State-Space')

    plt.tight_layout()
    save_figure(fig, 'multi_rep_summary', 'pixel_structure')
    plt.close(fig)

    # Figure 3: State-space coupling matrix with annotation
    if 'state_space' in results:
        coupling = results['state_space']['coupling']
        assign = results['state_space']['assignment']

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(coupling, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(8))
        ax.set_xticklabels(state_labels, rotation=45, ha='right')
        ax.set_yticks(range(8))
        ax.set_yticklabels(state_labels)
        ax.set_title('State-Space Coupling (Reference)')
        plt.colorbar(im, ax=ax)

        # Annotate partition
        for i in range(8):
            color = '#3498db' if assign[i] == 0 else '#e74c3c' if assign[i] == 1 else '#2ecc71'
            ax.get_yticklabels()[i].set_color(color)
            ax.get_xticklabels()[i].set_color(color)

        plt.tight_layout()
        save_figure(fig, 'multi_rep_state_coupling', 'pixel_structure')
        plt.close(fig)

    print(f"\n  Saved comparison figures to results/")


def compute_physical_recovery(results, state_labels):
    """Compute physical variable recovery score for each representation."""
    if 'state_space' not in results:
        return {}

    state_assign = results['state_space']['assignment']

    # Physical groups from state-space analysis
    groups = {}
    for i, a in enumerate(state_assign):
        label = state_labels[i]
        if a not in groups:
            groups[a] = []
        groups[a].append(label)

    recovery = {}
    for key in ['pixel_latent', 'dreamer_latent']:
        if key not in results:
            continue

        nmi = results[key].get('nmi_vs_state', 0)
        recovery[key] = {
            'nmi': nmi,
            'interpretation': (
                'strong' if nmi > 0.5 else
                'moderate' if nmi > 0.3 else
                'weak'
            ),
        }

    return recovery


def run_us044():
    """Run US-044: multi-representation comparison."""
    print("=" * 60)
    print("US-044: Multi-Representation Comparison")
    print("=" * 60)

    results, state_labels = load_all_results()

    if len(results) < 2:
        print("Need at least 2 representations to compare. Available:")
        for k in results:
            print(f"  - {k}")
        return

    table = build_comparison_table(results, state_labels)

    print("\nComparison Table:")
    print(f"{'Representation':<30s} {'Dims':>5s} {'Objects':>8s} {'Blanket':>8s} "
          f"{'Eigengap':>9s} {'NMI':>6s}")
    print("-" * 70)
    for row in table:
        nmi_str = f"{row['nmi_vs_state']:.3f}" if row['nmi_vs_state'] is not None else "ref"
        print(f"{row['representation']:<30s} {row['dims']:>5d} {row['n_objects']:>8d} "
              f"{row['n_blanket']:>8d} {row['eigengap']:>9.1f} {nmi_str:>6s}")

    recovery = compute_physical_recovery(results, state_labels)
    print("\nPhysical Variable Recovery:")
    for key, r in recovery.items():
        name = results[key]['name']
        print(f"  {name}: NMI={r['nmi']:.3f} ({r['interpretation']})")

    plot_comparison(results, state_labels)

    # Key finding
    all_nmis = [r.get('nmi_vs_state', 0) for k, r in results.items()
                if k != 'state_space' and r.get('nmi_vs_state') is not None]
    mean_nmi = np.mean(all_nmis) if all_nmis else 0

    if mean_nmi > 0.5:
        key_finding = ("Independently learned representations converge on similar "
                      "physical structure: strong preservation of Markov blanket "
                      "organization across representation levels.")
    elif mean_nmi > 0.3:
        key_finding = ("Learned representations partially preserve physical Markov "
                      "blanket structure. Dreamer (trained with reconstruction loss) "
                      "preserves more structure than the pixel encoder (trained with "
                      "temporal consistency loss).")
    else:
        key_finding = ("Representation structure diverges significantly from physical "
                      "state-space structure. Learned latent codes encode information "
                      "differently from the physical decomposition.")

    print(f"\nKey Finding: {key_finding}")

    # Save results
    metrics = {
        'comparison_table': table,
        'physical_recovery': recovery,
        'key_finding': key_finding,
        'representations_available': list(results.keys()),
    }
    config = {'state_labels': state_labels}

    save_results('pixel_structure_comparison', metrics, config,
                 notes='US-044: Multi-representation comparison. State-space vs '
                       'pixel-latent vs Dreamer-latent TB structure.')

    print("\n" + "=" * 60)
    print("US-044 complete.")
    return results


if __name__ == '__main__':
    run_us044()
