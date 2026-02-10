"""
US-033: Publication-Quality Figures for the Paper
==================================================

Generate all figures for the paper's experimental sections in consistent
publication style: matplotlib with publication font sizes, line widths,
color scheme. High-DPI PNG at 300 DPI.
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

from experiments.utils.results import load_results

FIGURES_DIR = os.path.join(NOUMENAL_DIR, 'paper', 'figures')
STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

# Publication style
PUB_RC = {
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
}

# Color scheme: TB in blues, baselines in grays/reds
COLORS = {
    'tb_gradient': '#2980b9',
    'tb_spectral': '#3498db',
    'tb_hybrid': '#1abc9c',
    'tb_coupling': '#27ae60',
    'dmbd': '#e74c3c',
    'axiom': '#e67e22',
    'glasso': '#95a5a6',
    'notears': '#8e44ad',
}


def find_result(pattern):
    """Find latest result matching pattern."""
    results_dir = os.path.join(NOUMENAL_DIR, 'results')
    matches = []
    for f in sorted(os.listdir(results_dir)):
        if pattern in f and f.endswith('.json'):
            matches.append(os.path.join(results_dir, f))
    return load_results(matches[-1]) if matches else {}


def save_pub_figure(fig, name):
    """Save figure at publication quality."""
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png")
    return path


# =========================================================================
# Figure 1: Coupling matrix on quadratic EBM
# =========================================================================

def figure_1():
    """Coupling matrix example on quadratic EBM (annotated)."""
    data = find_result('quadratic_toy_demo')
    if not data:
        print("  SKIP: no quadratic toy demo results")
        return

    metrics = data.get('metrics', {})
    # Get coupling from the TC result
    tb_result = metrics.get('topological_blankets', metrics.get('tc', {}))

    # Generate a clean coupling matrix from a simple quadratic
    from experiments.quadratic_toy_comparison import build_precision_matrix, QuadraticEBMConfig, langevin_sampling
    from topological_blankets.features import compute_geometric_features

    cfg = QuadraticEBMConfig(n_objects=2, vars_per_object=4, vars_per_blanket=2, blanket_strength=0.8)
    Theta = build_precision_matrix(cfg)
    samples, grads = langevin_sampling(Theta, n_samples=3000, n_steps=200, step_size=0.01, temp=0.1)
    features = compute_geometric_features(grads)
    coupling = features['coupling']

    n = coupling.shape[0]
    labels = []
    for i in range(cfg.n_objects):
        for j in range(cfg.vars_per_object):
            labels.append(f'O{i+1}_{j+1}')
    for j in range(cfg.vars_per_blanket):
        labels.append(f'B_{j+1}')

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(coupling), cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title('Coupling Matrix: 2-Object Quadratic EBM')

        # Draw partition boundaries
        for boundary in [cfg.vars_per_object, cfg.n_objects * cfg.vars_per_object]:
            ax.axhline(y=boundary - 0.5, color='cyan', linewidth=1.5, linestyle='--')
            ax.axvline(x=boundary - 0.5, color='cyan', linewidth=1.5, linestyle='--')

        plt.colorbar(im, ax=ax, shrink=0.85, label='|Coupling|')

    return save_pub_figure(fig, 'fig1_coupling_matrix')


# =========================================================================
# Figure 2: Strength sweep ARI comparison
# =========================================================================

def figure_2():
    """Strength sweep ARI comparison (TC vs baselines, error bars)."""
    data = find_result('v2_strength_sweep')
    if not data:
        print("  SKIP: no v2 strength sweep results")
        return

    metrics = data.get('metrics', {})

    # v2 format: keys are strength strings, values are method dicts
    strengths = sorted([float(k) for k in metrics.keys()])
    if not strengths:
        print("  SKIP: empty sweep results")
        return

    with plt.rc_context(PUB_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        for method_key, color, label in [
            ('gradient', COLORS['tb_gradient'], 'TB (gradient)'),
            ('coupling', COLORS['tb_coupling'], 'TB (coupling)'),
            ('dmbd', COLORS['dmbd'], 'DMBD'),
            ('axiom', COLORS['axiom'], 'AXIOM'),
        ]:
            aris = []
            f1s = []
            for s in strengths:
                method_data = metrics[str(s) if str(s) in metrics else f'{s:.1f}'].get(method_key, {})
                aris.append(method_data.get('mean_ari', 0))
                f1s.append(method_data.get('mean_f1', 0))

            if any(a > 0 for a in aris):
                ax1.plot(strengths, aris, 'o-', color=color, label=label)
            if any(f > 0 for f in f1s):
                ax2.plot(strengths, f1s, 'o-', color=color, label=label)

        ax1.set_xlabel('Blanket Strength')
        ax1.set_ylabel('Object ARI')
        ax1.set_title('Object Recovery vs Blanket Strength')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.1)

        ax2.set_xlabel('Blanket Strength')
        ax2.set_ylabel('Blanket F1')
        ax2.set_title('Blanket Detection vs Blanket Strength')
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.1)

    return save_pub_figure(fig, 'fig2_strength_sweep')


# =========================================================================
# Figure 3: Scaling heatmap
# =========================================================================

def figure_3():
    """Scaling heatmap (n_objects x vars_per_object)."""
    data = find_result('v2_scaling')
    if not data:
        print("  SKIP: no v2 scaling results")
        return

    metrics = data.get('metrics', {})

    # v2 format: keys like '2obj_3vpo', values have n_objects, vars_per_object, gradient dict
    if not metrics:
        print("  SKIP: empty scaling results")
        return

    entries = []
    for key, val in metrics.items():
        if isinstance(val, dict) and 'n_objects' in val:
            entries.append(val)

    if not entries:
        print("  SKIP: no scaling entries")
        return

    n_objects_vals = sorted(set(e['n_objects'] for e in entries))
    vars_vals = sorted(set(e['vars_per_object'] for e in entries))

    ari_matrix = np.zeros((len(n_objects_vals), len(vars_vals)))
    for e in entries:
        i = n_objects_vals.index(e['n_objects'])
        j = vars_vals.index(e['vars_per_object'])
        grad_data = e.get('gradient', {})
        ari_matrix[i, j] = grad_data.get('mean_ari', 0)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(ari_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(vars_vals)))
        ax.set_xticklabels(vars_vals)
        ax.set_yticks(range(len(n_objects_vals)))
        ax.set_yticklabels(n_objects_vals)
        ax.set_xlabel('Variables per Object')
        ax.set_ylabel('Number of Objects')
        ax.set_title('Scaling: Object ARI')

        for i in range(len(n_objects_vals)):
            for j in range(len(vars_vals)):
                ax.text(j, i, f'{ari_matrix[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if ari_matrix[i,j] > 0.5 else 'white')

        plt.colorbar(im, ax=ax, shrink=0.85, label='ARI')

    return save_pub_figure(fig, 'fig3_scaling_heatmap')


# =========================================================================
# Figure 4: Temperature sensitivity
# =========================================================================

def figure_4():
    """Temperature sensitivity curves."""
    data = find_result('v2_temperature')
    if not data:
        print("  SKIP: no v2 temperature results")
        return

    metrics = data.get('metrics', {})

    # v2 format: keys are temperature strings, values are method dicts
    temps = sorted([float(k) for k in metrics.keys()])
    if not temps:
        print("  SKIP: empty temperature results")
        return

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for method_key, color, label in [
            ('gradient', COLORS['tb_gradient'], 'TB (gradient)'),
            ('coupling', COLORS['tb_coupling'], 'TB (coupling)'),
            ('dmbd', COLORS['dmbd'], 'DMBD'),
            ('axiom', COLORS['axiom'], 'AXIOM'),
        ]:
            aris = []
            for t in temps:
                t_key = str(t) if str(t) in metrics else f'{t:.2f}'
                method_data = metrics.get(t_key, {}).get(method_key, {})
                aris.append(method_data.get('mean_ari', 0))
            if any(a > 0 for a in aris):
                ax.plot(temps, aris, 'o-', color=color, label=label)

        ax.set_xlabel('Temperature')
        ax.set_ylabel('Object ARI')
        ax.set_title('Geometric-to-Topological Transition')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

    return save_pub_figure(fig, 'fig4_temperature_sensitivity')


# =========================================================================
# Figure 5: 2D score model
# =========================================================================

def figure_5():
    """2D score model with blanket overlay (use pre-generated plots)."""
    # Just reference the existing plots; generate a clean version
    data = find_result('score_model_2d')
    if not data:
        print("  SKIP: no score model results")
        return

    # Use the existing PNG as-is, or generate a minimal version
    import glob
    score_pngs = glob.glob(os.path.join(NOUMENAL_DIR, 'results', '*score_model_2d_score_field_moons*.png'))
    cluster_pngs = glob.glob(os.path.join(NOUMENAL_DIR, 'results', '*score_model_2d_sample_clusters_moons*.png'))

    if score_pngs and cluster_pngs:
        # Copy latest
        import shutil
        shutil.copy(sorted(score_pngs)[-1], os.path.join(FIGURES_DIR, 'fig5_score_model_2d.png'))
        print("  Saved: fig5_score_model_2d.png (copied from results)")
    else:
        print("  SKIP: no score model PNGs found")


# =========================================================================
# Figure 6: Ising model
# =========================================================================

def figure_6():
    """Ising model at 3 temperatures with blanket overlay."""
    import glob
    ising_pngs = glob.glob(os.path.join(NOUMENAL_DIR, 'results', '*ising_model_ising_three_temps*.png'))
    if ising_pngs:
        import shutil
        shutil.copy(sorted(ising_pngs)[-1], os.path.join(FIGURES_DIR, 'fig6_ising_model.png'))
        print("  Saved: fig6_ising_model.png (copied from results)")
    else:
        print("  SKIP: no ising model PNGs found")


# =========================================================================
# Figure 7: LunarLander 8x8 coupling matrix
# =========================================================================

def figure_7():
    """LunarLander state-space coupling matrix (8x8, labeled axes)."""
    data = find_result('actinf_tb_analysis')
    if not data:
        print("  SKIP: no actinf TB results")
        return

    metrics = data.get('metrics', data)
    dynamics = metrics.get('dynamics', metrics)
    coupling = np.array(dynamics.get('coupling', []))

    if coupling.size == 0:
        print("  SKIP: no coupling data")
        return

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(coupling), cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(8))
        ax.set_xticklabels(STATE_LABELS, rotation=45, ha='right')
        ax.set_yticks(range(8))
        ax.set_yticklabels(STATE_LABELS)
        ax.set_title('Active Inference: Dynamics Coupling Matrix (8D)')

        for i in range(8):
            for j in range(8):
                val = np.abs(coupling[i, j])
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if val > np.max(np.abs(coupling)) * 0.6 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.85, label='|Coupling|')

    return save_pub_figure(fig, 'fig7_lunarlander_coupling')


# =========================================================================
# Figure 8: Dreamer 64x64 coupling matrix
# =========================================================================

def figure_8():
    """Dreamer latent-space coupling matrix (64x64)."""
    import glob
    dreamer_pngs = glob.glob(os.path.join(NOUMENAL_DIR, 'results', '*dreamer_coupling_64d*.png'))
    if dreamer_pngs:
        import shutil
        shutil.copy(sorted(dreamer_pngs)[-1], os.path.join(FIGURES_DIR, 'fig8_dreamer_coupling.png'))
        print("  Saved: fig8_dreamer_coupling.png (copied from results)")
    else:
        print("  SKIP: no dreamer coupling PNGs found")


# =========================================================================
# Figure 9: Multi-scale comparison
# =========================================================================

def figure_9():
    """Multi-scale comparison (3-panel)."""
    import glob
    multi_pngs = glob.glob(os.path.join(NOUMENAL_DIR, 'results', '*multi_scale_coupling_comparison*.png'))
    if multi_pngs:
        import shutil
        shutil.copy(sorted(multi_pngs)[-1], os.path.join(FIGURES_DIR, 'fig9_multi_scale_comparison.png'))
        print("  Saved: fig9_multi_scale_comparison.png (copied from results)")
    else:
        print("  SKIP: no multi-scale PNGs found")


# =========================================================================
# Main
# =========================================================================

def run_us033():
    """US-033: Generate all publication figures."""
    print("=" * 70)
    print("US-033: Publication-Quality Figures")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\nGenerating figures:")
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()

    # List all generated figures
    print(f"\nFigures in {FIGURES_DIR}:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f} ({size//1024}KB)")

    # LaTeX include commands
    print("\nLaTeX includegraphics commands:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        name = f.replace('.png', '')
        print(f"  \\includegraphics[width=\\textwidth]{{figures/{f}}}")

    print("\nUS-033 complete.")


if __name__ == '__main__':
    run_us033()
