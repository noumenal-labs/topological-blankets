"""
US-043: Apply TB to Pixel Encoder Latent Space (64D)
=====================================================

Run the full TB pipeline on dynamics prediction error gradients from
the pixel agent's 64D latent space. Compare discovered structure to
the state-space TB results (US-025).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectory_data"

sys.path.insert(0, str(PROJECT_ROOT))

from topological_blankets import TopologicalBlankets
from topological_blankets.features import compute_geometric_features
from topological_blankets.spectral import (
    build_adjacency_from_hessian,
    build_graph_laplacian,
    recursive_spectral_detection,
    compute_eigengap,
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


def load_data():
    """Load pixel trajectory data."""
    latents = np.load(TRAJECTORY_DIR / 'pixel_latents_50ep.npy')
    states = np.load(TRAJECTORY_DIR / 'pixel_states_50ep.npy')
    gradients = np.load(TRAJECTORY_DIR / 'pixel_dynamics_gradients_50ep.npy')
    print(f"Loaded pixel data: {latents.shape[0]} transitions")
    print(f"  Latents: {latents.shape}")
    print(f"  Gradients: {gradients.shape}")
    print(f"  States: {states.shape}")
    return latents, states, gradients


def run_tb_analysis(gradients, latents, states):
    """Run full TB pipeline on pixel latent gradients."""
    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    print("\n--- TB Pipeline on 64D Pixel Latent Gradients ---")

    # Compute features
    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']
    coupling = features['coupling']

    print(f"Hessian shape: {H_est.shape}")
    print(f"Coupling matrix range: [{coupling.min():.4f}, {coupling.max():.4f}]")

    # Spectral analysis
    from scipy.linalg import eigh
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    eigvals, eigvecs = eigh(L)

    n_clusters, eigengap = compute_eigengap(eigvals[:min(20, len(eigvals))])
    print(f"Spectral eigengap: {eigengap:.2f}")
    print(f"Suggested n_clusters: {n_clusters}")

    # Run TB with multiple methods
    results = {}
    for method in ['gradient', 'coupling', 'hybrid']:
        try:
            tb = TopologicalBlankets(method=method, n_objects=max(1, n_clusters - 1))
            tb.fit(gradients)
            assign = tb.get_assignment()
            blankets = tb.get_blankets()
            objects = tb.get_objects()

            n_blanket = len(blankets)
            obj_sizes = {str(k): len(v) for k, v in objects.items()}

            results[method] = {
                'assignment': assign.tolist(),
                'is_blanket': (assign == -1).tolist(),
                'n_blanket': n_blanket,
                'object_sizes': obj_sizes,
            }

            print(f"\n  {method}: {len(objects)} objects, {n_blanket} blanket dims")
            for obj_id, dims in objects.items():
                print(f"    Object {obj_id}: {len(dims)} dims")
        except Exception as e:
            print(f"  {method}: failed ({e})")
            results[method] = {'error': str(e)}

    # Hierarchical detection
    hierarchy = recursive_spectral_detection(H_est, max_levels=3)
    print(f"\n  Hierarchical: {len(hierarchy)} levels")
    for level in hierarchy:
        print(f"    Level {level['level']}: {len(level['internals'])} internal, "
              f"{len(level['blanket'])} blanket, eigengap={level['eigengap']:.2f}")

    # Latent-to-physical correlation
    corr_matrix = np.zeros((64, 8))
    for i in range(64):
        for j in range(8):
            c = np.corrcoef(latents[:, i], states[:, j])[0, 1]
            corr_matrix[i, j] = c if np.isfinite(c) else 0.0

    print(f"\n--- Latent-to-Physical Correlation ---")
    for j, label in enumerate(state_labels):
        top_dims = np.argsort(np.abs(corr_matrix[:, j]))[-3:][::-1]
        top_corrs = [corr_matrix[d, j] for d in top_dims]
        print(f"  {label}: dims {list(top_dims)} (r={[f'{c:.3f}' for c in top_corrs]})")

    # Gradient-to-physical correlation
    grad_corr_matrix = np.zeros((64, 8))
    for i in range(64):
        for j in range(8):
            c = np.corrcoef(gradients[:, i], states[:, j])[0, 1]
            grad_corr_matrix[i, j] = c if np.isfinite(c) else 0.0

    return {
        'features': features,
        'eigvals': eigvals,
        'eigengap': float(eigengap),
        'n_clusters': int(n_clusters),
        'methods': results,
        'hierarchy': hierarchy,
        'corr_matrix': corr_matrix,
        'grad_corr_matrix': grad_corr_matrix,
        'coupling': coupling,
    }


def compare_with_state_space(tb_results, states, latents):
    """Compare pixel-latent TB structure with state-space TB results."""
    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    print("\n--- Comparison with State-Space TB (US-025) ---")

    # Load actinf results
    import json
    actinf_files = sorted(RESULTS_DIR.glob('*actinf_tb_analysis.json'))
    if not actinf_files:
        print("  No state-space TB results found. Skipping comparison.")
        return {}

    with open(actinf_files[-1]) as f:
        actinf_data = json.load(f)

    state_assign = actinf_data['metrics']['dynamics']['gradient_method']['assignment']
    state_blanket = actinf_data['metrics']['dynamics']['gradient_method']['is_blanket']

    print(f"  State-space assignment: {state_assign}")
    print(f"  State-space blanket: {[state_labels[i] for i, b in enumerate(state_blanket) if b]}")

    # Project pixel-latent partition to physical space
    corr = tb_results['corr_matrix']  # (64, 8)
    gradient_method = tb_results['methods'].get('gradient', {})
    pixel_assign = gradient_method.get('assignment', [0]*64)

    # For each physical variable, find its best-correlated latent dim
    # Then use that latent dim's TB assignment as the projected assignment
    projected_assign = []
    for j in range(8):
        best_latent = np.argmax(np.abs(corr[:, j]))
        projected_assign.append(pixel_assign[best_latent])
    print(f"  Projected pixel assignment: {projected_assign}")

    # NMI between state and projected pixel partition
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(state_assign, projected_assign)
    print(f"  NMI (state vs projected pixel): {nmi:.3f}")

    # Physical variable recovery
    # Check if variables in the same state-space object end up in the same pixel cluster
    state_obj0 = [i for i, a in enumerate(state_assign) if a == 0]
    state_obj1 = [i for i, a in enumerate(state_assign) if a == 1]
    state_blanket_idx = [i for i, b in enumerate(state_blanket) if b]

    print(f"\n  Physical grouping recovery:")
    print(f"    State Object 0 {[state_labels[i] for i in state_obj0]}:")
    print(f"      -> Pixel: {[projected_assign[i] for i in state_obj0]}")
    print(f"    State Object 1 {[state_labels[i] for i in state_obj1]}:")
    print(f"      -> Pixel: {[projected_assign[i] for i in state_obj1]}")
    print(f"    State Blanket {[state_labels[i] for i in state_blanket_idx]}:")
    print(f"      -> Pixel: {[projected_assign[i] for i in state_blanket_idx]}")

    return {
        'state_assignment': state_assign,
        'projected_pixel_assignment': projected_assign,
        'nmi': float(nmi),
        'state_labels': state_labels,
    }


def plot_results(tb_results, comparison):
    """Generate visualization PNGs."""
    coupling = tb_results['coupling']
    corr = tb_results['corr_matrix']
    eigvals = tb_results['eigvals']
    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    # Figure 1: 64x64 coupling matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(coupling, cmap='hot', aspect='auto')
    ax.set_title('Pixel Encoder Latent Coupling Matrix (64x64)')
    ax.set_xlabel('Latent Dimension j')
    ax.set_ylabel('Latent Dimension i')
    plt.colorbar(im, ax=ax)

    # Mark TB partition boundaries
    gradient_method = tb_results['methods'].get('gradient', {})
    assign = gradient_method.get('assignment', [])
    if assign:
        blanket_dims = [i for i, a in enumerate(assign) if a == -1]
        for d in blanket_dims[:5]:  # Mark first few blanket dims
            ax.axhline(y=d, color='cyan', alpha=0.3, linewidth=0.5)
            ax.axvline(x=d, color='cyan', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_figure(fig, 'pixel_coupling_64d', 'pixel_tb')
    plt.close(fig)

    # Figure 2: Latent-to-physical correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(np.abs(corr), cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)
    ax.set_xlabel('Physical Variable')
    ax.set_ylabel('Latent Dimension')
    ax.set_xticks(range(8))
    ax.set_xticklabels(state_labels, rotation=45, ha='right')
    ax.set_title('|Correlation|: Latent Dims vs Physical State')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_figure(fig, 'pixel_latent_physical_corr', 'pixel_tb')
    plt.close(fig)

    # Figure 3: Eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eigvals[:20], 'o-', color='#e74c3c', markersize=4)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Graph Laplacian Spectrum (eigengap={tb_results["eigengap"]:.1f})')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='First gap')
    ax.legend()
    plt.tight_layout()
    save_figure(fig, 'pixel_eigenvalue_spectrum', 'pixel_tb')
    plt.close(fig)

    # Figure 4: Gradient magnitude by latent dimension
    features = tb_results['features']
    grad_mag = features['grad_magnitude']
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = []
    if assign:
        for a in assign:
            if a == -1:
                colors.append('#2ecc71')
            elif a == 0:
                colors.append('#3498db')
            else:
                colors.append('#e74c3c')
    else:
        colors = ['gray'] * 64
    ax.bar(range(64), grad_mag, color=colors)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean |gradient|')
    ax.set_title('Gradient Magnitude per Latent Dimension (color = TB assignment)')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#3498db', label='Object 0'),
        Patch(facecolor='#e74c3c', label='Object 1+'),
        Patch(facecolor='#2ecc71', label='Blanket'),
    ])
    plt.tight_layout()
    save_figure(fig, 'pixel_gradient_magnitude', 'pixel_tb')
    plt.close(fig)

    print(f"\n  Saved 4 figures to results/")


def run_us043():
    """Run US-043: TB on pixel encoder latent space."""
    print("=" * 60)
    print("US-043: Apply TB to Pixel Encoder Latent Space")
    print("=" * 60)

    latents, states, gradients = load_data()
    tb_results = run_tb_analysis(gradients, latents, states)
    comparison = compare_with_state_space(tb_results, states, latents)
    plot_results(tb_results, comparison)

    # Prepare results for JSON serialization
    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    metrics = {
        'eigengap': tb_results['eigengap'],
        'n_clusters_spectral': tb_results['n_clusters'],
        'eigenvalues': tb_results['eigvals'][:20].tolist(),
        'methods': tb_results['methods'],
        'hierarchy': [
            {
                'level': h['level'],
                'n_internals': len(h['internals']),
                'n_blanket': len(h['blanket']),
                'eigengap': float(h['eigengap']),
            }
            for h in tb_results['hierarchy']
        ],
        'latent_physical_correlations': {
            label: {
                'top_3_dims': np.argsort(np.abs(tb_results['corr_matrix'][:, j]))[-3:][::-1].tolist(),
                'top_3_corrs': [
                    float(tb_results['corr_matrix'][d, j])
                    for d in np.argsort(np.abs(tb_results['corr_matrix'][:, j]))[-3:][::-1]
                ],
            }
            for j, label in enumerate(state_labels)
        },
        'comparison': comparison,
    }

    config = {
        'latent_dim': 64,
        'n_transitions': int(gradients.shape[0]),
        'state_labels': state_labels,
    }

    save_results('pixel_tb_analysis', metrics, config,
                 notes='US-043: TB applied to pixel encoder 64D latent gradients. '
                       'Structure compared to state-space TB from US-025.')

    print("\n" + "=" * 60)
    print("US-043 complete.")
    return tb_results, comparison


if __name__ == '__main__':
    run_us043()
