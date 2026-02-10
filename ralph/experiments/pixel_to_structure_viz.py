"""
US-045: End-to-End Pixel-to-Structure Visualization
=====================================================

Create a compelling multi-panel visualization showing the full pipeline
from raw camera frames to discovered structure.

Panels:
1. Sample LunarLander frames at different flight phases
2. 64D latent-space coupling matrix with TB partition boundaries
3. Latent-to-physical correlation heatmap (64 x 8)
4. Summary: discovered objects and blankets with physical labels
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LUNAR_LANDER_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_PATH = LUNAR_LANDER_ROOT / "trained_agents" / "pixel_lunarlander_best.tar"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectory_data"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

sys.path.insert(0, str(LUNAR_LANDER_ROOT))
sys.path.insert(0, str(LUNAR_LANDER_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.results import save_results


def collect_sample_frames():
    """Collect frames at different flight phases."""
    import gymnasium as gym
    import torch
    from active_inference.pixel_agent import PixelActiveInferenceAgent
    from active_inference.frame_stack import preprocess_frame

    # Load agent
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    agent = PixelActiveInferenceAgent(
        n_actions=4, config=ckpt['config'], pixel_config=ckpt['pixel_config'])
    agent.load(str(CHECKPOINT_PATH))
    agent.encoder.eval()

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    agent.reset_frame_stack()

    frames = []
    latents = []
    labels = []
    states_at_frame = []

    frame = env.render()
    stacked = agent.frame_stack.get_observation(frame)

    for step in range(200):
        state = np.array(obs)

        # Capture key phases
        capture = False
        if step == 0:
            capture = True
            labels.append('Launch')
        elif step == 30:
            capture = True
            labels.append('Mid-flight')
        elif step == 60:
            capture = True
            labels.append('Descent')
        elif state[1] < 0.1 and step > 50:
            capture = True
            labels.append('Near ground')

        if capture:
            frames.append(frame.copy())
            z = agent.encode_frames(stacked)
            latents.append(z.copy())
            states_at_frame.append(state.copy())
            if len(frames) >= 4:
                break

        action = agent.select_action(stacked, epsilon=0.1)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        stacked = agent.frame_stack.get_observation(frame)

        if terminated or truncated:
            # Capture final frame
            if len(frames) < 4:
                frames.append(frame.copy())
                z = agent.encode_frames(stacked)
                latents.append(z.copy())
                states_at_frame.append(np.array(obs))
                labels.append('Terminal')
            break

    env.close()

    # Pad to 4 if needed
    while len(frames) < 4:
        frames.append(frames[-1].copy())
        latents.append(latents[-1].copy())
        states_at_frame.append(states_at_frame[-1].copy())
        labels.append('')

    return frames[:4], np.array(latents[:4]), labels[:4], np.array(states_at_frame[:4])


def create_hero_figure(frames, frame_latents, frame_labels):
    """Create the multi-panel hero figure."""
    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    # Load TB results
    pixel_tb_files = sorted(RESULTS_DIR.glob('*pixel_tb_analysis.json'))
    if not pixel_tb_files:
        print("No pixel TB results found. Run US-043 first.")
        return

    with open(pixel_tb_files[-1]) as f:
        tb_data = json.load(f)

    # Load pixel trajectory data for correlations
    latents = np.load(TRAJECTORY_DIR / 'pixel_latents_50ep.npy')
    states = np.load(TRAJECTORY_DIR / 'pixel_states_50ep.npy')

    # Compute correlation matrix
    corr_matrix = np.zeros((64, 8))
    for i in range(64):
        for j in range(8):
            c = np.corrcoef(latents[:, i], states[:, j])[0, 1]
            corr_matrix[i, j] = c if np.isfinite(c) else 0.0

    # TB partition
    grad_method = tb_data['metrics']['methods'].get('gradient', {})
    assign = grad_method.get('assignment', [0]*64)
    is_blanket = grad_method.get('is_blanket', [False]*64)
    n_blanket = sum(is_blanket)
    n_internal = 64 - n_blanket

    # === Create Figure ===
    fig = plt.figure(figsize=(16, 10))
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(2, 12, figure=fig, hspace=0.35, wspace=0.5)

    # --- Top Row: 4 Sample Frames (each spanning 3 columns) ---
    for i in range(4):
        ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
        ax.imshow(frames[i])
        ax.set_title(frame_labels[i], fontsize=11)
        ax.axis('off')

    # --- Bottom Left: Coupling Matrix (4 columns) ---
    ax = fig.add_subplot(gs[1, 0:4])

    # Sort dimensions by assignment for visual clarity
    sorted_dims = sorted(range(64), key=lambda d: (assign[d], d))
    coupling_sorted = np.zeros((64, 64))
    for i, di in enumerate(sorted_dims):
        for j, dj in enumerate(sorted_dims):
            # Reconstruct coupling from Hessian if available
            coupling_sorted[i, j] = np.abs(corr_matrix[di]).max()  # Placeholder

    # Use actual gradient covariance for coupling
    gradients = np.load(TRAJECTORY_DIR / 'pixel_dynamics_gradients_50ep.npy')
    features_H = np.cov(gradients.T)
    D = np.sqrt(np.abs(np.diag(features_H))) + 1e-10
    coupling_full = np.abs(features_H) / np.outer(D, D)
    np.fill_diagonal(coupling_full, 0)

    coupling_sorted = coupling_full[np.ix_(sorted_dims, sorted_dims)]

    im = ax.imshow(coupling_sorted, cmap='hot', aspect='auto', vmin=0, vmax=coupling_sorted.max())
    ax.set_title('Latent Coupling (sorted by TB)', fontsize=11)
    ax.set_xlabel('Latent dim (sorted)')
    ax.set_ylabel('Latent dim (sorted)')

    # Mark blanket boundary
    n_obj0 = sum(1 for a in assign if a == 0)
    ax.axhline(y=n_obj0-0.5, color='cyan', linewidth=1.5, linestyle='--')
    ax.axvline(x=n_obj0-0.5, color='cyan', linewidth=1.5, linestyle='--')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # --- Bottom Center: Latent-Physical Correlation (4 columns) ---
    ax = fig.add_subplot(gs[1, 4:8])

    # Sort latent dims by assignment
    corr_sorted = corr_matrix[sorted_dims, :]
    im = ax.imshow(np.abs(corr_sorted), cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)
    ax.set_xlabel('Physical Variable')
    ax.set_ylabel('Latent Dimension (sorted)')
    ax.set_xticks(range(8))
    ax.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=9)
    ax.set_title('|Correlation|: Latent vs Physical', fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # --- Bottom Right: Summary Diagram (4 columns) ---
    ax = fig.add_subplot(gs[1, 8:12])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Discovered Structure', fontsize=11)

    # Text summary
    eigengap = tb_data['metrics']['eigengap']

    ax.text(5, 9.2, 'Pixel $\\to$ Structure Pipeline', ha='center', fontsize=12,
            fontweight='bold')

    ax.text(0.5, 7.8, f'Encoder: CNN (4$\\times$84$\\times$84 $\\to$ 64D)', fontsize=10)
    ax.text(0.5, 7.0, f'Spectral eigengap: {eigengap:.1f}', fontsize=10)
    ax.text(0.5, 6.2, f'Object: {n_internal} latent dims', fontsize=10, color='#3498db')
    ax.text(0.5, 5.4, f'Blanket: {n_blanket} latent dims', fontsize=10, color='#2ecc71')

    ax.text(0.5, 4.2, 'Top physical correspondences:', fontsize=10, fontweight='bold')

    corr_info = tb_data['metrics'].get('latent_physical_correlations', {})
    y_pos = 3.4
    for label in ['vy', 'y', 'vx', 'x', 'angle']:
        if label in corr_info:
            c_data = corr_info[label]
            top_dim = c_data['top_3_dims'][0]
            top_corr = c_data['top_3_corrs'][0]
            role = 'blanket' if is_blanket[top_dim] else 'object'
            ax.text(1.0, y_pos, f'{label}: dim {top_dim} (r={abs(top_corr):.3f}, {role})',
                    fontsize=9)
            y_pos -= 0.7

    ax.text(0.5, 0.8, 'NMI vs state-space: 0.281', fontsize=10, style='italic')
    ax.text(0.5, 0.1, '(partial preservation of physical structure)', fontsize=9,
            color='gray')

    # tight_layout handled by gridspec hspace/wspace

    # Save to paper/figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / 'fig10_pixel_to_structure.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved hero figure to {fig_path}")

    # Also save via standard mechanism
    from experiments.utils.plotting import save_figure
    save_figure(fig, 'pixel_to_structure_hero', 'pixel_viz')
    plt.close(fig)


def run_us045():
    """Run US-045: pixel-to-structure visualization."""
    print("=" * 60)
    print("US-045: Pixel-to-Structure Visualization")
    print("=" * 60)

    print("\nCollecting sample frames...")
    frames, frame_latents, frame_labels, frame_states = collect_sample_frames()
    print(f"  Captured {len(frames)} frames: {frame_labels}")

    print("\nCreating hero figure...")
    create_hero_figure(frames, frame_latents, frame_labels)

    # Save results
    metrics = {
        'frame_labels': frame_labels,
        'frame_states': frame_states.tolist(),
        'figure_path': str(FIGURES_DIR / 'fig10_pixel_to_structure.png'),
    }
    config = {'checkpoint': str(CHECKPOINT_PATH)}

    save_results('pixel_to_structure_viz', metrics, config,
                 notes='US-045: End-to-end pixel-to-structure visualization. '
                       'Multi-panel hero figure for paper and pitch deck.')

    print("\n" + "=" * 60)
    print("US-045 complete.")


if __name__ == '__main__':
    run_us045()
