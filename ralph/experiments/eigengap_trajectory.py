"""
US-047: Track topology dynamics during training with eigengap trajectories
==========================================================================

Train a fresh Active Inference agent on LunarLander-v3 for 300 episodes,
saving checkpoints every 25 episodes (12 snapshots). At each checkpoint,
collect 20 episodes of trajectory data with a random policy, run TB on
dynamics gradients, and record: eigengap, number of detected objects,
blanket size, coupling matrix Frobenius norm, and the partition itself.
Plot eigengap trajectory and detect abrupt structural transitions.

Hypothesis: structure should emerge once the dynamics model starts learning
accurate physics (after ~50-100 episodes).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
NOUMENAL_DIR = os.path.dirname(RALPH_DIR)
LUNAR_LANDER_DIR = os.path.dirname(NOUMENAL_DIR)

sys.path.insert(0, os.path.join(LUNAR_LANDER_DIR, 'sharing'))
sys.path.insert(0, NOUMENAL_DIR)
sys.path.insert(0, RALPH_DIR)

from topological_blankets.features import compute_geometric_features
from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

EXPERIMENT_NAME = "eigengap_trajectory"

# Training configuration
N_TRAIN_EPISODES = 300
CHECKPOINT_INTERVAL = 25  # every 25 episodes
N_COLLECT_EPISODES = 20   # random-policy episodes per checkpoint
N_ENSEMBLE = 5
HIDDEN_DIM = 256
SEED = 42

# Derived constants
CHECKPOINT_EPISODES = list(range(CHECKPOINT_INTERVAL, N_TRAIN_EPISODES + 1, CHECKPOINT_INTERVAL))
N_CHECKPOINTS = len(CHECKPOINT_EPISODES)


# =========================================================================
# Training loop with periodic TB analysis
# =========================================================================

def create_fresh_agent():
    """Create a fresh (untrained) Active Inference agent."""
    from active_inference import LunarLanderActiveInference, ActiveInferenceConfig

    config = ActiveInferenceConfig(
        n_ensemble=N_ENSEMBLE,
        hidden_dim=HIDDEN_DIM,
        use_learned_reward=True,
        device='cpu',
        lr_dynamics=1e-3,
        lr_reward=1e-3,
        batch_size=128,
        buffer_capacity=100000,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_episodes=300,
        lambda_epistemic_start=1.0,
        lambda_epistemic_end=0.1,
        lambda_anneal_episodes=300,
    )
    agent = LunarLanderActiveInference(config)
    print(f"Created fresh Active Inference agent: {N_ENSEMBLE} ensemble, {HIDDEN_DIM} hidden")
    return agent


def collect_random_trajectories(n_episodes=20, seed=42):
    """
    Collect trajectory data using a random policy.

    Random policy ensures consistent exploration across checkpoints,
    isolating the effect of the dynamics model quality.
    """
    import gymnasium as gym

    env = gym.make('LunarLander-v3')
    all_states = []
    all_actions = []
    all_next_states = []
    episode_returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0.0

        while True:
            action = env.action_space.sample()
            next_state, reward, term, trunc, _ = env.step(action)

            all_states.append(state.copy())
            all_actions.append(action)
            all_next_states.append(next_state.copy())

            state = next_state
            ep_return += reward

            if term or trunc:
                break

        episode_returns.append(ep_return)

    env.close()

    return {
        'states': np.array(all_states),
        'actions': np.array(all_actions),
        'next_states': np.array(all_next_states),
        'episode_returns': np.array(episode_returns),
    }


def compute_dynamics_gradients(agent, traj_data):
    """
    Compute gradients of dynamics model prediction error w.r.t. state.

    grad_s ||f(s,a) - s'||^2
    """
    states = traj_data['states']
    actions = traj_data['actions']
    next_states = traj_data['next_states']
    n_samples = len(states)
    n_actions = 4

    ensemble = agent.ensemble
    ensemble.eval()

    gradients = np.zeros_like(states)

    batch_size = 256
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_s = torch.FloatTensor(states[start:end]).requires_grad_(True)
        batch_a = torch.zeros(end - start, n_actions)
        batch_a[range(end - start), actions[start:end]] = 1.0
        batch_ns = torch.FloatTensor(next_states[start:end])

        means, _ = ensemble.forward_all(batch_s, batch_a)
        pred_mean = means.mean(dim=0)

        loss = ((pred_mean - batch_ns) ** 2).sum()
        loss.backward()

        gradients[start:end] = batch_s.grad.detach().numpy()

    return gradients


def run_tb_analysis(gradients):
    """
    Run Topological Blankets analysis on dynamics gradients.

    Returns a dictionary of TB metrics for this checkpoint.
    """
    from scipy.linalg import eigh

    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']
    coupling = features['coupling']

    # Spectral analysis
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    eigvals, eigvecs = eigh(L)
    n_clusters, eigengap = compute_eigengap(eigvals[:8])

    # Run TB pipeline with coupling method
    result = tb_pipeline(gradients, n_objects=2, method='coupling')
    assignment = result['assignment']
    is_blanket = result['is_blanket']

    # Compute metrics
    n_objects = len(set(assignment[assignment >= 0]))
    blanket_size = int(np.sum(is_blanket))
    coupling_frobenius = float(np.linalg.norm(coupling, 'fro'))

    # Object-to-variable mapping
    obj_dims = {}
    for obj_id in set(assignment):
        if obj_id >= 0:
            obj_dims[int(obj_id)] = [STATE_LABELS[j] for j in range(8) if assignment[j] == obj_id]
    blanket_dims = [STATE_LABELS[j] for j in range(8) if is_blanket[j]]

    return {
        'eigengap': float(eigengap),
        'n_clusters_spectral': int(n_clusters),
        'n_objects': n_objects,
        'blanket_size': blanket_size,
        'coupling_frobenius': coupling_frobenius,
        'coupling_matrix': coupling.tolist(),
        'assignment': assignment.tolist(),
        'is_blanket': is_blanket.tolist(),
        'eigenvalues': eigvals.tolist(),
        'object_dims': obj_dims,
        'blanket_dims': blanket_dims,
        'hessian_est': H_est.tolist(),
        'grad_magnitude': features['grad_magnitude'].tolist(),
    }


def train_with_checkpoints():
    """
    Train a fresh agent for 300 episodes, running TB analysis every 25 episodes.

    Returns per-checkpoint TB metrics and training curve data.
    """
    import gymnasium as gym

    print("=" * 70)
    print("US-047: Eigengap Trajectory During Training")
    print("=" * 70)

    agent = create_fresh_agent()

    # Collect random trajectories once (same data for all checkpoints)
    print(f"\nCollecting {N_COLLECT_EPISODES} random-policy episodes for TB analysis...")
    traj_data = collect_random_trajectories(n_episodes=N_COLLECT_EPISODES, seed=SEED)
    print(f"  {len(traj_data['states'])} transitions collected")

    # Training state
    env = gym.make('LunarLander-v3')
    episode_returns = []
    episode_losses = []
    checkpoint_metrics = []
    training_start = time.time()

    # Run TB on the untrained model (episode 0)
    print("\n--- Checkpoint at episode 0 (untrained) ---")
    grads_0 = compute_dynamics_gradients(agent, traj_data)
    tb_metrics_0 = run_tb_analysis(grads_0)
    tb_metrics_0['episode'] = 0
    tb_metrics_0['mean_return'] = 0.0
    tb_metrics_0['mean_loss'] = 0.0
    checkpoint_metrics.append(tb_metrics_0)
    print(f"  eigengap={tb_metrics_0['eigengap']:.4f}, "
          f"n_objects={tb_metrics_0['n_objects']}, "
          f"blanket_size={tb_metrics_0['blanket_size']}, "
          f"coupling_frob={tb_metrics_0['coupling_frobenius']:.4f}")

    # Training loop
    # Use high epsilon (random policy) throughout to avoid expensive CEM
    # planning on CPU. The dynamics model learns from replay buffer updates;
    # random data is sufficient (and desirable for exploration diversity).
    for ep in range(1, N_TRAIN_EPISODES + 1):
        state, _ = env.reset(seed=SEED + ep + 1000)
        ep_return = 0.0
        done = False

        agent.update_lambda(ep)

        while not done:
            # Random policy: fastest training; dynamics model still learns
            action = env.action_space.sample()

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.add_experience(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward

        episode_returns.append(ep_return)

        # Train dynamics and reward models every episode
        ep_loss = 0.0
        if len(agent.memory) >= agent.config.batch_size:
            n_updates = 5  # more updates per episode for faster learning
            for _ in range(n_updates):
                loss = agent.train_dynamics(agent.config.batch_size)
                ep_loss += loss
            if agent.reward_model is not None:
                agent.train_reward(agent.config.batch_size)
            ep_loss /= n_updates

        episode_losses.append(ep_loss)

        # Log progress periodically
        if ep % 25 == 0 or ep % 50 == 0:
            recent_returns = episode_returns[-min(25, len(episode_returns)):]
            recent_losses = [l for l in episode_losses[-min(25, len(episode_losses)):] if l != 0]
            elapsed = time.time() - training_start
            if recent_losses:
                loss_str = f"mean_loss={np.mean(recent_losses):.4f}"
            else:
                loss_str = "no training yet"
            print(f"\n  Episode {ep}/{N_TRAIN_EPISODES}: "
                  f"mean_return={np.mean(recent_returns):.1f}, "
                  f"{loss_str} [{elapsed:.0f}s elapsed]")

        # Run TB analysis at checkpoint episodes
        if ep in CHECKPOINT_EPISODES:
            print(f"\n--- Checkpoint at episode {ep} ---")
            grads = compute_dynamics_gradients(agent, traj_data)
            tb_metrics = run_tb_analysis(grads)
            tb_metrics['episode'] = ep

            # Compute recent training stats
            window = min(CHECKPOINT_INTERVAL, len(episode_returns))
            recent_returns = episode_returns[-window:]
            recent_losses = [l for l in episode_losses[-window:] if l != 0]
            tb_metrics['mean_return'] = float(np.mean(recent_returns))
            tb_metrics['mean_loss'] = float(np.mean(recent_losses)) if recent_losses else 0.0

            checkpoint_metrics.append(tb_metrics)
            print(f"  eigengap={tb_metrics['eigengap']:.4f}, "
                  f"n_objects={tb_metrics['n_objects']}, "
                  f"blanket_size={tb_metrics['blanket_size']}, "
                  f"coupling_frob={tb_metrics['coupling_frobenius']:.4f}")
            print(f"  Objects: {tb_metrics['object_dims']}")
            print(f"  Blanket: {tb_metrics['blanket_dims']}")

    env.close()
    total_time = time.time() - training_start
    print(f"\nTraining complete: {total_time:.0f}s total")

    return {
        'checkpoint_metrics': checkpoint_metrics,
        'episode_returns': [float(r) for r in episode_returns],
        'episode_losses': [float(l) for l in episode_losses],
        'total_training_time_s': total_time,
    }


# =========================================================================
# Phase transition detection
# =========================================================================

def detect_phase_transitions(checkpoint_metrics):
    """
    Detect abrupt structural transitions where eigengap changes by >2x
    between adjacent checkpoints.
    """
    transitions = []
    for i in range(1, len(checkpoint_metrics)):
        prev = checkpoint_metrics[i - 1]
        curr = checkpoint_metrics[i]

        prev_eg = prev['eigengap']
        curr_eg = curr['eigengap']

        # Avoid division by zero
        if prev_eg > 1e-8:
            ratio = curr_eg / prev_eg
        else:
            ratio = float('inf') if curr_eg > 1e-8 else 1.0

        is_transition = ratio > 2.0 or ratio < 0.5

        transitions.append({
            'from_episode': prev['episode'],
            'to_episode': curr['episode'],
            'eigengap_from': prev_eg,
            'eigengap_to': curr_eg,
            'ratio': float(ratio) if ratio != float('inf') else 999.0,
            'is_transition': is_transition,
            'n_objects_from': prev['n_objects'],
            'n_objects_to': curr['n_objects'],
        })

    phase_transitions = [t for t in transitions if t['is_transition']]
    return transitions, phase_transitions


def compute_loss_structure_correlation(checkpoint_metrics, episode_losses):
    """
    Compute correlation between training loss convergence and structural emergence.
    """
    from scipy.stats import pearsonr, spearmanr

    # Compute smoothed loss at each checkpoint
    episodes = [m['episode'] for m in checkpoint_metrics]
    eigengaps = [m['eigengap'] for m in checkpoint_metrics]
    coupling_frobs = [m['coupling_frobenius'] for m in checkpoint_metrics]

    # For episode 0, use mean loss of first few episodes (or 0)
    checkpoint_losses = []
    for ep in episodes:
        if ep == 0:
            checkpoint_losses.append(0.0)
        else:
            # Average loss in the 25-episode window leading to this checkpoint
            start_idx = max(0, ep - CHECKPOINT_INTERVAL)
            end_idx = ep
            window_losses = [l for l in episode_losses[start_idx:end_idx] if l != 0]
            checkpoint_losses.append(float(np.mean(window_losses)) if window_losses else 0.0)

    # Only compute correlations where we have nonzero losses
    valid_mask = [l != 0 for l in checkpoint_losses]
    valid_losses = [l for l, v in zip(checkpoint_losses, valid_mask) if v]
    valid_eigengaps = [e for e, v in zip(eigengaps, valid_mask) if v]
    valid_frobs = [f for f, v in zip(coupling_frobs, valid_mask) if v]

    result = {
        'checkpoint_episodes': episodes,
        'checkpoint_losses': checkpoint_losses,
        'eigengaps': eigengaps,
        'coupling_frobenius': coupling_frobs,
    }

    if len(valid_losses) >= 3:
        # Pearson correlation: loss vs eigengap
        r_loss_eg, p_loss_eg = pearsonr(valid_losses, valid_eigengaps)
        result['pearson_loss_eigengap'] = {'r': float(r_loss_eg), 'p': float(p_loss_eg)}

        # Spearman correlation: loss vs eigengap (monotonic relationship)
        rho_loss_eg, p_rho_eg = spearmanr(valid_losses, valid_eigengaps)
        result['spearman_loss_eigengap'] = {'rho': float(rho_loss_eg), 'p': float(p_rho_eg)}

        # Pearson correlation: loss vs coupling Frobenius norm
        r_loss_frob, p_loss_frob = pearsonr(valid_losses, valid_frobs)
        result['pearson_loss_coupling_frob'] = {'r': float(r_loss_frob), 'p': float(p_loss_frob)}

        # Spearman correlation: loss vs coupling Frobenius norm
        rho_loss_frob, p_rho_frob = spearmanr(valid_losses, valid_frobs)
        result['spearman_loss_coupling_frob'] = {'rho': float(rho_loss_frob), 'p': float(p_rho_frob)}
    else:
        result['pearson_loss_eigengap'] = {'r': 0.0, 'p': 1.0}
        result['spearman_loss_eigengap'] = {'rho': 0.0, 'p': 1.0}
        result['pearson_loss_coupling_frob'] = {'r': 0.0, 'p': 1.0}
        result['spearman_loss_coupling_frob'] = {'rho': 0.0, 'p': 1.0}

    return result


# =========================================================================
# Visualization
# =========================================================================

def plot_eigengap_trajectory(checkpoint_metrics, phase_transitions):
    """Plot eigengap vs training episode with phase transitions marked."""
    episodes = [m['episode'] for m in checkpoint_metrics]
    eigengaps = [m['eigengap'] for m in checkpoint_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, eigengaps, 'o-', color='#2ecc71', markersize=7, linewidth=2,
            label='Eigengap')

    # Mark phase transitions
    for pt in phase_transitions:
        mid_ep = (pt['from_episode'] + pt['to_episode']) / 2.0
        ax.axvline(x=mid_ep, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.annotate(f"ratio={pt['ratio']:.1f}x",
                    xy=(mid_ep, max(eigengaps) * 0.9),
                    fontsize=8, color='red', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Eigengap', fontsize=11)
    ax.set_title('Eigengap Trajectory During Training', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def plot_n_objects_trajectory(checkpoint_metrics):
    """Plot number of detected objects vs training episode."""
    episodes = [m['episode'] for m in checkpoint_metrics]
    n_objects = [m['n_objects'] for m in checkpoint_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, n_objects, 's-', color='#3498db', markersize=7, linewidth=2,
            label='Number of objects')

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Detected Objects', fontsize=11)
    ax.set_title('Number of Detected Objects During Training', fontsize=13)
    ax.set_yticks(range(0, max(n_objects) + 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def plot_coupling_heatmaps(checkpoint_metrics, target_episodes=(25, 100, 200, 300)):
    """
    Side-by-side coupling matrix heatmaps at selected episodes.
    """
    # Find the closest checkpoint to each target episode
    selected = []
    for target in target_episodes:
        closest = min(checkpoint_metrics, key=lambda m: abs(m['episode'] - target))
        selected.append(closest)

    n_panels = len(selected)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, metrics in zip(axes, selected):
        coupling = np.array(metrics['coupling_matrix'])
        vmax = np.max(np.abs(coupling))
        if vmax == 0:
            vmax = 1.0

        im = ax.imshow(coupling, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(8))
        ax.set_xticklabels(STATE_LABELS, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(8))
        ax.set_yticklabels(STATE_LABELS, fontsize=7)
        ax.set_title(f'Episode {metrics["episode"]}\n'
                     f'eigengap={metrics["eigengap"]:.3f}, '
                     f'frob={metrics["coupling_frobenius"]:.2f}',
                     fontsize=9)

        # Annotate values
        for i in range(8):
            for j in range(8):
                val = coupling[i, j]
                color = 'white' if val > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=5, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Coupling Matrix Evolution During Training', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_training_and_structure(checkpoint_metrics, episode_returns, episode_losses):
    """
    Multi-panel figure: training curve alongside structural curves.
    """
    episodes_ckpt = [m['episode'] for m in checkpoint_metrics]
    eigengaps = [m['eigengap'] for m in checkpoint_metrics]
    coupling_frobs = [m['coupling_frobenius'] for m in checkpoint_metrics]
    blanket_sizes = [m['blanket_size'] for m in checkpoint_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Episode returns (smoothed)
    ax = axes[0, 0]
    ep_nums = np.arange(1, len(episode_returns) + 1)
    ax.plot(ep_nums, episode_returns, alpha=0.3, color='gray', linewidth=0.5)
    # Smoothed with running mean
    window = 25
    if len(episode_returns) >= window:
        smoothed = np.convolve(episode_returns, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window, len(episode_returns) + 1), smoothed,
                color='#e74c3c', linewidth=2, label=f'{window}-ep moving avg')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Episode Return')
    ax.set_title('Training Curve (Episode Returns)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Eigengap trajectory
    ax = axes[0, 1]
    ax.plot(episodes_ckpt, eigengaps, 'o-', color='#2ecc71', markersize=6, linewidth=2)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Eigengap')
    ax.set_title('Eigengap Trajectory')
    ax.grid(True, alpha=0.3)

    # Panel 3: Coupling Frobenius norm
    ax = axes[1, 0]
    ax.plot(episodes_ckpt, coupling_frobs, 'D-', color='#9b59b6', markersize=6, linewidth=2)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Coupling Matrix Frobenius Norm')
    ax.grid(True, alpha=0.3)

    # Panel 4: Training loss (smoothed) with eigengap overlay
    ax = axes[1, 1]
    nonzero_losses = [(i + 1, l) for i, l in enumerate(episode_losses) if l > 0]
    if nonzero_losses:
        loss_eps, loss_vals = zip(*nonzero_losses)
        ax.plot(loss_eps, loss_vals, alpha=0.3, color='gray', linewidth=0.5)
        if len(loss_vals) >= window:
            loss_smoothed = np.convolve(loss_vals, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(loss_eps[0] + window - 1, loss_eps[0] + window - 1 + len(loss_smoothed)),
                    loss_smoothed, color='#e67e22', linewidth=2, label=f'{window}-ep loss avg')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Dynamics Loss', color='#e67e22')
    ax.set_title('Training Loss with Eigengap Overlay')
    ax.grid(True, alpha=0.3)

    # Overlay eigengap on secondary axis
    ax2 = ax.twinx()
    ax2.plot(episodes_ckpt, eigengaps, 'o--', color='#2ecc71', markersize=5,
             linewidth=1.5, alpha=0.8, label='Eigengap')
    ax2.set_ylabel('Eigengap', color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    fig.suptitle('Training Dynamics and Structural Emergence', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_blanket_size_trajectory(checkpoint_metrics):
    """Plot blanket size vs training episode."""
    episodes = [m['episode'] for m in checkpoint_metrics]
    blanket_sizes = [m['blanket_size'] for m in checkpoint_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, blanket_sizes, '^-', color='#e67e22', markersize=7, linewidth=2,
            label='Blanket size (n variables)')

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Blanket Size', fontsize=11)
    ax.set_title('Blanket Size During Training', fontsize=13)
    ax.set_yticks(range(0, 9))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


# =========================================================================
# Main experiment
# =========================================================================

def run_experiment():
    """Run the full US-047 eigengap trajectory experiment."""

    # Phase 1: Train with checkpoints
    training_data = train_with_checkpoints()
    checkpoint_metrics = training_data['checkpoint_metrics']
    episode_returns = training_data['episode_returns']
    episode_losses = training_data['episode_losses']

    # Phase 2: Detect phase transitions
    print("\n" + "=" * 70)
    print("Phase Transition Detection")
    print("=" * 70)
    all_transitions, phase_transitions = detect_phase_transitions(checkpoint_metrics)
    print(f"  Found {len(phase_transitions)} phase transitions (eigengap >2x change)")
    for pt in phase_transitions:
        print(f"    Episode {pt['from_episode']} -> {pt['to_episode']}: "
              f"eigengap {pt['eigengap_from']:.4f} -> {pt['eigengap_to']:.4f} "
              f"(ratio={pt['ratio']:.1f}x)")

    # Phase 3: Loss-structure correlation
    print("\n" + "=" * 70)
    print("Loss-Structure Correlation")
    print("=" * 70)
    correlation = compute_loss_structure_correlation(checkpoint_metrics, episode_losses)
    print(f"  Pearson (loss vs eigengap): r={correlation['pearson_loss_eigengap']['r']:.3f}, "
          f"p={correlation['pearson_loss_eigengap']['p']:.4f}")
    print(f"  Spearman (loss vs eigengap): rho={correlation['spearman_loss_eigengap']['rho']:.3f}, "
          f"p={correlation['spearman_loss_eigengap']['p']:.4f}")
    print(f"  Pearson (loss vs coupling frob): r={correlation['pearson_loss_coupling_frob']['r']:.3f}, "
          f"p={correlation['pearson_loss_coupling_frob']['p']:.4f}")

    # Phase 4: Generate visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    fig1 = plot_eigengap_trajectory(checkpoint_metrics, phase_transitions)
    save_figure(fig1, 'eigengap_trajectory', EXPERIMENT_NAME)

    fig2 = plot_n_objects_trajectory(checkpoint_metrics)
    save_figure(fig2, 'n_objects_trajectory', EXPERIMENT_NAME)

    fig3 = plot_coupling_heatmaps(checkpoint_metrics)
    save_figure(fig3, 'coupling_evolution', EXPERIMENT_NAME)

    fig4 = plot_training_and_structure(checkpoint_metrics, episode_returns, episode_losses)
    save_figure(fig4, 'training_and_structure', EXPERIMENT_NAME)

    fig5 = plot_blanket_size_trajectory(checkpoint_metrics)
    save_figure(fig5, 'blanket_size_trajectory', EXPERIMENT_NAME)

    # Phase 5: Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    config = {
        'n_train_episodes': N_TRAIN_EPISODES,
        'checkpoint_interval': CHECKPOINT_INTERVAL,
        'n_collect_episodes': N_COLLECT_EPISODES,
        'n_ensemble': N_ENSEMBLE,
        'hidden_dim': HIDDEN_DIM,
        'seed': SEED,
        'checkpoint_episodes': CHECKPOINT_EPISODES,
    }

    # Build summary trajectory for easy parsing
    trajectory_summary = {
        'episodes': [m['episode'] for m in checkpoint_metrics],
        'eigengaps': [m['eigengap'] for m in checkpoint_metrics],
        'n_objects': [m['n_objects'] for m in checkpoint_metrics],
        'blanket_sizes': [m['blanket_size'] for m in checkpoint_metrics],
        'coupling_frobenius': [m['coupling_frobenius'] for m in checkpoint_metrics],
        'mean_returns': [m['mean_return'] for m in checkpoint_metrics],
        'mean_losses': [m['mean_loss'] for m in checkpoint_metrics],
    }

    metrics = {
        'trajectory_summary': trajectory_summary,
        'checkpoint_metrics': checkpoint_metrics,
        'phase_transitions': phase_transitions,
        'all_transitions': all_transitions,
        'correlation': correlation,
        'n_phase_transitions': len(phase_transitions),
        'total_training_time_s': training_data['total_training_time_s'],
    }

    notes = (
        'US-047: Eigengap trajectory during training. '
        f'Trained fresh Active Inference agent for {N_TRAIN_EPISODES} episodes, '
        f'TB analysis at {N_CHECKPOINTS} checkpoints (every {CHECKPOINT_INTERVAL} episodes). '
        f'{N_COLLECT_EPISODES} random-policy episodes per checkpoint. '
        f'Found {len(phase_transitions)} phase transitions (eigengap >2x change). '
        f'Pearson(loss,eigengap)={correlation["pearson_loss_eigengap"]["r"]:.3f}.'
    )

    save_results(EXPERIMENT_NAME, metrics, config, notes=notes)

    # Summary
    print("\n" + "=" * 70)
    print("US-047 SUMMARY")
    print("=" * 70)
    print(f"  Checkpoints analyzed: {len(checkpoint_metrics)} "
          f"(episodes {checkpoint_metrics[0]['episode']} to {checkpoint_metrics[-1]['episode']})")
    print(f"  Phase transitions detected: {len(phase_transitions)}")

    eigengaps = trajectory_summary['eigengaps']
    print(f"  Eigengap range: {min(eigengaps):.4f} to {max(eigengaps):.4f}")
    print(f"  Pearson(loss, eigengap): r={correlation['pearson_loss_eigengap']['r']:.3f}")
    print(f"  Spearman(loss, eigengap): rho={correlation['spearman_loss_eigengap']['rho']:.3f}")

    # Show the trajectory
    print("\n  Episode | Eigengap | Objects | Blanket | Coupling Frob | Return")
    print("  " + "-" * 75)
    for m in checkpoint_metrics:
        print(f"    {m['episode']:5d} | {m['eigengap']:8.4f} | {m['n_objects']:7d} | "
              f"{m['blanket_size']:7d} | {m['coupling_frobenius']:13.4f} | "
              f"{m['mean_return']:7.1f}")

    return metrics


if __name__ == '__main__':
    results = run_experiment()
