"""
US-024/025/026/027: World Model Analysis
=========================================

Load Active Inference checkpoint from lunar-lander, collect trajectory data,
and apply Topological Blankets to the 8D state space.

State variables: x, y, vx, vy, angle, angular_vel, left_leg, right_leg
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Project paths
LUNAR_LANDER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NOUMENAL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(LUNAR_LANDER_DIR, 'sharing'))
sys.path.insert(0, NOUMENAL_DIR)

from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import (
    detect_blankets_otsu, detect_blankets_spectral, detect_blankets_coupling
)
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap
)
from topological_blankets.clustering import cluster_internals
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']


# =========================================================================
# US-024: Load checkpoint and collect trajectories
# =========================================================================

def load_active_inference_agent():
    """Load the trained Active Inference agent."""
    from active_inference import LunarLanderActiveInference, ActiveInferenceConfig

    config = ActiveInferenceConfig(
        n_ensemble=5,
        hidden_dim=256,
        use_learned_reward=True,
        device='cpu',
    )
    agent = LunarLanderActiveInference(config)
    ckpt_path = os.path.join(LUNAR_LANDER_DIR, 'trained_agents', 'lunarlander_actinf_best.tar')
    agent.load(ckpt_path)
    print(f"Loaded Active Inference agent from episode {agent.episode}")
    return agent


def collect_trajectories(agent, n_episodes=50, seed=42, use_policy=False):
    """
    Run episodes and collect trajectory data.

    Uses mixed policy: epsilon=0.8 for state diversity with some structure.
    CEM planning on CPU is slow, so we use high epsilon for speed while
    still getting transitions shaped by the environment dynamics.
    """
    import gymnasium as gym

    env = gym.make('LunarLander-v3')
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    episode_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_len = 0

        while True:
            if use_policy:
                action = agent.select_action(state, epsilon=0.8)
            else:
                action = env.action_space.sample()
            next_state, reward, term, trunc, _ = env.step(action)

            all_states.append(state.copy())
            all_actions.append(action)
            all_next_states.append(next_state.copy())
            all_rewards.append(reward)

            state = next_state
            ep_return += reward
            ep_len += 1

            if term or trunc:
                break

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: return={ep_return:.1f}, len={ep_len}")

    env.close()

    states = np.array(all_states)
    actions = np.array(all_actions)
    next_states = np.array(all_next_states)
    rewards = np.array(all_rewards)

    print(f"\nCollected {len(states)} transitions from {n_episodes} episodes")
    print(f"Mean return: {np.mean(episode_returns):.1f} +/- {np.std(episode_returns):.1f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")

    return {
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'rewards': rewards,
        'episode_returns': np.array(episode_returns),
        'episode_lengths': np.array(episode_lengths),
    }


def compute_dynamics_gradients(agent, trajectory_data):
    """
    Compute gradients of dynamics model prediction error w.r.t. state.

    grad_s ||f(s,a) - s'||^2
    """
    states = trajectory_data['states']
    actions = trajectory_data['actions']
    next_states = trajectory_data['next_states']
    n_samples = len(states)

    ensemble = agent.ensemble
    ensemble.eval()

    gradients = np.zeros_like(states)
    n_actions = 4  # LunarLander discrete actions

    batch_size = 256
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_s = torch.FloatTensor(states[start:end]).requires_grad_(True)
        batch_a = torch.zeros(end - start, n_actions)
        batch_a[range(end - start), actions[start:end]] = 1.0
        batch_ns = torch.FloatTensor(next_states[start:end])

        # Forward through all ensemble members, take mean prediction
        means, _ = ensemble.forward_all(batch_s, batch_a)
        pred_mean = means.mean(dim=0)  # average across ensemble

        # Prediction error
        loss = ((pred_mean - batch_ns) ** 2).sum()
        loss.backward()

        gradients[start:end] = batch_s.grad.detach().numpy()

    print(f"Computed gradients for {n_samples} transitions")
    print(f"Gradient magnitude per dim: {np.mean(np.abs(gradients), axis=0).round(3)}")

    return gradients


def compute_reward_gradients(agent, trajectory_data):
    """
    Compute gradients of reward model w.r.t. state.

    grad_s R(s,a)
    """
    if agent.reward_model is None:
        print("No learned reward model available")
        return None

    states = trajectory_data['states']
    actions = trajectory_data['actions']
    n_samples = len(states)
    n_actions = 4

    agent.reward_model.eval()
    gradients = np.zeros_like(states)

    batch_size = 256
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_s = torch.FloatTensor(states[start:end]).requires_grad_(True)
        batch_a = torch.zeros(end - start, n_actions)
        batch_a[range(end - start), actions[start:end]] = 1.0

        reward_pred = agent.reward_model(batch_s, batch_a).sum()
        reward_pred.backward()

        gradients[start:end] = batch_s.grad.detach().numpy()

    print(f"Computed reward gradients for {n_samples} transitions")
    return gradients


def compute_ensemble_disagreement_gradients(agent, trajectory_data):
    """
    Compute gradients of ensemble disagreement (epistemic uncertainty).

    Disagreement = Var across ensemble predictions.
    grad_s Var_k[f_k(s,a)]
    """
    states = trajectory_data['states']
    actions = trajectory_data['actions']
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

        means, _ = ensemble.forward_all(batch_s, batch_a)
        # Disagreement: variance across ensemble members
        disagreement = means.var(dim=0).sum()
        disagreement.backward()

        gradients[start:end] = batch_s.grad.detach().numpy()

    print(f"Computed disagreement gradients for {n_samples} transitions")
    return gradients


# =========================================================================
# US-025: Apply TB to state space
# =========================================================================

def analyze_state_space(gradients, label="dynamics"):
    """Apply TB to 8D state-space gradients and return analysis results."""
    print(f"\n--- TB Analysis: {label} gradients ---")

    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']
    coupling = features['coupling']

    # Gradient method
    result_grad = tb_pipeline(gradients, n_objects=2, method='gradient')
    result_coupling = tb_pipeline(gradients, n_objects=2, method='coupling')
    result_hybrid = tb_pipeline(gradients, n_objects=2, method='hybrid')

    # Spectral analysis
    from scipy.linalg import eigh
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    eigvals, eigvecs = eigh(L)
    n_clusters, eigengap = compute_eigengap(eigvals[:8])

    # Hierarchical detection
    from topological_blankets.spectral import recursive_spectral_detection
    hierarchy = recursive_spectral_detection(H_est, max_levels=3)

    # Print results
    for method_name, result in [('gradient', result_grad),
                                  ('coupling', result_coupling),
                                  ('hybrid', result_hybrid)]:
        assign = result['assignment']
        blanket = result['is_blanket']
        obj_dims = {i: [STATE_LABELS[j] for j in range(8) if assign[j] == i]
                    for i in set(assign) if i >= 0}
        blanket_dims = [STATE_LABELS[j] for j in range(8) if blanket[j]]
        print(f"  {method_name}:")
        print(f"    Objects: {obj_dims}")
        print(f"    Blanket: {blanket_dims}")

    print(f"  Eigengap: {eigengap:.3f}, spectral clusters: {n_clusters}")
    print(f"  Hierarchy levels: {len(hierarchy)}")

    return {
        'hessian_est': H_est.tolist(),
        'coupling': coupling.tolist(),
        'grad_magnitude': features['grad_magnitude'].tolist(),
        'gradient_method': {
            'assignment': result_grad['assignment'].tolist(),
            'is_blanket': result_grad['is_blanket'].tolist(),
        },
        'coupling_method': {
            'assignment': result_coupling['assignment'].tolist(),
            'is_blanket': result_coupling['is_blanket'].tolist(),
        },
        'hybrid_method': {
            'assignment': result_hybrid['assignment'].tolist(),
            'is_blanket': result_hybrid['is_blanket'].tolist(),
        },
        'eigengap': float(eigengap),
        'n_clusters_spectral': int(n_clusters),
        'eigenvalues': eigvals.tolist(),
        'hierarchy': [{k: v.tolist() if hasattr(v, 'tolist') else v
                       for k, v in level.items()} for level in hierarchy],
    }


# =========================================================================
# Visualization
# =========================================================================

def plot_coupling_matrix(coupling, title, state_labels=None):
    """Plot labeled coupling matrix heatmap."""
    if state_labels is None:
        state_labels = STATE_LABELS

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.abs(coupling), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(state_labels)))
    ax.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(state_labels)))
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    for i in range(len(state_labels)):
        for j in range(len(state_labels)):
            val = np.abs(coupling[i][j]) if isinstance(coupling, list) else np.abs(coupling[i, j])
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if val > np.max(np.abs(coupling)) * 0.6 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_eigenvalue_spectrum(eigvals, title="Eigenvalue Spectrum"):
    """Plot eigenvalue spectrum with eigengap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(eigvals)), eigvals, 'o-', color='#2ecc71', markersize=6)
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_multi_analysis(dynamics_results, reward_results=None, disagreement_results=None):
    """Side-by-side coupling matrices for different energy landscapes."""
    panels = [('Dynamics Energy', dynamics_results)]
    if reward_results:
        panels.append(('Reward Landscape', reward_results))
    if disagreement_results:
        panels.append(('Ensemble Disagreement', disagreement_results))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (title, results) in zip(axes, panels):
        coupling = np.array(results['coupling'])
        im = ax.imshow(np.abs(coupling), cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(8))
        ax.set_xticklabels(STATE_LABELS, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(8))
        ax.set_yticklabels(STATE_LABELS, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig


# =========================================================================
# Main experiment
# =========================================================================

def run_us024():
    """US-024: Load checkpoint and collect trajectory data."""
    print("=" * 70)
    print("US-024: Load Active Inference Checkpoint")
    print("=" * 70)

    agent = load_active_inference_agent()
    trajectory_data = collect_trajectories(agent, n_episodes=50)

    # Compute gradients for three energy landscapes
    print("\n--- Computing dynamics gradients ---")
    dynamics_grads = compute_dynamics_gradients(agent, trajectory_data)

    print("\n--- Computing reward gradients ---")
    reward_grads = compute_reward_gradients(agent, trajectory_data)

    print("\n--- Computing disagreement gradients ---")
    disagreement_grads = compute_ensemble_disagreement_gradients(agent, trajectory_data)

    # Save trajectory data
    data_dir = os.path.join(NOUMENAL_DIR, 'results', 'trajectory_data')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'states.npy'), trajectory_data['states'])
    np.save(os.path.join(data_dir, 'actions.npy'), trajectory_data['actions'])
    np.save(os.path.join(data_dir, 'next_states.npy'), trajectory_data['next_states'])
    np.save(os.path.join(data_dir, 'dynamics_gradients.npy'), dynamics_grads)
    if reward_grads is not None:
        np.save(os.path.join(data_dir, 'reward_gradients.npy'), reward_grads)
    np.save(os.path.join(data_dir, 'disagreement_gradients.npy'), disagreement_grads)

    collection_meta = {
        'n_episodes': 50,
        'n_transitions': len(trajectory_data['states']),
        'mean_return': float(np.mean(trajectory_data['episode_returns'])),
        'std_return': float(np.std(trajectory_data['episode_returns'])),
        'mean_length': float(np.mean(trajectory_data['episode_lengths'])),
        'dynamics_grad_magnitude': np.mean(np.abs(dynamics_grads), axis=0).tolist(),
        'state_labels': STATE_LABELS,
    }

    save_results('actinf_trajectory_collection', collection_meta, {},
                 notes='US-024: Active Inference trajectory collection. 50 episodes, dynamics/reward/disagreement gradients.')

    print("\nUS-024 complete.")
    return agent, trajectory_data, dynamics_grads, reward_grads, disagreement_grads


def run_us025(dynamics_grads, reward_grads=None, disagreement_grads=None):
    """US-025/026/027: Apply TB to Active Inference state space."""
    print("\n" + "=" * 70)
    print("US-025: TB Analysis of Active Inference State Space")
    print("=" * 70)

    # Dynamics energy landscape
    dynamics_results = analyze_state_space(dynamics_grads, label="dynamics")

    fig_coupling = plot_coupling_matrix(
        np.array(dynamics_results['coupling']),
        'Active Inference: Dynamics Coupling Matrix (8D State)')
    save_figure(fig_coupling, 'actinf_dynamics_coupling', 'world_model')

    fig_eigvals = plot_eigenvalue_spectrum(
        dynamics_results['eigenvalues'][:8],
        'Active Inference: Eigenvalue Spectrum')
    save_figure(fig_eigvals, 'actinf_eigenvalue_spectrum', 'world_model')

    # Reward landscape (US-027)
    reward_results = None
    if reward_grads is not None:
        reward_results = analyze_state_space(reward_grads, label="reward")

    # Disagreement landscape (US-026)
    disagreement_results = None
    if disagreement_grads is not None:
        disagreement_results = analyze_state_space(disagreement_grads, label="disagreement")

    # Multi-panel comparison
    fig_multi = plot_multi_analysis(dynamics_results, reward_results, disagreement_results)
    save_figure(fig_multi, 'actinf_multi_landscape', 'world_model')

    # Combine all results
    all_results = {
        'dynamics': dynamics_results,
    }
    if reward_results:
        all_results['reward'] = reward_results
    if disagreement_results:
        all_results['disagreement'] = disagreement_results

    save_results('actinf_tb_analysis', all_results, {'state_labels': STATE_LABELS},
                 notes='US-025/026/027: TB analysis of Active Inference world model. Dynamics, reward, and disagreement landscapes.')

    print("\nUS-025/026/027 complete.")
    return all_results


# =========================================================================
# Entry point
# =========================================================================

if __name__ == '__main__':
    agent, traj_data, dyn_grads, rew_grads, dis_grads = run_us024()
    run_us025(dyn_grads, rew_grads, dis_grads)
