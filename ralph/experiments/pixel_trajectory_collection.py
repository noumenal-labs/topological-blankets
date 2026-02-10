"""
US-042: Collect Pixel Trajectory Data with Latent Encodings
=============================================================

Run 50 episodes with the V1 pixel agent, recording latent vectors,
true physical states, actions, and dynamics prediction error gradients
in latent space for TB analysis.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LUNAR_LANDER_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_PATH = LUNAR_LANDER_ROOT / "trained_agents" / "pixel_lunarlander_best.tar"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectory_data"

# Add lunar-lander paths for checkpoint unpickling and imports
sys.path.insert(0, str(LUNAR_LANDER_ROOT))
sys.path.insert(0, str(LUNAR_LANDER_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.results import save_results


def load_pixel_agent():
    """Load the V1 pixel agent from checkpoint."""
    from active_inference.pixel_agent import PixelActiveInferenceAgent

    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    config = ckpt['config']
    pixel_config = ckpt['pixel_config']

    agent = PixelActiveInferenceAgent(
        n_actions=4,
        config=config,
        pixel_config=pixel_config,
    )
    agent.load(str(CHECKPOINT_PATH))
    agent.encoder.eval()
    agent.ensemble.eval()

    print(f"Loaded pixel agent V1 (episode {ckpt['episode']})")
    print(f"  Encoder: CNNEncoder -> {pixel_config.latent_dim}D latent")
    print(f"  Ensemble: {config.n_ensemble} dynamics models")

    return agent


def compute_dynamics_gradients(agent, latent_t, action_t, latent_next_t):
    """
    Compute gradient of dynamics prediction error in latent space.

    grad_z ||f(z,a) - z'||^2

    where f is the ensemble mean dynamics prediction.

    Args:
        agent: PixelActiveInferenceAgent with loaded weights
        latent_t: Tensor (latent_dim,) - current latent
        action_t: int - action taken
        latent_next_t: Tensor (latent_dim,) - next latent (target)

    Returns:
        gradient: numpy array (latent_dim,)
    """
    z = latent_t.clone().detach().requires_grad_(True)
    z_batch = z.unsqueeze(0)  # (1, latent_dim)

    action_onehot = F.one_hot(
        torch.tensor([action_t], device=z.device, dtype=torch.long),
        agent.n_actions
    ).float()

    # Ensemble mean prediction
    means, _ = agent.ensemble.forward_all(z_batch, action_onehot)
    z_pred = means.mean(dim=0)  # (1, latent_dim)

    z_target = latent_next_t.detach().unsqueeze(0)
    loss = F.mse_loss(z_pred, z_target)
    loss.backward()

    grad = z.grad.detach().cpu().numpy()
    return grad


def collect_episodes(agent, n_episodes=50, max_steps=1000):
    """
    Collect trajectory data from n_episodes with the pixel agent.

    For each transition, record:
    - latent vector z (64D) from encoder
    - true physical state s (8D) from environment
    - action a
    - dynamics prediction error gradient in latent space
    """
    import gymnasium as gym

    all_latents = []
    all_states = []
    all_actions = []
    all_gradients = []
    all_rewards = []
    episode_lengths = []
    episode_returns = []

    device = agent.device

    for ep in range(n_episodes):
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        obs, info = env.reset(seed=42 + ep)
        agent.reset_frame_stack()

        frame = env.render()
        stacked = agent.frame_stack.get_observation(frame)

        ep_reward = 0.0
        ep_latents = []
        ep_states = []
        ep_actions = []
        ep_grads = []

        for step in range(max_steps):
            true_state = np.array(obs)

            # Encode current frame stack
            with torch.no_grad():
                frames_t = torch.tensor(
                    stacked[np.newaxis], dtype=torch.float32, device=device
                )
                z_t = agent.encoder(frames_t).squeeze(0)

            # Select action
            action = agent.select_action(stacked, epsilon=0.05)

            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Get next frame and encode
            frame_next = env.render()
            stacked_next = agent.frame_stack.get_observation(frame_next)

            with torch.no_grad():
                frames_next_t = torch.tensor(
                    stacked_next[np.newaxis], dtype=torch.float32, device=device
                )
                z_next_t = agent.encoder(frames_next_t).squeeze(0)

            # Compute dynamics gradient
            grad = compute_dynamics_gradients(agent, z_t, action, z_next_t)

            ep_latents.append(z_t.detach().cpu().numpy())
            ep_states.append(true_state)
            ep_actions.append(action)
            ep_grads.append(grad)

            stacked = stacked_next
            obs = obs_next
            done = terminated or truncated
            if done:
                break

        env.close()

        all_latents.extend(ep_latents)
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_gradients.extend(ep_grads)
        episode_lengths.append(step + 1)
        episode_returns.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes}: {step+1} steps, "
                  f"reward={ep_reward:.1f}, total transitions={len(all_latents)}")

    latents = np.array(all_latents)
    states = np.array(all_states)
    actions = np.array(all_actions)
    gradients = np.array(all_gradients)

    return {
        'latents': latents,
        'states': states,
        'actions': actions,
        'gradients': gradients,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
    }


def run_us042():
    """Run US-042: pixel trajectory collection."""
    print("=" * 60)
    print("US-042: Collect Pixel Trajectory Data")
    print("=" * 60)

    agent = load_pixel_agent()

    print(f"\nCollecting 50 episodes...")
    data = collect_episodes(agent, n_episodes=50)

    # Save trajectory data
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAJECTORY_DIR / 'pixel_latents_50ep.npy', data['latents'])
    np.save(TRAJECTORY_DIR / 'pixel_states_50ep.npy', data['states'])
    np.save(TRAJECTORY_DIR / 'pixel_actions_50ep.npy', data['actions'])
    np.save(TRAJECTORY_DIR / 'pixel_dynamics_gradients_50ep.npy', data['gradients'])

    print(f"\nData saved to {TRAJECTORY_DIR}")
    print(f"  Latents: {data['latents'].shape}")
    print(f"  States: {data['states'].shape}")
    print(f"  Gradients: {data['gradients'].shape}")

    # Summary statistics
    latents = data['latents']
    gradients = data['gradients']
    states = data['states']

    per_dim_latent_std = latents.std(axis=0)
    per_dim_grad_mag = np.mean(np.abs(gradients), axis=0)

    state_labels = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    # Latent-physical correlations
    corr_matrix = np.zeros((64, 8))
    for i in range(64):
        for j in range(8):
            corr_matrix[i, j] = np.abs(np.corrcoef(latents[:, i], states[:, j])[0, 1])

    print(f"\nSummary Statistics:")
    print(f"  Total transitions: {latents.shape[0]}")
    print(f"  Episode returns: {np.mean(data['episode_returns']):.1f} +/- {np.std(data['episode_returns']):.1f}")
    print(f"  Latent per-dim std: [{per_dim_latent_std.min():.4f}, {per_dim_latent_std.max():.4f}]")
    print(f"  Gradient magnitude: [{per_dim_grad_mag.min():.6f}, {per_dim_grad_mag.max():.6f}]")
    print(f"\n  Top latent-physical correlations:")
    for j, label in enumerate(state_labels):
        top_latent = np.argmax(corr_matrix[:, j])
        top_corr = corr_matrix[top_latent, j]
        print(f"    {label}: latent dim {top_latent} (r={top_corr:.3f})")

    # Save results
    metrics = {
        'total_transitions': int(latents.shape[0]),
        'n_episodes': 50,
        'mean_episode_return': float(np.mean(data['episode_returns'])),
        'std_episode_return': float(np.std(data['episode_returns'])),
        'mean_episode_length': float(np.mean(data['episode_lengths'])),
        'latent_std_range': [float(per_dim_latent_std.min()), float(per_dim_latent_std.max())],
        'gradient_magnitude_range': [float(per_dim_grad_mag.min()), float(per_dim_grad_mag.max())],
        'latent_physical_top_correlations': {
            label: {
                'best_latent_dim': int(np.argmax(corr_matrix[:, j])),
                'correlation': float(corr_matrix[np.argmax(corr_matrix[:, j]), j]),
            }
            for j, label in enumerate(state_labels)
        },
        'episode_returns': [float(r) for r in data['episode_returns']],
        'episode_lengths': [int(l) for l in data['episode_lengths']],
    }

    config = {
        'checkpoint': str(CHECKPOINT_PATH),
        'n_episodes': 50,
        'max_steps': 1000,
        'epsilon': 0.05,
        'state_labels': state_labels,
    }

    save_results('pixel_trajectory_collection', metrics, config,
                 notes='US-042: 50 episodes of pixel agent trajectory data with '
                       'latent encodings and dynamics prediction error gradients.')

    return data


if __name__ == '__main__':
    data = run_us042()
