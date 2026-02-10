"""
US-085: Ghost Trajectories for FetchPush Ensemble Predictions
=============================================================

Generates "ghost" predicted trajectories from each ensemble member,
showing prediction divergence as a visual measure of uncertainty.

For each of the 5 ensemble members, the CEM plan is rolled out in that
member's predicted dynamics for H=30 steps. Each member may predict a
different future state, so the ghost trajectories diverge where the
ensemble is uncertain and converge where it is confident.

This script:
  1. Loads the trained ensemble from pandas/data/push_demo/ (or creates
     a fresh random model if no checkpoint exists).
  2. Collects random trajectories from FetchPush-v4.
  3. For selected initial states, runs CEM planning to get an action plan.
  4. Rolls out that action plan independently through each ensemble member.
  5. Extracts per-member gripper xyz positions as "ghost" trajectories.
  6. Plots ghost trajectories in 3D state space (gripper xyz) using matplotlib.
  7. Computes ghost spread: mean pairwise distance between ensemble
     trajectories at each timestep.
  8. Generates comparison plots for confident vs uncertain episodes.
  9. Saves results JSON and PNGs to results/.

FetchPush-v4 observation space (25D):
  [0:3]   grip_pos (gripper xyz)
  [3:6]   object_pos (object xyz)
  [6:9]   object_rel_pos (object - gripper xyz)
  [9:11]  gripper_state (finger widths)
  [11:14] object_rot (euler angles)
  [14:17] object_velp (object linear velocity)
  [17:20] object_velr (object angular velocity)
  [20:22] grip_velp (gripper velocity xy)
  [22:25] (varies: additional gripper state / zeros)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sys
import os
import json
import warnings
import imageio
from itertools import combinations

warnings.filterwarnings('ignore')

# -- Path setup ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)                      # ralph/
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)                   # topological_blankets/

PANDAS_DIR = 'C:/Users/citiz/Documents/noumenal-labs/pandas'

# Add TB package parent and ralph root to sys.path
sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)
sys.path.insert(0, PANDAS_DIR)

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

import jax
import jax.numpy as jnp
import equinox as eqx
from panda.model import EnsembleModel, make_model, ModelConfig, DynamicsMember
from panda.planner import (
    cem_plan, CEMConfig, PlanningObjective, _predict_members
)
from panda.utils import Normalizer

# -- Constants -----------------------------------------------------------------
RUN_DIR = os.path.join(PANDAS_DIR, 'data', 'push_demo')
META_PATH = os.path.join(RUN_DIR, 'model.eqx.json')
MODEL_PATH = os.path.join(RUN_DIR, 'model.eqx')

EXPERIMENT_NAME = "pandas_ghost_trajectories"
HORIZON = 30
N_EPISODES_COLLECT = 20
MAX_STEPS_COLLECT = 50
N_GHOST_EPISODES = 10    # episodes to evaluate for ghost analysis
SEED = 42

# Gripper position indices in the 25D observation
GRIP_X, GRIP_Y, GRIP_Z = 0, 1, 2
OBJ_X, OBJ_Y, OBJ_Z = 3, 4, 5

# Member colors (one per ensemble member)
MEMBER_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4']
MEMBER_LABELS = ['Member 0', 'Member 1', 'Member 2', 'Member 3', 'Member 4']


# =============================================================================
# Model loading (mirrors US-076 pattern)
# =============================================================================

def load_trained_model():
    """Load trained ensemble from pandas/data/push_demo/."""
    with open(META_PATH, 'r') as f:
        meta = json.load(f)

    obs_dim = meta['obs_dim']
    action_dim = meta['action_dim']
    ag_dim = meta['achieved_goal_dim']

    normalizer = Normalizer.from_stats(
        jnp.zeros(obs_dim), jnp.ones(obs_dim),
        jnp.zeros(ag_dim), jnp.ones(ag_dim),
        jnp.zeros(action_dim), jnp.ones(action_dim),
        jnp.zeros(obs_dim), jnp.ones(obs_dim),
        jnp.zeros(ag_dim), jnp.ones(ag_dim),
    )

    cfg = ModelConfig(
        ensemble_size=meta['ensemble_size'],
        hidden_size=meta['hidden_size'],
        depth=meta['depth'],
    )
    key = jax.random.PRNGKey(0)
    model = make_model(obs_dim, action_dim, ag_dim, cfg, normalizer, key)
    model = eqx.tree_deserialise_leaves(MODEL_PATH, model)

    print(f"Loaded trained ensemble: {len(model.members)} members, "
          f"obs_dim={obs_dim}, action_dim={action_dim}, ag_dim={ag_dim}")
    return model, meta


def create_random_model():
    """Create a fresh random ensemble (fallback if no trained model)."""
    obs_dim, action_dim, ag_dim = 25, 4, 3
    normalizer = Normalizer.from_stats(
        jnp.zeros(obs_dim), jnp.ones(obs_dim),
        jnp.zeros(ag_dim), jnp.ones(ag_dim),
        jnp.zeros(action_dim), jnp.ones(action_dim),
        jnp.zeros(obs_dim), jnp.ones(obs_dim),
        jnp.zeros(ag_dim), jnp.ones(ag_dim),
    )
    cfg = ModelConfig(ensemble_size=5, hidden_size=256, depth=2)
    key = jax.random.PRNGKey(42)
    model = make_model(obs_dim, action_dim, ag_dim, cfg, normalizer, key)

    meta = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'achieved_goal_dim': ag_dim,
        'ensemble_size': 5,
        'hidden_size': 256,
        'depth': 2,
        'env_id': 'FetchPush-v4',
    }
    print("Created random (untrained) ensemble for synthetic analysis")
    return model, meta


def load_or_create_model():
    """Load trained model if available; otherwise create random."""
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            return load_trained_model(), True
        except Exception as e:
            print(f"Failed to load trained model: {e}")
    return create_random_model(), False


# =============================================================================
# Data collection
# =============================================================================

def collect_fetchpush_data(n_episodes=N_EPISODES_COLLECT,
                           max_steps=MAX_STEPS_COLLECT, seed=SEED):
    """Collect random trajectories from FetchPush-v4."""
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
    import gymnasium as gym

    env = gym.make('FetchPush-v4', max_episode_steps=max_steps, reward_type='dense')

    episodes = []
    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        episode_data = {
            'obs': [], 'achieved_goal': [], 'desired_goal': [], 'actions': []
        }
        dg = obs_dict['desired_goal'].copy()
        episode_data['desired_goal_fixed'] = dg

        for step in range(max_steps):
            obs = obs_dict['observation']
            ag = obs_dict['achieved_goal']

            action = env.action_space.sample()

            episode_data['obs'].append(obs.copy())
            episode_data['achieved_goal'].append(ag.copy())
            episode_data['desired_goal'].append(dg.copy())
            episode_data['actions'].append(action.copy())

            obs_dict, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        for key in ['obs', 'achieved_goal', 'desired_goal', 'actions']:
            episode_data[key] = np.array(episode_data[key], dtype=np.float32)

        episodes.append(episode_data)

    env.close()
    total_transitions = sum(len(ep['obs']) for ep in episodes)
    print(f"Collected {total_transitions} transitions from {n_episodes} episodes")
    return episodes


# =============================================================================
# Ghost trajectory rollout
# =============================================================================

def rollout_single_member(member, normalizer, obs0, ag0, actions):
    """
    Roll out a single ensemble member's dynamics for a given action sequence.

    Args:
        member: DynamicsMember instance
        normalizer: Normalizer from the EnsembleModel
        obs0: initial observation, shape (obs_dim,)
        ag0: initial achieved goal, shape (ag_dim,)
        actions: action sequence, shape (H, action_dim)

    Returns:
        obs_trajectory: shape (H+1, obs_dim) including obs0
        ag_trajectory: shape (H+1, ag_dim) including ag0
    """
    obs_traj = [np.array(obs0)]
    ag_traj = [np.array(ag0)]

    obs_curr = jnp.array(obs0)
    ag_curr = jnp.array(ag0)

    for t in range(actions.shape[0]):
        action_t = jnp.array(actions[t])
        # Get normalized prediction then denormalize
        delta_obs_norm, delta_ag_norm = member(
            obs_curr, ag_curr, action_t, normalizer
        )
        delta_obs = normalizer.denormalize_delta_obs(delta_obs_norm)
        delta_ag = normalizer.denormalize_delta_ag(delta_ag_norm)

        obs_curr = obs_curr + delta_obs
        ag_curr = ag_curr + delta_ag

        obs_traj.append(np.array(obs_curr))
        ag_traj.append(np.array(ag_curr))

    return np.stack(obs_traj, axis=0), np.stack(ag_traj, axis=0)


def rollout_all_members(model, obs0, ag0, actions):
    """
    Roll out the action plan through each ensemble member independently.

    Returns:
        obs_trajectories: shape (E, H+1, obs_dim)
        ag_trajectories: shape (E, H+1, ag_dim)
    """
    all_obs = []
    all_ag = []

    for i, member in enumerate(model.members):
        obs_traj, ag_traj = rollout_single_member(
            member, model.normalizer, obs0, ag0, actions
        )
        all_obs.append(obs_traj)
        all_ag.append(ag_traj)

    return np.stack(all_obs, axis=0), np.stack(all_ag, axis=0)


# =============================================================================
# Ghost spread metric
# =============================================================================

def compute_ghost_spread(obs_trajectories):
    """
    Compute ghost spread: mean pairwise distance between ensemble
    trajectories at each timestep, using gripper xyz positions.

    Args:
        obs_trajectories: shape (E, H+1, obs_dim)

    Returns:
        spread_per_timestep: shape (H+1,), mean pairwise L2 distance
            between gripper positions at each timestep
        max_spread_per_timestep: shape (H+1,), max pairwise distance
    """
    E = obs_trajectories.shape[0]
    H_plus_1 = obs_trajectories.shape[1]

    # Extract gripper xyz: shape (E, H+1, 3)
    grip_pos = obs_trajectories[:, :, GRIP_X:GRIP_Z + 1]

    spread = np.zeros(H_plus_1)
    max_spread = np.zeros(H_plus_1)

    for t in range(H_plus_1):
        pairwise_dists = []
        for i, j in combinations(range(E), 2):
            d = np.linalg.norm(grip_pos[i, t] - grip_pos[j, t])
            pairwise_dists.append(d)
        if pairwise_dists:
            spread[t] = np.mean(pairwise_dists)
            max_spread[t] = np.max(pairwise_dists)

    return spread, max_spread


def compute_object_ghost_spread(obs_trajectories):
    """
    Compute ghost spread for the object position (achieved goal / object xyz).

    Args:
        obs_trajectories: shape (E, H+1, obs_dim)

    Returns:
        spread_per_timestep: shape (H+1,), mean pairwise L2 distance
            between object positions at each timestep
    """
    E = obs_trajectories.shape[0]
    H_plus_1 = obs_trajectories.shape[1]

    obj_pos = obs_trajectories[:, :, OBJ_X:OBJ_Z + 1]

    spread = np.zeros(H_plus_1)
    for t in range(H_plus_1):
        pairwise_dists = []
        for i, j in combinations(range(E), 2):
            d = np.linalg.norm(obj_pos[i, t] - obj_pos[j, t])
            pairwise_dists.append(d)
        if pairwise_dists:
            spread[t] = np.mean(pairwise_dists)

    return spread


# =============================================================================
# CEM planning for ghost episodes
# =============================================================================

def plan_for_state(model, obs0, ag0, desired_goal, action_low, action_high):
    """Run CEM planning from a given state to get an action plan."""
    cem_cfg = CEMConfig(
        horizon=HORIZON,
        population=128,       # smaller than default for speed
        elite_frac=0.1,
        cem_iters=5,
        init_std=0.5,
        action_penalty=0.01,
        reward_weight=1.0,
        reward_mode='dense',
        epistemic_bonus_weight=0.0,
        seed=SEED,
    )

    best_actions, mean, std, _ = cem_plan(
        model, obs0, ag0, desired_goal,
        action_low, action_high, cem_cfg,
        distance_threshold=0.05,
        horizon=HORIZON,
        return_rollout=False,
    )

    return best_actions


# =============================================================================
# Visualization: 3D ghost trajectories
# =============================================================================

def plot_ghost_3d(obs_trajectories, desired_goal, initial_obs,
                  title="Ghost Trajectories (Gripper XYZ)", spread=None):
    """
    Plot ghost trajectories for gripper xyz in 3D space.

    Args:
        obs_trajectories: shape (E, H+1, obs_dim), per-member trajectories
        desired_goal: shape (3,), target goal position
        initial_obs: shape (obs_dim,), initial observation
        title: plot title
        spread: optional shape (H+1,) ghost spread per timestep

    Returns:
        matplotlib Figure
    """
    E = obs_trajectories.shape[0]
    fig = plt.figure(figsize=(12, 8))

    if spread is not None:
        # Use gridspec for 3D plot + spread subplot
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax_spread = fig.add_subplot(gs[1])
    else:
        ax3d = fig.add_subplot(111, projection='3d')

    # Plot each member's ghost trajectory
    for m in range(E):
        grip_x = obs_trajectories[m, :, GRIP_X]
        grip_y = obs_trajectories[m, :, GRIP_Y]
        grip_z = obs_trajectories[m, :, GRIP_Z]

        ax3d.plot(grip_x, grip_y, grip_z,
                  color=MEMBER_COLORS[m % len(MEMBER_COLORS)],
                  alpha=0.6, linewidth=1.5,
                  label=MEMBER_LABELS[m] if m < len(MEMBER_LABELS) else f'Member {m}')

        # Start point (slightly larger)
        ax3d.scatter([grip_x[0]], [grip_y[0]], [grip_z[0]],
                     color=MEMBER_COLORS[m % len(MEMBER_COLORS)],
                     s=40, marker='o', alpha=0.8)

        # End point (diamond)
        ax3d.scatter([grip_x[-1]], [grip_y[-1]], [grip_z[-1]],
                     color=MEMBER_COLORS[m % len(MEMBER_COLORS)],
                     s=60, marker='D', alpha=0.9)

    # Plot initial gripper position
    ax3d.scatter([initial_obs[GRIP_X]], [initial_obs[GRIP_Y]], [initial_obs[GRIP_Z]],
                 color='black', s=100, marker='*', label='Start', zorder=5)

    # Plot initial object position
    ax3d.scatter([initial_obs[OBJ_X]], [initial_obs[OBJ_Y]], [initial_obs[OBJ_Z]],
                 color='brown', s=80, marker='s', label='Object', zorder=5)

    # Plot desired goal
    ax3d.scatter([desired_goal[0]], [desired_goal[1]], [desired_goal[2]],
                 color='gold', s=120, marker='*', label='Goal', zorder=5,
                 edgecolors='black', linewidths=0.5)

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title(title, fontsize=11)
    ax3d.legend(fontsize=7, loc='upper left')

    # Add spread subplot if provided
    if spread is not None:
        timesteps = np.arange(len(spread))
        ax_spread.fill_between(timesteps, 0, spread, alpha=0.3, color='steelblue')
        ax_spread.plot(timesteps, spread, color='steelblue', linewidth=1.5)
        ax_spread.set_xlabel('Timestep')
        ax_spread.set_ylabel('Ghost Spread')
        ax_spread.set_title('Mean Pairwise Distance (gripper) per Timestep', fontsize=10)
        ax_spread.grid(True, alpha=0.3)

    return fig


def plot_ghost_2d_projections(obs_trajectories, desired_goal, initial_obs,
                              title="Ghost Trajectories (2D Projections)"):
    """
    Plot ghost trajectories as 2D projections: XY, XZ, YZ planes.

    Args:
        obs_trajectories: shape (E, H+1, obs_dim)
        desired_goal: shape (3,)
        initial_obs: shape (obs_dim,)
        title: overall title

    Returns:
        matplotlib Figure
    """
    E = obs_trajectories.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    projections = [
        (GRIP_X, GRIP_Y, 'X', 'Y', 'XY Plane (top view)'),
        (GRIP_X, GRIP_Z, 'X', 'Z', 'XZ Plane (side view)'),
        (GRIP_Y, GRIP_Z, 'Y', 'Z', 'YZ Plane (side view)'),
    ]

    for ax, (ix, iy, xlabel, ylabel, subtitle) in zip(axes, projections):
        for m in range(E):
            traj_x = obs_trajectories[m, :, ix]
            traj_y = obs_trajectories[m, :, iy]

            ax.plot(traj_x, traj_y,
                    color=MEMBER_COLORS[m % len(MEMBER_COLORS)],
                    alpha=0.5, linewidth=1.5,
                    label=MEMBER_LABELS[m] if m < len(MEMBER_LABELS) else f'Member {m}')

            # End points
            ax.scatter([traj_x[-1]], [traj_y[-1]],
                       color=MEMBER_COLORS[m % len(MEMBER_COLORS)],
                       s=40, marker='D', alpha=0.8, zorder=5)

        # Start position
        ax.scatter([initial_obs[ix]], [initial_obs[iy]],
                   color='black', s=80, marker='*', label='Start', zorder=6)

        # Object position
        obj_ix = ix + 3 if ix < 3 else ix  # map grip index to object index
        obj_iy = iy + 3 if iy < 3 else iy
        if obj_ix < 6 and obj_iy < 6:
            ax.scatter([initial_obs[obj_ix]], [initial_obs[obj_iy]],
                       color='brown', s=60, marker='s', label='Object', zorder=6)

        # Goal
        goal_map = {GRIP_X: 0, GRIP_Y: 1, GRIP_Z: 2}
        gx = goal_map.get(ix, 0)
        gy = goal_map.get(iy, 1)
        ax.scatter([desired_goal[gx]], [desired_goal[gy]],
                   color='gold', s=100, marker='*', label='Goal', zorder=6,
                   edgecolors='black', linewidths=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_spread_comparison(episode_spreads, episode_labels):
    """
    Plot ghost spread timeseries for multiple episodes to compare
    confident vs uncertain states.

    Args:
        episode_spreads: list of arrays, each shape (H+1,)
        episode_labels: list of strings

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(episode_spreads)))
    for i, (spread, label) in enumerate(zip(episode_spreads, episode_labels)):
        timesteps = np.arange(len(spread))
        ax.plot(timesteps, spread, color=colors_cycle[i], linewidth=2,
                label=label, alpha=0.8)

    ax.set_xlabel('Timestep (planning horizon)')
    ax.set_ylabel('Ghost Spread (mean pairwise L2 distance)')
    ax.set_title('Ghost Spread: Confident vs Uncertain Episodes', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spread_with_task_progress(spread_gripper, spread_object, title=""):
    """
    Plot ghost spread for gripper and object alongside each other,
    showing how uncertainty propagates from gripper to object.

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    timesteps = np.arange(len(spread_gripper))

    ax1.fill_between(timesteps, 0, spread_gripper, alpha=0.3, color='steelblue')
    ax1.plot(timesteps, spread_gripper, color='steelblue', linewidth=2)
    ax1.set_ylabel('Gripper Ghost Spread')
    ax1.set_title(f'Ghost Spread per Timestep{" - " + title if title else ""}',
                  fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(timesteps, 0, spread_object, alpha=0.3, color='forestgreen')
    ax2.plot(timesteps, spread_object, color='forestgreen', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Object Ghost Spread')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# GIF generation
# =============================================================================

def generate_ghost_gif(all_episode_data, gif_path):
    """
    Generate a GIF cycling through ghost trajectory snapshots
    for multiple episodes.

    Args:
        all_episode_data: list of dicts, each with:
            'obs_trajectories': shape (E, H+1, obs_dim)
            'desired_goal': shape (3,)
            'initial_obs': shape (obs_dim,)
            'spread': shape (H+1,)
            'label': str
        gif_path: output path
    """
    frames = []
    for ep_data in all_episode_data:
        # Generate a 2D projection frame
        fig = plot_ghost_2d_projections(
            ep_data['obs_trajectories'],
            ep_data['desired_goal'],
            ep_data['initial_obs'],
            title=ep_data['label']
        )
        # Render figure to image array
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        frames.append(buf)
        plt.close(fig)

    if frames:
        try:
            imageio.mimsave(gif_path, frames, duration=2.0, loop=0)
        except TypeError:
            # Fallback for imageio v3+ API
            imageio.mimsave(gif_path, frames, loop=0)
        print(f"GIF saved to {gif_path} ({len(frames)} frames)")


# =============================================================================
# Main experiment
# =============================================================================

def run_ghost_experiment(model, episodes, meta, is_trained):
    """
    Core experiment: for each selected episode starting state, plan with CEM,
    roll out through each member, compute ghost spread, generate plots.
    """
    model_status = "trained" if is_trained else "random (untrained)"
    obs_dim = meta['obs_dim']
    action_dim = meta['action_dim']

    # Action bounds for FetchPush-v4
    action_low = np.full(action_dim, -1.0, dtype=np.float32)
    action_high = np.full(action_dim, 1.0, dtype=np.float32)

    n_episodes = min(N_GHOST_EPISODES, len(episodes))
    print(f"\nRunning ghost trajectory analysis on {n_episodes} episodes...")

    all_episode_results = []
    all_spreads = []
    all_labels = []
    all_gif_data = []

    for ep_idx in range(n_episodes):
        ep = episodes[ep_idx]
        # Use the initial state of the episode
        obs0 = ep['obs'][0]
        ag0 = ep['achieved_goal'][0]
        dg = ep['desired_goal_fixed']

        print(f"\n  Episode {ep_idx}: grip=({obs0[0]:.3f}, {obs0[1]:.3f}, {obs0[2]:.3f}), "
              f"obj=({obs0[3]:.3f}, {obs0[4]:.3f}, {obs0[5]:.3f}), "
              f"goal=({dg[0]:.3f}, {dg[1]:.3f}, {dg[2]:.3f})")

        # 1. Plan with CEM
        print(f"    Planning (CEM, H={HORIZON})...", end=" ", flush=True)
        best_actions = plan_for_state(model, obs0, ag0, dg, action_low, action_high)
        print(f"done, action shape={best_actions.shape}")

        # 2. Roll out through each member
        print(f"    Rolling out through {len(model.members)} members...", end=" ", flush=True)
        obs_trajs, ag_trajs = rollout_all_members(model, obs0, ag0, best_actions)
        print(f"done, obs_traj shape={obs_trajs.shape}")

        # 3. Compute ghost spread
        grip_spread, grip_max_spread = compute_ghost_spread(obs_trajs)
        obj_spread = compute_object_ghost_spread(obs_trajs)

        mean_spread = float(np.mean(grip_spread))
        max_spread_val = float(np.max(grip_spread))
        final_spread = float(grip_spread[-1])

        print(f"    Ghost spread: mean={mean_spread:.5f}, "
              f"max={max_spread_val:.5f}, final={final_spread:.5f}")

        ep_result = {
            'episode_idx': ep_idx,
            'initial_grip_pos': obs0[:3].tolist(),
            'initial_obj_pos': obs0[3:6].tolist(),
            'desired_goal': dg.tolist(),
            'ghost_spread_mean': mean_spread,
            'ghost_spread_max': max_spread_val,
            'ghost_spread_final': final_spread,
            'ghost_spread_per_timestep': grip_spread.tolist(),
            'ghost_max_spread_per_timestep': grip_max_spread.tolist(),
            'object_spread_per_timestep': obj_spread.tolist(),
            'object_spread_mean': float(np.mean(obj_spread)),
        }
        all_episode_results.append(ep_result)
        all_spreads.append(grip_spread)
        all_labels.append(f"Ep {ep_idx} (spread={mean_spread:.4f})")

        all_gif_data.append({
            'obs_trajectories': obs_trajs,
            'desired_goal': dg,
            'initial_obs': obs0,
            'spread': grip_spread,
            'label': f"Episode {ep_idx} ({model_status}) - "
                     f"mean spread={mean_spread:.4f}",
        })

    # Sort episodes by mean spread to identify confident vs uncertain
    sorted_eps = sorted(all_episode_results, key=lambda x: x['ghost_spread_mean'])
    confident_idx = sorted_eps[0]['episode_idx']
    uncertain_idx = sorted_eps[-1]['episode_idx']
    median_idx = sorted_eps[len(sorted_eps) // 2]['episode_idx']

    print(f"\n  Most confident episode: {confident_idx} "
          f"(spread={sorted_eps[0]['ghost_spread_mean']:.5f})")
    print(f"  Most uncertain episode: {uncertain_idx} "
          f"(spread={sorted_eps[-1]['ghost_spread_mean']:.5f})")
    print(f"  Median episode: {median_idx} "
          f"(spread={sorted_eps[len(sorted_eps)//2]['ghost_spread_mean']:.5f})")

    return {
        'all_episode_results': all_episode_results,
        'all_spreads': all_spreads,
        'all_labels': all_labels,
        'all_gif_data': all_gif_data,
        'confident_idx': confident_idx,
        'uncertain_idx': uncertain_idx,
        'median_idx': median_idx,
    }


def main():
    print("=" * 70)
    print("US-085: Ghost Trajectories for FetchPush Ensemble Predictions")
    print("=" * 70)

    # 1. Load or create model
    (model, meta), is_trained = load_or_create_model()
    model_status = "trained" if is_trained else "random (untrained)"
    print(f"Model status: {model_status}")

    # 2. Collect data from FetchPush-v4
    print("\n--- Collecting FetchPush-v4 trajectory data ---")
    episodes = collect_fetchpush_data()

    # 3. Run ghost experiment
    print("\n--- Running Ghost Trajectory Analysis ---")
    results = run_ghost_experiment(model, episodes, meta, is_trained)

    all_episode_results = results['all_episode_results']
    all_spreads = results['all_spreads']
    all_labels = results['all_labels']
    all_gif_data = results['all_gif_data']
    confident_idx = results['confident_idx']
    uncertain_idx = results['uncertain_idx']
    median_idx = results['median_idx']

    # 4. Generate visualizations
    print("\n--- Generating Visualizations ---")

    # 4a. 3D ghost plot for the most confident episode
    conf_data = all_gif_data[confident_idx]
    fig_conf_3d = plot_ghost_3d(
        conf_data['obs_trajectories'],
        conf_data['desired_goal'],
        conf_data['initial_obs'],
        title=f"Confident Episode {confident_idx} - Ghosts Aligned ({model_status})",
        spread=conf_data['spread']
    )
    save_figure(fig_conf_3d, "ghost_3d_confident", EXPERIMENT_NAME)

    # 4b. 3D ghost plot for the most uncertain episode
    unc_data = all_gif_data[uncertain_idx]
    fig_unc_3d = plot_ghost_3d(
        unc_data['obs_trajectories'],
        unc_data['desired_goal'],
        unc_data['initial_obs'],
        title=f"Uncertain Episode {uncertain_idx} - Ghosts Diverge ({model_status})",
        spread=unc_data['spread']
    )
    save_figure(fig_unc_3d, "ghost_3d_uncertain", EXPERIMENT_NAME)

    # 4c. 2D projections for confident episode
    fig_conf_2d = plot_ghost_2d_projections(
        conf_data['obs_trajectories'],
        conf_data['desired_goal'],
        conf_data['initial_obs'],
        title=f"Confident Episode {confident_idx} - 2D Projections ({model_status})"
    )
    save_figure(fig_conf_2d, "ghost_2d_confident", EXPERIMENT_NAME)

    # 4d. 2D projections for uncertain episode
    fig_unc_2d = plot_ghost_2d_projections(
        unc_data['obs_trajectories'],
        unc_data['desired_goal'],
        unc_data['initial_obs'],
        title=f"Uncertain Episode {uncertain_idx} - 2D Projections ({model_status})"
    )
    save_figure(fig_unc_2d, "ghost_2d_uncertain", EXPERIMENT_NAME)

    # 4e. Spread comparison across all episodes
    fig_spread = plot_spread_comparison(all_spreads, all_labels)
    save_figure(fig_spread, "spread_comparison", EXPERIMENT_NAME)

    # 4f. Gripper vs object spread for the most uncertain episode
    unc_result = all_episode_results[uncertain_idx]
    fig_spread_detail = plot_spread_with_task_progress(
        np.array(unc_result['ghost_spread_per_timestep']),
        np.array(unc_result['object_spread_per_timestep']),
        title=f"Uncertain Episode {uncertain_idx}"
    )
    save_figure(fig_spread_detail, "spread_gripper_vs_object", EXPERIMENT_NAME)

    # 4g. Generate GIF with confident, uncertain, and median episodes
    print("\n--- Generating GIF ---")
    gif_episodes = []
    for idx in [confident_idx, median_idx, uncertain_idx]:
        gif_episodes.append(all_gif_data[idx])

    from experiments.utils.plotting import RESULTS_DIR
    gif_path = os.path.join(str(RESULTS_DIR), f"ghost_trajectories_{model_status.replace(' ', '_').replace('(', '').replace(')', '')}.gif")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    generate_ghost_gif(gif_episodes, gif_path)

    # 5. Save results JSON
    print("\n--- Saving Results ---")
    # Compute summary statistics across all episodes
    all_mean_spreads = [r['ghost_spread_mean'] for r in all_episode_results]
    all_max_spreads = [r['ghost_spread_max'] for r in all_episode_results]

    metrics = {
        'model_status': model_status,
        'n_episodes': len(all_episode_results),
        'horizon': HORIZON,
        'ensemble_size': int(meta['ensemble_size']),
        'obs_dim': int(meta['obs_dim']),
        'action_dim': int(meta['action_dim']),
        'ghost_spread_summary': {
            'mean_across_episodes': float(np.mean(all_mean_spreads)),
            'std_across_episodes': float(np.std(all_mean_spreads)),
            'min_episode_spread': float(np.min(all_mean_spreads)),
            'max_episode_spread': float(np.max(all_mean_spreads)),
            'confident_episode': confident_idx,
            'uncertain_episode': uncertain_idx,
            'median_episode': median_idx,
        },
        'per_episode': all_episode_results,
    }

    config = {
        'horizon': HORIZON,
        'n_episodes_collected': N_EPISODES_COLLECT,
        'n_ghost_episodes': N_GHOST_EPISODES,
        'cem_population': 128,
        'cem_iters': 5,
        'seed': SEED,
        'env_id': meta.get('env_id', 'FetchPush-v4'),
        'model_hidden_size': meta.get('hidden_size', 256),
        'model_depth': meta.get('depth', 2),
    }

    save_results(
        EXPERIMENT_NAME,
        metrics=metrics,
        config=config,
        notes=(
            f"Ghost trajectory analysis of pandas Bayes ensemble ({model_status}) "
            f"on FetchPush-v4. "
            f"Rolled out CEM plans through {meta['ensemble_size']} ensemble members "
            f"independently for H={HORIZON} steps. "
            f"Mean ghost spread across {len(all_episode_results)} episodes: "
            f"{np.mean(all_mean_spreads):.5f} "
            f"(confident ep {confident_idx}: {np.min(all_mean_spreads):.5f}, "
            f"uncertain ep {uncertain_idx}: {np.max(all_mean_spreads):.5f}). "
            f"Ghost trajectories visualized in 3D gripper-xyz space (matplotlib fallback)."
        )
    )

    # 6. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_status}")
    print(f"  Episodes analyzed: {len(all_episode_results)}")
    print(f"  Planning horizon: {HORIZON} steps")
    print(f"  Ensemble size: {meta['ensemble_size']} members")
    print(f"  Mean ghost spread: {np.mean(all_mean_spreads):.5f} "
          f"(+/- {np.std(all_mean_spreads):.5f})")
    print(f"  Most confident episode: {confident_idx} "
          f"(spread={np.min(all_mean_spreads):.5f})")
    print(f"  Most uncertain episode: {uncertain_idx} "
          f"(spread={np.max(all_mean_spreads):.5f})")
    print(f"  Visualizations: 3D ghost plots (confident + uncertain), "
          f"2D projections, spread comparison, GIF")
    print("=" * 70)
    print("US-085 complete.")

    return metrics


if __name__ == '__main__':
    main()
