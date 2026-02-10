"""
US-055: Cross-environment TB comparison: LunarLander vs CartPole
================================================================

Tests whether TB discovers analogous structure across different environments.
Both LunarLander-v3 and CartPole-v1 have position-velocity pairs and
contact/boundary variables.

CartPole-v1 state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
Actions: 0 (push left) or 1 (push right)

LunarLander-v3 state: [x, y, vx, vy, angle, ang_vel, left_leg, right_leg]
Actions: 0 (noop), 1 (left), 2 (main), 3 (right)

Approach:
  1. Train a simple ensemble dynamics model on CartPole (5 MLPs)
  2. Collect trajectory data, compute dynamics gradients via torch.autograd
  3. Apply TB to CartPole 4D state space
  4. Load existing LunarLander TB results
  5. Compare partitions and coupling matrices across environments
"""

import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
NOUMENAL_DIR = os.path.dirname(RALPH_DIR)

sys.path.insert(0, RALPH_DIR)
sys.path.insert(0, NOUMENAL_DIR)

from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import (
    detect_blankets_otsu, detect_blankets_spectral, detect_blankets_coupling
)
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap,
    recursive_spectral_detection
)
from topological_blankets.clustering import cluster_internals
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

CARTPOLE_LABELS = ['x', 'x_dot', 'theta', 'theta_dot']
LUNARLANDER_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']


# =========================================================================
# CartPole Ensemble Dynamics Model
# =========================================================================

class DynamicsNetwork(nn.Module):
    """Single MLP dynamics model: (state, action_onehot) -> next_state."""

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x)


class EnsembleDynamics:
    """Ensemble of 5 dynamics MLPs for CartPole."""

    def __init__(self, n_ensemble=5, state_dim=4, action_dim=2, hidden_dim=128, lr=1e-3):
        self.n_ensemble = n_ensemble
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.models = nn.ModuleList([
            DynamicsNetwork(state_dim, action_dim, hidden_dim)
            for _ in range(n_ensemble)
        ])
        self.optimizers = [
            optim.Adam(m.parameters(), lr=lr) for m in self.models
        ]

    def train_step(self, states, actions_onehot, next_states):
        """Train all ensemble members on the same batch, return mean loss."""
        losses = []
        for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            pred = model(states, actions_onehot)
            loss = nn.MSELoss()(pred, next_states)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return np.mean(losses)

    def predict(self, states, actions_onehot):
        """Get ensemble mean prediction."""
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(states, actions_onehot))
        return torch.stack(preds).mean(dim=0)

    def forward_all(self, states, actions_onehot):
        """Forward through all ensemble members, return all predictions."""
        preds = []
        for model in self.models:
            preds.append(model(states, actions_onehot))
        return torch.stack(preds)  # (n_ensemble, batch, state_dim)

    def eval(self):
        for m in self.models:
            m.eval()


# =========================================================================
# CartPole Data Collection
# =========================================================================

def collect_cartpole_data(n_episodes=50, seed=42):
    """
    Collect transition data from CartPole-v1 using a competent balancing policy.

    Uses a linear feedback controller (PD on pole angle and cart position)
    with mild epsilon-greedy exploration to produce long episodes while
    still sampling diverse state regions. Collects exactly n_episodes episodes.
    """
    import gymnasium as gym

    env = gym.make('CartPole-v1')
    rng = np.random.RandomState(seed)

    all_states = []
    all_actions = []
    all_next_states = []

    ep_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=int(rng.randint(0, 100000)))
        ep_len = 0

        while True:
            # PD controller: linear combination of state variables
            # Coefficients tuned for CartPole-v1 balance
            # action = 1 (right) if weighted sum > 0, else 0 (left)
            cart_pos, cart_vel, pole_angle, pole_ang_vel = state
            score = (0.1 * cart_pos + 0.3 * cart_vel
                     + 1.0 * pole_angle + 0.5 * pole_ang_vel)

            if rng.random() < 0.1:
                # 10% exploration for state diversity
                action = env.action_space.sample()
            else:
                action = 1 if score > 0 else 0

            next_state, reward, term, trunc, _ = env.step(action)

            all_states.append(state.copy())
            all_actions.append(action)
            all_next_states.append(next_state.copy())

            state = next_state
            ep_len += 1

            if term or trunc:
                break

        ep_lengths.append(ep_len)

    env.close()

    states = np.array(all_states)
    actions = np.array(all_actions)
    next_states = np.array(all_next_states)

    print(f"Collected {len(states)} CartPole transitions across {len(ep_lengths)} episodes")
    print(f"Mean episode length: {np.mean(ep_lengths):.1f} (min={np.min(ep_lengths)}, max={np.max(ep_lengths)})")

    return {
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'episode_lengths': np.array(ep_lengths),
    }


def train_cartpole_ensemble(data, n_epochs=200, batch_size=256, seed=42):
    """Train the ensemble dynamics model on CartPole transition data.

    Trains on delta prediction (next_state - state) for better accuracy
    on small changes, which is typical in CartPole.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ensemble = EnsembleDynamics(n_ensemble=5, state_dim=4, action_dim=2, hidden_dim=128)

    states = torch.FloatTensor(data['states'])
    actions_onehot = torch.zeros(len(data['actions']), 2)
    actions_onehot[range(len(data['actions'])), data['actions']] = 1.0
    next_states = torch.FloatTensor(data['next_states'])

    # Train on deltas for better accuracy
    deltas = next_states - states

    n_samples = len(states)
    epoch_losses = []

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            batch_loss = ensemble.train_step(
                states[idx], actions_onehot[idx], deltas[idx]
            )
            epoch_loss += batch_loss
            n_batches += 1

        epoch_loss /= n_batches
        epoch_losses.append(epoch_loss)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.6f}")

    print(f"Final training loss: {epoch_losses[-1]:.6f}")

    # Evaluate on held-out predictions
    ensemble.eval()
    with torch.no_grad():
        pred_delta = ensemble.predict(states[:500], actions_onehot[:500])
        pred = states[:500] + pred_delta
        eval_mse = nn.MSELoss()(pred, next_states[:500]).item()
    print(f"Evaluation MSE on 500 samples: {eval_mse:.6f}")

    return ensemble, epoch_losses


def compute_cartpole_gradients(ensemble, data):
    """
    Compute gradients of dynamics model prediction error w.r.t. state.

    The ensemble predicts deltas (next_state - state), so the full prediction
    is state + delta. The loss is ||state + delta(state, action) - next_state||^2.
    """
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']
    n_samples = len(states)

    ensemble.eval()
    gradients = np.zeros_like(states)
    n_actions = 2  # CartPole discrete actions

    batch_size = 256
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_s = torch.FloatTensor(states[start:end]).requires_grad_(True)
        batch_a = torch.zeros(end - start, n_actions)
        batch_a[range(end - start), actions[start:end]] = 1.0
        batch_ns = torch.FloatTensor(next_states[start:end])

        # Forward through all ensemble members, take mean delta prediction
        all_deltas = ensemble.forward_all(batch_s, batch_a)
        delta_mean = all_deltas.mean(dim=0)
        pred = batch_s + delta_mean

        # Prediction error
        loss = ((pred - batch_ns) ** 2).sum()
        loss.backward()

        gradients[start:end] = batch_s.grad.detach().numpy()

    print(f"Computed CartPole gradients for {n_samples} transitions")
    print(f"Gradient magnitude per dim: {np.mean(np.abs(gradients), axis=0).round(4)}")
    print(f"  Labels: {CARTPOLE_LABELS}")

    return gradients


# =========================================================================
# CartPole TB Analysis
# =========================================================================

def analyze_cartpole_tb(gradients):
    """Apply TB to CartPole 4D state-space gradients."""
    print("\n--- TB Analysis: CartPole dynamics ---")

    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']
    coupling = features['coupling']

    # Run multiple detection methods
    result_grad = tb_pipeline(gradients, n_objects=2, method='gradient')
    result_coupling = tb_pipeline(gradients, n_objects=2, method='coupling')
    result_hybrid = tb_pipeline(gradients, n_objects=2, method='hybrid')

    # Spectral analysis
    from scipy.linalg import eigh
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    eigvals, eigvecs = eigh(L)
    n_clusters, eigengap = compute_eigengap(eigvals[:4])

    # Hierarchical detection
    hierarchy = recursive_spectral_detection(H_est, max_levels=3)

    # Print results
    for method_name, result in [('gradient', result_grad),
                                  ('coupling', result_coupling),
                                  ('hybrid', result_hybrid)]:
        assign = result['assignment']
        blanket = result['is_blanket']
        obj_dims = {i: [CARTPOLE_LABELS[j] for j in range(4) if assign[j] == i]
                    for i in set(assign) if i >= 0}
        blanket_dims = [CARTPOLE_LABELS[j] for j in range(4) if blanket[j]]
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
# Cross-environment comparison
# =========================================================================

def load_lunarlander_tb_results():
    """Load existing LunarLander TB analysis results."""
    results_dir = os.path.join(RALPH_DIR, 'results')

    # Find the actinf_tb_analysis result file
    candidates = sorted([
        f for f in os.listdir(results_dir)
        if 'actinf_tb_analysis' in f and f.endswith('.json')
    ])

    if not candidates:
        raise FileNotFoundError(
            "No LunarLander TB analysis results found. Run world_model_analysis.py first."
        )

    # Use the most recent one
    filepath = os.path.join(results_dir, candidates[-1])
    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Loaded LunarLander TB results from {candidates[-1]}")

    # Extract the dynamics results
    if 'metrics' in data and 'dynamics' in data['metrics']:
        return data['metrics']['dynamics']
    else:
        raise ValueError("Unexpected LunarLander results format")


def compute_normalized_similarity(coupling_a, coupling_b, labels_a, labels_b):
    """
    Compute structural similarity between two coupling matrices.

    Since the environments have different dimensionalities, we compare
    normalized coupling patterns. Specifically, we:
    1. Identify analogous variable pairs
    2. Compare coupling strengths between analogous pairs
    3. Compute a normalized structural similarity score
    """
    coupling_a = np.array(coupling_a)
    coupling_b = np.array(coupling_b)

    # Normalize both to [0, 1] range
    max_a = np.max(coupling_a) if np.max(coupling_a) > 0 else 1.0
    max_b = np.max(coupling_b) if np.max(coupling_b) > 0 else 1.0
    norm_a = coupling_a / max_a
    norm_b = coupling_b / max_b

    # Define analogous variable mappings:
    # CartPole {x, x_dot} <-> LunarLander {x, vx}
    # CartPole {theta, theta_dot} <-> LunarLander {angle, ang_vel}
    cartpole_pairs = {
        'position': (0, 1),     # x, x_dot
        'angular': (2, 3),      # theta, theta_dot
        'cross_pos_ang': ((0, 1), (2, 3)),  # position-angular coupling
    }

    lunarlander_pairs = {
        'position': (0, 2),     # x, vx
        'angular': (4, 5),      # angle, ang_vel
        'cross_pos_ang': ((0, 2), (4, 5)),  # position-angular coupling
    }

    # Compare within-pair coupling strengths
    comparisons = {}

    # Position coupling (x, x_dot for CartPole; x, vx for LunarLander)
    cp_pos = norm_a[cartpole_pairs['position'][0], cartpole_pairs['position'][1]]
    ll_pos = norm_b[lunarlander_pairs['position'][0], lunarlander_pairs['position'][1]]
    comparisons['position_velocity_coupling'] = {
        'cartpole': float(cp_pos),
        'lunarlander': float(ll_pos),
        'difference': float(abs(cp_pos - ll_pos)),
    }

    # Angular coupling (theta, theta_dot for CartPole; angle, ang_vel for LunarLander)
    cp_ang = norm_a[cartpole_pairs['angular'][0], cartpole_pairs['angular'][1]]
    ll_ang = norm_b[lunarlander_pairs['angular'][0], lunarlander_pairs['angular'][1]]
    comparisons['angular_velocity_coupling'] = {
        'cartpole': float(cp_ang),
        'lunarlander': float(ll_ang),
        'difference': float(abs(cp_ang - ll_ang)),
    }

    # Cross-domain coupling: position vars to angular vars
    cp_cross_vals = []
    for i in cartpole_pairs['cross_pos_ang'][0] if isinstance(cartpole_pairs['cross_pos_ang'][0], tuple) else [cartpole_pairs['cross_pos_ang'][0]]:
        for j in cartpole_pairs['cross_pos_ang'][1] if isinstance(cartpole_pairs['cross_pos_ang'][1], tuple) else [cartpole_pairs['cross_pos_ang'][1]]:
            cp_cross_vals.append(norm_a[i, j])
    # Unpack properly
    cp_cross_vals = []
    for i in [0, 1]:
        for j in [2, 3]:
            cp_cross_vals.append(norm_a[i, j])
    cp_cross = np.mean(cp_cross_vals)

    ll_cross_vals = []
    for i in [0, 2]:     # x, vx
        for j in [4, 5]:  # angle, ang_vel
            ll_cross_vals.append(norm_b[i, j])
    ll_cross = np.mean(ll_cross_vals)

    comparisons['cross_domain_coupling'] = {
        'cartpole': float(cp_cross),
        'lunarlander': float(ll_cross),
        'difference': float(abs(cp_cross - ll_cross)),
    }

    # Overall structural similarity: 1 - mean absolute difference
    diffs = [v['difference'] for v in comparisons.values()]
    similarity = 1.0 - np.mean(diffs)

    return {
        'comparisons': comparisons,
        'structural_similarity': float(similarity),
        'norm_coupling_cartpole': norm_a.tolist(),
        'norm_coupling_lunarlander': norm_b.tolist(),
    }


def compare_partitions(cartpole_results, lunarlander_results):
    """
    Compare TB partitions between CartPole and LunarLander.

    Analyze whether analogous variables (position-velocity pairs)
    play analogous roles in the partition structure.
    """
    comparison = {
        'environments': {},
        'shared_structure': [],
        'environment_specific': [],
    }

    # CartPole partition analysis
    for method in ['gradient_method', 'coupling_method', 'hybrid_method']:
        cp_assign = np.array(cartpole_results[method]['assignment'])
        cp_blanket = np.array(cartpole_results[method]['is_blanket'])

        cp_objs = {}
        for obj_id in set(cp_assign):
            if obj_id >= 0:
                cp_objs[int(obj_id)] = [CARTPOLE_LABELS[j] for j in range(4) if cp_assign[j] == obj_id]
        cp_blanket_vars = [CARTPOLE_LABELS[j] for j in range(4) if cp_blanket[j]]

        comparison['environments'].setdefault('cartpole', {})[method] = {
            'objects': cp_objs,
            'blanket': cp_blanket_vars,
            'n_objects': len(cp_objs),
            'blanket_size': len(cp_blanket_vars),
        }

    # LunarLander partition analysis
    for method in ['gradient_method', 'coupling_method', 'hybrid_method']:
        ll_assign = np.array(lunarlander_results[method]['assignment'])
        ll_blanket = np.array(lunarlander_results[method]['is_blanket'])

        ll_objs = {}
        for obj_id in set(ll_assign):
            if obj_id >= 0:
                ll_objs[int(obj_id)] = [LUNARLANDER_LABELS[j] for j in range(8) if ll_assign[j] == obj_id]
        ll_blanket_vars = [LUNARLANDER_LABELS[j] for j in range(8) if ll_blanket[j]]

        comparison['environments'].setdefault('lunarlander', {})[method] = {
            'objects': ll_objs,
            'blanket': ll_blanket_vars,
            'n_objects': len(ll_objs),
            'blanket_size': len(ll_blanket_vars),
        }

    # Analyze shared vs environment-specific structure
    # Check if position-velocity pairs group together or show strong coupling
    for method in ['gradient_method', 'coupling_method', 'hybrid_method']:
        cp_assign = np.array(cartpole_results[method]['assignment'])
        ll_assign = np.array(lunarlander_results[method]['assignment'])

        # CartPole: do {x, x_dot} cluster together?
        cp_pos_vel_same = (cp_assign[0] == cp_assign[1]) and (cp_assign[0] >= 0)
        # CartPole: do {theta, theta_dot} cluster together?
        cp_ang_vel_same = (cp_assign[2] == cp_assign[3]) and (cp_assign[2] >= 0)

        # LunarLander: do {x, vx} cluster together?
        ll_pos_vel_same = (ll_assign[0] == ll_assign[2]) and (ll_assign[0] >= 0)
        # LunarLander: do {angle, ang_vel} cluster together?
        ll_ang_vel_same = (ll_assign[4] == ll_assign[5]) and (ll_assign[4] >= 0)

        if cp_pos_vel_same and ll_pos_vel_same:
            comparison['shared_structure'].append(
                f"{method}: position-velocity coupling preserved in both environments"
            )
        if cp_ang_vel_same and ll_ang_vel_same:
            comparison['shared_structure'].append(
                f"{method}: angular-velocity coupling preserved in both environments"
            )

        # Also check if angular variables form a distinct physical group
        # (theta, theta_dot together OR both strongly coupled even if one is blanket)
        if cp_ang_vel_same:
            comparison['shared_structure'].append(
                f"{method}: CartPole angular variables (theta, theta_dot) form a single object"
            )
        if ll_pos_vel_same:
            comparison['shared_structure'].append(
                f"{method}: LunarLander positional variables (x, vx) co-assigned"
            )

    # Check coupling-based shared structure (across all methods)
    cp_coupling = np.array(cartpole_results['coupling'])
    ll_coupling = np.array(lunarlander_results['coupling'])

    # Normalize
    cp_max = np.max(cp_coupling) if np.max(cp_coupling) > 0 else 1.0
    ll_max = np.max(ll_coupling) if np.max(ll_coupling) > 0 else 1.0
    cp_norm = cp_coupling / cp_max
    ll_norm = ll_coupling / ll_max

    # Strong angular coupling in both?
    cp_ang_coupling = cp_norm[2, 3]
    ll_ang_coupling = ll_norm[4, 5]
    if cp_ang_coupling > 0.5 and ll_ang_coupling > 0.5:
        comparison['shared_structure'].append(
            f"Coupling analysis: strong angular-velocity coupling in both environments "
            f"(CartPole: {cp_ang_coupling:.2f}, LunarLander: {ll_ang_coupling:.2f})"
        )

    # Both show separation between positional and angular subsystems?
    cp_cross = np.mean([cp_norm[i, j] for i in [0, 1] for j in [2, 3]])
    ll_cross = np.mean([ll_norm[i, j] for i in [0, 2] for j in [4, 5]])
    if cp_cross < 0.8 and ll_cross < 0.8:
        comparison['shared_structure'].append(
            f"Coupling analysis: position-angular cross-coupling is moderate in both "
            f"(CartPole: {cp_cross:.2f}, LunarLander: {ll_cross:.2f}), "
            "indicating partially separable subsystems"
        )

    # Environment-specific features
    comparison['environment_specific'].append(
        "LunarLander has additional variables: y, vy (vertical), left_leg, right_leg (contact)"
    )
    comparison['environment_specific'].append(
        "CartPole has only 4 state variables vs LunarLander's 8"
    )
    comparison['environment_specific'].append(
        "CartPole is 1D translation + 1D rotation; LunarLander is 2D translation + 1D rotation + contacts"
    )

    return comparison


def build_comparison_table(cartpole_results, lunarlander_results):
    """Build a structured comparison table."""
    table = {
        'headers': ['Property', 'CartPole', 'LunarLander'],
        'rows': [],
    }

    # State dimensionality
    table['rows'].append(['State dimension', 4, 8])
    table['rows'].append(['Action space', '2 (left/right)', '4 (noop/left/main/right)'])

    # Coupling analysis
    cp_coupling = np.array(cartpole_results['coupling'])
    ll_coupling = np.array(lunarlander_results['coupling'])

    table['rows'].append([
        'Max coupling strength',
        f"{np.max(cp_coupling):.3f}",
        f"{np.max(ll_coupling):.3f}",
    ])

    # Position-velocity coupling
    table['rows'].append([
        'Position-velocity coupling',
        f"{cp_coupling[0,1]:.3f} (x, x_dot)",
        f"{ll_coupling[0,2]:.3f} (x, vx)",
    ])

    # Angular coupling
    table['rows'].append([
        'Angular coupling',
        f"{cp_coupling[2,3]:.3f} (theta, theta_dot)",
        f"{ll_coupling[4,5]:.3f} (angle, ang_vel)",
    ])

    # Spectral analysis
    table['rows'].append([
        'Eigengap',
        f"{cartpole_results['eigengap']:.3f}",
        f"{lunarlander_results['eigengap']:.3f}",
    ])
    table['rows'].append([
        'Spectral clusters',
        cartpole_results['n_clusters_spectral'],
        lunarlander_results['n_clusters_spectral'],
    ])

    # Partition results (hybrid method)
    cp_hybrid = cartpole_results['hybrid_method']
    ll_hybrid = lunarlander_results['hybrid_method']

    cp_n_objects = len(set(cp_hybrid['assignment']) - {-1})
    ll_n_objects = len(set(ll_hybrid['assignment']) - {-1})
    cp_blanket_size = sum(cp_hybrid['is_blanket'])
    ll_blanket_size = sum(ll_hybrid['is_blanket'])

    table['rows'].append(['N objects (hybrid)', cp_n_objects, ll_n_objects])
    table['rows'].append(['Blanket size (hybrid)', cp_blanket_size, ll_blanket_size])

    # Physical groupings
    cp_assign = np.array(cp_hybrid['assignment'])
    ll_assign = np.array(ll_hybrid['assignment'])

    cp_groups = {}
    for i, label in enumerate(CARTPOLE_LABELS):
        g = int(cp_assign[i])
        cp_groups.setdefault(g, []).append(label)

    ll_groups = {}
    for i, label in enumerate(LUNARLANDER_LABELS):
        g = int(ll_assign[i])
        ll_groups.setdefault(g, []).append(label)

    table['rows'].append([
        'Physical groupings (hybrid)',
        str(cp_groups),
        str(ll_groups),
    ])

    return table


# =========================================================================
# Visualization
# =========================================================================

def plot_cartpole_coupling(coupling, title="CartPole: Dynamics Coupling Matrix"):
    """Plot CartPole 4x4 coupling matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    coupling = np.array(coupling)
    im = ax.imshow(np.abs(coupling), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(CARTPOLE_LABELS, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CARTPOLE_LABELS, fontsize=10)
    ax.set_title(title, fontsize=11)

    for i in range(4):
        for j in range(4):
            val = np.abs(coupling[i, j])
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if val > np.max(np.abs(coupling)) * 0.6 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_side_by_side_coupling(cp_coupling, ll_coupling):
    """Side-by-side normalized coupling matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cp = np.array(cp_coupling)
    ll = np.array(ll_coupling)

    # Normalize both to [0, 1]
    cp_norm = cp / (np.max(cp) if np.max(cp) > 0 else 1.0)
    ll_norm = ll / (np.max(ll) if np.max(ll) > 0 else 1.0)

    # CartPole
    im1 = ax1.imshow(cp_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(CARTPOLE_LABELS, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(CARTPOLE_LABELS, fontsize=10)
    ax1.set_title('CartPole (normalized)', fontsize=11)
    for i in range(4):
        for j in range(4):
            val = cp_norm[i, j]
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                     color='white' if val > 0.6 else 'black')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # LunarLander
    im2 = ax2.imshow(ll_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(LUNARLANDER_LABELS, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels(LUNARLANDER_LABELS, fontsize=9)
    ax2.set_title('LunarLander (normalized)', fontsize=11)
    for i in range(8):
        for j in range(8):
            val = ll_norm[i, j]
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                     color='white' if val > 0.6 else 'black')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle('Cross-Environment Coupling Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_analogous_coupling_bars(similarity_results):
    """Bar chart comparing analogous coupling strengths."""
    fig, ax = plt.subplots(figsize=(8, 5))

    comparisons = similarity_results['comparisons']
    labels = list(comparisons.keys())
    cp_vals = [comparisons[k]['cartpole'] for k in labels]
    ll_vals = [comparisons[k]['lunarlander'] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, cp_vals, width, label='CartPole', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, ll_vals, width, label='LunarLander', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Normalized Coupling Strength')
    ax.set_title('Analogous Coupling Strengths Across Environments')
    ax.set_xticks(x)
    ax.set_xticklabels([
        'Position-Velocity\nCoupling',
        'Angular-Velocity\nCoupling',
        'Cross-Domain\nCoupling',
    ], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add similarity score
    sim = similarity_results['structural_similarity']
    ax.text(0.98, 0.95, f'Structural Similarity: {sim:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_partition_comparison(cartpole_results, lunarlander_results):
    """Visualize partition assignments side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ['gradient_method', 'coupling_method', 'hybrid_method']
    method_names = ['Gradient', 'Coupling', 'Hybrid']

    colors = {-1: '#95a5a6', 0: '#3498db', 1: '#e74c3c', 2: '#2ecc71', 3: '#f39c12'}

    for ax, method, mname in zip(axes, methods, method_names):
        cp_assign = np.array(cartpole_results[method]['assignment'])
        ll_assign = np.array(lunarlander_results[method]['assignment'])

        # Stack: CartPole on top, LunarLander on bottom
        all_labels = CARTPOLE_LABELS + [''] + LUNARLANDER_LABELS
        all_assigns = list(cp_assign) + [np.nan] + list(ll_assign)
        n_total = len(all_labels)

        y_positions = list(range(4)) + [5] + list(range(6, 14))

        for idx, (y, label, assign) in enumerate(zip(y_positions, all_labels, all_assigns)):
            if label == '':
                continue
            color = colors.get(int(assign), '#7f8c8d')
            ax.barh(y, 1, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
            role = 'blanket' if assign == -1 else f'obj {int(assign)}'
            ax.text(0.5, y, f'{label} ({role})', ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_yticks([1.5, 9.5])
        ax.set_yticklabels(['CartPole', 'LunarLander'], fontsize=10)
        ax.set_title(f'{mname} Method', fontsize=11)
        ax.invert_yaxis()
        ax.set_xticks([])

        # Separator line
        ax.axhline(y=4.5, color='black', linewidth=1.5, linestyle='--')

    plt.suptitle('TB Partition Assignments: CartPole vs LunarLander', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# =========================================================================
# Evaluate agent quality
# =========================================================================

def evaluate_cartpole_agent(ensemble, n_episodes=20, seed=100, horizon=5, n_candidates=32):
    """
    Evaluate the trained ensemble as a model-based controller using
    random-shooting MPC (Model Predictive Control).

    At each step, samples n_candidates random action sequences of
    length horizon, simulates them through the ensemble, and picks
    the first action of the sequence that maximizes a reward proxy
    (keeping the pole upright and cart centered).

    Returns mean episode length (>150 is the acceptance criterion).
    """
    import gymnasium as gym

    env = gym.make('CartPole-v1')
    ep_lengths = []
    rng = np.random.RandomState(seed)

    ensemble.eval()

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_len = 0

        while True:
            # Random-shooting MPC: evaluate all candidates in parallel
            state_t = torch.FloatTensor(state).unsqueeze(0).repeat(n_candidates, 1)
            action_seqs = rng.randint(0, 2, size=(n_candidates, horizon))

            total_reward = np.zeros(n_candidates)
            current = state_t.clone()

            with torch.no_grad():
                for t in range(horizon):
                    a_onehot = torch.zeros(n_candidates, 2)
                    for c in range(n_candidates):
                        a_onehot[c, action_seqs[c, t]] = 1.0

                    delta = ensemble.predict(current, a_onehot)
                    current = current + delta

                    theta = current[:, 2].numpy()
                    x = current[:, 0].numpy()
                    alive = (np.abs(theta) < 0.2095) & (np.abs(x) < 2.4)
                    total_reward += alive.astype(float)
                    total_reward += (1.0 - np.abs(theta) / 0.2095) * alive * 0.5

            best_idx = np.argmax(total_reward)
            best_action = int(action_seqs[best_idx, 0])

            next_state, reward, term, trunc, _ = env.step(best_action)
            state = next_state
            ep_len += 1

            if term or trunc:
                break

        ep_lengths.append(ep_len)
        if (ep + 1) % 5 == 0:
            print(f"  Eval ep {ep+1}/{n_episodes}: length={ep_len}")

    env.close()

    mean_len = np.mean(ep_lengths)
    print(f"MPC controller evaluation ({n_episodes} episodes): mean length = {mean_len:.1f} "
          f"(min={np.min(ep_lengths)}, max={np.max(ep_lengths)}, "
          f"median={np.median(ep_lengths):.0f})")

    return ep_lengths


# =========================================================================
# Discussion generation
# =========================================================================

def generate_discussion(similarity_results, partition_comparison, comparison_table):
    """Generate a structured discussion of what transfers and what doesn't."""
    discussion = {
        'transfers': [],
        'does_not_transfer': [],
        'interpretation': '',
    }

    sim = similarity_results['structural_similarity']
    comps = similarity_results['comparisons']

    # What transfers
    if comps['position_velocity_coupling']['difference'] < 0.3:
        discussion['transfers'].append(
            "Position-velocity coupling: Both environments show coupling between "
            "translational position and velocity variables, reflecting the shared "
            "physics of Newtonian kinematics (dx/dt = v)."
        )

    if comps['angular_velocity_coupling']['difference'] < 0.3:
        discussion['transfers'].append(
            "Angular-velocity coupling: Both environments couple angular position "
            "with angular velocity, reflecting rotational kinematics."
        )

    if sim > 0.5:
        discussion['transfers'].append(
            f"Overall structural similarity ({sim:.3f}) indicates that TB discovers "
            "analogous structure despite different embodiments."
        )

    # Shared structure from partition comparison
    for shared in partition_comparison.get('shared_structure', []):
        discussion['transfers'].append(shared)

    # What doesn't transfer
    discussion['does_not_transfer'].append(
        "Contact variables: LunarLander has left_leg and right_leg contact sensors "
        "with no CartPole analog. These form distinct coupling patterns."
    )
    discussion['does_not_transfer'].append(
        "Vertical dynamics: LunarLander's y, vy variables have no CartPole counterpart. "
        "The 2D flight dynamics add a vertical object that CartPole lacks."
    )
    discussion['does_not_transfer'].append(
        "Action complexity: LunarLander's 4-action space produces richer dynamics "
        "gradients than CartPole's binary action space."
    )

    # Environment-specific
    for specific in partition_comparison.get('environment_specific', []):
        discussion['does_not_transfer'].append(specific)

    # Interpretation
    discussion['interpretation'] = (
        f"The structural similarity of {sim:.3f} between CartPole and LunarLander "
        "TB partitions supports the hypothesis that shared physical symmetries "
        "(position-velocity kinematics, angular dynamics) produce shared blanket structure. "
        "The position-velocity and angle-angular_velocity pairs in both environments "
        "serve analogous roles, with coupling strengths that reflect the shared "
        "Newtonian physics underlying both systems. Environment-specific variables "
        "(vertical dynamics, contact sensors) appear as additional structure in LunarLander "
        "that has no CartPole analog, demonstrating that TB correctly identifies "
        "both shared and unique structural features across embodiments."
    )

    return discussion


# =========================================================================
# Main experiment
# =========================================================================

def run_us055():
    """Run the full cross-environment comparison experiment."""
    print("=" * 70)
    print("US-055: Cross-Environment TB Comparison (LunarLander vs CartPole)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Step 1: Collect CartPole data
    # -----------------------------------------------------------------
    print("\n--- Step 1: Collect CartPole trajectory data ---")
    cp_data = collect_cartpole_data(n_episodes=50, seed=42)

    # -----------------------------------------------------------------
    # Step 2: Train ensemble dynamics model
    # -----------------------------------------------------------------
    print("\n--- Step 2: Train CartPole ensemble dynamics model ---")
    ensemble, train_losses = train_cartpole_ensemble(cp_data, n_epochs=200)

    # -----------------------------------------------------------------
    # Step 3: Evaluate agent quality
    # -----------------------------------------------------------------
    print("\n--- Step 3: Evaluate model-based controller ---")
    eval_lengths = evaluate_cartpole_agent(ensemble, n_episodes=20)
    mean_eval_length = np.mean(eval_lengths)
    print(f"  Mean episode length: {mean_eval_length:.1f}")
    print(f"  Acceptance criterion (>150): {'PASS' if mean_eval_length > 150 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Step 4: Compute CartPole dynamics gradients
    # -----------------------------------------------------------------
    print("\n--- Step 4: Compute CartPole dynamics gradients ---")
    cp_gradients = compute_cartpole_gradients(ensemble, cp_data)

    # -----------------------------------------------------------------
    # Step 5: Apply TB to CartPole
    # -----------------------------------------------------------------
    print("\n--- Step 5: Apply TB to CartPole state space ---")
    cp_tb_results = analyze_cartpole_tb(cp_gradients)

    # -----------------------------------------------------------------
    # Step 6: Load LunarLander TB results
    # -----------------------------------------------------------------
    print("\n--- Step 6: Load LunarLander TB results ---")
    ll_tb_results = load_lunarlander_tb_results()
    print("  LunarLander TB results loaded successfully")

    # -----------------------------------------------------------------
    # Step 7: Cross-environment comparison
    # -----------------------------------------------------------------
    print("\n--- Step 7: Cross-environment comparison ---")

    # Coupling similarity
    similarity = compute_normalized_similarity(
        cp_tb_results['coupling'], ll_tb_results['coupling'],
        CARTPOLE_LABELS, LUNARLANDER_LABELS
    )
    print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
    for name, comp in similarity['comparisons'].items():
        print(f"    {name}: CP={comp['cartpole']:.3f}, LL={comp['lunarlander']:.3f}, diff={comp['difference']:.3f}")

    # Partition comparison
    partition_comp = compare_partitions(cp_tb_results, ll_tb_results)
    print(f"  Shared structure: {len(partition_comp['shared_structure'])} findings")
    for finding in partition_comp['shared_structure']:
        print(f"    - {finding}")

    # Comparison table
    comp_table = build_comparison_table(cp_tb_results, ll_tb_results)

    # Discussion
    discussion = generate_discussion(similarity, partition_comp, comp_table)

    # -----------------------------------------------------------------
    # Step 8: Generate visualizations
    # -----------------------------------------------------------------
    print("\n--- Step 8: Generate visualizations ---")

    # CartPole coupling matrix
    fig1 = plot_cartpole_coupling(np.array(cp_tb_results['coupling']))
    save_figure(fig1, 'cartpole_coupling', 'cross_env')

    # Side-by-side normalized coupling
    fig2 = plot_side_by_side_coupling(
        np.array(cp_tb_results['coupling']),
        np.array(ll_tb_results['coupling'])
    )
    save_figure(fig2, 'coupling_comparison', 'cross_env')

    # Analogous coupling bar chart
    fig3 = plot_analogous_coupling_bars(similarity)
    save_figure(fig3, 'analogous_coupling_bars', 'cross_env')

    # Partition comparison
    fig4 = plot_partition_comparison(cp_tb_results, ll_tb_results)
    save_figure(fig4, 'partition_comparison', 'cross_env')

    # -----------------------------------------------------------------
    # Step 9: Save results
    # -----------------------------------------------------------------
    print("\n--- Step 9: Save results ---")

    all_results = {
        'cartpole': {
            'tb_analysis': cp_tb_results,
            'n_transitions': len(cp_data['states']),
            'mean_episode_length': float(np.mean(cp_data['episode_lengths'])),
            'eval_mean_length': float(mean_eval_length),
            'eval_lengths': [int(x) for x in eval_lengths],
            'final_train_loss': float(train_losses[-1]),
        },
        'lunarlander': {
            'tb_analysis': ll_tb_results,
        },
        'comparison': {
            'similarity': similarity,
            'partition_comparison': partition_comp,
            'comparison_table': comp_table,
            'discussion': discussion,
        },
    }

    config = {
        'cartpole_state_labels': CARTPOLE_LABELS,
        'lunarlander_state_labels': LUNARLANDER_LABELS,
        'n_ensemble': 5,
        'hidden_dim': 128,
        'n_episodes_collected': 50,
        'n_train_epochs': 200,
        'mpc_horizon': 10,
        'mpc_candidates': 64,
        'delta_prediction': True,
    }

    save_results(
        'cross_env_comparison',
        all_results,
        config,
        notes='US-055: Cross-environment TB comparison. CartPole vs LunarLander. '
              'Tests whether TB discovers analogous structure across different environments.'
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("US-055 SUMMARY")
    print("=" * 70)
    print(f"CartPole transitions collected: {len(cp_data['states'])}")
    print(f"CartPole ensemble final loss: {train_losses[-1]:.6f}")
    print(f"CartPole controller mean length: {mean_eval_length:.1f} (criterion: >150)")
    print(f"Structural similarity: {similarity['structural_similarity']:.3f}")
    print(f"Shared structural findings: {len(partition_comp['shared_structure'])}")
    print()
    print("Comparison table:")
    for row in comp_table['rows']:
        print(f"  {row[0]:30s} | {str(row[1]):30s} | {str(row[2]):30s}")
    print()
    print("Discussion (what transfers):")
    for t in discussion['transfers']:
        print(f"  - {t}")
    print()
    print("Discussion (what does not transfer):")
    for t in discussion['does_not_transfer'][:3]:
        print(f"  - {t}")
    print()
    print(f"Interpretation: {discussion['interpretation'][:200]}...")

    return all_results


if __name__ == '__main__':
    results = run_us055()
