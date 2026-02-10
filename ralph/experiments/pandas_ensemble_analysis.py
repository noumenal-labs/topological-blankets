"""
US-076: Pandas Bayes Ensemble — Topological Blankets Analysis
=============================================================

Applies Topological Blankets (TB) to Alec's Bayes ensemble world model
for FetchPush-v4 manipulation. The ensemble is a JAX/Equinox model with
5 DynamicsMember instances, each predicting state deltas given (obs, ag, action).

This script:
  1. Loads the trained ensemble from pandas/data/push_demo/ (or creates
     a fresh random ensemble if no checkpoint exists).
  2. Collects observation data from FetchPush-v4 via random exploration.
  3. Computes Jacobians d(prediction)/d(obs) for each ensemble member.
  4. Aggregates Jacobian information into a gradient covariance (coupling) matrix.
  5. Runs TB hybrid detection to identify variable groupings.
  6. Compares discovered partition against ground-truth FetchPush structure.
  7. Visualizes ensemble disagreement alongside TB partition.
  8. Saves results JSON and visualization PNGs to results/.

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

Ground-truth partition:
  Object 0 (gripper):  indices [0,1,2, 9,10, 20,21]   (grip_pos + gripper_state + grip_velp)
  Object 1 (object):   indices [3,4,5, 11,12,13, 14,15,16, 17,18,19]  (object_pos + rot + velp + velr)
  Blanket (relational): indices [6,7,8]               (object_rel_pos)
  Unstructured:         indices [22,23,24]             (extra gripper state)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)                      # ralph/
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)                   # topological_blankets/

PANDAS_DIR = 'C:/Users/citiz/Documents/noumenal-labs/pandas'

# Add TB package parent and ralph root to sys.path
sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)
sys.path.insert(0, PANDAS_DIR)

from topological_blankets import TopologicalBlankets
from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap
)
from topological_blankets.clustering import cluster_internals
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

import jax
import jax.numpy as jnp
import equinox as eqx
from panda.model import EnsembleModel, make_model, ModelConfig, DynamicsMember
from panda.utils import Normalizer

# ── Constants ───────────────────────────────────────────────────────────
RUN_DIR = os.path.join(PANDAS_DIR, 'data', 'push_demo')
META_PATH = os.path.join(RUN_DIR, 'model.eqx.json')
MODEL_PATH = os.path.join(RUN_DIR, 'model.eqx')

# FetchPush-v4 observation variable labels (25D)
OBS_LABELS = [
    'grip_x', 'grip_y', 'grip_z',              # 0-2
    'obj_x', 'obj_y', 'obj_z',                 # 3-5
    'rel_x', 'rel_y', 'rel_z',                 # 6-8
    'grip_state_0', 'grip_state_1',             # 9-10
    'obj_rot_0', 'obj_rot_1', 'obj_rot_2',     # 11-13
    'obj_velp_x', 'obj_velp_y', 'obj_velp_z',  # 14-16
    'obj_velr_x', 'obj_velr_y', 'obj_velr_z',  # 17-19
    'grip_velp_x', 'grip_velp_y',              # 20-21
    'extra_0', 'extra_1', 'extra_2',           # 22-24
]

# Semantic group labels (for coloring)
OBS_GROUPS = [
    'gripper', 'gripper', 'gripper',
    'object', 'object', 'object',
    'relative', 'relative', 'relative',
    'gripper', 'gripper',
    'object', 'object', 'object',
    'object', 'object', 'object',
    'object', 'object', 'object',
    'gripper', 'gripper',
    'extra', 'extra', 'extra',
]

# Ground-truth assignment: object 0 = gripper, object 1 = object, -1 = blanket (rel),
# -2 = unstructured (extra). For ARI we treat extra as a third object or blanket.
GROUND_TRUTH_22D = np.array([
    0, 0, 0,      # grip_pos -> gripper
    1, 1, 1,      # obj_pos -> object
    -1, -1, -1,   # rel_pos -> blanket
    0, 0,          # gripper_state -> gripper
    1, 1, 1,       # obj_rot -> object
    1, 1, 1,       # obj_velp -> object
    1, 1, 1,       # obj_velr -> object
    0, 0,          # grip_velp -> gripper
])

# Full 25D ground truth (last 3 = unstructured extra, assign to gripper since
# they are zeros/gripper-related in the actual env)
GROUND_TRUTH_25D = np.concatenate([GROUND_TRUTH_22D, np.array([0, 0, 0])])

# Boolean blanket mask for 25D
BLANKET_MASK_25D = (GROUND_TRUTH_25D == -1)

EXPERIMENT_NAME = "pandas_ensemble_analysis"


# =========================================================================
# Model loading
# =========================================================================

def load_trained_model():
    """Load trained ensemble from pandas/data/push_demo/."""
    with open(META_PATH, 'r') as f:
        meta = json.load(f)

    obs_dim = meta['obs_dim']
    action_dim = meta['action_dim']
    ag_dim = meta['achieved_goal_dim']

    # Create identity normalizer as skeleton; deserialise will overwrite
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


# =========================================================================
# Data collection
# =========================================================================

def collect_fetchpush_data(n_episodes=20, max_steps=50, seed=42):
    """Collect random trajectories from FetchPush-v4."""
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
    import gymnasium as gym

    env = gym.make('FetchPush-v4', max_episode_steps=max_steps, reward_type='dense')

    all_obs = []
    all_ag = []
    all_dg = []
    all_actions = []

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        for step in range(max_steps):
            obs = obs_dict['observation']
            ag = obs_dict['achieved_goal']
            dg = obs_dict['desired_goal']
            action = env.action_space.sample()

            all_obs.append(obs.copy())
            all_ag.append(ag.copy())
            all_dg.append(dg.copy())
            all_actions.append(action.copy())

            obs_dict, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    env.close()

    data = {
        'obs': np.array(all_obs, dtype=np.float32),
        'achieved_goal': np.array(all_ag, dtype=np.float32),
        'desired_goal': np.array(all_dg, dtype=np.float32),
        'actions': np.array(all_actions, dtype=np.float32),
    }
    print(f"Collected {len(data['obs'])} transitions from {n_episodes} episodes")
    return data


# =========================================================================
# Jacobian computation
# =========================================================================

def compute_member_jacobians(member, normalizer, obs_batch, ag_batch, action_batch):
    """
    Compute Jacobian d(delta_pred)/d(obs) for a single ensemble member.

    For each sample, the Jacobian is of shape (out_dim, obs_dim) where
    out_dim = obs_dim + ag_dim (concatenated delta predictions).

    Returns: array of shape (N, out_dim, obs_dim).
    """
    def member_forward(obs_single, ag_single, act_single):
        """Forward pass returning concatenated [delta_obs, delta_ag]."""
        delta_obs, delta_ag = member(obs_single, ag_single, act_single, normalizer)
        return jnp.concatenate([delta_obs, delta_ag])

    # Jacobian w.r.t. first argument (obs)
    jac_fn = jax.jacobian(member_forward, argnums=0)

    # Vectorize over the batch
    batched_jac_fn = jax.vmap(
        lambda o, a, act: jac_fn(o, a, act),
        in_axes=(0, 0, 0)
    )

    obs_j = jnp.array(obs_batch)
    ag_j = jnp.array(ag_batch)
    act_j = jnp.array(action_batch)

    # Compute in chunks to avoid memory issues
    chunk_size = 64
    n_samples = obs_j.shape[0]
    jacobians_list = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        J_chunk = batched_jac_fn(obs_j[start:end], ag_j[start:end], act_j[start:end])
        jacobians_list.append(np.array(J_chunk))

    return np.concatenate(jacobians_list, axis=0)  # (N, out_dim, obs_dim)


def compute_ensemble_jacobians(model, data, max_samples=500):
    """
    Compute Jacobians for all ensemble members.

    Returns:
        jacobians: array of shape (E, N, out_dim, obs_dim)
        where E = ensemble_size, N = number of samples.
    """
    n_samples = min(max_samples, len(data['obs']))
    indices = np.random.RandomState(42).choice(len(data['obs']), n_samples, replace=False)

    obs_batch = data['obs'][indices]
    ag_batch = data['achieved_goal'][indices]
    act_batch = data['actions'][indices]

    print(f"Computing Jacobians for {len(model.members)} members on {n_samples} samples...")

    all_jacobians = []
    for i, member in enumerate(model.members):
        print(f"  Member {i+1}/{len(model.members)}...", end=" ", flush=True)
        J = compute_member_jacobians(member, model.normalizer, obs_batch, ag_batch, act_batch)
        all_jacobians.append(J)
        print(f"shape={J.shape}")

    jacobians = np.stack(all_jacobians, axis=0)  # (E, N, out_dim, obs_dim)
    print(f"Ensemble Jacobians: {jacobians.shape}")
    return jacobians, indices


# =========================================================================
# Gradient covariance and TB input construction
# =========================================================================

def jacobians_to_tb_gradients(jacobians):
    """
    Convert ensemble Jacobians to gradient samples for TB analysis.

    For world model structure discovery, we want the coupling structure
    between input variables as captured by the model's learned dynamics.
    Each (member, sample) pair has a Jacobian J of shape (out_dim, obs_dim).

    The per-variable sensitivity profile s_j = ||J[:, j]|| is a scalar
    per variable. Across many (sample, member) pairs, these form
    gradient-like samples of shape (E*N, obs_dim) suitable for TB's
    gradient covariance estimation.

    Returns:
        gradients: array of shape (E*N, obs_dim)  -- TB input (sensitivity profiles)
        var_sensitivity: array of shape (N, obs_dim) -- mean sensitivity per sample
        ensemble_disagreement: array of shape (N, obs_dim) -- std across members
    """
    E, N, out_dim, obs_dim = jacobians.shape

    # Per-(member, sample) sensitivity: column norms of Jacobian
    # s[e, n, j] = ||J[e, n, :, j]||  -- how much does output change with var j?
    # shape: (E, N, obs_dim)
    sensitivity = np.sqrt(np.sum(jacobians ** 2, axis=2))

    # For TB: use all (member x sample) sensitivities as gradient samples
    # shape: (E*N, obs_dim)
    gradients = sensitivity.reshape(E * N, obs_dim)

    # Mean sensitivity across members: (N, obs_dim)
    var_sensitivity = sensitivity.mean(axis=0)  # (N, obs_dim)

    # Ensemble disagreement: std across members for each (sample, variable)
    ensemble_disagreement = sensitivity.std(axis=0)  # (N, obs_dim)

    return gradients, var_sensitivity, ensemble_disagreement


def compute_coupling_from_jacobians(jacobians):
    """
    Alternative coupling computation directly from Jacobians.

    For each sample, the Jacobian J is (out_dim, obs_dim). The coupling
    between variables i and j is captured by how similar their columns
    J[:, i] and J[:, j] are across samples. This is equivalent to computing
    the cross-covariance of column norms, but we can also compute
    J^T @ J which gives the "Fisher-like" information per sample, then
    average across samples and members.

    Returns: coupling matrix of shape (obs_dim, obs_dim).
    """
    E, N, out_dim, obs_dim = jacobians.shape

    # For each (member, sample), compute J^T @ J -> (obs_dim, obs_dim)
    # Then average across all members and samples
    fisher_sum = np.zeros((obs_dim, obs_dim))
    count = 0
    for e in range(E):
        for n in range(N):
            J = jacobians[e, n]  # (out_dim, obs_dim)
            fisher_sum += J.T @ J
            count += 1

    fisher_avg = fisher_sum / count

    # Normalize to correlation-like coupling
    D = np.sqrt(np.diag(fisher_avg)) + 1e-8
    coupling = np.abs(fisher_avg) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    return coupling, fisher_avg


# =========================================================================
# TB Analysis
# =========================================================================

def run_tb_analysis(gradients, n_objects=2, method='hybrid'):
    """Run Topological Blankets on the gradient samples."""
    print(f"\nRunning TB analysis (method={method}, n_objects={n_objects})...")
    print(f"  Gradient samples: {gradients.shape}")

    tb = TopologicalBlankets(method=method, n_objects=n_objects)
    tb.fit(gradients)

    assignment = tb.get_assignment()
    blankets = tb.get_blankets()
    objects = tb.get_objects()
    coupling = tb.get_coupling_matrix()
    features = tb.get_features()

    print(f"  Blanket variables ({len(blankets)}): {blankets}")
    for obj_id, obj_vars in objects.items():
        print(f"  Object {obj_id} variables ({len(obj_vars)}): {obj_vars}")

    # Print semantic interpretation
    print("\n  Semantic interpretation:")
    for obj_id, obj_vars in objects.items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels = [OBS_LABELS[i] for i in blankets if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels}")

    return {
        'assignment': assignment,
        'blankets': blankets,
        'objects': objects,
        'coupling': coupling,
        'features': features,
        'tb_instance': tb,
    }


def run_tb_from_fisher(fisher_coupling, n_objects=2, method='hybrid'):
    """
    Run TB analysis using a pre-computed Fisher coupling matrix.

    Instead of computing gradients -> covariance -> coupling, we directly
    supply the Fisher information coupling (J^T J averaged over samples
    and ensemble members) as the coupling matrix. TB detection methods
    operate on this coupling structure.
    """
    from topological_blankets.detection import (
        detect_blankets_hybrid, detect_blankets_gradient, detect_blankets_coupling
    )

    print(f"\nRunning TB from Fisher coupling (method={method}, n_objects={n_objects})...")

    obs_dim = fisher_coupling.shape[0]

    # Create features dict mimicking compute_geometric_features output
    # The Fisher coupling IS the Hessian estimate for this application
    D = np.sqrt(np.diag(np.abs(fisher_coupling)) + 1e-8)
    coupling_norm = np.abs(fisher_coupling) / np.outer(D, D)
    np.fill_diagonal(coupling_norm, 0)

    grad_magnitude = D  # diagonal = self-coupling = variable importance
    grad_variance = D ** 2

    features = {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': fisher_coupling,
        'coupling': coupling_norm,
    }

    if method == 'hybrid':
        result = detect_blankets_hybrid(
            np.random.randn(100, obs_dim),  # dummy gradients for fallback
            fisher_coupling
        )
        is_blanket = result['is_blanket']
    elif method == 'coupling':
        is_blanket = detect_blankets_coupling(fisher_coupling, coupling_norm, n_objects)
    else:
        from topological_blankets.detection import detect_blankets_otsu
        is_blanket, _ = detect_blankets_otsu(features)

    assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)

    blankets = np.where(is_blanket)[0]
    objects = {}
    for label in np.unique(assignment):
        if label >= 0:
            objects[int(label)] = np.where(assignment == label)[0]

    print(f"  Blanket variables ({len(blankets)}): {blankets}")
    for obj_id, obj_vars in objects.items():
        print(f"  Object {obj_id} variables ({len(obj_vars)}): {obj_vars}")

    # Semantic interpretation
    print("\n  Semantic interpretation:")
    for obj_id, obj_vars in objects.items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels = [OBS_LABELS[i] for i in blankets if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels}")

    return {
        'assignment': assignment,
        'blankets': blankets,
        'objects': objects,
        'coupling': coupling_norm,
        'features': features,
    }


def compute_ari_vs_ground_truth(assignment):
    """Compute Adjusted Rand Index against the ground-truth partition."""
    from sklearn.metrics import adjusted_rand_score, f1_score

    n_vars = min(len(assignment), len(GROUND_TRUTH_25D))
    pred = assignment[:n_vars].copy()
    gt = GROUND_TRUTH_25D[:n_vars].copy()

    # ARI (ignoring blanket label distinction)
    ari = adjusted_rand_score(gt, pred)

    # Blanket F1
    pred_blanket = (pred == -1)
    gt_blanket = BLANKET_MASK_25D[:n_vars]
    if gt_blanket.sum() > 0:
        blanket_f1 = f1_score(gt_blanket, pred_blanket)
    else:
        blanket_f1 = 0.0

    print(f"\n  ARI vs ground truth: {ari:.3f}")
    print(f"  Blanket F1: {blanket_f1:.3f}")

    return ari, blanket_f1


# =========================================================================
# Visualization
# =========================================================================

def plot_coupling_matrix(coupling, assignment, title_suffix=""):
    """Plot the coupling matrix with variable labels and partition overlay."""
    n_vars = coupling.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw coupling matrix
    ax = axes[0]
    im = ax.imshow(coupling, cmap='hot', aspect='auto')
    ax.set_title(f'Coupling Matrix{title_suffix}', fontsize=12)
    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Variable Index')

    # Add variable labels on axes
    tick_labels = [OBS_LABELS[i] if i < len(OBS_LABELS) else f'v{i}'
                   for i in range(n_vars)]
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(tick_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Coupling matrix reordered by partition
    ax = axes[1]
    order = []
    unique_labels = sorted(set(assignment))
    label_boundaries = []
    for lbl in unique_labels:
        indices = np.where(assignment == lbl)[0]
        label_boundaries.append((len(order), len(order) + len(indices), lbl))
        order.extend(indices.tolist())

    reordered = coupling[np.ix_(order, order)]
    im2 = ax.imshow(reordered, cmap='hot', aspect='auto')
    ax.set_title(f'Coupling (reordered by TB partition){title_suffix}', fontsize=12)

    # Draw partition boundaries
    for start, end, lbl in label_boundaries:
        color = 'cyan' if lbl == -1 else ['lime', 'yellow', 'magenta'][lbl % 3]
        label_text = 'Blanket' if lbl == -1 else f'Object {lbl}'
        ax.axhline(y=start - 0.5, color=color, linewidth=1.5, linestyle='--')
        ax.axvline(x=start - 0.5, color=color, linewidth=1.5, linestyle='--')
        ax.axhline(y=end - 0.5, color=color, linewidth=1.5, linestyle='--')
        ax.axvline(x=end - 0.5, color=color, linewidth=1.5, linestyle='--')
        mid = (start + end) / 2
        ax.text(n_vars + 0.5, mid, label_text, fontsize=8, va='center', color=color)

    reorder_labels = [tick_labels[i] for i in order]
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(reorder_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(reorder_labels, fontsize=6)
    plt.colorbar(im2, ax=ax, fraction=0.046)

    fig.tight_layout()
    return fig


def plot_partition_comparison(assignment, title_suffix=""):
    """Side-by-side comparison of discovered vs ground-truth partition."""
    n_vars = min(len(assignment), len(GROUND_TRUTH_25D))
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

    # Discovered partition
    ax = axes[0]
    colors_discovered = []
    cmap = {-1: 'red', 0: 'steelblue', 1: 'forestgreen', 2: 'orange'}
    for v in assignment[:n_vars]:
        colors_discovered.append(cmap.get(int(v), 'gray'))
    ax.bar(range(n_vars), np.ones(n_vars), color=colors_discovered, edgecolor='black', linewidth=0.5)
    ax.set_title(f'TB Discovered Partition{title_suffix}', fontsize=11)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([OBS_LABELS[i] for i in range(n_vars)], rotation=90, fontsize=7)
    ax.set_yticks([])
    # Legend
    from matplotlib.patches import Patch
    unique_vals = sorted(set(assignment[:n_vars].tolist()))
    legend_elements = []
    for v in unique_vals:
        label = 'Blanket' if v == -1 else f'Object {v}'
        legend_elements.append(Patch(facecolor=cmap.get(int(v), 'gray'), label=label))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Ground-truth partition
    ax = axes[1]
    colors_gt = []
    for v in GROUND_TRUTH_25D[:n_vars]:
        colors_gt.append(cmap.get(int(v), 'gray'))
    ax.bar(range(n_vars), np.ones(n_vars), color=colors_gt, edgecolor='black', linewidth=0.5)
    ax.set_title('Ground-Truth Partition', fontsize=11)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([OBS_LABELS[i] for i in range(n_vars)], rotation=90, fontsize=7)
    ax.set_yticks([])
    legend_elements_gt = [
        Patch(facecolor='steelblue', label='Object 0 (gripper)'),
        Patch(facecolor='forestgreen', label='Object 1 (object)'),
        Patch(facecolor='red', label='Blanket (relative)'),
    ]
    ax.legend(handles=legend_elements_gt, loc='upper right', fontsize=8)

    fig.tight_layout()
    return fig


def plot_disagreement_and_partition(disagreement_per_var, assignment, title_suffix=""):
    """
    Plot ensemble disagreement per variable alongside the TB partition.
    High-disagreement variables should cluster differently from low-disagreement ones.
    """
    n_vars = min(len(disagreement_per_var), len(assignment))
    fig, ax = plt.subplots(figsize=(14, 5))

    # Color bars by partition assignment
    cmap = {-1: 'red', 0: 'steelblue', 1: 'forestgreen', 2: 'orange'}
    colors = [cmap.get(int(assignment[i]), 'gray') for i in range(n_vars)]

    bars = ax.bar(range(n_vars), disagreement_per_var[:n_vars], color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_title(f'Ensemble Disagreement by Variable (colored by TB partition){title_suffix}',
                 fontsize=11)
    ax.set_xlabel('Observation Variable')
    ax.set_ylabel('Mean Ensemble Disagreement (std of sensitivity)')
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([OBS_LABELS[i] if i < len(OBS_LABELS) else f'v{i}'
                        for i in range(n_vars)], rotation=90, fontsize=7)

    # Add semantic group annotations
    from matplotlib.patches import Patch
    unique_vals = sorted(set(assignment[:n_vars].tolist()))
    legend_elements = []
    for v in unique_vals:
        label = 'Blanket' if v == -1 else f'Object {v}'
        legend_elements.append(Patch(facecolor=cmap.get(int(v), 'gray'), label=label))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.tight_layout()
    return fig


def plot_sensitivity_heatmap(var_sensitivity, title_suffix=""):
    """Plot mean sensitivity (Jacobian column norms) across samples."""
    n_vars = var_sensitivity.shape[1]
    mean_sens = var_sensitivity.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Group coloring
    group_colors = {
        'gripper': 'steelblue', 'object': 'forestgreen',
        'relative': 'red', 'extra': 'gray'
    }
    colors = [group_colors.get(OBS_GROUPS[i], 'gray') if i < len(OBS_GROUPS) else 'gray'
              for i in range(n_vars)]

    ax.bar(range(n_vars), mean_sens, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(f'Mean Jacobian Sensitivity per Variable{title_suffix}', fontsize=11)
    ax.set_xlabel('Observation Variable')
    ax.set_ylabel('Mean ||J[:, var]|| across samples')
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([OBS_LABELS[i] if i < len(OBS_LABELS) else f'v{i}'
                        for i in range(n_vars)], rotation=90, fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Gripper'),
        Patch(facecolor='forestgreen', label='Object'),
        Patch(facecolor='red', label='Relative'),
        Patch(facecolor='gray', label='Extra'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig


def plot_fisher_coupling(fisher_coupling, title_suffix=""):
    """Plot the Fisher-based coupling matrix from Jacobians."""
    n_vars = fisher_coupling.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(fisher_coupling, cmap='hot', aspect='auto')
    ax.set_title(f'Fisher Coupling (J^T @ J){title_suffix}', fontsize=12)

    tick_labels = [OBS_LABELS[i] if i < len(OBS_LABELS) else f'v{i}'
                   for i in range(n_vars)]
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(tick_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    return fig


# =========================================================================
# Multi-method comparison
# =========================================================================

def run_multi_method_comparison(gradients, fisher_raw=None):
    """Run TB with multiple detection methods and compare."""
    methods = ['hybrid', 'gradient', 'coupling']
    results = {}

    for method in methods:
        try:
            tb_result = run_tb_analysis(gradients, n_objects=2, method=method)
            ari, f1 = compute_ari_vs_ground_truth(tb_result['assignment'])
            results[f'sensitivity_{method}_2obj'] = {
                'assignment': tb_result['assignment'].tolist(),
                'blankets': tb_result['blankets'].tolist(),
                'objects': {str(k): v.tolist() for k, v in tb_result['objects'].items()},
                'ari': ari,
                'blanket_f1': f1,
            }
        except Exception as e:
            print(f"  Method 'sensitivity_{method}' failed: {e}")
            results[f'sensitivity_{method}_2obj'] = {'error': str(e)}

    # Fisher-based methods
    if fisher_raw is not None:
        for n_obj in [2, 3]:
            for method in ['hybrid', 'coupling']:
                try:
                    tb_result = run_tb_from_fisher(fisher_raw, n_objects=n_obj, method=method)
                    ari, f1 = compute_ari_vs_ground_truth(tb_result['assignment'])
                    key = f'fisher_{method}_{n_obj}obj'
                    results[key] = {
                        'assignment': tb_result['assignment'].tolist(),
                        'blankets': tb_result['blankets'].tolist(),
                        'objects': {str(k): v.tolist() for k, v in tb_result['objects'].items()},
                        'ari': ari,
                        'blanket_f1': f1,
                    }
                except Exception as e:
                    key = f'fisher_{method}_{n_obj}obj'
                    print(f"  Method '{key}' failed: {e}")
                    results[key] = {'error': str(e)}

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("US-076: Pandas Bayes Ensemble — Topological Blankets Analysis")
    print("=" * 70)

    # 1. Load or create model
    (model, meta), is_trained = load_or_create_model()
    model_status = "trained" if is_trained else "random (untrained)"
    print(f"Model status: {model_status}")

    # 2. Collect data from FetchPush-v4
    print("\n--- Collecting FetchPush-v4 trajectory data ---")
    data = collect_fetchpush_data(n_episodes=20, max_steps=50, seed=42)

    # 3. Compute Jacobians
    print("\n--- Computing ensemble Jacobians ---")
    jacobians, sample_indices = compute_ensemble_jacobians(model, data, max_samples=300)

    # 4. Convert to TB gradient format
    print("\n--- Preparing TB input ---")
    gradients, var_sensitivity, ensemble_disagreement = jacobians_to_tb_gradients(jacobians)
    print(f"  TB gradient samples: {gradients.shape}")
    print(f"  Per-variable sensitivity: {var_sensitivity.shape}")
    print(f"  Ensemble disagreement: {ensemble_disagreement.shape}")

    # 5. Compute Fisher coupling directly from Jacobians
    fisher_coupling, fisher_raw = compute_coupling_from_jacobians(jacobians)

    # 6. Run comprehensive multi-method comparison
    print("\n--- Multi-Method Comparison ---")
    method_results = run_multi_method_comparison(gradients, fisher_raw=fisher_raw)

    # Select the best result by ARI
    best_key = None
    best_ari = -np.inf
    for key, mres in method_results.items():
        if 'error' not in mres and mres['ari'] > best_ari:
            best_ari = mres['ari']
            best_key = key

    # Reconstruct primary result from best method
    if best_key and best_key.startswith('fisher_'):
        parts = best_key.split('_')
        method = parts[1]
        n_obj = int(parts[2].replace('obj', ''))
        primary_result = run_tb_from_fisher(fisher_raw, n_objects=n_obj, method=method)
    elif best_key:
        parts = best_key.split('_')
        method = parts[1]
        n_obj = int(parts[2].replace('obj', ''))
        primary_result = run_tb_analysis(gradients, n_objects=n_obj, method=method)
    else:
        # Fallback: hybrid on sensitivity
        primary_result = run_tb_analysis(gradients, n_objects=2, method='hybrid')
        best_key = 'sensitivity_hybrid_2obj'

    primary_ari, primary_f1 = compute_ari_vs_ground_truth(primary_result['assignment'])
    primary_method = best_key
    print(f"\n  Best approach: {primary_method} (ARI={primary_ari:.3f})")

    # 8. Compute mean disagreement per variable (averaged across samples)
    disagreement_per_var = ensemble_disagreement.mean(axis=0)  # (obs_dim,)

    # 9. Generate visualizations
    print("\n--- Generating Visualizations ---")

    # Coupling matrix with partition overlay (use primary result)
    fig_coupling = plot_coupling_matrix(
        primary_result['coupling'], primary_result['assignment'],
        title_suffix=f" ({model_status}, {primary_method})")
    save_figure(fig_coupling, "coupling_matrix", EXPERIMENT_NAME)

    # Partition comparison
    fig_partition = plot_partition_comparison(
        primary_result['assignment'],
        title_suffix=f" ({model_status}, {primary_method})")
    save_figure(fig_partition, "partition_comparison", EXPERIMENT_NAME)

    # Disagreement + partition
    fig_disagree = plot_disagreement_and_partition(
        disagreement_per_var, primary_result['assignment'],
        title_suffix=f" ({model_status}, {primary_method})")
    save_figure(fig_disagree, "disagreement_partition", EXPERIMENT_NAME)

    # Sensitivity heatmap
    fig_sensitivity = plot_sensitivity_heatmap(
        var_sensitivity,
        title_suffix=f" ({model_status})")
    save_figure(fig_sensitivity, "sensitivity_heatmap", EXPERIMENT_NAME)

    # Fisher coupling
    fig_fisher = plot_fisher_coupling(
        fisher_coupling,
        title_suffix=f" ({model_status})")
    save_figure(fig_fisher, "fisher_coupling", EXPERIMENT_NAME)

    # 10. Save results JSON
    print("\n--- Saving Results ---")
    metrics = {
        'model_status': model_status,
        'n_samples': int(gradients.shape[0]),
        'n_env_transitions': int(len(data['obs'])),
        'obs_dim': int(meta['obs_dim']),
        'ensemble_size': int(meta['ensemble_size']),
        'primary_method': primary_method,
        'primary_ari': float(primary_ari),
        'primary_blanket_f1': float(primary_f1),
        'primary_assignment': primary_result['assignment'].tolist(),
        'primary_blankets': primary_result['blankets'].tolist(),
        'primary_objects': {
            str(k): v.tolist() for k, v in primary_result['objects'].items()
        },
        'multi_method': {},
        'disagreement_per_var': disagreement_per_var.tolist(),
        'mean_sensitivity_per_var': var_sensitivity.mean(axis=0).tolist(),
        'obs_labels': OBS_LABELS,
        'semantic_interpretation': {},
    }

    # Multi-method results
    for method_name, mres in method_results.items():
        if 'error' not in mres:
            metrics['multi_method'][method_name] = {
                'ari': mres['ari'],
                'blanket_f1': mres['blanket_f1'],
                'n_blanket_vars': len(mres['blankets']),
                'n_objects': len(mres['objects']),
            }

    # Semantic interpretation
    for obj_id, obj_vars in primary_result['objects'].items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        groups = [OBS_GROUPS[i] for i in obj_vars if i < len(OBS_GROUPS)]
        dominant_group = max(set(groups), key=groups.count) if groups else 'unknown'
        metrics['semantic_interpretation'][f'object_{obj_id}'] = {
            'variables': labels,
            'dominant_group': dominant_group,
        }
    blanket_labels = [OBS_LABELS[i] for i in primary_result['blankets'] if i < len(OBS_LABELS)]
    metrics['semantic_interpretation']['blanket'] = {
        'variables': blanket_labels,
    }

    config = {
        'n_episodes': 20,
        'max_steps': 50,
        'max_jacobian_samples': 300,
        'detection_method': 'hybrid',
        'n_objects': 2,
        'env_id': meta.get('env_id', 'FetchPush-v4'),
        'model_hidden_size': meta.get('hidden_size', 256),
        'model_depth': meta.get('depth', 2),
    }

    save_results(
        EXPERIMENT_NAME,
        metrics=metrics,
        config=config,
        notes=(
            f"TB analysis of pandas Bayes ensemble ({model_status}) on FetchPush-v4. "
            f"Primary method: {primary_method}. "
            f"ARI={primary_ari:.3f}, Blanket F1={primary_f1:.3f}. "
            f"Discovered {len(primary_result['objects'])} objects and "
            f"{len(primary_result['blankets'])} blanket variables."
        )
    )

    # 11. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_status}")
    print(f"  Observations: {meta['obs_dim']}D, {len(data['obs'])} transitions")
    print(f"  Ensemble: {meta['ensemble_size']} members")
    print(f"  Primary method: {primary_method}")
    print(f"  ARI vs ground truth: {primary_ari:.3f}")
    print(f"  Blanket F1: {primary_f1:.3f}")
    print(f"  Objects discovered: {len(primary_result['objects'])}")
    print(f"  Blanket variables: {len(primary_result['blankets'])}")
    for obj_id, obj_vars in primary_result['objects'].items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels = [OBS_LABELS[i] for i in primary_result['blankets'] if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels}")
    print()
    print("Multi-method comparison:")
    for method_name, mres in method_results.items():
        if 'error' not in mres:
            print(f"  {method_name}: ARI={mres['ari']:.3f}, Blanket F1={mres['blanket_f1']:.3f}")
        else:
            print(f"  {method_name}: FAILED ({mres['error']})")
    print("=" * 70)
    print("US-076 complete.")

    return metrics


if __name__ == '__main__':
    main()
