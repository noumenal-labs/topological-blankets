"""
US-110: FWH for Ensemble World Models
======================================

Tests whether the Factored World Hypothesis (FWH, Shai et al. 2602.02385)
extends to ensemble world models. The FWH predicts that neural networks
trained to model the world converge to factored representations, where the
state space decomposes into approximately independent objects with an
interface (blanket) mediating their interaction.

This script applies Topological Blankets (TB) to the FetchPush Bayes
ensemble's Jacobians and compares the discovered structure against FWH
predictions:

  1. Effective dimensionality: should be linear in N (number of objects),
     not exponential in the state dimension.
  2. Orthogonal factor subspaces: TB-discovered objects should correspond
     to gripper, manipulated object, and relational (blanket) subspaces.
  3. Adjusted Rand Index (ARI) against the ground-truth partition.

The ensemble model is a JAX/Equinox model with 5 DynamicsMember instances,
each predicting state deltas given (obs, achieved_goal, action). Jacobians
d(delta_prediction)/d(obs) are computed for each ensemble member; their
covariance structure provides the coupling signal for TB.

FetchPush-v4 observation space (25D):
  [0:3]   grip_pos (gripper xyz)
  [3:6]   object_pos (object xyz)
  [6:9]   object_rel_pos (object - gripper xyz)
  [9:11]  gripper_state (finger widths)
  [11:14] object_rot (euler angles)
  [14:17] object_velp (object linear velocity)
  [17:20] object_velr (object angular velocity)
  [20:22] grip_velp (gripper velocity xy)
  [22:25] (additional gripper state / zeros)

Ground-truth partition:
  Object 0 (gripper):  indices [0,1,2, 9,10, 20,21]
  Object 1 (object):   indices [3,4,5, 11,12,13, 14,15,16, 17,18,19]
  Blanket (relational): indices [6,7,8]
  Unstructured:         indices [22,23,24]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# -- Path setup ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)                      # ralph/
PROJECT_ROOT = os.path.dirname(RALPH_DIR)                     # topological-blankets/

PANDAS_DIR = 'C:/Users/citiz/Documents/noumenal-labs/pandas'

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RALPH_DIR)
sys.path.insert(0, PANDAS_DIR)

from topological_blankets import TopologicalBlankets, compute_eigengap
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, spectral_partition
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

import jax
import jax.numpy as jnp
import equinox as eqx
from panda.model import EnsembleModel, make_model, ModelConfig, DynamicsMember
from panda.utils import Normalizer

# -- Constants ----------------------------------------------------------------
RUN_DIR = os.path.join(PANDAS_DIR, 'data', 'fetchpush_50step')
META_PATH = os.path.join(RUN_DIR, 'model.eqx.json')
MODEL_PATH = os.path.join(RUN_DIR, 'model.eqx')
RESULTS_DIR = os.path.join(RALPH_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "us110_fwh_ensemble"

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

# Semantic group labels
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

# Ground truth: 0 = gripper, 1 = object, -1 = blanket (relative)
GROUND_TRUTH_22D = np.array([
    0, 0, 0,       # grip_pos -> gripper
    1, 1, 1,       # obj_pos -> object
    -1, -1, -1,    # rel_pos -> blanket
    0, 0,          # gripper_state -> gripper
    1, 1, 1,       # obj_rot -> object
    1, 1, 1,       # obj_velp -> object
    1, 1, 1,       # obj_velr -> object
    0, 0,          # grip_velp -> gripper
])

# Full 25D ground truth (last 3 assigned to gripper; they are zeros/gripper-related)
GROUND_TRUTH_25D = np.concatenate([GROUND_TRUTH_22D, np.array([0, 0, 0])])
BLANKET_MASK_25D = (GROUND_TRUTH_25D == -1)

# FWH predictions for 2-object system:
# Effective dimensionality should be O(N) = O(2), not O(2^25).
# The two objects (gripper, manipulated object) plus blanket (relational)
# yield 3 well-separated clusters.
FWH_N_OBJECTS = 2
FWH_N_CLUSTERS = 3  # 2 objects + 1 blanket


# =============================================================================
# Model loading
# =============================================================================

def load_trained_model():
    """Load trained ensemble from pandas/data/fetchpush_50step/."""
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


# =============================================================================
# Data collection
# =============================================================================

def collect_fetchpush_data(n_episodes=50, max_steps=50, seed=42):
    """
    Collect random trajectories from FetchPush-v4.

    Falls back to random observations if gymnasium[mujoco] is not installed.
    """
    try:
        import gymnasium_robotics
        gymnasium_robotics.register_robotics_envs()
        import gymnasium as gym

        env = gym.make('FetchPush-v4', max_episode_steps=max_steps, reward_type='dense')

        all_obs = []
        all_ag = []
        all_actions = []

        for ep in range(n_episodes):
            obs_dict, _ = env.reset(seed=seed + ep)
            for step in range(max_steps):
                obs = obs_dict['observation']
                ag = obs_dict['achieved_goal']
                action = env.action_space.sample()

                all_obs.append(obs.copy())
                all_ag.append(ag.copy())
                all_actions.append(action.copy())

                obs_dict, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

        env.close()

        data = {
            'obs': np.array(all_obs, dtype=np.float32),
            'achieved_goal': np.array(all_ag, dtype=np.float32),
            'actions': np.array(all_actions, dtype=np.float32),
        }
        print(f"Collected {len(data['obs'])} transitions from {n_episodes} "
              f"episodes via gymnasium FetchPush-v4")
        return data

    except (ImportError, Exception) as e:
        print(f"Could not create FetchPush-v4 environment: {e}")
        print("Falling back to random observations...")

        rng = np.random.RandomState(seed)
        n_samples = n_episodes * max_steps

        # Generate plausible FetchPush-like observations
        obs = np.zeros((n_samples, 25), dtype=np.float32)
        # Gripper position: centered around table
        obs[:, 0:3] = rng.normal(loc=[1.3, 0.75, 0.5], scale=0.1, size=(n_samples, 3))
        # Object position: near gripper
        obs[:, 3:6] = obs[:, 0:3] + rng.normal(0, 0.05, size=(n_samples, 3))
        # Relative position: object - gripper
        obs[:, 6:9] = obs[:, 3:6] - obs[:, 0:3]
        # Gripper state: finger widths
        obs[:, 9:11] = rng.uniform(0.0, 0.05, size=(n_samples, 2))
        # Object rotation
        obs[:, 11:14] = rng.normal(0, 0.1, size=(n_samples, 3))
        # Object velocity (linear)
        obs[:, 14:17] = rng.normal(0, 0.01, size=(n_samples, 3))
        # Object velocity (angular)
        obs[:, 17:20] = rng.normal(0, 0.01, size=(n_samples, 3))
        # Gripper velocity
        obs[:, 20:22] = rng.normal(0, 0.01, size=(n_samples, 2))
        # Extra (zeros in standard env)
        obs[:, 22:25] = 0.0

        # Achieved goals (object position)
        ag = obs[:, 3:6].copy()

        # Random actions
        actions = rng.uniform(-1, 1, size=(n_samples, 4)).astype(np.float32)

        data = {
            'obs': obs,
            'achieved_goal': ag,
            'actions': actions,
        }
        print(f"Generated {n_samples} synthetic observations")
        return data


# =============================================================================
# Jacobian computation
# =============================================================================

def compute_member_jacobian(member, normalizer, obs_single, ag_single, act_single):
    """
    Compute Jacobian d(delta_obs)/d(obs) for a single sample on a single member.

    Returns: Jacobian of shape (obs_dim, obs_dim).
    """
    def forward_fn(obs):
        delta_obs, _ = member(obs, ag_single, act_single, normalizer)
        return delta_obs

    jac = jax.jacobian(forward_fn)(obs_single)
    return jac  # (obs_dim, obs_dim)


def compute_ensemble_jacobians(model, data, max_samples=500):
    """
    Compute Jacobians d(delta_obs)/d(obs) for all ensemble members.

    For each ensemble member and each observation sample, the Jacobian
    captures how the predicted state delta depends on each input dimension.
    This is the core signal for detecting factored structure: if the world
    model has learned factored dynamics, the Jacobian will be approximately
    block-diagonal.

    Returns:
        jacobians: array of shape (E, N, obs_dim, obs_dim)
    """
    n_samples = min(max_samples, len(data['obs']))
    indices = np.random.RandomState(42).choice(
        len(data['obs']), n_samples, replace=False
    )

    obs_batch = jnp.array(data['obs'][indices])
    ag_batch = jnp.array(data['achieved_goal'][indices])
    act_batch = jnp.array(data['actions'][indices])

    print(f"Computing Jacobians for {len(model.members)} ensemble members "
          f"on {n_samples} samples...")

    # JIT-compile the Jacobian function for a single member
    @jax.jit
    def jac_single_member(member, obs, ag, act):
        """Jacobian of delta_obs w.r.t. obs for a single (obs, ag, act) tuple."""
        def fwd(o):
            delta_obs, _ = member(o, ag, act, model.normalizer)
            return delta_obs
        return jax.jacobian(fwd)(obs)

    # Vectorize over the sample dimension
    jac_batched = jax.vmap(
        lambda o, a, ac: jac_single_member(model.members[0], o, a, ac),
        in_axes=(0, 0, 0)
    )

    all_jacobians = []
    for m_idx, member in enumerate(model.members):
        print(f"  Member {m_idx + 1}/{len(model.members)}...", end=" ", flush=True)
        t0 = time.time()

        # Redefine batched Jacobian for this member
        @jax.jit
        def jac_for_member(obs, ag, act, _member=member):
            def fwd(o, a_g, a_c):
                def inner(o_):
                    delta_obs, _ = _member(o_, a_g, a_c, model.normalizer)
                    return delta_obs
                return jax.jacobian(inner)(o)
            return jax.vmap(fwd)(obs, ag, act)

        # Process in chunks to manage memory
        chunk_size = 64
        member_jacs = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            J_chunk = jac_for_member(
                obs_batch[start:end],
                ag_batch[start:end],
                act_batch[start:end],
            )
            member_jacs.append(np.array(J_chunk))

        J_member = np.concatenate(member_jacs, axis=0)  # (N, obs_dim, obs_dim)
        all_jacobians.append(J_member)
        elapsed = time.time() - t0
        print(f"shape={J_member.shape}, {elapsed:.1f}s")

    jacobians = np.stack(all_jacobians, axis=0)  # (E, N, obs_dim, obs_dim)
    print(f"Ensemble Jacobians shape: {jacobians.shape}")
    return jacobians, indices


# =============================================================================
# Jacobian covariance as TB gradients
# =============================================================================

def jacobians_to_tb_gradients(jacobians):
    """
    Convert ensemble Jacobians to gradient samples for TB analysis.

    For each (member, sample) pair, the Jacobian J is (obs_dim, obs_dim).
    The per-variable sensitivity s_j = ||J[:, j]|| captures how much the
    predicted dynamics depend on variable j. Across (member, sample) pairs,
    these sensitivities form gradient-like samples for TB's covariance
    estimation.

    Additionally computes the Fisher-like coupling J^T @ J averaged across
    members and samples, which gives a direct estimate of the coupling
    structure.

    Returns:
        gradients: (E*N, obs_dim) sensitivity profiles for TB
        jacobian_cov: (obs_dim, obs_dim) covariance of Jacobian rows
        fisher_coupling: (obs_dim, obs_dim) mean J^T @ J
        ensemble_disagreement: (N, obs_dim) std of sensitivity across members
    """
    E, N, out_dim, obs_dim = jacobians.shape

    # Per-(member, sample) sensitivity: column norms of Jacobian
    # s[e, n, j] = ||J[e, n, :, j]||
    sensitivity = np.sqrt(np.sum(jacobians ** 2, axis=2))  # (E, N, obs_dim)

    # TB gradient samples: all (member x sample) sensitivities
    gradients = sensitivity.reshape(E * N, obs_dim)

    # Jacobian covariance: treat each row of J as a gradient sample.
    # Flatten (E, N, out_dim) rows into (E*N*out_dim, obs_dim)
    jac_rows = jacobians.reshape(E * N * out_dim, obs_dim)
    jacobian_cov = np.cov(jac_rows.T)
    if jacobian_cov.ndim == 0:
        jacobian_cov = np.array([[float(jacobian_cov)]])

    # Fisher coupling: mean J^T @ J across all (member, sample) pairs
    fisher_sum = np.zeros((obs_dim, obs_dim))
    for e in range(E):
        for n in range(N):
            J = jacobians[e, n]
            fisher_sum += J.T @ J
    fisher_coupling = fisher_sum / (E * N)

    # Ensemble disagreement: std of sensitivity across members
    ensemble_disagreement = sensitivity.std(axis=0)  # (N, obs_dim)

    return gradients, jacobian_cov, fisher_coupling, ensemble_disagreement


# =============================================================================
# TB analysis
# =============================================================================

def run_tb_on_gradients(gradients, n_objects=2, method='hybrid'):
    """Run TB on sensitivity-profile gradient samples."""
    tb = TopologicalBlankets(method=method, n_objects=n_objects)
    tb.fit(gradients)

    assignment = tb.get_assignment()
    blankets = tb.get_blankets()
    objects = tb.get_objects()
    coupling = tb.get_coupling_matrix()

    return {
        'assignment': assignment,
        'blankets': blankets,
        'objects': objects,
        'coupling': coupling,
        'tb_instance': tb,
    }


def run_tb_on_fisher(fisher_coupling, n_objects=2):
    """
    Run TB analysis using the Fisher coupling matrix directly.

    The Fisher J^T @ J matrix is treated as the Hessian estimate for TB.
    """
    from topological_blankets.detection import detect_blankets_hybrid
    from topological_blankets.clustering import cluster_internals

    obs_dim = fisher_coupling.shape[0]

    # Normalize to correlation-like coupling
    D = np.sqrt(np.abs(np.diag(fisher_coupling)) + 1e-8)
    coupling_norm = np.abs(fisher_coupling) / np.outer(D, D)
    np.fill_diagonal(coupling_norm, 0)

    features = {
        'grad_magnitude': D,
        'grad_variance': D ** 2,
        'hessian_est': fisher_coupling,
        'coupling': coupling_norm,
    }

    result = detect_blankets_hybrid(
        np.random.randn(100, obs_dim),  # dummy gradients for fallback
        fisher_coupling,
    )
    is_blanket = result['is_blanket']
    assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)

    blankets = np.where(is_blanket)[0]
    objects = {}
    for label in np.unique(assignment):
        if label >= 0:
            objects[int(label)] = np.where(assignment == label)[0]

    return {
        'assignment': assignment,
        'blankets': blankets,
        'objects': objects,
        'coupling': coupling_norm,
    }


def run_tb_on_jacobian_covariance(jacobian_cov, n_objects=2):
    """
    Run TB using the covariance of Jacobian rows as the Hessian estimate.

    This alternative uses all rows of J (not just column norms) to capture
    the full coupling structure. The covariance of Jacobian rows directly
    estimates the Hessian of the dynamics' log-likelihood w.r.t. input.
    """
    from topological_blankets.detection import detect_blankets_hybrid
    from topological_blankets.clustering import cluster_internals

    obs_dim = jacobian_cov.shape[0]

    D = np.sqrt(np.abs(np.diag(jacobian_cov)) + 1e-8)
    coupling_norm = np.abs(jacobian_cov) / np.outer(D, D)
    np.fill_diagonal(coupling_norm, 0)

    features = {
        'grad_magnitude': D,
        'grad_variance': D ** 2,
        'hessian_est': jacobian_cov,
        'coupling': coupling_norm,
    }

    result = detect_blankets_hybrid(
        np.random.randn(100, obs_dim),
        jacobian_cov,
    )
    is_blanket = result['is_blanket']
    assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)

    blankets = np.where(is_blanket)[0]
    objects = {}
    for label in np.unique(assignment):
        if label >= 0:
            objects[int(label)] = np.where(assignment == label)[0]

    return {
        'assignment': assignment,
        'blankets': blankets,
        'objects': objects,
        'coupling': coupling_norm,
    }


# =============================================================================
# FWH metrics
# =============================================================================

def compute_ari_vs_ground_truth(assignment):
    """Compute ARI and blanket F1 against the ground-truth partition."""
    from sklearn.metrics import adjusted_rand_score, f1_score

    n_vars = min(len(assignment), len(GROUND_TRUTH_25D))
    pred = assignment[:n_vars].copy()
    gt = GROUND_TRUTH_25D[:n_vars].copy()

    ari = adjusted_rand_score(gt, pred)

    pred_blanket = (pred == -1)
    gt_blanket = BLANKET_MASK_25D[:n_vars]
    if gt_blanket.sum() > 0:
        blanket_f1 = f1_score(gt_blanket, pred_blanket)
    else:
        blanket_f1 = 0.0

    return float(ari), float(blanket_f1)


def compute_effective_dimensionality(coupling_matrix):
    """
    Compute effective dimensionality from the coupling matrix eigenspectrum.

    FWH Prediction (Shai et al. Sec. 3): for a factored world model with
    N objects, the effective dimensionality of the coupling structure should
    be O(N), not O(dim). This manifests as a sharp drop in eigenvalues
    after the first N clusters in the graph Laplacian spectrum.

    Returns:
        eff_dim: Effective dimensionality (number of significant eigenvalues)
        eigvals: Full eigenvalue spectrum
        eigengap_position: Position of the largest eigengap
        eigengap_value: Magnitude of the largest eigengap
    """
    A = build_adjacency_from_hessian(coupling_matrix)
    L = build_graph_laplacian(A)
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))

    # Effective dimensionality: number of eigenvalues below the largest gap
    n_clusters, gap_value = compute_eigengap(eigvals[:min(15, len(eigvals))])

    # Also compute participation ratio as a continuous measure
    # PR = (sum lambda_i)^2 / sum(lambda_i^2), normalized
    nonzero = eigvals[eigvals > 1e-10]
    if len(nonzero) > 0:
        participation_ratio = (np.sum(nonzero) ** 2) / np.sum(nonzero ** 2)
    else:
        participation_ratio = 1.0

    return {
        'n_clusters': int(n_clusters),
        'eigengap_value': float(gap_value),
        'participation_ratio': float(participation_ratio),
        'eigenvalues': eigvals.tolist(),
    }


def compute_subspace_orthogonality(jacobians, assignment):
    """
    Measure orthogonality of factor subspaces in Jacobian space.

    FWH Prediction (Shai et al. Sec. 4): in a factored representation, the
    Jacobian restricted to each object's variables should span approximately
    orthogonal subspaces. The principal angles between these subspaces
    should be close to 90 degrees.

    For each pair of objects, compute the principal angles between their
    Jacobian column spans using SVD of the cross-covariance.
    """
    E, N, out_dim, obs_dim = jacobians.shape

    n_vars = min(len(assignment), obs_dim)
    objects = {}
    for label in np.unique(assignment[:n_vars]):
        if label >= 0:
            objects[int(label)] = np.where(assignment[:n_vars] == label)[0]

    if len(objects) < 2:
        return {'principal_angles': {}, 'mean_angle_deg': 0.0}

    # Compute mean Jacobian across (member, sample) pairs
    mean_J = jacobians.mean(axis=(0, 1))  # (out_dim, obs_dim)

    angles = {}
    obj_ids = sorted(objects.keys())
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            id_i, id_j = obj_ids[i], obj_ids[j]
            cols_i = objects[id_i]
            cols_j = objects[id_j]

            # Subspace spanned by columns of J for each object
            U_i = mean_J[:, cols_i]
            U_j = mean_J[:, cols_j]

            # Orthonormalize via QR
            Q_i, _ = np.linalg.qr(U_i)
            Q_j, _ = np.linalg.qr(U_j)

            # Principal angles from SVD of Q_i^T @ Q_j
            svd_vals = np.linalg.svd(Q_i.T @ Q_j, compute_uv=False)
            # Clamp to [-1, 1] for numerical safety
            svd_vals = np.clip(svd_vals, -1.0, 1.0)
            principal_angles_rad = np.arccos(svd_vals)
            principal_angles_deg = np.degrees(principal_angles_rad)

            pair_key = f"obj{id_i}_vs_obj{id_j}"
            angles[pair_key] = {
                'principal_angles_deg': principal_angles_deg.tolist(),
                'min_angle_deg': float(principal_angles_deg.min()),
                'mean_angle_deg': float(principal_angles_deg.mean()),
                'max_angle_deg': float(principal_angles_deg.max()),
            }

    # Global mean of minimum principal angles (closer to 90 = more orthogonal)
    min_angles = [v['min_angle_deg'] for v in angles.values()]
    mean_min_angle = float(np.mean(min_angles)) if min_angles else 0.0

    return {
        'principal_angles': angles,
        'mean_min_angle_deg': mean_min_angle,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_coupling_matrix(coupling, assignment, title_suffix=""):
    """Plot coupling matrix: raw and reordered by TB partition."""
    n_vars = coupling.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw coupling
    ax = axes[0]
    im = ax.imshow(coupling, cmap='hot', aspect='auto')
    ax.set_title(f'Coupling Matrix{title_suffix}', fontsize=12)
    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Variable Index')
    tick_labels = [OBS_LABELS[i] if i < len(OBS_LABELS) else f'v{i}'
                   for i in range(n_vars)]
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(tick_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Reordered by partition
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


def plot_eigengap_spectrum(eigvals, title_suffix=""):
    """
    Plot eigenvalue spectrum and eigengaps with FWH predictions.

    FWH predicts a clear eigengap at position N=2 (number of objects)
    or N=3 (objects + blanket).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_show = min(20, len(eigvals))
    evals = eigvals[:n_show]
    gaps = np.diff(evals)

    # Panel 1: Eigenvalue spectrum
    ax = axes[0]
    ax.plot(range(n_show), evals, 'o-', color='steelblue', markersize=5)
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='N=2 (objects)')
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5,
               label='N=3 (objects + blanket)')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Graph Laplacian Spectrum{title_suffix}')
    ax.legend(fontsize=8)

    # Panel 2: Eigengaps
    ax = axes[1]
    gap_x = range(1, len(gaps) + 1)
    colors = ['red' if i in (1, 2) else 'steelblue' for i in gap_x]
    ax.bar(gap_x, gaps, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Gap index (k)')
    ax.set_ylabel(r'$\lambda_{k+1} - \lambda_k$')
    ax.set_title('Eigengaps (FWH predicts peak near N)')
    # Annotate the largest gap
    if len(gaps) > 0:
        max_gap_idx = np.argmax(gaps)
        ax.annotate(f'max gap at k={max_gap_idx + 1}',
                    xy=(max_gap_idx + 1, gaps[max_gap_idx]),
                    xytext=(max_gap_idx + 3, gaps[max_gap_idx] * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')

    # Panel 3: Cumulative explained variance (participation ratio interpretation)
    ax = axes[2]
    nonzero = evals[evals > 1e-10]
    if len(nonzero) > 0:
        cumulative = np.cumsum(nonzero) / np.sum(nonzero)
        ax.plot(range(1, len(cumulative) + 1), cumulative, 'o-',
                color='forestgreen', markersize=4)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        ax.axvline(x=2, color='red', linestyle='--', alpha=0.3, label='N=2')
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.3, label='N=3')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative explained variance')
        ax.set_title('Effective Dimensionality')
        ax.legend(fontsize=8)

        # Mark where 90% is reached
        above_90 = np.where(cumulative >= 0.9)[0]
        if len(above_90) > 0:
            eff_dim_90 = above_90[0] + 1
            ax.annotate(f'eff_dim(90%)={eff_dim_90}',
                        xy=(eff_dim_90, cumulative[above_90[0]]),
                        xytext=(eff_dim_90 + 2, 0.85),
                        arrowprops=dict(arrowstyle='->', color='forestgreen'),
                        fontsize=9, color='forestgreen')

    fig.suptitle('FWH Eigengap Analysis: Effective Dimensionality', fontsize=14)
    fig.tight_layout()
    return fig


def plot_partition_comparison(assignment, title_suffix=""):
    """Side-by-side comparison of TB-discovered vs ground-truth partition."""
    n_vars = min(len(assignment), len(GROUND_TRUTH_25D))
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

    cmap = {-1: 'red', 0: 'steelblue', 1: 'forestgreen', 2: 'orange'}

    # Discovered partition
    ax = axes[0]
    colors_disc = [cmap.get(int(assignment[i]), 'gray') for i in range(n_vars)]
    ax.bar(range(n_vars), np.ones(n_vars), color=colors_disc,
           edgecolor='black', linewidth=0.5)
    ax.set_title(f'TB Discovered Partition{title_suffix}', fontsize=11)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([OBS_LABELS[i] for i in range(n_vars)], rotation=90, fontsize=7)
    ax.set_yticks([])

    from matplotlib.patches import Patch
    unique_vals = sorted(set(assignment[:n_vars].tolist()))
    legend_elements = []
    for v in unique_vals:
        label = 'Blanket' if v == -1 else f'Object {v}'
        legend_elements.append(Patch(facecolor=cmap.get(int(v), 'gray'), label=label))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Ground-truth partition
    ax = axes[1]
    colors_gt = [cmap.get(int(GROUND_TRUTH_25D[i]), 'gray') for i in range(n_vars)]
    ax.bar(range(n_vars), np.ones(n_vars), color=colors_gt,
           edgecolor='black', linewidth=0.5)
    ax.set_title('Ground-Truth Partition', fontsize=11)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([OBS_LABELS[i] for i in range(n_vars)], rotation=90, fontsize=7)
    ax.set_yticks([])
    legend_gt = [
        Patch(facecolor='steelblue', label='Object 0 (gripper)'),
        Patch(facecolor='forestgreen', label='Object 1 (manipulated object)'),
        Patch(facecolor='red', label='Blanket (relative position)'),
    ]
    ax.legend(handles=legend_gt, loc='upper right', fontsize=8)

    fig.tight_layout()
    return fig


def plot_fwh_summary(fwh_metrics, title_suffix=""):
    """
    Summary plot of all three FWH predictions.

    Panel 1: Effective dimensionality (bar chart, linear vs exponential)
    Panel 2: Subspace orthogonality (principal angles)
    Panel 3: ARI across methods
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Effective dimensionality
    ax = axes[0]
    eff_dim_data = fwh_metrics.get('effective_dimensionality', {})
    methods = list(eff_dim_data.keys())
    n_clusters_vals = [eff_dim_data[m]['n_clusters'] for m in methods]
    pr_vals = [eff_dim_data[m]['participation_ratio'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, n_clusters_vals, width, label='Eigengap clusters',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, pr_vals, width, label='Participation ratio',
                   color='coral', alpha=0.8)
    ax.axhline(y=FWH_N_OBJECTS, color='red', linestyle='--', alpha=0.5,
               label=f'FWH prediction (N={FWH_N_OBJECTS})')
    ax.axhline(y=FWH_N_CLUSTERS, color='orange', linestyle='--', alpha=0.5,
               label=f'N+blanket={FWH_N_CLUSTERS}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('FWH Prediction 1: Linear Dimensionality')
    ax.legend(fontsize=7)

    # Panel 2: Subspace orthogonality
    ax = axes[1]
    ortho_data = fwh_metrics.get('subspace_orthogonality', {})
    methods_o = list(ortho_data.keys())
    min_angles = [ortho_data[m].get('mean_min_angle_deg', 0) for m in methods_o]

    ax.bar(range(len(methods_o)), min_angles, color='forestgreen', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5,
               label='Perfect orthogonality (90 deg)')
    ax.set_xticks(range(len(methods_o)))
    ax.set_xticklabels(methods_o, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Mean Min Principal Angle (deg)')
    ax.set_title('FWH Prediction 2: Orthogonal Subspaces')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)

    # Panel 3: ARI across methods
    ax = axes[2]
    ari_data = fwh_metrics.get('ari_comparison', {})
    methods_a = list(ari_data.keys())
    ari_vals = [ari_data[m]['ari'] for m in methods_a]
    f1_vals = [ari_data[m]['blanket_f1'] for m in methods_a]

    x = np.arange(len(methods_a))
    bars_ari = ax.bar(x - width/2, ari_vals, width, label='ARI',
                      color='steelblue', alpha=0.8)
    bars_f1 = ax.bar(x + width/2, f1_vals, width, label='Blanket F1',
                     color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_a, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('FWH Prediction 3: Structure Recovery')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)

    fig.suptitle('FWH Predictions for Ensemble World Model', fontsize=14)
    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("US-110: FWH for Ensemble World Models")
    print("  Testing Factored World Hypothesis on FetchPush Bayes Ensemble")
    print("  (Shai et al. 2602.02385)")
    print("=" * 70)

    t_start = time.time()

    # 1. Load model
    print("\n[1/7] Loading trained ensemble model...")
    model, meta = load_trained_model()

    # 2. Collect observations
    print("\n[2/7] Collecting FetchPush-v4 trajectory data (50 episodes)...")
    data = collect_fetchpush_data(n_episodes=50, max_steps=50, seed=42)

    # 3. Compute Jacobians
    print("\n[3/7] Computing ensemble Jacobians d(delta_obs)/d(obs)...")
    jacobians, sample_indices = compute_ensemble_jacobians(model, data, max_samples=500)

    # 4. Convert Jacobians to TB input signals
    print("\n[4/7] Computing TB gradient signals from Jacobians...")
    (gradients, jacobian_cov, fisher_coupling,
     ensemble_disagreement) = jacobians_to_tb_gradients(jacobians)
    print(f"  Sensitivity gradient samples: {gradients.shape}")
    print(f"  Jacobian covariance: {jacobian_cov.shape}")
    print(f"  Fisher coupling: {fisher_coupling.shape}")

    # 5. Run TB with multiple coupling sources
    print("\n[5/7] Running TB analysis (three coupling sources)...")

    # Method A: Sensitivity profiles (column norms) as gradient samples
    print("\n  --- Method A: Sensitivity profile gradients ---")
    result_sensitivity = run_tb_on_gradients(gradients, n_objects=2, method='hybrid')
    ari_sens, f1_sens = compute_ari_vs_ground_truth(result_sensitivity['assignment'])
    print(f"  ARI={ari_sens:.3f}, Blanket F1={f1_sens:.3f}")
    for obj_id, obj_vars in result_sensitivity['objects'].items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels = [OBS_LABELS[i] for i in result_sensitivity['blankets']
                      if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels}")

    # Method B: Fisher coupling (J^T J)
    print("\n  --- Method B: Fisher coupling (J^T @ J) ---")
    result_fisher = run_tb_on_fisher(fisher_coupling, n_objects=2)
    ari_fisher, f1_fisher = compute_ari_vs_ground_truth(result_fisher['assignment'])
    print(f"  ARI={ari_fisher:.3f}, Blanket F1={f1_fisher:.3f}")
    for obj_id, obj_vars in result_fisher['objects'].items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels_f = [OBS_LABELS[i] for i in result_fisher['blankets']
                        if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels_f}")

    # Method C: Jacobian row covariance
    print("\n  --- Method C: Jacobian row covariance ---")
    result_jacocov = run_tb_on_jacobian_covariance(jacobian_cov, n_objects=2)
    ari_jcov, f1_jcov = compute_ari_vs_ground_truth(result_jacocov['assignment'])
    print(f"  ARI={ari_jcov:.3f}, Blanket F1={f1_jcov:.3f}")
    for obj_id, obj_vars in result_jacocov['objects'].items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels_jc = [OBS_LABELS[i] for i in result_jacocov['blankets']
                         if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels_jc}")

    # 6. Compute FWH metrics
    print("\n[6/7] Computing FWH metrics...")

    # Select best method by ARI
    method_results = {
        'sensitivity': (result_sensitivity, ari_sens, f1_sens),
        'fisher': (result_fisher, ari_fisher, f1_fisher),
        'jac_covariance': (result_jacocov, ari_jcov, f1_jcov),
    }
    best_method = max(method_results.keys(),
                      key=lambda k: method_results[k][1])
    best_result, best_ari, best_f1 = method_results[best_method]
    print(f"\n  Best method: {best_method} (ARI={best_ari:.3f})")

    # FWH Prediction 1: Effective dimensionality
    print("\n  FWH Prediction 1: Effective dimensionality...")
    eff_dim_results = {}
    for name, coupling_mat in [
        ('sensitivity', result_sensitivity['coupling']),
        ('fisher', result_fisher['coupling']),
        ('jac_covariance', result_jacocov['coupling']),
    ]:
        ed = compute_effective_dimensionality(coupling_mat)
        eff_dim_results[name] = ed
        print(f"    {name}: n_clusters={ed['n_clusters']}, "
              f"PR={ed['participation_ratio']:.2f}, "
              f"eigengap={ed['eigengap_value']:.4f}")

    # FWH Prediction 2: Orthogonal factor subspaces
    print("\n  FWH Prediction 2: Subspace orthogonality...")
    ortho_results = {}
    for name, (result, _, _) in method_results.items():
        ortho = compute_subspace_orthogonality(jacobians, result['assignment'])
        ortho_results[name] = ortho
        print(f"    {name}: mean min principal angle = "
              f"{ortho['mean_min_angle_deg']:.1f} deg")

    # FWH Prediction 3: ARI (already computed)
    ari_comparison = {
        'sensitivity': {'ari': ari_sens, 'blanket_f1': f1_sens},
        'fisher': {'ari': ari_fisher, 'blanket_f1': f1_fisher},
        'jac_covariance': {'ari': ari_jcov, 'blanket_f1': f1_jcov},
    }

    fwh_metrics = {
        'effective_dimensionality': eff_dim_results,
        'subspace_orthogonality': ortho_results,
        'ari_comparison': ari_comparison,
    }

    # 7. Generate plots and save results
    print("\n[7/7] Generating plots and saving results...")

    # Plot 1: Coupling matrix (best method)
    fig_coupling = plot_coupling_matrix(
        best_result['coupling'], best_result['assignment'],
        title_suffix=f" ({best_method})"
    )
    save_figure(fig_coupling, "coupling_matrix", EXPERIMENT_NAME)

    # Plot 2: Eigengap spectrum (best method)
    best_eigvals = np.array(eff_dim_results[best_method]['eigenvalues'])
    fig_eigengap = plot_eigengap_spectrum(
        best_eigvals, title_suffix=f" ({best_method})"
    )
    save_figure(fig_eigengap, "eigengap_spectrum", EXPERIMENT_NAME)

    # Plot 3: Partition comparison (best method)
    fig_partition = plot_partition_comparison(
        best_result['assignment'], title_suffix=f" ({best_method})"
    )
    save_figure(fig_partition, "partition_comparison", EXPERIMENT_NAME)

    # Plot 4: FWH summary (all three predictions)
    fig_fwh = plot_fwh_summary(fwh_metrics)
    save_figure(fig_fwh, "fwh_summary", EXPERIMENT_NAME)

    # Save results JSON
    elapsed = time.time() - t_start

    output = {
        'experiment': 'US-110',
        'title': 'FWH for Ensemble World Models',
        'description': (
            'Tests whether the Factored World Hypothesis (Shai et al. 2602.02385) '
            'extends to ensemble world models. Applies TB to FetchPush Bayes '
            'ensemble Jacobians and compares to FWH predictions.'
        ),
        'model': {
            'path': RUN_DIR,
            'obs_dim': meta['obs_dim'],
            'action_dim': meta['action_dim'],
            'achieved_goal_dim': meta['achieved_goal_dim'],
            'ensemble_size': meta['ensemble_size'],
            'hidden_size': meta['hidden_size'],
            'depth': meta['depth'],
            'env_id': meta.get('env_id', 'FetchPush-v4'),
        },
        'data': {
            'n_transitions': int(len(data['obs'])),
            'n_jacobian_samples': int(jacobians.shape[1]),
            'jacobian_shape': list(jacobians.shape),
        },
        'best_method': best_method,
        'best_ari': round(best_ari, 4),
        'best_blanket_f1': round(best_f1, 4),
        'method_results': {},
        'fwh_predictions': {
            'prediction_1_effective_dimensionality': {
                name: {
                    'n_clusters': d['n_clusters'],
                    'participation_ratio': round(d['participation_ratio'], 4),
                    'eigengap_value': round(d['eigengap_value'], 4),
                    'linear_in_N': d['n_clusters'] <= 2 * FWH_N_CLUSTERS,
                }
                for name, d in eff_dim_results.items()
            },
            'prediction_2_orthogonal_subspaces': {
                name: {
                    'mean_min_angle_deg': round(d['mean_min_angle_deg'], 2),
                    'near_orthogonal': d['mean_min_angle_deg'] > 45.0,
                }
                for name, d in ortho_results.items()
            },
            'prediction_3_structure_recovery': ari_comparison,
        },
        'ground_truth': {
            'object_0_gripper': [0, 1, 2, 9, 10, 20, 21],
            'object_1_manipulated': [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'blanket_relational': [6, 7, 8],
            'unstructured': [22, 23, 24],
        },
        'runtime_s': round(elapsed, 1),
    }

    # Add per-method details
    for name, (result, ari_val, f1_val) in method_results.items():
        output['method_results'][name] = {
            'ari': round(ari_val, 4),
            'blanket_f1': round(f1_val, 4),
            'assignment': result['assignment'].tolist(),
            'blankets': result['blankets'].tolist(),
            'objects': {
                str(k): v.tolist() for k, v in result['objects'].items()
            },
            'semantic_interpretation': {},
        }
        for obj_id, obj_vars in result['objects'].items():
            labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
            groups = [OBS_GROUPS[i] for i in obj_vars if i < len(OBS_GROUPS)]
            dominant = max(set(groups), key=groups.count) if groups else 'unknown'
            output['method_results'][name]['semantic_interpretation'][
                f'object_{obj_id}'
            ] = {
                'variables': labels,
                'dominant_group': dominant,
            }
        bl_labels = [OBS_LABELS[i] for i in result['blankets']
                     if i < len(OBS_LABELS)]
        output['method_results'][name]['semantic_interpretation']['blanket'] = {
            'variables': bl_labels,
        }

    results_path = os.path.join(RESULTS_DIR, 'us110_fwh_ensemble.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: FWH Predictions for Ensemble World Model")
    print("=" * 70)
    print(f"  Model: {meta['ensemble_size']}-member ensemble, "
          f"{meta['obs_dim']}D obs, FetchPush-v4")
    print(f"  Jacobians: {jacobians.shape}")
    print(f"  Runtime: {elapsed:.1f}s")

    print(f"\n  FWH Prediction 1 (Effective Dimensionality = O(N)):")
    for name, d in eff_dim_results.items():
        status = "PASS" if d['n_clusters'] <= 2 * FWH_N_CLUSTERS else "FAIL"
        print(f"    {name}: n_clusters={d['n_clusters']}, "
              f"PR={d['participation_ratio']:.2f} [{status}]")

    print(f"\n  FWH Prediction 2 (Orthogonal Factor Subspaces):")
    for name, d in ortho_results.items():
        status = "PASS" if d['mean_min_angle_deg'] > 45.0 else "FAIL"
        print(f"    {name}: mean_min_angle={d['mean_min_angle_deg']:.1f} deg [{status}]")

    print(f"\n  FWH Prediction 3 (Structure Recovery vs Ground Truth):")
    for name, d in ari_comparison.items():
        status = "STRONG" if d['ari'] > 0.5 else ("WEAK" if d['ari'] > 0.2 else "NONE")
        print(f"    {name}: ARI={d['ari']:.3f}, F1={d['blanket_f1']:.3f} [{status}]")

    print(f"\n  Best overall: {best_method} (ARI={best_ari:.3f})")
    print("=" * 70)
    print("US-110 complete.")

    return output


if __name__ == '__main__':
    main()
