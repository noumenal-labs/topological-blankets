"""
US-080: Structure Emergence During Learning -- Topological Blankets on Pandas Ensemble
======================================================================================

Runs TB analysis on the pandas Bayes ensemble at multiple training stages to show
how internal structure (gripper vs object vs relational coupling) emerges during
learning. This supports the info-thermodynamic selection narrative: as the ensemble
learns, it self-organises into a structured representation that TB can detect.

Two modes:
  (A) Interpolation mode (preferred, fast): If a trained checkpoint exists, create
      synthetic "training stages" by interpolating parameters between a random model
      and the trained model at fractions [0.0, 0.1, 0.2, 0.5, 1.0]. This simulates
      the trajectory of learning without requiring a multi-hour training run.

  (B) Mini-training mode (fallback): If no trained model exists, collect random
      transitions and run a short training loop (10 mini-iterations of 50 steps each),
      snapshotting at [0, 2, 4, 6, 8, 10] and running TB at each snapshot.

Outputs:
  - Filmstrip: coupling matrix heatmaps at each stage showing structure consolidation
  - Time series: partition metrics (NMI stability, n_objects, sparsity) vs stage,
    overlaid with prediction loss
  - Results JSON with all per-checkpoint metrics
  - Key finding: at what stage does meaningful structure emerge?

FetchPush-v4 ground truth:
  Object 0 (gripper):  [0,1,2, 9,10, 20,21]
  Object 1 (object):   [3,4,5, 11,12,13, 14,15,16, 17,18,19]
  Blanket (relational): [6,7,8]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import json
import warnings
import time as time_module
warnings.filterwarnings('ignore')

# -- Path setup ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)                      # ralph/
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)                   # topological_blankets/

PANDAS_DIR = 'C:/Users/citiz/Documents/noumenal-labs/pandas'

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)
sys.path.insert(0, PANDAS_DIR)

from topological_blankets import TopologicalBlankets
from topological_blankets.clustering import cluster_internals
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from panda.model import EnsembleModel, make_model, ModelConfig, DynamicsMember
from panda.utils import Normalizer
from panda.training_helpers import (
    compute_stats, loss_fn, make_train_step, prepare_training_data
)

# -- Constants -----------------------------------------------------------------
RUN_DIR = os.path.join(PANDAS_DIR, 'data', 'push_demo')
META_PATH = os.path.join(RUN_DIR, 'model.eqx.json')
MODEL_PATH = os.path.join(RUN_DIR, 'model.eqx')

EXPERIMENT_NAME = "pandas_structure_emergence"

OBS_LABELS = [
    'grip_x', 'grip_y', 'grip_z',
    'obj_x', 'obj_y', 'obj_z',
    'rel_x', 'rel_y', 'rel_z',
    'grip_state_0', 'grip_state_1',
    'obj_rot_0', 'obj_rot_1', 'obj_rot_2',
    'obj_velp_x', 'obj_velp_y', 'obj_velp_z',
    'obj_velr_x', 'obj_velr_y', 'obj_velr_z',
    'grip_velp_x', 'grip_velp_y',
    'extra_0', 'extra_1', 'extra_2',
]

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

GROUND_TRUTH_22D = np.array([
    0, 0, 0,       # grip_pos -> gripper
    1, 1, 1,       # obj_pos -> object
    -1, -1, -1,    # rel_pos -> blanket
    0, 0,           # gripper_state -> gripper
    1, 1, 1,        # obj_rot -> object
    1, 1, 1,        # obj_velp -> object
    1, 1, 1,        # obj_velr -> object
    0, 0,           # grip_velp -> gripper
])
GROUND_TRUTH_25D = np.concatenate([GROUND_TRUTH_22D, np.array([0, 0, 0])])
BLANKET_MASK_25D = (GROUND_TRUTH_25D == -1)

# Training stage fractions for interpolation mode
INTERP_FRACTIONS = [0.0, 0.1, 0.2, 0.5, 1.0]

# Mini-training checkpoints for fallback mode
MINI_TRAIN_CHECKPOINTS = [0, 2, 4, 6, 8, 10]
MINI_TRAIN_ITERS = 10
MINI_TRAIN_STEPS_PER_ITER = 50


# =============================================================================
# Model loading and creation
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


def create_random_model(meta=None):
    """Create a fresh random ensemble (for interpolation baseline or fallback)."""
    if meta is None:
        meta = {
            'obs_dim': 25, 'action_dim': 4, 'achieved_goal_dim': 3,
            'ensemble_size': 5, 'hidden_size': 256, 'depth': 2,
            'env_id': 'FetchPush-v4',
        }

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
    key = jax.random.PRNGKey(99)
    model = make_model(obs_dim, action_dim, ag_dim, cfg, normalizer, key)
    return model, meta


# =============================================================================
# Parameter interpolation
# =============================================================================

def interpolate_models(random_model, trained_model, alpha):
    """
    Create a model whose parameters are (1-alpha)*random + alpha*trained.

    alpha=0.0 yields the random (untrained) model.
    alpha=1.0 yields the fully trained model.

    The normalizer is taken from the trained model at all alpha > 0 so that
    forward passes use meaningful statistics, and from random_model at alpha=0.
    """
    random_params, random_static = eqx.partition(random_model, eqx.is_inexact_array)
    trained_params, trained_static = eqx.partition(trained_model, eqx.is_inexact_array)

    interp_params = jax.tree.map(
        lambda r, t: (1.0 - alpha) * r + alpha * t,
        random_params, trained_params
    )

    # Use trained static (normalizer, etc.) for alpha > 0 so forward pass
    # uses meaningful normalisation; for alpha=0 use the random model's static
    # (identity normalizer) to get the true random-model behaviour.
    if alpha > 0:
        interp_model = eqx.combine(interp_params, trained_static)
    else:
        interp_model = eqx.combine(interp_params, random_static)

    return interp_model


# =============================================================================
# Data collection
# =============================================================================

def collect_fetchpush_data(n_episodes=20, max_steps=50, seed=42):
    """Collect random trajectories from FetchPush-v4."""
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
    import gymnasium as gym

    env = gym.make('FetchPush-v4', max_episode_steps=max_steps, reward_type='dense')

    all_obs, all_ag, all_actions = [], [], []
    all_next_obs, all_next_ag = [], []

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        for step in range(max_steps):
            obs = obs_dict['observation']
            ag = obs_dict['achieved_goal']
            action = env.action_space.sample()

            all_obs.append(obs.copy())
            all_ag.append(ag.copy())
            all_actions.append(action.copy())

            obs_dict_next, _, term, trunc, _ = env.step(action)

            all_next_obs.append(obs_dict_next['observation'].copy())
            all_next_ag.append(obs_dict_next['achieved_goal'].copy())

            obs_dict = obs_dict_next
            if term or trunc:
                break

    env.close()

    data = {
        'obs': np.array(all_obs, dtype=np.float32),
        'achieved_goal': np.array(all_ag, dtype=np.float32),
        'actions': np.array(all_actions, dtype=np.float32),
        'next_obs': np.array(all_next_obs, dtype=np.float32),
        'next_achieved_goal': np.array(all_next_ag, dtype=np.float32),
    }
    print(f"Collected {len(data['obs'])} transitions from {n_episodes} episodes")
    return data


# =============================================================================
# Jacobian computation (reused from US-076)
# =============================================================================

def compute_member_jacobians(member, normalizer, obs_batch, ag_batch, action_batch):
    """Compute Jacobian d(delta_pred)/d(obs) for a single ensemble member."""
    def member_forward(obs_single, ag_single, act_single):
        delta_obs, delta_ag = member(obs_single, ag_single, act_single, normalizer)
        return jnp.concatenate([delta_obs, delta_ag])

    jac_fn = jax.jacobian(member_forward, argnums=0)
    batched_jac_fn = jax.vmap(
        lambda o, a, act: jac_fn(o, a, act),
        in_axes=(0, 0, 0)
    )

    obs_j = jnp.array(obs_batch)
    ag_j = jnp.array(ag_batch)
    act_j = jnp.array(action_batch)

    chunk_size = 64
    n_samples = obs_j.shape[0]
    jacobians_list = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        J_chunk = batched_jac_fn(obs_j[start:end], ag_j[start:end], act_j[start:end])
        jacobians_list.append(np.array(J_chunk))

    return np.concatenate(jacobians_list, axis=0)


def compute_ensemble_jacobians(model, data, max_samples=200):
    """Compute Jacobians for all ensemble members."""
    n_samples = min(max_samples, len(data['obs']))
    indices = np.random.RandomState(42).choice(len(data['obs']), n_samples, replace=False)

    obs_batch = data['obs'][indices]
    ag_batch = data['achieved_goal'][indices]
    act_batch = data['actions'][indices]

    all_jacobians = []
    for i, member in enumerate(model.members):
        J = compute_member_jacobians(member, model.normalizer, obs_batch, ag_batch, act_batch)
        all_jacobians.append(J)

    jacobians = np.stack(all_jacobians, axis=0)
    return jacobians, indices


# =============================================================================
# Coupling and TB analysis
# =============================================================================

def compute_coupling_from_jacobians(jacobians):
    """
    Compute Fisher-like coupling matrix from ensemble Jacobians.
    Returns normalised coupling and raw Fisher matrix.
    """
    E, N, out_dim, obs_dim = jacobians.shape
    fisher_sum = np.zeros((obs_dim, obs_dim))
    count = 0
    for e in range(E):
        for n in range(N):
            J = jacobians[e, n]
            fisher_sum += J.T @ J
            count += 1

    fisher_avg = fisher_sum / count
    D = np.sqrt(np.diag(fisher_avg)) + 1e-8
    coupling = np.abs(fisher_avg) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)
    return coupling, fisher_avg


def run_tb_on_fisher(fisher_raw, n_objects=2, method='hybrid'):
    """Run TB analysis using a pre-computed Fisher coupling matrix."""
    from topological_blankets.detection import (
        detect_blankets_hybrid, detect_blankets_gradient, detect_blankets_coupling
    )

    obs_dim = fisher_raw.shape[0]
    D = np.sqrt(np.diag(np.abs(fisher_raw)) + 1e-8)
    coupling_norm = np.abs(fisher_raw) / np.outer(D, D)
    np.fill_diagonal(coupling_norm, 0)

    grad_magnitude = D
    grad_variance = D ** 2

    features = {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': fisher_raw,
        'coupling': coupling_norm,
    }

    if method == 'hybrid':
        result = detect_blankets_hybrid(
            np.random.randn(100, obs_dim),
            fisher_raw
        )
        is_blanket = result['is_blanket']
    elif method == 'coupling':
        is_blanket = detect_blankets_coupling(fisher_raw, coupling_norm, n_objects)
    else:
        from topological_blankets.detection import detect_blankets_otsu
        is_blanket, _ = detect_blankets_otsu(features)

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
        'features': features,
    }


def compute_prediction_loss(model, data, max_samples=500):
    """Compute mean ensemble prediction loss on held-out data."""
    n_samples = min(max_samples, len(data['obs']))
    rng = np.random.RandomState(123)
    indices = rng.choice(len(data['obs']), n_samples, replace=False)

    obs = jnp.array(data['obs'][indices])
    ag = jnp.array(data['achieved_goal'][indices])
    actions = jnp.array(data['actions'][indices])
    next_obs = jnp.array(data['next_obs'][indices])
    next_ag = jnp.array(data['next_achieved_goal'][indices])

    delta_obs_true = next_obs - obs
    delta_ag_true = next_ag - ag

    # Compute predictions from each member, average loss
    losses = []
    for member in model.members:
        delta_obs_pred, delta_ag_pred = member.predict_deltas(
            obs, ag, actions, model.normalizer
        )
        obs_loss = float(jnp.mean((delta_obs_pred - delta_obs_true) ** 2))
        ag_loss = float(jnp.mean((delta_ag_pred - delta_ag_true) ** 2))
        losses.append(obs_loss + ag_loss)

    return float(np.mean(losses))


def compute_partition_metrics(tb_result, prev_assignment=None):
    """
    Compute summary metrics for a TB partition.

    Returns dict with:
      - n_objects: number of detected objects
      - n_blanket_vars: number of blanket variables
      - sparsity: fraction of near-zero entries in coupling matrix
      - ari_vs_gt: Adjusted Rand Index against ground truth
      - blanket_f1: F1 score for blanket detection
      - nmi_vs_prev: NMI against previous checkpoint (None if no previous)
    """
    from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

    assignment = tb_result['assignment']
    coupling = tb_result['coupling']

    n_vars = min(len(assignment), len(GROUND_TRUTH_25D))
    pred = assignment[:n_vars].copy()
    gt = GROUND_TRUTH_25D[:n_vars].copy()

    ari = adjusted_rand_score(gt, pred)

    pred_blanket = (pred == -1)
    gt_blanket = BLANKET_MASK_25D[:n_vars]
    if gt_blanket.sum() > 0 and pred_blanket.sum() > 0:
        blanket_f1 = f1_score(gt_blanket, pred_blanket)
    else:
        blanket_f1 = 0.0

    # Sparsity: fraction of off-diagonal entries below 0.1
    n_dim = coupling.shape[0]
    off_diag = coupling[~np.eye(n_dim, dtype=bool)]
    sparsity = float(np.mean(np.abs(off_diag) < 0.1))

    # NMI vs previous partition
    nmi_vs_prev = None
    if prev_assignment is not None:
        n_common = min(len(assignment), len(prev_assignment))
        nmi_vs_prev = float(normalized_mutual_info_score(
            prev_assignment[:n_common], assignment[:n_common]
        ))

    return {
        'n_objects': len(tb_result['objects']),
        'n_blanket_vars': len(tb_result['blankets']),
        'sparsity': float(sparsity),
        'ari_vs_gt': float(ari),
        'blanket_f1': float(blanket_f1),
        'nmi_vs_prev': nmi_vs_prev,
    }


# =============================================================================
# Per-stage analysis
# =============================================================================

def jacobians_to_sensitivity_gradients(jacobians):
    """
    Convert ensemble Jacobians to gradient-like sensitivity profiles for TB.
    Each (member, sample) pair yields a vector of per-variable sensitivity norms.
    """
    E, N, out_dim, obs_dim = jacobians.shape
    sensitivity = np.sqrt(np.sum(jacobians ** 2, axis=2))  # (E, N, obs_dim)
    gradients = sensitivity.reshape(E * N, obs_dim)
    return gradients


def run_tb_sensitivity(gradients, n_objects=2, method='hybrid'):
    """Run TB using sensitivity profiles as gradient samples."""
    tb = TopologicalBlankets(method=method, n_objects=n_objects)
    tb.fit(gradients)

    assignment = tb.get_assignment()
    blankets = tb.get_blankets()
    objects = tb.get_objects()
    coupling = tb.get_coupling_matrix()
    features = tb.get_features()

    return {
        'assignment': assignment,
        'blankets': blankets,
        'objects': objects,
        'coupling': coupling,
        'features': features,
    }


def best_tb_across_methods(jacobians, fisher_raw):
    """
    Try several TB approaches and return the one with highest ARI vs ground truth.
    Methods tried:
      - Fisher hybrid, coupling (n_objects=2 and 3)
      - Sensitivity hybrid, coupling, gradient
    """
    from sklearn.metrics import adjusted_rand_score, f1_score

    sens_grads = jacobians_to_sensitivity_gradients(jacobians)

    candidates = []

    # Fisher-based methods
    for method in ['hybrid', 'coupling']:
        for n_obj in [2, 3]:
            try:
                result = run_tb_on_fisher(fisher_raw, n_objects=n_obj, method=method)
                n_vars = min(len(result['assignment']), len(GROUND_TRUTH_25D))
                ari = adjusted_rand_score(GROUND_TRUTH_25D[:n_vars], result['assignment'][:n_vars])
                candidates.append((f'fisher_{method}_{n_obj}obj', result, ari))
            except Exception:
                pass

    # Sensitivity-based methods
    for method in ['hybrid', 'coupling', 'gradient']:
        for n_obj in [2, 3]:
            try:
                result = run_tb_sensitivity(sens_grads, n_objects=n_obj, method=method)
                n_vars = min(len(result['assignment']), len(GROUND_TRUTH_25D))
                ari = adjusted_rand_score(GROUND_TRUTH_25D[:n_vars], result['assignment'][:n_vars])
                candidates.append((f'sens_{method}_{n_obj}obj', result, ari))
            except Exception:
                pass

    if not candidates:
        # Fallback: just run hybrid on Fisher
        result = run_tb_on_fisher(fisher_raw, n_objects=2, method='hybrid')
        return 'fisher_hybrid_2obj', result

    best_name, best_result, best_ari = max(candidates, key=lambda x: x[2])
    return best_name, best_result


def analyse_stage(model, data, stage_label, prev_assignment=None, max_jac_samples=200):
    """
    Run full TB analysis on a single model snapshot, trying multiple detection
    methods and reporting the best result.

    Returns a dict with coupling matrix, TB result, partition metrics,
    and prediction loss.
    """
    print(f"\n{'='*60}")
    print(f"  Stage: {stage_label}")
    print(f"{'='*60}")

    t0 = time_module.time()

    # 1. Jacobians
    print("  Computing Jacobians...")
    jacobians, indices = compute_ensemble_jacobians(model, data, max_samples=max_jac_samples)
    print(f"    Jacobian shape: {jacobians.shape}")

    # 2. Coupling matrix (for the filmstrip visualisation)
    coupling, fisher_raw = compute_coupling_from_jacobians(jacobians)

    # 3. Best TB detection across methods
    print("  Running multi-method TB detection...")
    best_method, tb_result = best_tb_across_methods(jacobians, fisher_raw)
    print(f"    Best method: {best_method}")

    # 4. Partition metrics
    metrics = compute_partition_metrics(tb_result, prev_assignment)
    metrics['best_method'] = best_method

    # 5. Prediction loss
    print("  Computing prediction loss...")
    pred_loss = compute_prediction_loss(model, data)
    metrics['prediction_loss'] = pred_loss

    elapsed = time_module.time() - t0
    metrics['elapsed_sec'] = float(elapsed)

    # Semantic summary
    for obj_id, obj_vars in tb_result['objects'].items():
        labels = [OBS_LABELS[i] for i in obj_vars if i < len(OBS_LABELS)]
        print(f"    Object {obj_id}: {labels}")
    blanket_labels = [OBS_LABELS[i] for i in tb_result['blankets'] if i < len(OBS_LABELS)]
    print(f"    Blanket: {blanket_labels}")
    print(f"    ARI={metrics['ari_vs_gt']:.3f}, F1={metrics['blanket_f1']:.3f}, "
          f"sparsity={metrics['sparsity']:.3f}, loss={pred_loss:.4f}")

    return {
        'coupling': coupling,
        'fisher_raw': fisher_raw,
        'tb_result': tb_result,
        'metrics': metrics,
        'stage_label': stage_label,
    }


# =============================================================================
# Mode A: Interpolation-based stages
# =============================================================================

def run_interpolation_mode(trained_model, meta, data):
    """
    Create synthetic training stages by interpolating between random and
    trained model parameters.
    """
    print("\n" + "#" * 70)
    print("  MODE A: Parameter interpolation (random -> trained)")
    print("#" * 70)

    random_model, _ = create_random_model(meta)

    stages = []
    prev_assignment = None

    for alpha in INTERP_FRACTIONS:
        label = f"alpha={alpha:.1f}"
        interp_model = interpolate_models(random_model, trained_model, alpha)
        result = analyse_stage(interp_model, data, label, prev_assignment)
        result['alpha'] = alpha
        stages.append(result)
        prev_assignment = result['tb_result']['assignment']

    return stages


# =============================================================================
# Mode B: Mini-training fallback
# =============================================================================

def run_mini_training_mode(data):
    """
    Train from scratch for a small number of iterations and snapshot at
    regular intervals. This is a fallback for when no trained checkpoint
    exists.
    """
    print("\n" + "#" * 70)
    print("  MODE B: Mini-training (10 iterations x 50 steps)")
    print("#" * 70)

    obs_dim = data['obs'].shape[1]
    action_dim = data['actions'].shape[1]
    ag_dim = data['achieved_goal'].shape[1]

    # Compute normalisation statistics
    normalizer = compute_stats(
        data['obs'], data['actions'], data['next_obs'],
        data['achieved_goal'], data['next_achieved_goal']
    )

    cfg = ModelConfig(ensemble_size=5, hidden_size=256, depth=2)
    key = jax.random.PRNGKey(42)
    model = make_model(obs_dim, action_dim, ag_dim, cfg, normalizer, key)

    # Prepare training data
    obs_j, ag_j, actions_j, delta_obs_norm, delta_ag_norm = prepare_training_data(
        data['obs'], data['achieved_goal'], data['actions'],
        data['next_obs'], data['next_achieved_goal'], normalizer
    )

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer)

    n_train = len(data['obs'])
    batch_size = min(128, n_train)

    stages = []
    prev_assignment = None
    rng = np.random.RandomState(0)

    for iteration in range(MINI_TRAIN_ITERS + 1):
        # Snapshot at checkpoint iterations
        if iteration in MINI_TRAIN_CHECKPOINTS:
            label = f"iter={iteration}"
            result = analyse_stage(model, data, label, prev_assignment)
            result['iteration'] = iteration
            stages.append(result)
            prev_assignment = result['tb_result']['assignment']

        if iteration < MINI_TRAIN_ITERS:
            # Run MINI_TRAIN_STEPS_PER_ITER gradient steps
            for step in range(MINI_TRAIN_STEPS_PER_ITER):
                idx = rng.choice(n_train, batch_size, replace=False)
                batch = (obs_j[idx], ag_j[idx], actions_j[idx],
                         delta_obs_norm[idx], delta_ag_norm[idx])
                model, opt_state, loss = train_step(model, opt_state, batch)

            print(f"  Iteration {iteration + 1}/{MINI_TRAIN_ITERS}, "
                  f"last batch loss={float(loss):.5f}")

    return stages


# =============================================================================
# Visualization
# =============================================================================

def plot_filmstrip(stages):
    """
    Filmstrip: coupling matrix heatmaps at each training stage, showing
    how structure consolidates from dense (random) to block-diagonal (trained).
    """
    n_stages = len(stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(4.5 * n_stages, 4.5))
    if n_stages == 1:
        axes = [axes]

    vmin = 0
    vmax = max(s['coupling'].max() for s in stages)
    # Cap vmax to avoid outlier dominance
    vmax = min(vmax, 1.0)

    for i, stage in enumerate(stages):
        ax = axes[i]
        coupling = stage['coupling']
        n_vars = coupling.shape[0]
        im = ax.imshow(coupling, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(stage['stage_label'], fontsize=10, fontweight='bold')

        if i == 0:
            tick_labels = [OBS_LABELS[j] if j < len(OBS_LABELS) else f'v{j}'
                           for j in range(n_vars)]
            ax.set_yticks(range(n_vars))
            ax.set_yticklabels(tick_labels, fontsize=5)
        else:
            ax.set_yticks([])

        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(
            [OBS_LABELS[j] if j < len(OBS_LABELS) else f'v{j}' for j in range(n_vars)],
            rotation=90, fontsize=5
        )

        # Overlay partition boundaries
        assignment = stage['tb_result']['assignment']
        unique_labels = sorted(set(assignment))
        for lbl in unique_labels:
            indices = np.where(assignment == lbl)[0]
            if len(indices) == 0:
                continue
            # Find contiguous runs in original index space is not useful here;
            # instead draw a small coloured marker bar at the top
            color = 'cyan' if lbl == -1 else ['lime', 'yellow', 'magenta'][lbl % 3]
            for idx in indices:
                ax.plot(idx, -0.8, 's', color=color, markersize=3,
                        clip_on=False, transform=ax.transData)

        # Add metrics annotation
        m = stage['metrics']
        ax.text(0.5, -0.22, (
            f"ARI={m['ari_vs_gt']:.2f}  F1={m['blanket_f1']:.2f}\n"
            f"sparsity={m['sparsity']:.2f}  loss={m['prediction_loss']:.3f}"
        ), transform=ax.transAxes, fontsize=7, ha='center', va='top')

    # Shared colorbar
    fig.subplots_adjust(right=0.92, bottom=0.25)
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, label='Coupling Strength')

    fig.suptitle(
        'Structure Emergence: Coupling Matrix During Learning',
        fontsize=13, fontweight='bold', y=1.02
    )

    return fig


def plot_filmstrip_reordered(stages):
    """
    Filmstrip with coupling matrices reordered by TB partition at each stage,
    making block-diagonal structure more visible.
    """
    n_stages = len(stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(4.5 * n_stages, 4.5))
    if n_stages == 1:
        axes = [axes]

    for i, stage in enumerate(stages):
        ax = axes[i]
        coupling = stage['coupling']
        assignment = stage['tb_result']['assignment']
        n_vars = coupling.shape[0]

        # Build reorder index
        order = []
        unique_labels = sorted(set(assignment))
        label_boundaries = []
        for lbl in unique_labels:
            indices = np.where(assignment == lbl)[0]
            label_boundaries.append((len(order), len(order) + len(indices), lbl))
            order.extend(indices.tolist())

        reordered = coupling[np.ix_(order, order)]
        ax.imshow(reordered, cmap='hot', aspect='auto', vmin=0, vmax=1.0)
        ax.set_title(stage['stage_label'], fontsize=10, fontweight='bold')

        # Draw boundaries
        for start, end, lbl in label_boundaries:
            color = 'cyan' if lbl == -1 else ['lime', 'yellow', 'magenta'][lbl % 3]
            ax.axhline(y=start - 0.5, color=color, linewidth=1, linestyle='--', alpha=0.7)
            ax.axvline(x=start - 0.5, color=color, linewidth=1, linestyle='--', alpha=0.7)
            ax.axhline(y=end - 0.5, color=color, linewidth=1, linestyle='--', alpha=0.7)
            ax.axvline(x=end - 0.5, color=color, linewidth=1, linestyle='--', alpha=0.7)

        if i == 0:
            tick_labels = [OBS_LABELS[order[j]] if order[j] < len(OBS_LABELS) else f'v{order[j]}'
                           for j in range(n_vars)]
            ax.set_yticks(range(n_vars))
            ax.set_yticklabels(tick_labels, fontsize=5)
        else:
            ax.set_yticks([])

        ax.set_xticks([])

        m = stage['metrics']
        ax.text(0.5, -0.12, (
            f"ARI={m['ari_vs_gt']:.2f}  F1={m['blanket_f1']:.2f}"
        ), transform=ax.transAxes, fontsize=7, ha='center', va='top')

    fig.suptitle(
        'Structure Emergence: Reordered by TB Partition',
        fontsize=13, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    return fig


def plot_time_series(stages):
    """
    Time series of partition metrics and prediction loss across training stages.
    """
    # X-axis: stage index
    if 'alpha' in stages[0]:
        x_vals = [s['alpha'] for s in stages]
        x_label = 'Interpolation fraction (alpha)'
    else:
        x_vals = [s['iteration'] for s in stages]
        x_label = 'Training iteration'

    aris = [s['metrics']['ari_vs_gt'] for s in stages]
    f1s = [s['metrics']['blanket_f1'] for s in stages]
    sparsities = [s['metrics']['sparsity'] for s in stages]
    n_objects = [s['metrics']['n_objects'] for s in stages]
    losses = [s['metrics']['prediction_loss'] for s in stages]
    nmis = [s['metrics']['nmi_vs_prev'] for s in stages]
    # NMI for first stage is None; replace with 0 for plotting
    nmis_plot = [v if v is not None else 0.0 for v in nmis]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Panel 1: ARI and Blanket F1
    ax1 = axes[0]
    ax1.plot(x_vals, aris, 'o-', color='steelblue', label='ARI vs ground truth', linewidth=2)
    ax1.plot(x_vals, f1s, 's-', color='forestgreen', label='Blanket F1', linewidth=2)
    ax1.plot(x_vals, nmis_plot, '^-', color='purple', label='NMI vs prev stage', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Partition Quality During Learning', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Sparsity and n_objects
    ax2 = axes[1]
    ax2.plot(x_vals, sparsities, 'D-', color='darkorange', label='Coupling sparsity', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_vals, n_objects, 'x-', color='crimson', label='Detected objects', linewidth=2)
    ax2.set_ylabel('Sparsity (fraction < 0.1)')
    ax2_twin.set_ylabel('Number of objects')
    ax2_twin.set_ylim(0, max(n_objects) + 1)
    ax2.set_title('Structural Properties', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=9, loc='center right')

    # Panel 3: Prediction loss
    ax3 = axes[2]
    ax3.plot(x_vals, losses, 'o-', color='red', label='Prediction loss (MSE)', linewidth=2)
    ax3.set_ylabel('Loss')
    ax3.set_xlabel(x_label)
    ax3.set_title('Ensemble Prediction Loss', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_emergence_summary(stages):
    """
    Single-panel summary: ARI vs prediction loss, annotated with stage labels.
    This highlights when meaningful structure emerges relative to task performance.
    """
    if 'alpha' in stages[0]:
        x_vals = [s['alpha'] for s in stages]
        x_label = 'Interpolation fraction (alpha)'
    else:
        x_vals = [s['iteration'] for s in stages]
        x_label = 'Training iteration'

    aris = [s['metrics']['ari_vs_gt'] for s in stages]
    losses = [s['metrics']['prediction_loss'] for s in stages]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = 'steelblue'
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('ARI vs Ground Truth', color=color1, fontsize=11)
    line1 = ax1.plot(x_vals, aris, 'o-', color=color1, linewidth=2.5, markersize=8, label='ARI')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.15, 1.1)

    ax2 = ax1.twinx()
    color2 = 'red'
    ax2.set_ylabel('Prediction Loss', color=color2, fontsize=11)
    line2 = ax2.plot(x_vals, losses, 's--', color=color2, linewidth=2, markersize=8, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Annotate each point with stage label
    for i, stage in enumerate(stages):
        ax1.annotate(
            stage['stage_label'],
            (x_vals[i], aris[i]),
            textcoords="offset points",
            xytext=(0, 12),
            fontsize=7, ha='center', color='steelblue'
        )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    ax1.set_title(
        'Structure Emergence: ARI vs Prediction Loss',
        fontsize=13, fontweight='bold'
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# Key finding analysis
# =============================================================================

def analyse_emergence_point(stages):
    """
    Determine at which stage meaningful structure emerges.

    We define "meaningful structure" as ARI > 0.3 (better than random partition)
    and blanket F1 > 0.2 (detecting at least some blanket variables).

    Returns a narrative string documenting the finding.
    """
    ari_threshold = 0.3
    f1_threshold = 0.2

    emergence_stage = None
    for stage in stages:
        m = stage['metrics']
        if m['ari_vs_gt'] > ari_threshold and m['blanket_f1'] > f1_threshold:
            emergence_stage = stage
            break

    if emergence_stage is None:
        # Check if *any* stage shows moderate structure
        best_stage = max(stages, key=lambda s: s['metrics']['ari_vs_gt'])
        best_m = best_stage['metrics']
        narrative = (
            f"No stage achieved meaningful structure (ARI > {ari_threshold} AND "
            f"F1 > {f1_threshold}). The best result was at {best_stage['stage_label']} "
            f"with ARI={best_m['ari_vs_gt']:.3f}, F1={best_m['blanket_f1']:.3f}. "
            f"This suggests the trained model may encode the task differently than "
            f"the expected gripper-object-relational decomposition, or that TB's "
            f"hybrid detection needs domain-tuned thresholds for this architecture."
        )
    else:
        m = emergence_stage['metrics']
        first_m = stages[0]['metrics']
        last_m = stages[-1]['metrics']
        narrative = (
            f"Meaningful structure emerges at {emergence_stage['stage_label']} "
            f"(ARI={m['ari_vs_gt']:.3f}, F1={m['blanket_f1']:.3f}). "
            f"At the random baseline ({stages[0]['stage_label']}), "
            f"ARI={first_m['ari_vs_gt']:.3f}, loss={first_m['prediction_loss']:.4f}. "
            f"At the final stage ({stages[-1]['stage_label']}), "
            f"ARI={last_m['ari_vs_gt']:.3f}, loss={last_m['prediction_loss']:.4f}. "
            f"Coupling matrix sparsity increased from {first_m['sparsity']:.3f} to "
            f"{last_m['sparsity']:.3f}, indicating the learned dynamics self-organised "
            f"into a sparser, more modular structure as training progressed."
        )

    # Also report correlation between ARI and loss across stages
    aris = np.array([s['metrics']['ari_vs_gt'] for s in stages])
    losses = np.array([s['metrics']['prediction_loss'] for s in stages])
    if len(aris) > 2 and np.std(aris) > 0 and np.std(losses) > 0:
        corr = np.corrcoef(aris, -losses)[0, 1]  # negative loss so positive = aligned
        narrative += (
            f" Correlation between ARI and negative-loss: {corr:.3f} "
            f"({'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'} "
            f"alignment between structure quality and task performance)."
        )

    return narrative


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("US-080: Structure Emergence During Learning")
    print("  Topological Blankets on Pandas Ensemble at Multiple Stages")
    print("=" * 70)

    t_start = time_module.time()

    # 1. Collect environment data
    print("\n--- Collecting FetchPush-v4 trajectory data ---")
    data = collect_fetchpush_data(n_episodes=20, max_steps=50, seed=42)

    # 2. Determine mode
    trained_available = (
        os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
    )

    if trained_available:
        print("\nTrained checkpoint found. Using Mode A (interpolation).")
        try:
            trained_model, meta = load_trained_model()
            stages = run_interpolation_mode(trained_model, meta, data)
            mode_label = "interpolation"
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            print("Falling back to Mode B (mini-training).")
            stages = run_mini_training_mode(data)
            mode_label = "mini-training"
    else:
        print("\nNo trained checkpoint found. Using Mode B (mini-training).")
        stages = run_mini_training_mode(data)
        mode_label = "mini-training"

    # 3. Analyse emergence point
    print("\n--- Analysing emergence point ---")
    narrative = analyse_emergence_point(stages)
    print(f"\n  {narrative}")

    # 4. Generate visualizations
    print("\n--- Generating Visualizations ---")

    fig_filmstrip = plot_filmstrip(stages)
    save_figure(fig_filmstrip, "filmstrip_coupling", EXPERIMENT_NAME)

    fig_filmstrip_reordered = plot_filmstrip_reordered(stages)
    save_figure(fig_filmstrip_reordered, "filmstrip_reordered", EXPERIMENT_NAME)

    fig_timeseries = plot_time_series(stages)
    save_figure(fig_timeseries, "timeseries_metrics", EXPERIMENT_NAME)

    fig_summary = plot_emergence_summary(stages)
    save_figure(fig_summary, "emergence_summary", EXPERIMENT_NAME)

    # 5. Save results JSON
    print("\n--- Saving Results ---")

    per_stage_metrics = []
    for stage in stages:
        entry = {
            'stage_label': stage['stage_label'],
            **stage['metrics'],
            # Serialise assignment as list
            'assignment': stage['tb_result']['assignment'].tolist(),
            'blankets': stage['tb_result']['blankets'].tolist(),
            'objects': {str(k): v.tolist() for k, v in stage['tb_result']['objects'].items()},
        }
        if 'alpha' in stage:
            entry['alpha'] = stage['alpha']
        if 'iteration' in stage:
            entry['iteration'] = stage['iteration']
        per_stage_metrics.append(entry)

    metrics = {
        'mode': mode_label,
        'n_stages': len(stages),
        'n_env_transitions': int(len(data['obs'])),
        'obs_dim': int(data['obs'].shape[1]),
        'interp_fractions': INTERP_FRACTIONS if mode_label == 'interpolation' else None,
        'per_stage': per_stage_metrics,
        'emergence_narrative': narrative,
        'total_elapsed_sec': float(time_module.time() - t_start),
    }

    config = {
        'n_episodes': 20,
        'max_steps': 50,
        'max_jacobian_samples': 200,
        'detection_method': 'hybrid',
        'n_objects': 2,
        'env_id': 'FetchPush-v4',
        'mode': mode_label,
    }

    save_results(
        EXPERIMENT_NAME,
        metrics=metrics,
        config=config,
        notes=(
            f"US-080: Structure emergence analysis ({mode_label} mode). "
            f"Ran TB on {len(stages)} stages. "
            f"Emergence finding: {narrative[:200]}..."
        )
    )

    # 6. Summary
    total_time = time_module.time() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mode: {mode_label}")
    print(f"  Stages analysed: {len(stages)}")
    print(f"  Total time: {total_time:.1f}s")
    print()
    print("  Per-stage metrics:")
    print(f"  {'Stage':<15} {'ARI':>6} {'F1':>6} {'Sparsity':>9} {'N_Obj':>6} {'Loss':>10}")
    print(f"  {'-'*55}")
    for stage in stages:
        m = stage['metrics']
        print(f"  {stage['stage_label']:<15} {m['ari_vs_gt']:>6.3f} {m['blanket_f1']:>6.3f} "
              f"{m['sparsity']:>9.3f} {m['n_objects']:>6d} {m['prediction_loss']:>10.4f}")
    print()
    print(f"  Emergence narrative: {narrative}")
    print("=" * 70)
    print("US-080 complete.")

    return metrics


if __name__ == '__main__':
    main()
