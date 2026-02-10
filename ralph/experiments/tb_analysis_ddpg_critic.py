"""
US-106: Post-hoc TB Analysis of DDPG+HER Critic Network
========================================================

Applies Topological Blankets to the learned Q-function (critic) of a
DDPG+HER agent trained on FetchPush-v4. The critic maps (obs, action) -> Q,
so its gradients dQ/d(obs) encode which observation dimensions the policy
considers coupled for value estimation.

This answers the question: does a model-free agent implicitly learn
Markov blanket structure, even without an explicit world model?

The analysis:
  1. Loads the trained DDPG+HER model from stable-baselines3 checkpoint.
  2. Collects observation-action pairs from the trained policy.
  3. Computes dQ/d(obs) for each (obs, action) pair via PyTorch autograd.
  4. Runs TB.fit() on the critic gradients.
  5. Compares discovered partition to:
     (a) Ground-truth FetchPush structure
     (b) TB partition from the Bayes ensemble (US-076/096)
  6. Saves comparison results, coupling matrices, and figures.

Key hypothesis: The critic should learn that gripper and object positions
are coupled through relative position (the blanket), because Q-value
depends on whether the gripper can reach and push the object. TB should
detect this coupling structure, even though DDPG never builds an explicit
dynamics model.

If confirmed, this validates TB as a universal structure discovery tool
that works across learning paradigms (model-based and model-free).

Usage:
    python tb_analysis_ddpg_critic.py [--ddpg-dir ../pandas/data/ddpg_her_fetchpush]
                                       [--ensemble-dir ../pandas/data/fetchpush_50step]
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)
PANDAS_DIR = os.path.join(os.path.dirname(TB_PACKAGE_DIR), 'pandas')

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from topological_blankets import TopologicalBlankets
from topological_blankets.features import compute_geometric_features
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

# ── FetchPush ground truth ──────────────────────────────────────────────
FETCHPUSH_OBS_LABELS = [
    'grip_x', 'grip_y', 'grip_z',          # 0-2
    'obj_x', 'obj_y', 'obj_z',             # 3-5
    'rel_x', 'rel_y', 'rel_z',             # 6-8
    'finger_L', 'finger_R',                 # 9-10
    'obj_rot_x', 'obj_rot_y', 'obj_rot_z', # 11-13
    'obj_vx', 'obj_vy', 'obj_vz',          # 14-16
    'obj_wx', 'obj_wy', 'obj_wz',          # 17-19
    'grip_vx', 'grip_vy',                   # 20-21
    'extra_0', 'extra_1', 'extra_2',        # 22-24
]

GROUND_TRUTH = {
    'gripper': [0, 1, 2, 9, 10, 20, 21],
    'object': [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'blanket': [6, 7, 8],
    'unstructured': [22, 23, 24],
}


def collect_critic_gradients(model, env, n_samples=2000):
    """Collect dQ/d(obs) gradients from the DDPG critic."""
    import torch

    gradients = []
    obs_list = []

    obs, info = env.reset()
    collected = 0

    while collected < n_samples:
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=False)

        # Extract flat observation (just the 'observation' key)
        obs_flat = obs['observation']
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).requires_grad_(True)

        # Build critic input: need to replicate what SB3 does internally
        # The critic takes the full observation dict processed through the feature extractor
        # For simplicity, compute Q via the critic directly
        with torch.no_grad():
            # Get the processed observation through the policy's feature extractor
            ag = torch.FloatTensor(obs['achieved_goal']).unsqueeze(0)
            dg = torch.FloatTensor(obs['desired_goal']).unsqueeze(0)
            act = torch.FloatTensor(action).unsqueeze(0)

        # We need gradients w.r.t. observation only
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).requires_grad_(True)
        ag_tensor = torch.FloatTensor(obs['achieved_goal']).unsqueeze(0)
        dg_tensor = torch.FloatTensor(obs['desired_goal']).unsqueeze(0)
        act_tensor = torch.FloatTensor(action).unsqueeze(0)

        # Construct the combined observation as the critic sees it
        combined = torch.cat([obs_tensor, ag_tensor, dg_tensor], dim=1)

        try:
            # Access the critic network
            critic = model.critic
            # SB3 DDPG critic takes (obs_features, action)
            # The feature extractor for MultiInputPolicy concatenates all inputs
            q_value = critic.q_networks[0](torch.cat([combined, act_tensor], dim=1))
            q_value.backward()

            if obs_tensor.grad is not None:
                grad = obs_tensor.grad.detach().numpy().flatten()
                gradients.append(grad)
                obs_list.append(obs_flat.copy())
                collected += 1
        except Exception as e:
            # Fallback: use finite differences
            pass

        obs_tensor.grad = None if obs_tensor.grad is not None else None

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    return np.array(gradients), np.array(obs_list)


def collect_critic_gradients_finite_diff(model, env, n_samples=2000, eps=1e-4):
    """Fallback: finite-difference dQ/d(obs) if autograd fails."""
    import torch

    gradients = []
    obs_list = []
    obs, info = env.reset()
    collected = 0
    obs_dim = obs['observation'].shape[0]

    while collected < n_samples:
        action, _ = model.predict(obs, deterministic=False)

        # Base Q-value
        with torch.no_grad():
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}
            act_tensor = torch.FloatTensor(action).unsqueeze(0)
            q_base = model.critic(obs_tensor, act_tensor)[0].item()

        grad = np.zeros(obs_dim)
        for d in range(obs_dim):
            perturbed_obs = {k: v.copy() for k, v in obs.items()}
            perturbed_obs['observation'][d] += eps
            with torch.no_grad():
                p_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in perturbed_obs.items()}
                q_pert = model.critic(p_tensor, act_tensor)[0].item()
            grad[d] = (q_pert - q_base) / eps

        gradients.append(grad)
        obs_list.append(obs['observation'].copy())
        collected += 1

        if collected % 200 == 0:
            print(f"  Collected {collected}/{n_samples} critic gradients")

        next_obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    return np.array(gradients), np.array(obs_list)


def run_tb_analysis(gradients, label=""):
    """Run TB on gradients and return results dict."""
    tb = TopologicalBlankets(method='hybrid', n_objects=None)
    tb.fit(gradients)

    assignment = tb.get_assignment()
    blankets = tb.get_blankets()
    coupling = tb.get_coupling_matrix()
    info = tb.get_detection_info()

    objects = tb.get_objects()
    n_objects = len(set(assignment[~blankets]))

    # Compute metrics vs ground truth
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score

    # Build ground-truth labels (0=gripper, 1=object, 2=blanket, 3=unstructured)
    gt_labels = np.zeros(25, dtype=int)
    for idx in GROUND_TRUTH['gripper']:
        gt_labels[idx] = 0
    for idx in GROUND_TRUTH['object']:
        gt_labels[idx] = 1
    for idx in GROUND_TRUTH['blanket']:
        gt_labels[idx] = 2
    for idx in GROUND_TRUTH['unstructured']:
        gt_labels[idx] = 3

    # Predicted labels
    pred_labels = assignment.copy()
    pred_labels[blankets] = max(pred_labels) + 1  # blanket gets its own label

    ari = adjusted_rand_score(gt_labels, pred_labels)
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)

    # Blanket F1
    gt_blanket = np.zeros(len(blankets), dtype=int)
    for idx in GROUND_TRUTH['blanket']:
        if idx < len(gt_blanket):
            gt_blanket[idx] = 1
    blanket_f1 = f1_score(gt_blanket, blankets.astype(int), zero_division=0.0)

    eigengap = info.get('eigengap', 0.0)

    result = {
        'label': label,
        'n_objects': n_objects,
        'n_blanket_vars': int(blankets.sum()),
        'blanket_indices': np.where(blankets)[0].tolist(),
        'assignment': assignment.tolist(),
        'ari': round(float(ari), 4),
        'nmi': round(float(nmi), 4),
        'blanket_f1': round(float(blanket_f1), 4),
        'eigengap': round(float(eigengap), 4),
    }

    print(f"\n  [{label}] TB Results:")
    print(f"    Objects: {n_objects}, Blanket vars: {int(blankets.sum())}")
    print(f"    Blanket indices: {np.where(blankets)[0].tolist()}")
    blanket_names = [FETCHPUSH_OBS_LABELS[i] for i in np.where(blankets)[0]]
    print(f"    Blanket variables: {blanket_names}")
    print(f"    ARI: {ari:.4f}, NMI: {nmi:.4f}, Blanket F1: {blanket_f1:.4f}")

    return result, coupling, blankets, assignment


def plot_comparison(coupling_critic, coupling_ensemble,
                    blankets_critic, blankets_ensemble,
                    assign_critic, assign_ensemble,
                    results_critic, results_ensemble):
    """Side-by-side coupling matrix comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coupling, blankets, assign, res, title in [
        (axes[0], coupling_critic, blankets_critic, assign_critic,
         results_critic, "DDPG+HER Critic"),
        (axes[1], coupling_ensemble, blankets_ensemble, assign_ensemble,
         results_ensemble, "Bayes Ensemble"),
    ]:
        im = ax.imshow(np.abs(coupling), cmap='viridis', aspect='equal')
        ax.set_title(f"{title}\nARI={res['ari']:.3f}  F1={res['blanket_f1']:.3f}  "
                     f"Objects={res['n_objects']}", fontsize=11)
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Variable index")

        # Mark blanket variables
        blanket_idx = np.where(blankets)[0]
        for idx in blanket_idx:
            ax.axhline(y=idx, color='red', alpha=0.3, linewidth=0.5)
            ax.axvline(x=idx, color='red', alpha=0.3, linewidth=0.5)

        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("TB Structure Discovery: Model-Free vs Model-Based\n"
                 "Red lines = detected blanket variables", fontsize=13)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc TB analysis of DDPG+HER critic")
    parser.add_argument("--ddpg-dir", type=str,
                        default=os.path.join(PANDAS_DIR, 'data', 'ddpg_her_fetchpush'))
    parser.add_argument("--ensemble-dir", type=str,
                        default=os.path.join(PANDAS_DIR, 'data', 'fetchpush_50step'))
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of gradient samples to collect")
    args = parser.parse_args()

    print("=" * 60)
    print("US-106: Post-hoc TB Analysis of DDPG+HER Critic")
    print("=" * 60)

    # ── Load DDPG model ─────────────────────────────────────────────────
    print("\n1. Loading DDPG+HER model...")
    from stable_baselines3 import DDPG
    import gymnasium as gym

    ddpg_path = os.path.join(args.ddpg_dir, 'final_model.zip')
    if not os.path.exists(ddpg_path):
        best_path = os.path.join(args.ddpg_dir, 'best_model', 'best_model.zip')
        if os.path.exists(best_path):
            ddpg_path = best_path
        else:
            print(f"  ERROR: No DDPG model found at {ddpg_path} or {best_path}")
            sys.exit(1)

    env = gym.make("FetchPush-v4", max_episode_steps=50)
    ddpg_model = DDPG.load(ddpg_path, env=env)
    print(f"  Loaded from {ddpg_path}")

    # ── Collect critic gradients ────────────────────────────────────────
    print(f"\n2. Collecting {args.n_samples} critic gradients (finite differences)...")
    critic_grads, critic_obs = collect_critic_gradients_finite_diff(
        ddpg_model, env, n_samples=args.n_samples)
    print(f"  Gradient shape: {critic_grads.shape}")
    print(f"  Gradient magnitude: mean={np.linalg.norm(critic_grads, axis=1).mean():.4f}")

    # ── Run TB on critic gradients ──────────────────────────────────────
    print("\n3. Running TB on critic gradients...")
    results_critic, coupling_critic, blankets_critic, assign_critic = \
        run_tb_analysis(critic_grads, label="DDPG_Critic")

    # ── Load ensemble and run TB for comparison ─────────────────────────
    print("\n4. Loading Bayes ensemble for comparison...")
    results_ensemble = None
    coupling_ensemble = None
    blankets_ensemble = None
    assign_ensemble = None

    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        sys.path.insert(0, PANDAS_DIR)
        from panda.model import EnsembleModel, make_model, ModelConfig
        from panda.utils import Normalizer

        meta_path = os.path.join(args.ensemble_dir, 'model.eqx.json')
        model_path = os.path.join(args.ensemble_dir, 'model.eqx')

        if os.path.exists(meta_path) and os.path.exists(model_path):
            with open(meta_path) as f:
                meta = json.load(f)
            config = ModelConfig(
                obs_dim=meta['obs_dim'],
                action_dim=meta['action_dim'],
                achieved_goal_dim=meta['achieved_goal_dim'],
                ensemble_size=meta['ensemble_size'],
                hidden_size=meta['hidden_size'],
                depth=meta['depth'],
            )
            key = jax.random.PRNGKey(0)
            model_template = make_model(config, key)
            ensemble = eqx.tree_deserialise_leaves(model_path, model_template)
            print(f"  Loaded ensemble from {args.ensemble_dir}")

            # Collect Jacobian-based gradients from ensemble
            print(f"  Collecting {args.n_samples} ensemble Jacobians...")
            ens_grads = []
            for obs_vec in critic_obs[:args.n_samples]:
                action = np.random.randn(meta['action_dim']).astype(np.float32) * 0.1
                ag = obs_vec[3:6].astype(np.float32)  # object pos as achieved goal
                x = jnp.concatenate([jnp.array(obs_vec), jnp.array(ag),
                                     jnp.array(action)])

                def predict_fn(obs_part):
                    full = jnp.concatenate([obs_part, jnp.array(ag),
                                           jnp.array(action)])
                    preds = []
                    for member in ensemble.members:
                        pred = member(full)
                        preds.append(pred)
                    return jnp.stack(preds).mean(axis=0)

                jac = jax.jacobian(predict_fn)(jnp.array(obs_vec))
                # Flatten Jacobian to get gradient-like features
                ens_grads.append(np.array(jac).flatten()[:meta['obs_dim']])

            ens_grads = np.array(ens_grads)
            if ens_grads.shape[1] >= 25:
                ens_grads = ens_grads[:, :25]
            print(f"  Ensemble gradient shape: {ens_grads.shape}")

            results_ensemble, coupling_ensemble, blankets_ensemble, assign_ensemble = \
                run_tb_analysis(ens_grads, label="Bayes_Ensemble")
        else:
            print(f"  Ensemble checkpoint not found at {args.ensemble_dir}")
    except ImportError as e:
        print(f"  Cannot load ensemble (JAX/Equinox not available): {e}")
    except Exception as e:
        print(f"  Ensemble analysis failed: {e}")

    # ── Comparison ──────────────────────────────────────────────────────
    print("\n5. Comparison summary:")
    print(f"  {'Metric':<20} {'DDPG Critic':>15} {'Bayes Ensemble':>15}")
    print(f"  {'-'*50}")
    print(f"  {'ARI':<20} {results_critic['ari']:>15.4f}"
          f" {results_ensemble['ari'] if results_ensemble else 'N/A':>15}")
    print(f"  {'NMI':<20} {results_critic['nmi']:>15.4f}"
          f" {results_ensemble['nmi'] if results_ensemble else 'N/A':>15}")
    print(f"  {'Blanket F1':<20} {results_critic['blanket_f1']:>15.4f}"
          f" {results_ensemble['blanket_f1'] if results_ensemble else 'N/A':>15}")
    print(f"  {'N objects':<20} {results_critic['n_objects']:>15}"
          f" {results_ensemble['n_objects'] if results_ensemble else 'N/A':>15}")
    print(f"  {'N blanket vars':<20} {results_critic['n_blanket_vars']:>15}"
          f" {results_ensemble['n_blanket_vars'] if results_ensemble else 'N/A':>15}")

    # ── Visualization ───────────────────────────────────────────────────
    if coupling_ensemble is not None:
        print("\n6. Generating comparison figure...")
        fig = plot_comparison(
            coupling_critic, coupling_ensemble,
            blankets_critic, blankets_ensemble,
            assign_critic, assign_ensemble,
            results_critic, results_ensemble)
        save_figure(fig, "tb_model_free_vs_model_based", "us106_ddpg_critic_tb")
        plt.close(fig)

    # ── Save results ────────────────────────────────────────────────────
    combined_results = {
        'ddpg_critic': results_critic,
        'bayes_ensemble': results_ensemble,
        'n_gradient_samples': args.n_samples,
        'obs_labels': FETCHPUSH_OBS_LABELS,
        'ground_truth': {k: v for k, v in GROUND_TRUTH.items()},
    }
    save_results("us106_ddpg_critic_tb_analysis", combined_results,
                 config={'ddpg_dir': args.ddpg_dir, 'ensemble_dir': args.ensemble_dir},
                 notes="Post-hoc TB analysis of DDPG+HER critic vs Bayes ensemble. "
                       "Tests whether model-free RL learns implicit Markov blanket structure.")

    print("\nDone.")
    env.close()


if __name__ == "__main__":
    main()
