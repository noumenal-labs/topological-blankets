"""
US-078: Structure Emergence During Training — TB Checkpoint Analysis
====================================================================

Wraps the pandas training loop to save checkpoints at regular intervals,
then runs TB analysis on each checkpoint. This produces the "structure
crystallization" timelapse: how the coupling matrix transitions from
dense (random model, no structure) to sparse/block-diagonal (trained
model, clear gripper-object separation).

The script:
  1. Trains a fresh ensemble on FetchPush-v4 from scratch.
  2. Saves a checkpoint every `--checkpoint-every` iterations.
  3. After training, runs TB on each checkpoint's Jacobians.
  4. Plots training curves overlaid with TB structural metrics:
     - Eigengap vs iteration (structure sharpness)
     - Number of detected objects vs iteration
     - Blanket F1 vs iteration (structure correctness)
     - Coupling matrix sparsity vs iteration
     - Ensemble prediction loss vs iteration
  5. Generates a coupling matrix timelapse figure.

This is the most compelling visual for the info-thermodynamic selection
narrative: the agent self-organizes into a structured representation,
and TB detects the emergence of that structure geometrically.

Usage (on GPU):
    python train_with_tb_checkpoints.py --iterations 100 --checkpoint-every 10

Output saved to pandas/data/fetchpush_emergence/
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
sys.path.insert(0, PANDAS_DIR)

FETCHPUSH_OBS_LABELS = [
    'grip_x', 'grip_y', 'grip_z',
    'obj_x', 'obj_y', 'obj_z',
    'rel_x', 'rel_y', 'rel_z',
    'finger_L', 'finger_R',
    'obj_rot_x', 'obj_rot_y', 'obj_rot_z',
    'obj_vx', 'obj_vy', 'obj_vz',
    'obj_wx', 'obj_wy', 'obj_wz',
    'grip_vx', 'grip_vy',
    'extra_0', 'extra_1', 'extra_2',
]

GROUND_TRUTH_BLANKET = [6, 7, 8]  # rel_x, rel_y, rel_z


def analyze_checkpoint(checkpoint_dir, iteration, obs_data):
    """Run TB on a single checkpoint and return metrics."""
    from topological_blankets import TopologicalBlankets
    from sklearn.metrics import f1_score

    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        from panda.model import make_model, ModelConfig

        meta_path = os.path.join(checkpoint_dir, 'model.eqx.json')
        model_path = os.path.join(checkpoint_dir, 'model.eqx')

        if not os.path.exists(model_path):
            return None

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

        # Collect Jacobians
        n_samples = min(500, len(obs_data))
        grads = []
        for obs_vec in obs_data[:n_samples]:
            action = np.random.randn(meta['action_dim']).astype(np.float32) * 0.1
            ag = obs_vec[3:6].astype(np.float32)

            def predict_fn(obs_part):
                full = jnp.concatenate([obs_part, jnp.array(ag),
                                       jnp.array(action)])
                preds = jnp.stack([m(full) for m in ensemble.members])
                return preds.mean(axis=0)

            jac = jax.jacobian(predict_fn)(jnp.array(obs_vec))
            grads.append(np.array(jac).flatten()[:meta['obs_dim']])

        grads = np.array(grads)

        # Run TB
        tb = TopologicalBlankets(method='hybrid', n_objects=None)
        tb.fit(grads)

        blankets = tb.get_blankets()
        assignment = tb.get_assignment()
        coupling = tb.get_coupling_matrix()
        info = tb.get_detection_info()
        n_objects = len(set(assignment[~blankets]))

        # Blanket F1
        gt_blanket = np.zeros(len(blankets), dtype=int)
        for idx in GROUND_TRUTH_BLANKET:
            if idx < len(gt_blanket):
                gt_blanket[idx] = 1
        blanket_f1 = f1_score(gt_blanket, blankets.astype(int), zero_division=0.0)

        # Coupling sparsity (fraction of near-zero entries)
        abs_coupling = np.abs(coupling)
        threshold = 0.1 * abs_coupling.max() if abs_coupling.max() > 0 else 0.1
        sparsity = (abs_coupling < threshold).sum() / abs_coupling.size

        eigengap = float(info.get('eigengap', 0.0))

        return {
            'iteration': iteration,
            'n_objects': n_objects,
            'n_blanket_vars': int(blankets.sum()),
            'blanket_indices': np.where(blankets)[0].tolist(),
            'blanket_f1': round(blanket_f1, 4),
            'eigengap': round(eigengap, 4),
            'coupling_sparsity': round(float(sparsity), 4),
            'coupling_matrix': coupling.tolist(),
        }

    except Exception as e:
        print(f"    Error analyzing iteration {iteration}: {e}")
        return None


def plot_emergence(metrics_list, training_log, out_dir):
    """Plot training curves with structural metrics overlay."""
    iters = [m['iteration'] for m in metrics_list]
    eigengaps = [m['eigengap'] for m in metrics_list]
    n_objects = [m['n_objects'] for m in metrics_list]
    blanket_f1 = [m['blanket_f1'] for m in metrics_list]
    sparsity = [m['coupling_sparsity'] for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Eigengap (structure sharpness)
    ax = axes[0, 0]
    ax.plot(iters, eigengaps, 'b-o', markersize=4, linewidth=2)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Eigengap', color='b')
    ax.set_title('Structure Sharpness (Eigengap)')
    ax.grid(True, alpha=0.3)

    # Top right: Blanket F1 (structure correctness)
    ax = axes[0, 1]
    ax.plot(iters, blanket_f1, 'r-o', markersize=4, linewidth=2)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Blanket F1', color='r')
    ax.set_title('Blanket Detection Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom left: N objects + coupling sparsity
    ax = axes[1, 0]
    ax.plot(iters, n_objects, 'g-s', markersize=4, linewidth=2, label='N objects')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('N objects', color='g')
    ax.set_title('Partition Complexity')
    ax2 = ax.twinx()
    ax2.plot(iters, sparsity, 'purple', linestyle='--', linewidth=1.5,
             label='Coupling sparsity')
    ax2.set_ylabel('Coupling sparsity', color='purple')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom right: Training loss if available
    ax = axes[1, 1]
    if training_log:
        train_iters = [e['iteration'] for e in training_log]
        train_loss = [e.get('test_loss', e.get('train_loss', 0)) for e in training_log]
        success = [e.get('success_rate', 0) for e in training_log]
        if any(v > 0 for v in train_loss):
            ax.plot(train_iters, train_loss, 'k-', linewidth=1.5, label='Test loss')
            ax.set_ylabel('Prediction loss', color='k')
        if any(v > 0 for v in success):
            ax3 = ax.twinx()
            ax3.plot(train_iters, success, 'orange', linewidth=1.5, label='Success rate')
            ax3.set_ylabel('Success rate', color='orange')
            ax3.legend(loc='upper right')
        ax.legend(loc='upper left')
    ax.set_xlabel('Training Iteration')
    ax.set_title('Training Performance')
    ax.grid(True, alpha=0.3)

    fig.suptitle('TB Structure Emergence During Ensemble Training\n'
                 'FetchPush-v4 | 5-member Bayes ensemble | 50-step episodes',
                 fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'structure_emergence_curves.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved training curves: {out_path}")
    return out_path


def plot_coupling_timelapse(metrics_list, out_dir):
    """Plot coupling matrix at each checkpoint as a timelapse grid."""
    n = len(metrics_list)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
        axes = np.array(axes).reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, m in enumerate(metrics_list):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        coupling = np.array(m['coupling_matrix'])
        im = ax.imshow(np.abs(coupling), cmap='viridis', aspect='equal',
                       vmin=0, vmax=max(0.1, np.abs(coupling).max()))
        ax.set_title(f"Iter {m['iteration']}\n"
                     f"eigengap={m['eigengap']:.1f} F1={m['blanket_f1']:.2f}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle('Coupling Matrix Evolution During Training\n'
                 'Dense (random) → Sparse/Block-diagonal (trained)',
                 fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'coupling_matrix_timelapse.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved coupling timelapse: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble with TB checkpoint analysis")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PANDAS_DIR, 'data', 'fetchpush_emergence'))
    parser.add_argument("--n-obs-samples", type=int, default=500,
                        help="Observation samples for TB analysis per checkpoint")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("US-078: Structure Emergence During Training")
    print("=" * 60)
    print(f"  Iterations: {args.iterations}")
    print(f"  Checkpoint every: {args.checkpoint_every} iterations")
    print(f"  Output: {out_dir}")

    # ── Phase 1: Train with periodic checkpoints ────────────────────────
    # We run the training in a subprocess and periodically copy checkpoints
    print("\nPhase 1: Training with periodic checkpoints...")

    train_cmd = [
        sys.executable, os.path.join(PANDAS_DIR, 'train.py'),
        '--env-id', 'FetchPush-v4',
        '--max-episode-steps', '50',
        '--reward-mode', 'dense',
        '--ensemble-size', '5',
        '--run-dir', str(out_dir / 'live'),
        '--iterations', str(args.iterations),
        '--symbolic-task', 'push',
        '--no-train-use-epistemic-bonus',
        '--no-eval-use-epistemic-bonus',
        '--no-use-wandb',
    ]

    live_dir = out_dir / 'live'
    live_dir.mkdir(exist_ok=True)

    # Start training as subprocess
    t0 = time.time()
    proc = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    training_log = []
    last_checkpoint_iter = -1
    current_iter = 0

    log_file = open(out_dir / 'training_full.log', 'w')

    try:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()

            # Parse iteration number
            if 'iteration' in line and '/' in line:
                try:
                    parts = line.strip().split('|')[0].strip()
                    iter_str = parts.split('/')[0].split()[-1]
                    current_iter = int(iter_str)
                except (ValueError, IndexError):
                    pass

            # Parse training loss
            if 'test' in line and 'train_raw' not in line:
                try:
                    for part in line.strip().split('|'):
                        if 'test' in part:
                            test_loss = float(part.split()[-1])
                            training_log.append({
                                'iteration': current_iter,
                                'test_loss': test_loss,
                            })
                except (ValueError, IndexError):
                    pass

            # Parse success info
            if 'successful episodes' in line:
                try:
                    parts = line.strip().split('|')
                    for part in parts:
                        if 'successful episodes' in part:
                            n_success = int(part.split()[-1])
                        if 'episodes' in part and 'successful' not in part:
                            n_episodes = int(part.split()[-1])
                    rate = n_success / max(1, n_episodes)
                    if training_log and training_log[-1]['iteration'] == current_iter:
                        training_log[-1]['success_rate'] = rate
                    else:
                        training_log.append({
                            'iteration': current_iter,
                            'success_rate': rate,
                        })
                except (ValueError, IndexError):
                    pass

            # Save checkpoint at intervals
            if current_iter > last_checkpoint_iter and \
               current_iter % args.checkpoint_every == 0:
                ckpt_dir = checkpoints_dir / f'iter_{current_iter:04d}'
                ckpt_dir.mkdir(exist_ok=True)
                # Copy current model files
                for fname in ['model.eqx', 'model.eqx.json', 'run_config.json']:
                    src = live_dir / fname
                    if src.exists():
                        shutil.copy2(str(src), str(ckpt_dir / fname))
                last_checkpoint_iter = current_iter
                elapsed = time.time() - t0
                print(f"  Checkpoint saved: iter {current_iter} ({elapsed:.0f}s elapsed)")

    except KeyboardInterrupt:
        proc.terminate()
        print("\nTraining interrupted.")
    finally:
        log_file.close()
        proc.wait()

    elapsed_total = time.time() - t0
    print(f"\nTraining complete in {elapsed_total:.0f}s")

    # Save final checkpoint
    ckpt_dir = checkpoints_dir / f'iter_{args.iterations:04d}'
    ckpt_dir.mkdir(exist_ok=True)
    for fname in ['model.eqx', 'model.eqx.json', 'run_config.json']:
        src = live_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(ckpt_dir / fname))

    # ── Phase 2: Collect observation data ───────────────────────────────
    print("\nPhase 2: Collecting observation data for TB analysis...")
    import gymnasium as gym
    env = gym.make("FetchPush-v4", max_episode_steps=50)
    obs_data = []
    obs, _ = env.reset()
    for _ in range(args.n_obs_samples):
        action = env.action_space.sample()
        obs_data.append(obs['observation'].copy())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    obs_data = np.array(obs_data, dtype=np.float32)
    env.close()
    print(f"  Collected {len(obs_data)} observation samples")

    # ── Phase 3: TB analysis on each checkpoint ─────────────────────────
    print("\nPhase 3: Running TB analysis on each checkpoint...")
    checkpoint_dirs = sorted(checkpoints_dir.iterdir())
    metrics_list = []

    for ckpt_dir in checkpoint_dirs:
        if not ckpt_dir.is_dir():
            continue
        iter_num = int(ckpt_dir.name.split('_')[1])
        print(f"  Analyzing checkpoint iter_{iter_num:04d}...")
        result = analyze_checkpoint(str(ckpt_dir), iter_num, obs_data)
        if result is not None:
            metrics_list.append(result)

    # Also analyze iteration 0 (random model) if we have it
    print(f"\n  Analyzed {len(metrics_list)} checkpoints")

    # ── Phase 4: Visualization ──────────────────────────────────────────
    print("\nPhase 4: Generating visualizations...")
    if metrics_list:
        curves_path = plot_emergence(metrics_list, training_log, str(out_dir))
        timelapse_path = plot_coupling_timelapse(metrics_list, str(out_dir))

    # ── Phase 5: Save results ───────────────────────────────────────────
    results = {
        'checkpoints': metrics_list,
        'training_log': training_log,
        'total_training_time_s': round(elapsed_total, 1),
        'config': {
            'iterations': args.iterations,
            'checkpoint_every': args.checkpoint_every,
            'n_obs_samples': args.n_obs_samples,
        }
    }
    results_path = out_dir / 'structure_emergence_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Summary
    if metrics_list:
        first = metrics_list[0]
        last = metrics_list[-1]
        print(f"\n{'='*60}")
        print(f"Structure Emergence Summary")
        print(f"{'='*60}")
        print(f"  Iteration {first['iteration']} -> {last['iteration']}:")
        print(f"    Eigengap:        {first['eigengap']:.2f} -> {last['eigengap']:.2f}")
        print(f"    Blanket F1:      {first['blanket_f1']:.3f} -> {last['blanket_f1']:.3f}")
        print(f"    N objects:       {first['n_objects']} -> {last['n_objects']}")
        print(f"    Coupling sparsity: {first['coupling_sparsity']:.3f} -> {last['coupling_sparsity']:.3f}")
        print(f"    Blanket vars:    {first['blanket_indices']} -> {last['blanket_indices']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
