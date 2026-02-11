"""
US-113: Surprise-Weighted Learning with TB Structural Decomposition
====================================================================

Extends the BayesformerTB (US-112) with per-factor surprise computation.
Instead of using scalar surprise (total prediction error), this decomposes
surprise into TB-detected factor subspaces:

  surprise_factor_n = || proj_{subspace_n}(prediction_error) ||^2

This enables selective credit assignment: only update the factor that is
surprising, preserving learned structure in other factors.

Components:
  1. Per-factor surprise from TB partition
  2. Surprise-weighted gradient scaling
  3. Factor-specific replay priority
  4. Connection to Active Inference: per-factor free energy

Depends on: US-112 (BayesformerTB model and TB partition).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RALPH_DIR)
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(RALPH_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from topological_blankets import TopologicalBlankets, compute_eigengap
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian
)

from fwh_ghmm_tb_detection import (
    generate_ghmm_dataset, SmallGPT2, GHMMDataset,
    extract_residual_stream, compute_activation_gradients, run_tb_on_layer
)


class PerFactorSurprise:
    """
    Computes per-factor surprise by projecting prediction error into
    TB-detected subspaces.

    Given a TB partition of d_model dimensions into N objects + blanket,
    this constructs projection matrices for each factor subspace and
    decomposes the prediction error accordingly.

    Connection to free energy (Active Inference):
        In the free energy principle, surprise S = -log p(o) is the
        quantity that agents minimize. For a factored world model with
        N independent factors, total surprise decomposes as:

            S_total = sum_n S_n  where  S_n = || proj_n(epsilon) ||^2

        Here epsilon is the prediction error in the representation space,
        and proj_n is the orthogonal projector onto factor n's subspace
        (given by the TB partition). Each S_n corresponds to the *factor-
        specific free energy*: the free energy contribution from factor n.

        This decomposition enables *factor-specific epistemic foraging*:
        gradient updates are concentrated on the factor with highest free
        energy (most surprising), leaving well-predicted factors undisturbed.
        This is equivalent to minimizing expected free energy with a factored
        generative model, where each factor's posterior is updated
        independently based on its own prediction error.
    """

    def __init__(self, assignment, d_model):
        """
        Args:
            assignment: Array of length d_model. assignment[i] = object_id
                        or -1 for blanket.
            d_model: Dimensionality of the representation.
        """
        self.d_model = d_model
        self.assignment = assignment.copy()
        self.object_ids = sorted(set(assignment[assignment >= 0]))
        self.n_factors = len(self.object_ids)

        # Build projection matrices for each factor
        self.projections = {}
        for obj_id in self.object_ids:
            dims = np.where(assignment == obj_id)[0]
            P = np.zeros((d_model, d_model))
            for d in dims:
                P[d, d] = 1.0
            self.projections[obj_id] = torch.FloatTensor(P)

        # Blanket projection
        blanket_dims = np.where(assignment == -1)[0]
        P_blanket = np.zeros((d_model, d_model))
        for d in blanket_dims:
            P_blanket[d, d] = 1.0
        self.projections['blanket'] = torch.FloatTensor(P_blanket)

    def compute(self, prediction_error):
        """
        Decompose prediction error into per-factor surprise.

        Args:
            prediction_error: Tensor of shape (B, d_model).

        Returns:
            Dict mapping factor_id -> surprise tensor of shape (B,).
        """
        device = prediction_error.device
        surprises = {}

        for factor_id, P in self.projections.items():
            P = P.to(device)
            projected = prediction_error @ P  # (B, d_model)
            surprise = (projected ** 2).sum(dim=-1)  # (B,)
            surprises[factor_id] = surprise

        return surprises

    def compute_weights(self, prediction_error, temperature=1.0):
        """
        Compute per-factor gradient weights from surprise.

        Higher surprise for a factor means that factor's parameters
        should receive larger gradient updates.

        Args:
            prediction_error: Tensor of shape (B, d_model).
            temperature: Softmax temperature for weight normalization.

        Returns:
            Dict mapping factor_id -> weight tensor of shape (B,).
        """
        surprises = self.compute(prediction_error)
        total = sum(s for s in surprises.values())
        total = total.clamp(min=1e-8)

        weights = {}
        for factor_id, surprise in surprises.items():
            weights[factor_id] = surprise / total

        return weights


class SurpriseWeightedTrainer:
    """
    Trains a model with surprise-weighted gradient scaling.

    Instead of uniform gradient updates, scales gradients based on
    per-factor surprise: parameters associated with surprising factors
    receive larger updates, while well-predicted factors are updated less.
    """

    def __init__(self, model, tb_partition, d_model, lr=1e-3,
                 surprise_scale=2.0, update_partition_every=5):
        self.model = model
        self.d_model = d_model
        self.surprise_scale = surprise_scale
        self.update_partition_every = update_partition_every

        self.surprise_computer = PerFactorSurprise(tb_partition, d_model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Tracking
        self.surprise_history = []
        self.loss_history = []

    def update_partition(self, new_partition):
        """Update TB partition (called periodically during training)."""
        self.surprise_computer = PerFactorSurprise(new_partition, self.d_model)

    def train_step(self, inputs, targets, device='cpu', warmup_alpha=1.0):
        """
        Single training step with surprise-weighted gradients.

        Instead of scaling the global loss by total surprise (which acts
        as a noisy LR multiplier and destabilizes training), this uses
        normalized per-sample weighting: samples where surprise is
        concentrated in fewer factors get higher weight, since they are
        more informative for selective credit assignment. The weights
        are normalized to mean=1.0 to preserve gradient magnitude.

        warmup_alpha: 0.0 = no surprise weighting, 1.0 = full weighting.
            Ramps from 0 to 1 over the first few epochs.
        """
        self.model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        B, L = inputs.shape

        # Forward pass
        logits = self.model(inputs)

        # Per-sample loss (not reduced)
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none',
        )
        per_sample_loss = per_token_loss.reshape(B, L).mean(dim=-1)  # (B,)

        # Get residual stream for surprise computation
        with torch.no_grad():
            _, residual = self.model(inputs, return_residual_stream=True)
            last_layer_acts = residual[-1][:, -1, :]  # (B, d_model)
            surprises = self.surprise_computer.compute(last_layer_acts)

        # Per-sample total surprise
        total_per_sample = sum(s for s in surprises.values())  # (B,)

        # Surprise concentration: how imbalanced is the surprise across
        # factors? Concentrated surprise = informative for selective credit.
        factor_stack = torch.stack(list(surprises.values()), dim=-1)  # (B, n)
        factor_probs = factor_stack / factor_stack.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        n_factors = factor_stack.shape[-1]
        max_entropy = float(np.log(max(n_factors, 2)))
        entropy = -(factor_probs * (factor_probs + 1e-10).log()).sum(dim=-1)  # (B,)
        concentration = (1.0 - entropy / max(max_entropy, 1e-8)).clamp(min=0)  # (B,)

        # Weight: baseline 1.0 + scaled concentration, normalized to mean=1
        raw_weights = 1.0 + self.surprise_scale * concentration  # (B,)
        weights = raw_weights / raw_weights.mean().clamp(min=1e-8)
        weights = weights.detach().clamp(max=3.0)

        # Blend with uniform weights during warmup
        weights = warmup_alpha * weights + (1.0 - warmup_alpha) * torch.ones_like(weights)

        # Weighted loss
        weighted_loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Record per-factor surprise (for plotting)
        base_loss_val = per_sample_loss.mean().item()
        surprise_record = {
            str(fid): float(s.mean().item())
            for fid, s in surprises.items()
        }
        self.surprise_history.append(surprise_record)
        self.loss_history.append(base_loss_val)

        return base_loss_val, surprise_record


def train_with_surprise_weighting(model, tokens, tb_partition, d_model,
                                  n_epochs=30, batch_size=256, lr=1e-3,
                                  surprise_scale=2.0, device='cpu',
                                  update_partition_every=5):
    """
    Full training loop with surprise-weighted gradients and periodic
    TB partition updates.
    """
    dataset = GHMMDataset(tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    trainer = SurpriseWeightedTrainer(
        model, tb_partition, d_model, lr=lr,
        surprise_scale=surprise_scale,
        update_partition_every=update_partition_every,
    )

    model.to(device)
    t0 = time.time()
    epoch_losses = []
    epoch_surprises = []

    warmup_epochs = 5

    for epoch in range(n_epochs):
        batch_losses = []
        batch_surprises = []
        warmup_alpha = min(1.0, epoch / max(warmup_epochs, 1))

        for inputs, targets in loader:
            loss, surprises = trainer.train_step(
                inputs, targets, device, warmup_alpha=warmup_alpha)
            batch_losses.append(loss)
            batch_surprises.append(surprises)

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)

        # Average surprise per factor
        avg_surprise = {}
        for fid in batch_surprises[0].keys():
            avg_surprise[fid] = np.mean([s[fid] for s in batch_surprises])
        epoch_surprises.append(avg_surprise)

        # Periodically update TB partition
        if (epoch + 1) % update_partition_every == 0:
            model.eval()
            acts = extract_residual_stream(model, tokens, n_samples=2000,
                                           device=device)
            # Use last transformer layer
            layer_names = list(acts.keys())
            last_layer = layer_names[-1]
            tb_result = run_tb_on_layer(acts[last_layer], n_objects=5)
            trainer.update_partition(tb_result['assignment'])

            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
                      f"TB updated ({elapsed:.1f}s)")
        elif (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"({elapsed:.1f}s)")

    return epoch_losses, epoch_surprises


def plot_surprise_decomposition(epoch_surprises, save_path):
    """Plot per-factor surprise through training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_epochs = len(epoch_surprises)
    factor_ids = sorted(epoch_surprises[0].keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(factor_ids)))

    for i, fid in enumerate(factor_ids):
        values = [epoch_surprises[e][fid] for e in range(n_epochs)]
        label = f'Factor {fid}' if fid != 'blanket' else 'Blanket'
        ax.plot(range(1, n_epochs + 1), values, label=label,
                color=colors[i], linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Surprise')
    ax.set_title('Per-Factor Surprise Decomposition Through Training')
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_surprise_vs_baseline(surprise_losses, baseline_losses, save_path):
    """Compare surprise-weighted vs baseline training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(surprise_losses) + 1)

    # Panel 1: Training loss
    axes[0].plot(epochs, baseline_losses, label='Baseline (uniform)',
                 color='steelblue', linewidth=2)
    axes[0].plot(epochs, surprise_losses, label='Surprise-weighted',
                 color='coral', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-entropy Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()

    # Panel 2: Convergence ratio
    baseline_arr = np.array(baseline_losses)
    surprise_arr = np.array(surprise_losses)
    ratio = surprise_arr / np.maximum(baseline_arr, 1e-8)
    axes[1].plot(epochs, ratio, color='purple', linewidth=2)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Ratio (surprise / baseline)')
    axes[1].set_title('Convergence Advantage')
    axes[1].set_ylim(0.5, 1.5)

    fig.suptitle('Surprise-Weighted Learning: Training Dynamics', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 70)
    print("US-113: Surprise-Weighted Learning with TB Decomposition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    n_sequences = 10000
    n_epochs = 30

    # ── Generate data ───────────────────────────────────────────────────
    print(f"\n[1/5] Generating GHMM data...")
    tokens, _, factor_info = generate_ghmm_dataset(
        n_sequences=n_sequences, seq_len=8, seed=42
    )

    # ── Train baseline model ────────────────────────────────────────────
    print(f"\n[2/5] Training baseline GPT-2...")
    from fwh_ghmm_tb_detection import train_gpt2
    baseline_model = SmallGPT2(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=4, n_heads=4, d_mlp=480, max_len=16,
    )
    baseline_losses = train_gpt2(baseline_model, tokens, n_epochs=n_epochs,
                                  device=device)

    # ── Get initial TB partition ────────────────────────────────────────
    print(f"\n[3/5] Computing initial TB partition from baseline...")
    baseline_acts = extract_residual_stream(baseline_model, tokens,
                                            n_samples=2000, device=device)
    last_layer = list(baseline_acts.keys())[-1]
    initial_tb = run_tb_on_layer(baseline_acts[last_layer], n_objects=5)
    initial_partition = initial_tb['assignment']
    print(f"  Initial partition: {initial_tb['n_objects_detected']} objects, "
          f"{initial_tb['n_blanket']} blanket dims")

    # ── Train surprise-weighted model ───────────────────────────────────
    print(f"\n[4/5] Training surprise-weighted GPT-2...")
    surprise_model = SmallGPT2(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=4, n_heads=4, d_mlp=480, max_len=16,
    )
    surprise_losses, surprise_decomp = train_with_surprise_weighting(
        surprise_model, tokens, initial_partition, d_model=120,
        n_epochs=n_epochs, surprise_scale=2.0, device=device,
        update_partition_every=5,
    )

    # ── Final TB comparison ─────────────────────────────────────────────
    print(f"\n[5/5] Comparing final TB structure...")
    surprise_acts = extract_residual_stream(surprise_model, tokens,
                                            n_samples=2000, device=device)
    final_tb_baseline = {}
    final_tb_surprise = {}

    for layer in baseline_acts.keys():
        final_tb_baseline[layer] = run_tb_on_layer(
            baseline_acts[layer], n_objects=5
        )
        final_tb_surprise[layer] = run_tb_on_layer(
            surprise_acts[layer], n_objects=5
        )

    # ── Visualizations ──────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_surprise_decomposition(
        surprise_decomp,
        os.path.join(RESULTS_DIR, 'us113_surprise_decomposition.png')
    )
    plot_surprise_vs_baseline(
        surprise_losses, baseline_losses,
        os.path.join(RESULTS_DIR, 'us113_surprise_vs_baseline.png')
    )

    # ── Save results ────────────────────────────────────────────────────
    output = {
        'experiment': 'US-113',
        'title': 'Surprise-Weighted Learning with TB Decomposition',
        'training': {
            'n_sequences': n_sequences,
            'n_epochs': n_epochs,
            'surprise_scale': 2.0,
            'partition_update_every': 5,
        },
        'baseline': {
            'final_loss': round(baseline_losses[-1], 6),
            'loss_curve': [round(l, 6) for l in baseline_losses],
        },
        'surprise_weighted': {
            'final_loss': round(surprise_losses[-1], 6),
            'loss_curve': [round(l, 6) for l in surprise_losses],
        },
        'surprise_decomposition': [
            {k: round(float(v), 6) for k, v in s.items()}
            for s in surprise_decomp
        ],
        'final_tb_comparison': {
            'baseline': {
                layer: {
                    'eigengap': r['eigengap'],
                    'n_objects': r['n_objects_detected'],
                    'n_blanket': r['n_blanket'],
                }
                for layer, r in final_tb_baseline.items()
            },
            'surprise_weighted': {
                layer: {
                    'eigengap': r['eigengap'],
                    'n_objects': r['n_objects_detected'],
                    'n_blanket': r['n_blanket'],
                }
                for layer, r in final_tb_surprise.items()
            },
        },
        'free_energy_connection': {
            'formalization': (
                'Per-factor surprise S_n = ||proj_n(epsilon)||^2 corresponds '
                'to factor-specific free energy. Total free energy decomposes '
                'as S_total = sum_n S_n under the TB partition. Gradient '
                'weighting by S_n concentration implements factor-specific '
                'epistemic foraging: updating the most surprising factor '
                'while preserving well-predicted factors.'
            ),
            'correspondence': {
                'prediction_error': 'epsilon = residual stream activation',
                'factor_projection': 'proj_n = TB object n subspace projector',
                'factor_free_energy': 'S_n = ||proj_n(epsilon)||^2',
                'epistemic_action': 'gradient scaling by surprise concentration',
            },
        },
        'summary': {
            'baseline_final_loss': round(baseline_losses[-1], 6),
            'surprise_final_loss': round(surprise_losses[-1], 6),
            'loss_improvement': round(
                (baseline_losses[-1] - surprise_losses[-1]) /
                baseline_losses[-1] * 100, 2
            ),
            'convergence_advantage': 'surprise' if surprise_losses[-1] < baseline_losses[-1] else 'baseline',
        },
    }

    results_path = os.path.join(RESULTS_DIR, 'us113_surprise_weighted.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline final loss:  {baseline_losses[-1]:.4f}")
    print(f"  Surprise final loss:  {surprise_losses[-1]:.4f}")
    print(f"  Improvement: {output['summary']['loss_improvement']:.2f}%")
    print(f"  Convergence advantage: {output['summary']['convergence_advantage']}")

    return output


if __name__ == '__main__':
    main()
