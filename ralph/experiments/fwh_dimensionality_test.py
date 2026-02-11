"""
US-109: FWH Dimensionality Test — TB Eigengap Tracks Factored-to-Joint Transition
==================================================================================

Tests whether TB detects the factored-to-joint representation transition described
by Shai et al. (2602.02385). Two conditions:

  1. Clean (eps=0): Standard GHMM with 5 conditionally independent factors.
     The model should converge to a factored representation; TB eigengap at
     position 5 should remain high throughout training.

  2. Corrupted (eps=0.2): 20% of tokens are replaced with random tokens,
     breaking conditional independence. The model cannot fully factor the
     representation; TB eigengap should decrease compared to clean.

TB eigengap is measured at 10 evenly-spaced training checkpoints. The
"factored attractor" phenomenon (Shai et al. Section 4) predicts that even
corrupted models dwell at the factored solution for a time before departing;
TB should capture this temporal profile.

Depends on: US-107 (GHMM data generation, GPT-2 model, TB analysis pipeline).
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

# Reuse from US-107
from fwh_ghmm_tb_detection import (
    generate_ghmm_dataset, SmallGPT2, GHMMDataset, train_gpt2,
    extract_residual_stream, compute_activation_gradients
)


def corrupt_tokens(tokens, eps=0.2, vocab_size=433, seed=123):
    """
    Corrupt GHMM tokens by replacing a fraction eps with random tokens.

    This breaks the conditional independence between factors: when a token
    is replaced randomly, its sub-token decomposition is no longer consistent
    with the generating GHMM. The model cannot learn a perfectly factored
    representation from this data.

    Args:
        tokens: Array (n_sequences, seq_len+1) with BOS at position 0.
        eps: Fraction of non-BOS tokens to corrupt.
        vocab_size: Vocabulary size (for sampling random tokens).
        seed: Random seed.

    Returns:
        corrupted: Copy of tokens with eps fraction replaced.
    """
    rng = np.random.default_rng(seed)
    corrupted = tokens.copy()

    # Only corrupt non-BOS positions
    n_seq, seq_len_plus_1 = corrupted.shape
    mask = rng.random((n_seq, seq_len_plus_1 - 1)) < eps
    random_tokens = rng.integers(1, vocab_size, size=(n_seq, seq_len_plus_1 - 1))
    corrupted[:, 1:] = np.where(mask, random_tokens, corrupted[:, 1:])

    n_corrupted = mask.sum()
    total = mask.size
    print(f"  Corrupted {n_corrupted}/{total} tokens ({n_corrupted/total*100:.1f}%)")

    return corrupted


def train_with_checkpoints(model, tokens, n_epochs=30, n_checkpoints=10,
                           batch_size=256, lr=1e-3, device='cpu'):
    """
    Train model and save checkpoints at evenly-spaced intervals.

    Returns:
        loss_history: List of per-epoch losses.
        checkpoints: List of (epoch, state_dict) tuples.
    """
    dataset = GHMMDataset(tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    checkpoint_epochs = set(
        int(round(i * (n_epochs - 1) / (n_checkpoints - 1)))
        for i in range(n_checkpoints)
    )
    # Always include first and last
    checkpoint_epochs.add(0)
    checkpoint_epochs.add(n_epochs - 1)

    loss_history = []
    checkpoints = []
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if epoch in checkpoint_epochs:
            checkpoints.append((epoch, {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }))
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"[checkpoint saved] ({elapsed:.1f}s)")
        elif (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"({elapsed:.1f}s)")

    return loss_history, checkpoints


def tb_eigengap_at_checkpoint(model, tokens, n_objects=5, n_samples=2000,
                              device='cpu'):
    """
    Compute TB eigengap at each layer for the current model state.

    Returns dict: layer_name -> {eigengap, gap_at_5, n_blanket, n_objects}
    """
    activations = extract_residual_stream(model, tokens, n_samples=n_samples,
                                          device=device)
    results = {}
    for name, acts in activations.items():
        gradients = compute_activation_gradients(acts)

        tb = TopologicalBlankets(method='hybrid', n_objects=n_objects)
        tb.fit(gradients)

        H = tb._features['hessian_est']
        A = build_adjacency_from_hessian(H)
        L_mat = build_graph_laplacian(A)
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_mat)))

        gaps = np.diff(eigenvalues[:min(30, len(eigenvalues))])
        gap_at_5 = float(gaps[4]) if len(gaps) > 4 else 0.0
        max_gap_idx, max_gap_val = compute_eigengap(eigenvalues)

        blanket_indices = tb.get_blankets()

        results[name] = {
            'eigengap': float(max_gap_val),
            'eigengap_index': int(max_gap_idx),
            'gap_at_5': gap_at_5,
            'n_blanket': int(len(blanket_indices)),
            'n_objects': len(tb.get_objects()),
            'eigenvalues_top10': eigenvalues[:10].tolist(),
        }

    return results


def plot_eigengap_trajectories(clean_trajectory, corrupt_trajectory,
                               clean_losses, corrupt_losses, save_path):
    """
    Plot eigengap trajectory through training for clean vs corrupted.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = ['embedding', 'layer_1', 'layer_2', 'layer_3', 'layer_4']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

    # Panel 1: Max eigengap trajectory (clean)
    for i, layer in enumerate(layers):
        epochs = [t['epoch'] for t in clean_trajectory]
        gaps = [t['tb'][layer]['eigengap'] for t in clean_trajectory]
        axes[0, 0].plot(epochs, gaps, label=layer, color=colors[i],
                        marker='o', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Max Eigengap')
    axes[0, 0].set_title('Clean GHMM (eps=0): Eigengap Trajectory')
    axes[0, 0].legend(fontsize=7)

    # Panel 2: Max eigengap trajectory (corrupted)
    for i, layer in enumerate(layers):
        epochs = [t['epoch'] for t in corrupt_trajectory]
        gaps = [t['tb'][layer]['eigengap'] for t in corrupt_trajectory]
        axes[0, 1].plot(epochs, gaps, label=layer, color=colors[i],
                        marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Max Eigengap')
    axes[0, 1].set_title('Corrupted GHMM (eps=0.2): Eigengap Trajectory')
    axes[0, 1].legend(fontsize=7)

    # Panel 3: Gap at position 5 comparison
    for i, layer in enumerate(layers):
        clean_gap5 = [t['tb'][layer]['gap_at_5'] for t in clean_trajectory]
        corrupt_gap5 = [t['tb'][layer]['gap_at_5'] for t in corrupt_trajectory]
        clean_epochs = [t['epoch'] for t in clean_trajectory]
        corrupt_epochs = [t['epoch'] for t in corrupt_trajectory]

        axes[1, 0].plot(clean_epochs, clean_gap5, color=colors[i],
                        linestyle='-', marker='o', markersize=3,
                        label=f'{layer} (clean)')
        axes[1, 0].plot(corrupt_epochs, corrupt_gap5, color=colors[i],
                        linestyle='--', marker='s', markersize=3,
                        label=f'{layer} (corrupt)')

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Eigengap at k=5')
    axes[1, 0].set_title('Eigengap at N=5 Factors: Clean (solid) vs Corrupted (dashed)')
    axes[1, 0].legend(fontsize=6, ncol=2)

    # Panel 4: Training loss comparison
    axes[1, 1].plot(range(len(clean_losses)), clean_losses, label='Clean (eps=0)',
                    color='steelblue')
    axes[1, 1].plot(range(len(corrupt_losses)), corrupt_losses,
                    label='Corrupted (eps=0.2)', color='coral')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cross-entropy Loss')
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].legend()

    fig.suptitle('FWH Dimensionality Test: Factored vs Joint Representation',
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_blanket_trajectory(clean_trajectory, corrupt_trajectory, save_path):
    """
    Plot blanket dimension count through training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = ['embedding', 'layer_1', 'layer_2', 'layer_3', 'layer_4']
    colors = plt.cm.Set2(np.linspace(0, 1, len(layers)))

    for ax, trajectory, title in [
        (axes[0], clean_trajectory, 'Clean (eps=0)'),
        (axes[1], corrupt_trajectory, 'Corrupted (eps=0.2)'),
    ]:
        for i, layer in enumerate(layers):
            epochs = [t['epoch'] for t in trajectory]
            blankets = [t['tb'][layer]['n_blanket'] for t in trajectory]
            ax.plot(epochs, blankets, label=layer, color=colors[i],
                    marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Blanket Dimensions')
        ax.set_title(f'{title}: Blanket Count Trajectory')
        ax.legend(fontsize=8)

    fig.suptitle('Blanket Variable Count Through Training', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 70)
    print("US-109: FWH Dimensionality Test")
    print("  TB Eigengap Tracks Factored-to-Joint Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    n_sequences = 10000
    n_epochs = 30
    n_checkpoints = 10

    # ── Generate clean GHMM data ────────────────────────────────────────
    print(f"\n[1/6] Generating clean GHMM data ({n_sequences} seqs)...")
    clean_tokens, _, factor_info = generate_ghmm_dataset(
        n_sequences=n_sequences, seq_len=8, seed=42
    )

    # ── Generate corrupted GHMM data ────────────────────────────────────
    print(f"\n[2/6] Generating corrupted GHMM data (eps=0.2)...")
    corrupted_tokens = corrupt_tokens(
        clean_tokens, eps=0.2, vocab_size=factor_info['vocab_size'], seed=123
    )

    # ── Train clean model with checkpoints ──────────────────────────────
    print(f"\n[3/6] Training clean model ({n_epochs} epochs, "
          f"{n_checkpoints} checkpoints)...")
    clean_model = SmallGPT2(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=4, n_heads=4, d_mlp=480, max_len=16,
    )
    clean_losses, clean_checkpoints = train_with_checkpoints(
        clean_model, clean_tokens, n_epochs=n_epochs,
        n_checkpoints=n_checkpoints, device=device,
    )

    # ── Train corrupted model with checkpoints ──────────────────────────
    print(f"\n[4/6] Training corrupted model ({n_epochs} epochs, "
          f"{n_checkpoints} checkpoints)...")
    corrupt_model = SmallGPT2(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=4, n_heads=4, d_mlp=480, max_len=16,
    )
    corrupt_losses, corrupt_checkpoints = train_with_checkpoints(
        corrupt_model, corrupted_tokens, n_epochs=n_epochs,
        n_checkpoints=n_checkpoints, device=device,
    )

    # ── TB analysis at each checkpoint ──────────────────────────────────
    print(f"\n[5/6] Running TB at each checkpoint...")

    clean_trajectory = []
    for epoch, state_dict in clean_checkpoints:
        print(f"\n  Clean checkpoint epoch {epoch+1}:")
        clean_model.load_state_dict(state_dict)
        clean_model.to(device)
        tb_result = tb_eigengap_at_checkpoint(
            clean_model, clean_tokens, device=device
        )
        clean_trajectory.append({
            'epoch': epoch + 1,
            'loss': clean_losses[epoch],
            'tb': tb_result,
        })
        # Print summary for best layer
        best_layer = max(tb_result.keys(), key=lambda k: tb_result[k]['gap_at_5'])
        print(f"    Best gap@5: {tb_result[best_layer]['gap_at_5']:.4f} "
              f"({best_layer})")

    corrupt_trajectory = []
    for epoch, state_dict in corrupt_checkpoints:
        print(f"\n  Corrupted checkpoint epoch {epoch+1}:")
        corrupt_model.load_state_dict(state_dict)
        corrupt_model.to(device)
        tb_result = tb_eigengap_at_checkpoint(
            corrupt_model, corrupted_tokens, device=device
        )
        corrupt_trajectory.append({
            'epoch': epoch + 1,
            'loss': corrupt_losses[epoch],
            'tb': tb_result,
        })
        best_layer = max(tb_result.keys(), key=lambda k: tb_result[k]['gap_at_5'])
        print(f"    Best gap@5: {tb_result[best_layer]['gap_at_5']:.4f} "
              f"({best_layer})")

    # ── Visualizations ──────────────────────────────────────────────────
    print(f"\n[6/6] Generating visualizations...")

    plot_eigengap_trajectories(
        clean_trajectory, corrupt_trajectory,
        clean_losses, corrupt_losses,
        os.path.join(RESULTS_DIR, 'us109_eigengap_trajectories.png')
    )
    plot_blanket_trajectory(
        clean_trajectory, corrupt_trajectory,
        os.path.join(RESULTS_DIR, 'us109_blanket_trajectories.png')
    )

    # ── Save results ────────────────────────────────────────────────────
    def trajectory_to_json(traj):
        return [{
            'epoch': t['epoch'],
            'loss': round(t['loss'], 6),
            'tb': {
                layer: {k: v for k, v in data.items()}
                for layer, data in t['tb'].items()
            },
        } for t in traj]

    output = {
        'experiment': 'US-109',
        'title': 'FWH Dimensionality Test: Factored-to-Joint Transition',
        'conditions': {
            'clean': {'eps': 0.0, 'description': 'Standard 5-factor GHMM'},
            'corrupt': {'eps': 0.2, 'description': '20% random token corruption'},
        },
        'model': {
            'architecture': 'GPT-2',
            'n_layers': 4,
            'd_model': 120,
            'n_params': sum(p.numel() for p in clean_model.parameters()),
        },
        'training': {
            'n_sequences': n_sequences,
            'n_epochs': n_epochs,
            'n_checkpoints': n_checkpoints,
            'clean_final_loss': round(clean_losses[-1], 6),
            'corrupt_final_loss': round(corrupt_losses[-1], 6),
        },
        'clean_trajectory': trajectory_to_json(clean_trajectory),
        'corrupt_trajectory': trajectory_to_json(corrupt_trajectory),
        'summary': {},
    }

    # Compute summary statistics
    clean_final = clean_trajectory[-1]['tb']
    corrupt_final = corrupt_trajectory[-1]['tb']

    layers = ['embedding', 'layer_1', 'layer_2', 'layer_3', 'layer_4']
    clean_gap5_final = {l: clean_final[l]['gap_at_5'] for l in layers}
    corrupt_gap5_final = {l: corrupt_final[l]['gap_at_5'] for l in layers}

    # Mean eigengap difference across layers
    gap_diffs = [clean_gap5_final[l] - corrupt_gap5_final[l] for l in layers]
    mean_gap_diff = float(np.mean(gap_diffs))

    output['summary'] = {
        'clean_gap5_final': clean_gap5_final,
        'corrupt_gap5_final': corrupt_gap5_final,
        'mean_gap5_difference': round(mean_gap_diff, 4),
        'clean_higher_gap5': mean_gap_diff > 0,
        'factored_attractor_detected': any(
            # Check if corrupted model's eigengap is high early, then drops
            len(corrupt_trajectory) >= 3 and
            corrupt_trajectory[len(corrupt_trajectory)//3]['tb'][l]['gap_at_5'] >
            corrupt_trajectory[-1]['tb'][l]['gap_at_5']
            for l in layers
        ),
    }

    results_path = os.path.join(RESULTS_DIR, 'us109_fwh_dimensionality_test.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Clean final loss:   {clean_losses[-1]:.4f}")
    print(f"  Corrupt final loss: {corrupt_losses[-1]:.4f}")
    print(f"\n  Gap at k=5 (final checkpoint):")
    for l in layers:
        print(f"    {l}: clean={clean_gap5_final[l]:.4f}, "
              f"corrupt={corrupt_gap5_final[l]:.4f}, "
              f"diff={clean_gap5_final[l] - corrupt_gap5_final[l]:.4f}")
    print(f"\n  Mean gap difference: {mean_gap_diff:.4f}")
    print(f"  Clean has higher gap: {mean_gap_diff > 0}")
    print(f"  Factored attractor detected: "
          f"{output['summary']['factored_attractor_detected']}")

    return output


if __name__ == '__main__':
    main()
