"""
US-111: Cross-Architecture Factoring — Transformer vs RNN vs Ensemble
=====================================================================

Compares TB-detected factored structure across three architectures trained
on equivalent data. If TB finds similar partitions across architectures,
then factored structure is a property of the data (not the architecture),
and TB becomes a universal detector.

Architectures:
  1. GPT-2 Transformer (from US-107): 4-layer, d_model=120
  2. LSTM RNN: 2-layer, hidden_size=120
  3. MLP Ensemble: 5 members, hidden_size=256

All trained on the same 5-factor GHMM data (Shai et al. 2602.02385).

TB is applied to internal representations from each architecture.
Partitions are compared via NMI and ARI to check cross-architecture
consistency and alignment with the transformer's factored structure.

Depends on: US-107 (GHMM data, transformer), US-110 (ensemble pattern).
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from topological_blankets import TopologicalBlankets, compute_eigengap
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian
)

from fwh_ghmm_tb_detection import (
    generate_ghmm_dataset, SmallGPT2, GHMMDataset, train_gpt2,
    extract_residual_stream, compute_activation_gradients, run_tb_on_layer
)


# ══════════════════════════════════════════════════════════════════════════
# Architecture 2: LSTM RNN
# ══════════════════════════════════════════════════════════════════════════

class LSTMNextToken(nn.Module):
    """LSTM-based next-token predictor with same capacity as GPT-2."""

    def __init__(self, vocab_size=433, d_model=120, n_layers=2, max_len=16):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight  # tie weights

    def forward(self, x, return_hidden=False):
        B, L = x.shape
        h = self.token_embed(x)
        output, (h_n, c_n) = self.lstm(h)
        output = self.ln(output)
        logits = self.head(output)

        if return_hidden:
            return logits, output
        return logits

    def extract_hidden(self, x):
        """Extract hidden state at last position."""
        _, output = self.forward(x, return_hidden=True)
        return output[:, -1, :]  # (B, d_model)


# ══════════════════════════════════════════════════════════════════════════
# Architecture 3: MLP Ensemble
# ══════════════════════════════════════════════════════════════════════════

class MLPMember(nn.Module):
    """Single MLP member for next-token prediction."""

    def __init__(self, vocab_size=433, d_model=120, hidden_size=256,
                 context_len=7):
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model)
        input_dim = d_model * context_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, return_hidden=False):
        B, L = x.shape
        h = self.token_embed(x)  # (B, L, d_model)
        h_flat = h.reshape(B, -1)  # (B, L*d_model)
        hidden = self.mlp(h_flat)  # (B, d_model)
        hidden = self.ln(hidden)
        logits = self.head(hidden)  # (B, vocab_size)

        if return_hidden:
            return logits, hidden
        return logits


class MLPEnsemble(nn.Module):
    """Ensemble of MLP members for next-token prediction."""

    def __init__(self, n_members=5, vocab_size=433, d_model=120,
                 hidden_size=256, context_len=7):
        super().__init__()
        self.d_model = d_model
        self.n_members = n_members
        self.members = nn.ModuleList([
            MLPMember(vocab_size, d_model, hidden_size, context_len)
            for _ in range(n_members)
        ])

    def forward(self, x, return_hidden=False):
        """Average prediction across members."""
        all_logits = []
        all_hidden = []
        for member in self.members:
            if return_hidden:
                logits, hidden = member(x, return_hidden=True)
                all_hidden.append(hidden)
            else:
                logits = member(x)
            all_logits.append(logits)

        avg_logits = torch.stack(all_logits).mean(dim=0)

        if return_hidden:
            # Concatenate hidden states from all members
            concat_hidden = torch.cat(all_hidden, dim=-1)  # (B, n*d_model)
            return avg_logits, concat_hidden
        return avg_logits


def train_lstm(model, tokens, n_epochs=30, batch_size=256, lr=1e-3,
               device='cpu'):
    """Train LSTM model on GHMM data."""
    dataset = GHMMDataset(tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    loss_history = []
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"({time.time()-t0:.1f}s)")

    print(f"    Final loss: {loss_history[-1]:.4f}")
    return loss_history


def train_ensemble(model, tokens, n_epochs=30, batch_size=256, lr=1e-3,
                   device='cpu'):
    """Train MLP ensemble on GHMM data (each member gets full context)."""
    # For ensemble: input is full context, target is next token only
    all_inputs = torch.tensor(tokens[:, :-1], dtype=torch.long)
    all_targets = torch.tensor(tokens[:, -1], dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    loss_history = []
    n = len(all_inputs)
    t0 = time.time()

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            inputs = all_inputs[idx].to(device)
            targets = all_targets[idx].to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"({time.time()-t0:.1f}s)")

    print(f"    Final loss: {loss_history[-1]:.4f}")
    return loss_history


def extract_representations(model, tokens, arch_type, n_samples=5000,
                            device='cpu'):
    """
    Extract internal representations from each architecture.
    Returns dict of layer_name -> (n_samples, d_model) arrays.
    """
    model.eval()
    model.to(device)

    rng = np.random.default_rng(0)
    indices = rng.choice(len(tokens), size=min(n_samples, len(tokens)),
                         replace=False)
    sample_tokens = tokens[indices]

    if arch_type == 'transformer':
        return extract_residual_stream(model, tokens, n_samples=n_samples,
                                       device=device)

    elif arch_type == 'lstm':
        inputs = torch.tensor(sample_tokens[:, :-1], dtype=torch.long,
                              device=device)
        all_hidden = []
        with torch.no_grad():
            for start in range(0, len(inputs), 512):
                batch = inputs[start:start + 512]
                _, hidden = model(batch, return_hidden=True)
                all_hidden.append(hidden[:, -1, :].cpu().numpy())
        return {'lstm_hidden': np.concatenate(all_hidden, axis=0)}

    elif arch_type == 'ensemble':
        inputs = torch.tensor(sample_tokens[:, :-1], dtype=torch.long,
                              device=device)
        all_hidden = []
        with torch.no_grad():
            for start in range(0, len(inputs), 512):
                batch = inputs[start:start + 512]
                _, hidden = model(batch, return_hidden=True)
                all_hidden.append(hidden.cpu().numpy())
        return {'ensemble_hidden': np.concatenate(all_hidden, axis=0)}

    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def run_tb_on_representations(activations, n_objects=5):
    """Run TB on each representation layer."""
    results = {}
    for name, acts in activations.items():
        gradients = compute_activation_gradients(acts)
        n_dims = acts.shape[1]

        tb = TopologicalBlankets(method='hybrid', n_objects=n_objects)
        tb.fit(gradients)

        coupling = tb.get_coupling_matrix()
        objects_dict = tb.get_objects()
        blanket_indices = tb.get_blankets()

        assignment = np.full(n_dims, -1, dtype=int)
        for obj_id, dim_indices in objects_dict.items():
            for idx in dim_indices:
                assignment[idx] = obj_id

        H = tb._features['hessian_est']
        A = build_adjacency_from_hessian(H)
        L_mat = build_graph_laplacian(A)
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_mat)))

        gaps = np.diff(eigenvalues[:min(30, len(eigenvalues))])
        gap_at_5 = float(gaps[4]) if len(gaps) > 4 else 0.0
        max_gap_idx, max_gap_val = compute_eigengap(eigenvalues)

        results[name] = {
            'coupling': coupling,
            'assignment': assignment,
            'eigengap': float(max_gap_val),
            'gap_at_5': gap_at_5,
            'n_objects': len(objects_dict),
            'n_blanket': int(len(blanket_indices)),
            'eigenvalues_top15': eigenvalues[:15].tolist(),
        }

    return results


def plot_cross_architecture_comparison(all_results, save_path):
    """
    Compare coupling matrices and partitions across architectures.
    """
    archs = list(all_results.keys())
    n_archs = len(archs)

    fig, axes = plt.subplots(2, n_archs, figsize=(5 * n_archs, 9))

    for i, arch in enumerate(archs):
        # Get the "best" layer for this architecture
        results = all_results[arch]
        best_layer = max(results.keys(),
                         key=lambda k: results[k]['gap_at_5'])
        r = results[best_layer]

        # Top row: coupling matrix
        coupling = r['coupling']
        im = axes[0, i].imshow(np.abs(coupling), cmap='hot', aspect='auto',
                                vmin=0,
                                vmax=np.percentile(np.abs(coupling), 95))
        axes[0, i].set_title(f'{arch}\n{best_layer}\n'
                              f'gap@5={r["gap_at_5"]:.3f}')
        axes[0, i].set_xlabel('Dimension')
        if i == 0:
            axes[0, i].set_ylabel('Coupling Matrix')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

        # Bottom row: partition strip
        assignment = r['assignment']
        img = np.zeros((3, len(assignment)))
        for row in range(3):
            img[row, :] = assignment
        axes[1, i].imshow(img, cmap='tab10', aspect='auto', vmin=-1, vmax=5)
        axes[1, i].set_title(f'{r["n_objects"]} objects, '
                              f'{r["n_blanket"]} blanket')
        axes[1, i].set_yticks([])
        axes[1, i].set_xlabel('Dimension')
        if i == 0:
            axes[1, i].set_ylabel('TB Partition')

    fig.suptitle('Cross-Architecture Factor Detection Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_eigengap_comparison(all_results, save_path):
    """Compare eigenvalue spectra across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    archs = list(all_results.keys())
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(archs)))

    # Panel 1: Eigenvalue spectra
    for i, arch in enumerate(archs):
        results = all_results[arch]
        best_layer = max(results.keys(),
                         key=lambda k: results[k]['gap_at_5'])
        evals = results[best_layer]['eigenvalues_top15']
        axes[0].plot(range(len(evals)), evals,
                     label=f'{arch} ({best_layer})',
                     color=colors[i], marker='o', markersize=4)

    axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.5,
                     label='N=5 factors')
    axes[0].set_xlabel('Eigenvalue index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Graph Laplacian Spectrum by Architecture')
    axes[0].legend(fontsize=8)

    # Panel 2: Summary metrics
    arch_names = []
    gap5_vals = []
    n_blanket_vals = []

    for arch in archs:
        results = all_results[arch]
        best_layer = max(results.keys(),
                         key=lambda k: results[k]['gap_at_5'])
        arch_names.append(f'{arch}\n({best_layer})')
        gap5_vals.append(results[best_layer]['gap_at_5'])
        n_blanket_vals.append(results[best_layer]['n_blanket'])

    x = np.arange(len(arch_names))
    width = 0.35

    ax1 = axes[1]
    ax2 = ax1.twinx()
    bars = ax1.bar(x - width/2, gap5_vals, width, color='steelblue',
                    alpha=0.8, label='Gap at k=5')
    dots = ax2.bar(x + width/2, n_blanket_vals, width, color='coral',
                    alpha=0.8, label='Blanket dims')

    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_names, fontsize=9)
    ax1.set_ylabel('Eigengap at k=5', color='steelblue')
    ax2.set_ylabel('Blanket dimensions', color='coral')
    ax1.set_title('Factor Detection Quality by Architecture')
    ax1.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    fig.suptitle('Cross-Architecture Factored Structure Analysis', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_cross_architecture_nmi(all_results):
    """Compute NMI between every pair of architectures."""
    archs = list(all_results.keys())
    nmi_matrix = {}

    for i, arch_a in enumerate(archs):
        results_a = all_results[arch_a]
        best_a = max(results_a.keys(),
                     key=lambda k: results_a[k]['gap_at_5'])
        assign_a = results_a[best_a]['assignment']

        for j, arch_b in enumerate(archs):
            results_b = all_results[arch_b]
            best_b = max(results_b.keys(),
                         key=lambda k: results_b[k]['gap_at_5'])
            assign_b = results_b[best_b]['assignment']

            # Assignments may have different lengths (different hidden sizes)
            # Compare only if same dimension
            if len(assign_a) == len(assign_b):
                nmi = normalized_mutual_info_score(assign_a, assign_b)
                ari = adjusted_rand_score(assign_a, assign_b)
            else:
                nmi = float('nan')
                ari = float('nan')

            nmi_matrix[f'{arch_a}_vs_{arch_b}'] = {
                'nmi': round(float(nmi), 4) if not np.isnan(nmi) else 'N/A (dim mismatch)',
                'ari': round(float(ari), 4) if not np.isnan(ari) else 'N/A (dim mismatch)',
                'dim_a': len(assign_a),
                'dim_b': len(assign_b),
            }

    return nmi_matrix


def main():
    print("=" * 70)
    print("US-111: Cross-Architecture Factoring Comparison")
    print("  Transformer vs LSTM vs MLP Ensemble")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    n_sequences = 10000
    n_epochs = 30

    # ── Generate GHMM data ──────────────────────────────────────────────
    print(f"\n[1/5] Generating GHMM data ({n_sequences} seqs)...")
    tokens, _, factor_info = generate_ghmm_dataset(
        n_sequences=n_sequences, seq_len=8, seed=42
    )

    all_results = {}
    all_losses = {}

    # ── Architecture 1: Transformer ─────────────────────────────────────
    print(f"\n[2/5] Training Transformer (GPT-2, 4L, d=120)...")
    transformer = SmallGPT2(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=4, n_heads=4, d_mlp=480, max_len=16,
    )
    n_params_trans = sum(p.numel() for p in transformer.parameters())
    print(f"  Parameters: {n_params_trans:,}")
    trans_losses = train_gpt2(transformer, tokens, n_epochs=n_epochs,
                              device=device)
    all_losses['transformer'] = trans_losses

    print("  Extracting representations...")
    trans_acts = extract_representations(transformer, tokens, 'transformer',
                                         device=device)
    print("  Running TB...")
    all_results['transformer'] = run_tb_on_representations(trans_acts)

    # ── Architecture 2: LSTM ────────────────────────────────────────────
    print(f"\n[3/5] Training LSTM (2L, hidden=120)...")
    lstm = LSTMNextToken(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=2, max_len=16,
    )
    n_params_lstm = sum(p.numel() for p in lstm.parameters())
    print(f"  Parameters: {n_params_lstm:,}")
    lstm_losses = train_lstm(lstm, tokens, n_epochs=n_epochs, device=device)
    all_losses['lstm'] = lstm_losses

    print("  Extracting representations...")
    lstm_acts = extract_representations(lstm, tokens, 'lstm', device=device)
    print("  Running TB...")
    all_results['lstm'] = run_tb_on_representations(lstm_acts)

    # ── Architecture 3: MLP Ensemble ────────────────────────────────────
    print(f"\n[4/5] Training MLP Ensemble (5 members, hidden=256)...")
    ensemble = MLPEnsemble(
        n_members=5,
        vocab_size=factor_info['vocab_size'],
        d_model=120, hidden_size=256, context_len=8,
    )
    n_params_ens = sum(p.numel() for p in ensemble.parameters())
    print(f"  Parameters: {n_params_ens:,}")
    ens_losses = train_ensemble(ensemble, tokens, n_epochs=n_epochs,
                                device=device)
    all_losses['ensemble'] = ens_losses

    print("  Extracting representations...")
    ens_acts = extract_representations(ensemble, tokens, 'ensemble',
                                       device=device)
    print("  Running TB...")
    all_results['ensemble'] = run_tb_on_representations(ens_acts)

    # ── Cross-architecture comparison ───────────────────────────────────
    print(f"\n[5/5] Cross-architecture comparison...")
    nmi_matrix = compute_cross_architecture_nmi(all_results)

    # ── Visualizations ──────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_cross_architecture_comparison(
        all_results,
        os.path.join(RESULTS_DIR, 'us111_cross_architecture.png')
    )
    plot_eigengap_comparison(
        all_results,
        os.path.join(RESULTS_DIR, 'us111_eigengap_comparison.png')
    )

    # ── Save results ────────────────────────────────────────────────────
    json_results = {}
    for arch, results in all_results.items():
        json_results[arch] = {}
        for layer, r in results.items():
            json_results[arch][layer] = {
                k: v for k, v in r.items()
                if k not in ('coupling', 'assignment')
            }
            json_results[arch][layer]['assignment'] = r['assignment'].tolist()

    output = {
        'experiment': 'US-111',
        'title': 'Cross-Architecture Factoring Comparison',
        'architectures': {
            'transformer': {
                'type': 'GPT-2',
                'n_params': n_params_trans,
                'final_loss': round(trans_losses[-1], 6),
            },
            'lstm': {
                'type': 'LSTM',
                'n_params': n_params_lstm,
                'final_loss': round(lstm_losses[-1], 6),
            },
            'ensemble': {
                'type': 'MLP Ensemble (5 members)',
                'n_params': n_params_ens,
                'final_loss': round(ens_losses[-1], 6),
            },
        },
        'tb_results': json_results,
        'cross_architecture_nmi': nmi_matrix,
        'summary': {},
    }

    # Summary: which architectures find similar structure?
    for arch in all_results:
        results = all_results[arch]
        best_layer = max(results.keys(),
                         key=lambda k: results[k]['gap_at_5'])
        output['summary'][arch] = {
            'best_layer': best_layer,
            'gap_at_5': results[best_layer]['gap_at_5'],
            'n_objects': results[best_layer]['n_objects'],
            'n_blanket': results[best_layer]['n_blanket'],
        }

    results_path = os.path.join(RESULTS_DIR, 'us111_cross_architecture.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for arch in all_results:
        s = output['summary'][arch]
        print(f"  {arch}: gap@5={s['gap_at_5']:.4f}, "
              f"objects={s['n_objects']}, blanket={s['n_blanket']}")

    print(f"\nCross-architecture NMI:")
    for pair, metrics in nmi_matrix.items():
        if '_vs_' in pair:
            a, b = pair.split('_vs_')
            if a != b:
                print(f"  {pair}: NMI={metrics['nmi']}, ARI={metrics['ari']}")

    return output


if __name__ == '__main__':
    main()
