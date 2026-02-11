"""
US-108: TB on Transformer Residual Stream — Layer-by-Layer Factored Structure
=============================================================================

Extends US-107 with proper loss-gradient-based TB analysis. The key improvement:
instead of using activation differences as gradient proxies (which produces
isotropic covariance after LayerNorm), this script computes actual d(loss)/d(h_l)
gradients at each layer. These capture the true coupling structure in the loss
landscape, which is what TB is designed to analyze.

Additionally provides:
  - Per-layer blanket variable identification (which dimensions mediate
    cross-factor information flow)
  - Eigengap analysis at position N=5 specifically (not just max eigengap)
  - Full eigenvalue spectrum comparison across layers
  - 5-panel coupling matrix figure with reordered dimensions

Depends on: US-107 (complete; provides GHMM data generation and GPT-2 model).
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
from sklearn.decomposition import PCA

from topological_blankets import TopologicalBlankets, compute_eigengap
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian
)

# Reuse data generation and model from US-107
from fwh_ghmm_tb_detection import (
    generate_ghmm_dataset, SmallGPT2, GHMMDataset, train_gpt2
)


def collect_loss_gradients(model, tokens, n_samples=2000, device='cpu'):
    """
    Collect d(loss)/d(h_l) for each layer l.

    For each input sequence, this computes the gradient of the cross-entropy
    loss with respect to the residual stream activations at each layer.
    These gradients capture the true coupling structure: if two dimensions
    are jointly important for prediction, their loss gradients will be
    correlated, producing off-diagonal coupling in the Hessian estimate.

    This is superior to activation differences because:
    - Activation differences reflect representation variance (which is
      isotropic after LayerNorm).
    - Loss gradients reflect *functional* coupling: which dimensions must
      change together to reduce prediction error.
    """
    model.eval()
    model.to(device)

    rng = np.random.default_rng(0)
    indices = rng.choice(len(tokens), size=min(n_samples, len(tokens)), replace=False)
    sample_tokens = tokens[indices]

    layer_names = ['embedding'] + [f'layer_{i+1}' for i in range(model.n_layers)]
    all_grads = {name: [] for name in layer_names}

    batch_size = 64
    for start in range(0, len(sample_tokens), batch_size):
        batch_tokens = sample_tokens[start:start + batch_size]
        inputs = torch.tensor(batch_tokens[:, :-1], dtype=torch.long, device=device)
        targets = torch.tensor(batch_tokens[:, 1:], dtype=torch.long, device=device)

        B, L = inputs.shape
        positions = torch.arange(L, device=device).unsqueeze(0)

        # Forward pass with gradient tracking at each layer
        h = model.token_embed(inputs) + model.pos_embed(positions)
        h.requires_grad_(True)
        h.retain_grad()
        layer_activations = [h]

        current = h
        for block in model.blocks:
            current = block(current)
            current.retain_grad()
            layer_activations.append(current)

        # Compute loss
        final = model.ln_final(current)
        logits = model.head(final)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))

        # Backprop
        model.zero_grad()
        loss.backward()

        # Collect gradients at each layer (last token position)
        for i, (name, act) in enumerate(zip(layer_names, layer_activations)):
            if act.grad is not None:
                grad = act.grad[:, -1, :].detach().cpu().numpy()
                all_grads[name].append(grad)
            else:
                # Fallback if gradient didn't flow
                all_grads[name].append(np.zeros((B, model.d_model)))

    gradients = {}
    for name in layer_names:
        if all_grads[name]:
            gradients[name] = np.concatenate(all_grads[name], axis=0)
        else:
            gradients[name] = np.zeros((0, model.d_model))

    return gradients


def run_tb_analysis(gradients, n_objects=5, method='hybrid'):
    """
    Run TB on loss gradients and return detailed analysis.
    """
    n_dims = gradients.shape[1]

    tb = TopologicalBlankets(method=method, n_objects=n_objects)
    tb.fit(gradients)

    coupling = tb.get_coupling_matrix()
    objects_dict = tb.get_objects()
    blanket_indices = tb.get_blankets()

    # Build assignment array
    assignment = np.full(n_dims, -1, dtype=int)
    for obj_id, dim_indices in objects_dict.items():
        for idx in dim_indices:
            assignment[idx] = obj_id
    is_blanket = (assignment == -1)

    # Full eigenvalue analysis
    H = tb._features['hessian_est']
    A = build_adjacency_from_hessian(H)
    L_mat = build_graph_laplacian(A)
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_mat)))

    # Eigengap at each position
    gaps = np.diff(eigenvalues[:min(30, len(eigenvalues))])
    max_gap_idx, max_gap_val = compute_eigengap(eigenvalues)

    # Eigengap specifically at position 5 (for 5 factors)
    gap_at_5 = float(gaps[4]) if len(gaps) > 4 else 0.0
    # Relative gap: gap_5 / mean(gaps)
    mean_gap = float(np.mean(gaps[:20])) if len(gaps) >= 20 else float(np.mean(gaps))
    relative_gap_at_5 = gap_at_5 / mean_gap if mean_gap > 0 else 0.0

    # Coupling matrix sparsity (fraction of near-zero entries)
    coupling_abs = np.abs(coupling)
    threshold = 0.01 * coupling_abs.max()
    sparsity = float((coupling_abs < threshold).sum()) / coupling_abs.size

    # Per-object statistics
    object_stats = {}
    for obj_id, dim_indices in objects_dict.items():
        dims = list(dim_indices)
        # Within-object coupling strength
        if len(dims) > 1:
            within_coupling = np.mean([
                abs(coupling[i, j]) for i in dims for j in dims if i != j
            ])
        else:
            within_coupling = 0.0

        # Cross-object coupling (to all other dims)
        other_dims = [d for d in range(n_dims) if d not in dims]
        if other_dims and dims:
            cross_coupling = np.mean([
                abs(coupling[i, j]) for i in dims for j in other_dims
            ])
        else:
            cross_coupling = 0.0

        object_stats[f'object_{obj_id}'] = {
            'dims': dims,
            'n_dims': len(dims),
            'within_coupling': round(float(within_coupling), 6),
            'cross_coupling': round(float(cross_coupling), 6),
            'isolation_ratio': round(
                float(within_coupling / cross_coupling) if cross_coupling > 0 else float('inf'),
                4
            ),
        }

    # Blanket analysis
    blanket_dims = sorted(blanket_indices.tolist()) if len(blanket_indices) > 0 else []
    blanket_coupling_profile = {}
    for b_dim in blanket_dims:
        # For each blanket dim, measure coupling to each object
        couplings_to_objects = {}
        for obj_id, dim_indices in objects_dict.items():
            c = np.mean([abs(coupling[b_dim, d]) for d in dim_indices])
            couplings_to_objects[f'object_{obj_id}'] = round(float(c), 6)
        blanket_coupling_profile[str(b_dim)] = couplings_to_objects

    return {
        'coupling': coupling,
        'assignment': assignment,
        'is_blanket': is_blanket,
        'eigenvalues': eigenvalues[:30].tolist(),
        'eigengaps': gaps[:20].tolist(),
        'max_eigengap': float(max_gap_val),
        'max_eigengap_index': int(max_gap_idx),
        'gap_at_5': gap_at_5,
        'relative_gap_at_5': round(relative_gap_at_5, 4),
        'coupling_sparsity': round(sparsity, 4),
        'n_blanket': int(is_blanket.sum()),
        'n_objects_detected': len(objects_dict),
        'blanket_dims': blanket_dims,
        'object_stats': object_stats,
        'blanket_coupling_profile': blanket_coupling_profile,
    }


def plot_5panel_coupling(results, save_path):
    """
    5-panel coupling matrix figure with dimensions reordered by TB partition.
    """
    layer_names = list(results.keys())
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for i, name in enumerate(layer_names):
        r = results[name]
        coupling = r['coupling']
        assignment = r['assignment']

        # Reorder dimensions: group by object, blanket at end
        obj_ids = sorted(set(assignment[assignment >= 0]))
        order = []
        for oid in obj_ids:
            order.extend(np.where(assignment == oid)[0].tolist())
        order.extend(np.where(assignment == -1)[0].tolist())
        order = np.array(order)

        reordered = coupling[np.ix_(order, order)]

        im = axes[i].imshow(np.abs(reordered), cmap='hot', aspect='auto',
                            vmin=0, vmax=np.percentile(np.abs(coupling), 95))
        axes[i].set_title(f"{name}\neigengap@5={r['gap_at_5']:.2f}\n"
                         f"blanket={r['n_blanket']} dims")
        axes[i].set_xlabel('Dimension (reordered)')
        if i == 0:
            axes[i].set_ylabel('Dimension (reordered)')

        # Draw partition boundaries
        boundary = 0
        for oid in obj_ids:
            n_dims = sum(1 for a in assignment if a == oid)
            boundary += n_dims
            if boundary < len(order):
                axes[i].axhline(y=boundary - 0.5, color='cyan', linewidth=0.8, alpha=0.7)
                axes[i].axvline(x=boundary - 0.5, color='cyan', linewidth=0.8, alpha=0.7)

    fig.suptitle('TB Coupling Matrix by Layer (Reordered by Partition)', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_eigengap_spectrum(results, save_path):
    """
    Eigenvalue spectra across layers with N=5 marker.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    layer_names = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layer_names)))

    # Panel 1: Full eigenvalue spectrum (first 25)
    for i, name in enumerate(layer_names):
        evals = results[name]['eigenvalues'][:25]
        axes[0].plot(range(len(evals)), evals, label=name,
                     color=colors[i], marker='o', markersize=3)
    axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.5, label='N=5 factors')
    axes[0].set_xlabel('Eigenvalue index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Graph Laplacian Spectrum')
    axes[0].legend(fontsize=7)

    # Panel 2: Eigengaps (consecutive differences)
    for i, name in enumerate(layer_names):
        gaps = results[name]['eigengaps'][:15]
        axes[1].plot(range(1, len(gaps) + 1), gaps, label=name,
                     color=colors[i], marker='s', markersize=3)
    axes[1].axvline(x=5, color='red', linestyle='--', alpha=0.5, label='N=5')
    axes[1].set_xlabel('Gap index (k)')
    axes[1].set_ylabel('lambda_{k+1} - lambda_k')
    axes[1].set_title('Eigengaps')
    axes[1].legend(fontsize=7)

    # Panel 3: Key metrics across layers
    x = range(len(layer_names))
    gap5 = [results[name]['gap_at_5'] for name in layer_names]
    n_blanket = [results[name]['n_blanket'] for name in layer_names]
    sparsity = [results[name]['coupling_sparsity'] for name in layer_names]

    ax3a = axes[2]
    ax3b = ax3a.twinx()
    bars = ax3a.bar(x, gap5, alpha=0.6, color='steelblue', label='Gap at k=5')
    line = ax3b.plot(x, n_blanket, 'ro-', label='Blanket dims', markersize=6)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax3a.set_ylabel('Eigengap at k=5', color='steelblue')
    ax3b.set_ylabel('Blanket dimensions', color='red')
    ax3a.set_title('Factor Detection by Layer')
    ax3a.legend(loc='upper left', fontsize=8)
    ax3b.legend(loc='upper right', fontsize=8)

    fig.suptitle('Layer-by-Layer Factored Structure Analysis', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_blanket_mediation(results, save_path):
    """
    Blanket variable mediation: which objects each blanket dim connects.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    layer_names = list(results.keys())
    for i, name in enumerate(layer_names):
        profile = results[name]['blanket_coupling_profile']
        if not profile:
            axes[i].text(0.5, 0.5, 'No blanket vars', ha='center', va='center',
                        transform=axes[i].transAxes)
            axes[i].set_title(name)
            continue

        # Build heatmap: blanket_dim x object
        b_dims = sorted(profile.keys(), key=int)
        obj_names = sorted(next(iter(profile.values())).keys())

        data = np.zeros((len(b_dims), len(obj_names)))
        for bi, bd in enumerate(b_dims):
            for oj, on in enumerate(obj_names):
                data[bi, oj] = profile[bd].get(on, 0)

        im = axes[i].imshow(data, cmap='YlOrRd', aspect='auto')
        axes[i].set_yticks(range(len(b_dims)))
        axes[i].set_yticklabels([f'd{d}' for d in b_dims], fontsize=7)
        axes[i].set_xticks(range(len(obj_names)))
        axes[i].set_xticklabels([n.replace('object_', 'O') for n in obj_names],
                                fontsize=8)
        axes[i].set_title(f'{name}\n({len(b_dims)} blanket dims)')
        axes[i].set_xlabel('Object')
        if i == 0:
            axes[i].set_ylabel('Blanket dimension')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    fig.suptitle('Blanket Variable Mediation Profile', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_object_isolation(results, save_path):
    """
    Object isolation ratio (within-coupling / cross-coupling) across layers.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    layer_names = list(results.keys())
    x = np.arange(len(layer_names))
    n_objects = results[layer_names[0]]['n_objects_detected']
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, n_objects))

    for obj_id in range(n_objects):
        ratios = []
        for name in layer_names:
            stats = results[name]['object_stats'].get(f'object_{obj_id}', {})
            r = stats.get('isolation_ratio', 0)
            if r == float('inf'):
                r = 10.0  # cap for display
            ratios.append(min(r, 10.0))
        ax.bar(x + obj_id * width, ratios, width, label=f'Object {obj_id}',
               color=colors[obj_id], alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
               label='isolation=1 (no separation)')
    ax.set_xticks(x + width * (n_objects - 1) / 2)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Isolation Ratio (within/cross coupling)')
    ax.set_title('Object Isolation Across Layers')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 70)
    print("US-108: Layer-by-Layer TB on Transformer Residual Stream")
    print("  (Loss-gradient-based TB analysis)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ── Generate data and train model ────────────────────────────────────
    n_sequences = 10000
    n_epochs = 30
    print(f"\n[1/4] Generating GHMM data and training GPT-2...")
    tokens, factor_subtokens, factor_info = generate_ghmm_dataset(
        n_sequences=n_sequences, seq_len=8, seed=42
    )

    model = SmallGPT2(
        vocab_size=factor_info['vocab_size'],
        d_model=120, n_layers=4, n_heads=4, d_mlp=480, max_len=16,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    loss_history = train_gpt2(model, tokens, n_epochs=n_epochs,
                              batch_size=256, lr=1e-3, device=device)

    # ── Collect loss gradients at each layer ─────────────────────────────
    print(f"\n[2/4] Collecting loss gradients at each layer...")
    gradients = collect_loss_gradients(model, tokens, n_samples=2000, device=device)
    for name, g in gradients.items():
        print(f"  {name}: gradient shape {g.shape}, "
              f"norm range [{np.linalg.norm(g, axis=1).min():.4f}, "
              f"{np.linalg.norm(g, axis=1).max():.4f}]")

    # ── TB analysis at each layer ────────────────────────────────────────
    print(f"\n[3/4] Running TB analysis with loss gradients...")
    tb_results = {}

    for name, grads in gradients.items():
        print(f"\n  === {name} ===")
        t0 = time.time()
        result = run_tb_analysis(grads, n_objects=5, method='hybrid')
        elapsed = time.time() - t0
        result['runtime_s'] = round(elapsed, 3)

        # Remove large arrays from stored result (keep coupling for plots only)
        tb_results[name] = result

        print(f"  Eigengap@max: {result['max_eigengap']:.4f} "
              f"(at index {result['max_eigengap_index']})")
        print(f"  Eigengap@5: {result['gap_at_5']:.4f} "
              f"(relative: {result['relative_gap_at_5']:.4f})")
        print(f"  Objects: {result['n_objects_detected']}, "
              f"Blanket: {result['n_blanket']} dims")
        print(f"  Coupling sparsity: {result['coupling_sparsity']:.4f}")
        print(f"  Blanket dims: {result['blanket_dims'][:10]}"
              f"{'...' if len(result['blanket_dims']) > 10 else ''}")

        # Object isolation
        for obj_name, stats in result['object_stats'].items():
            iso = stats['isolation_ratio']
            iso_str = f"{iso:.2f}" if iso != float('inf') else "inf"
            print(f"    {obj_name}: {stats['n_dims']} dims, "
                  f"isolation={iso_str}")

    # ── Generate plots ───────────────────────────────────────────────────
    print(f"\n[4/4] Generating visualizations...")

    plot_5panel_coupling(tb_results,
                         os.path.join(RESULTS_DIR, 'us108_coupling_5panel.png'))
    plot_eigengap_spectrum(tb_results,
                          os.path.join(RESULTS_DIR, 'us108_eigengap_spectrum.png'))
    plot_blanket_mediation(tb_results,
                           os.path.join(RESULTS_DIR, 'us108_blanket_mediation.png'))
    plot_object_isolation(tb_results,
                          os.path.join(RESULTS_DIR, 'us108_object_isolation.png'))

    # ── Save results ─────────────────────────────────────────────────────
    # Prepare JSON-serializable results (remove numpy arrays)
    json_results = {}
    for name, r in tb_results.items():
        json_results[name] = {
            k: v for k, v in r.items()
            if k not in ('coupling', 'assignment', 'is_blanket')
        }
        # Add assignment as list
        json_results[name]['assignment'] = r['assignment'].tolist()
        json_results[name]['is_blanket'] = r['is_blanket'].tolist()

    output = {
        'experiment': 'US-108',
        'title': 'Layer-by-Layer TB on Transformer Residual Stream',
        'method': 'Loss-gradient TB (d(CE_loss)/d(h_l))',
        'model': {
            'architecture': 'GPT-2',
            'n_layers': 4,
            'd_model': 120,
            'n_params': n_params,
        },
        'training': {
            'n_sequences': n_sequences,
            'n_epochs': n_epochs,
            'final_loss': round(loss_history[-1], 6),
        },
        'n_gradient_samples': 2000,
        'results_by_layer': json_results,
        'summary': {
            'eigengap_at_5_trajectory': {
                name: r['gap_at_5'] for name, r in tb_results.items()
            },
            'blanket_count_trajectory': {
                name: r['n_blanket'] for name, r in tb_results.items()
            },
            'coupling_sparsity_trajectory': {
                name: r['coupling_sparsity'] for name, r in tb_results.items()
            },
            'best_layer_for_factoring': max(
                tb_results.keys(),
                key=lambda k: tb_results[k]['relative_gap_at_5']
            ),
        },
    }

    results_path = os.path.join(RESULTS_DIR, 'us108_layer_by_layer_tb.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Layer-by-Layer Factored Structure")
    print("=" * 70)
    for name in gradients.keys():
        r = tb_results[name]
        print(f"  {name}: gap@5={r['gap_at_5']:.4f}, "
              f"rel_gap={r['relative_gap_at_5']:.4f}, "
              f"blanket={r['n_blanket']}, "
              f"sparsity={r['coupling_sparsity']:.4f}")
    print(f"\n  Best layer for factoring: "
          f"{output['summary']['best_layer_for_factoring']}")

    return output


if __name__ == '__main__':
    main()
