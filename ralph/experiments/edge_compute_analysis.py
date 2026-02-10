"""
US-034: Edge-Compute Factorization Analysis
============================================

Quantify compute savings from TB-discovered structure.
Monolithic vs factored update cost analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


def compute_factored_cost(object_sizes, blanket_size):
    """
    Compute factored update cost.

    Monolithic: O(n^2) where n = total vars
    Factored: O(sum(n_i^2) + n_b^2 + k * n_b * max(n_i))
    """
    n_total = sum(object_sizes) + blanket_size
    k = len(object_sizes)

    monolithic = n_total ** 2

    # Object-local updates (independent, parallelizable)
    object_cost = sum(ni ** 2 for ni in object_sizes)
    # Blanket self-update
    blanket_cost = blanket_size ** 2
    # Cross-terms: blanket communicates with each object
    cross_cost = k * blanket_size * max(object_sizes) if object_sizes else 0

    factored = object_cost + blanket_cost + cross_cost

    speedup = monolithic / factored if factored > 0 else float('inf')

    return {
        'n_total': int(n_total),
        'k_objects': int(k),
        'object_sizes': [int(s) for s in object_sizes],
        'blanket_size': int(blanket_size),
        'monolithic_flops': int(monolithic),
        'factored_flops': int(factored),
        'object_cost': int(object_cost),
        'blanket_cost': int(blanket_cost),
        'cross_cost': int(cross_cost),
        'speedup': float(speedup),
    }


def run_edge_compute_analysis():
    """Run the full edge-compute factorization analysis."""
    print("=" * 70)
    print("US-034: Edge-Compute Factorization Analysis")
    print("=" * 70)

    results = {}

    # Active Inference: 8D state space
    # Typical partition from US-025: 2 objects of ~3 vars each + 2 blanket
    print("\n--- Active Inference (8D) ---")
    configs_8d = [
        ([3, 3], 2, "AI: 3+3+2b"),
        ([4, 2], 2, "AI: 4+2+2b"),
        ([3, 3, 2], 0, "AI: no blanket"),
    ]
    for obj_sizes, b_size, label in configs_8d:
        r = compute_factored_cost(obj_sizes, b_size)
        results[label] = r
        print(f"  {label}: monolithic={r['monolithic_flops']}, "
              f"factored={r['factored_flops']}, speedup={r['speedup']:.2f}x")

    # Dreamer: 64D latent space
    print("\n--- Dreamer (64D) ---")
    configs_64d = [
        ([20, 20, 20], 4, "Dreamer: 20+20+20+4b"),
        ([16, 16, 16, 16], 0, "Dreamer: 4x16 no blanket"),
        ([30, 30], 4, "Dreamer: 30+30+4b"),
        ([10, 10, 10, 10, 10, 10], 4, "Dreamer: 6x10+4b"),
    ]
    for obj_sizes, b_size, label in configs_64d:
        r = compute_factored_cost(obj_sizes, b_size)
        results[label] = r
        print(f"  {label}: monolithic={r['monolithic_flops']}, "
              f"factored={r['factored_flops']}, speedup={r['speedup']:.2f}x")

    # Extrapolation to higher dimensions
    print("\n--- Scaling Extrapolation ---")
    dims = [8, 64, 256, 1024, 4096]
    n_objects_per_dim = {8: 2, 64: 4, 256: 8, 1024: 16, 4096: 32}
    scaling_results = {}

    for d in dims:
        k = n_objects_per_dim[d]
        obj_size = (d - 4) // k  # blanket of 4
        obj_sizes = [obj_size] * k
        b_size = d - obj_size * k
        r = compute_factored_cost(obj_sizes, b_size)
        scaling_results[str(d)] = r
        print(f"  d={d:>5d}, k={k:>3d}: speedup={r['speedup']:.1f}x "
              f"(monolithic={r['monolithic_flops']:>12,d}, factored={r['factored_flops']:>12,d})")

    results['scaling'] = scaling_results

    # Memory savings
    print("\n--- Memory Savings (Sparse Storage) ---")
    for d in dims:
        k = n_objects_per_dim[d]
        dense = d * d
        # Sparse: only store object-internal blocks + blanket column
        obj_size = (d - 4) // k
        sparse = k * obj_size ** 2 + 4 * d  # blocks + blanket row/col
        ratio = sparse / dense
        print(f"  d={d:>5d}: dense={dense:>12,d}, sparse={sparse:>12,d}, "
              f"ratio={ratio:.3f} ({100*(1-ratio):.0f}% savings)")

    # Plot speedup curve
    fig, ax = plt.subplots(figsize=(8, 5))
    dims_plot = [int(d) for d in scaling_results.keys()]
    speedups = [scaling_results[str(d)]['speedup'] for d in dims_plot]
    ax.plot(dims_plot, speedups, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax.set_xlabel('Total Dimension')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Factored vs Monolithic Update: Speedup vs Dimension')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    for d, s in zip(dims_plot, speedups):
        ax.annotate(f'{s:.1f}x', (d, s), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    save_figure(fig, 'edge_compute_speedup', 'edge_compute')

    config = {
        'dims_tested': dims,
        'formula': 'factored = sum(n_i^2) + n_b^2 + k * n_b * max(n_i)',
    }

    save_results('edge_compute_analysis', results, config,
                 notes='US-034: Compute savings from TB-discovered structure. Speedup grows with dimension and number of objects.')

    print("\nUS-034 complete.")
    return results


if __name__ == '__main__':
    run_edge_compute_analysis()
