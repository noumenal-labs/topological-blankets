"""
US-020: Ising Model Demonstration
==================================

2D Ising model with Metropolis sampling at T < T_c, T ~ T_c, T > T_c.
TB detects domain boundaries (blankets between magnetic domains).
Shows geometric-to-topological transition (Section 6.7 of paper).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import detect_blankets_otsu, detect_blankets_spectral
from topological_blankets.spectral import compute_eigengap, build_adjacency_from_hessian, build_graph_laplacian
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# Ising model
# =========================================================================

TC_ISING = 2.0 / np.log(1 + np.sqrt(2))  # ~2.269 for 2D square lattice


def ising_energy(spins, J=1.0, h=0.0):
    """
    Compute 2D Ising energy with periodic boundary conditions.
    E = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
    """
    L = spins.shape[0]
    energy = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            # Right and down neighbors (periodic BC)
            energy -= J * s * spins[i, (j + 1) % L]
            energy -= J * s * spins[(i + 1) % L, j]
            energy -= h * s
    return energy


def metropolis_sample(L, T, J=1.0, h=0.0, n_steps=50000, n_burn=10000):
    """
    Metropolis sampling of 2D Ising model.

    Returns array of spin configurations after burn-in.
    """
    spins = np.random.choice([-1, 1], size=(L, L))
    beta = 1.0 / T

    configs = []
    for step in range(n_steps + n_burn):
        # Random spin flip
        i, j = np.random.randint(0, L, size=2)
        s = spins[i, j]
        # Neighbor sum
        nn_sum = (spins[(i + 1) % L, j] + spins[(i - 1) % L, j] +
                  spins[i, (j + 1) % L] + spins[i, (j - 1) % L])
        dE = 2 * J * s * nn_sum + 2 * h * s

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s

        if step >= n_burn and step % 50 == 0:
            configs.append(spins.copy())

    return np.array(configs)


def compute_ising_gradients(configs, J=1.0, h=0.0):
    """
    Compute energy gradients for each spin variable across configurations.

    The "gradient" for spin i is the local field: dE/ds_i = -J * sum_nn s_j - h.
    Flattened to (n_configs, L*L).
    """
    n_configs, L, _ = configs.shape
    n_vars = L * L
    gradients = np.zeros((n_configs, n_vars))

    for t in range(n_configs):
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                nn_sum = (configs[t, (i + 1) % L, j] + configs[t, (i - 1) % L, j] +
                          configs[t, i, (j + 1) % L] + configs[t, i, (j - 1) % L])
                gradients[t, idx] = -J * nn_sum - h

    return gradients


# =========================================================================
# Analysis
# =========================================================================

def analyze_ising_at_temperature(T, L=8, n_steps=80000, n_burn=20000):
    """Run Ising + TB analysis at a given temperature."""
    configs = metropolis_sample(L, T, n_steps=n_steps, n_burn=n_burn)
    gradients = compute_ising_gradients(configs)
    features = compute_geometric_features(gradients)

    # Blanket detection
    is_blanket_otsu, tau = detect_blankets_otsu(features)

    # Spectral analysis
    H_est = features['hessian_est']
    from scipy.linalg import eigh
    A = build_adjacency_from_hessian(H_est)
    Lap = build_graph_laplacian(A)
    eigvals, eigvecs = eigh(Lap)
    n_clusters, eigengap = compute_eigengap(eigvals[:min(20, len(eigvals))])

    # Count effective objects
    n_blanket = int(np.sum(is_blanket_otsu))
    n_internal = L * L - n_blanket

    return {
        'temperature': T,
        'T_over_Tc': float(T / TC_ISING),
        'n_blanket': n_blanket,
        'n_internal': n_internal,
        'blanket_fraction': float(n_blanket / (L * L)),
        'eigengap': float(eigengap),
        'n_clusters_spectral': int(n_clusters),
        'eigenvalues': eigvals[:10].tolist(),
        'is_blanket': is_blanket_otsu,
        'final_config': configs[-1],
        'coupling': features['coupling'],
    }


# =========================================================================
# Visualization
# =========================================================================

def plot_ising_results(results_by_temp, L):
    """Three-panel figure: Ising config + blanket overlay at each temperature."""
    temps = sorted(results_by_temp.keys())
    n_temps = len(temps)

    fig, axes = plt.subplots(2, n_temps, figsize=(5 * n_temps, 10))

    for col, T in enumerate(temps):
        r = results_by_temp[T]
        config = r['final_config']
        is_blanket = r['is_blanket'].reshape(L, L)

        # Top row: Ising configuration
        ax = axes[0, col]
        ax.imshow(config, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
        label = 'ordered' if T < TC_ISING * 0.8 else ('critical' if abs(T - TC_ISING) < 0.5 else 'disordered')
        ax.set_title(f'T={T:.2f} ({label})\nT/Tc={T/TC_ISING:.2f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Bottom row: blanket overlay
        ax = axes[1, col]
        ax.imshow(config, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest', alpha=0.4)
        blanket_overlay = np.ma.masked_where(~is_blanket, np.ones((L, L)))
        ax.imshow(blanket_overlay, cmap='Greens', vmin=0, vmax=1, interpolation='nearest', alpha=0.7)
        n_b = int(np.sum(is_blanket))
        ax.set_title(f'Blanket overlay ({n_b}/{L*L} vars)\neigengap={r["eigengap"]:.3f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_ising_phase_diagram(sweep_results):
    """Temperature vs n_objects and eigengap curves."""
    temps = sorted(sweep_results.keys())
    n_blankets = [sweep_results[T]['n_blanket'] for T in temps]
    eigengaps = [sweep_results[T]['eigengap'] for T in temps]
    n_clusters = [sweep_results[T]['n_clusters_spectral'] for T in temps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Blanket fraction vs T
    ax = axes[0]
    blanket_frac = [sweep_results[T]['blanket_fraction'] for T in temps]
    ax.plot(temps, blanket_frac, 'o-', color='#2ecc71', linewidth=2)
    ax.axvline(x=TC_ISING, color='red', linestyle='--', alpha=0.5, label=f'Tc={TC_ISING:.2f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Blanket Fraction')
    ax.set_title('Blanket Fraction vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eigengap vs T
    ax = axes[1]
    ax.plot(temps, eigengaps, 's-', color='#9b59b6', linewidth=2)
    ax.axvline(x=TC_ISING, color='red', linestyle='--', alpha=0.5, label=f'Tc={TC_ISING:.2f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Eigengap')
    ax.set_title('Spectral Gap vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # N_clusters vs T
    ax = axes[2]
    ax.plot(temps, n_clusters, 'D-', color='#e74c3c', linewidth=2)
    ax.axvline(x=TC_ISING, color='red', linestyle='--', alpha=0.5, label=f'Tc={TC_ISING:.2f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Detected Clusters (spectral)')
    ax.set_title('Detected Objects vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =========================================================================
# Main experiment
# =========================================================================

def run_ising_experiment():
    """Run the full Ising model demonstration."""
    print("=" * 70)
    print("US-020: Ising Model Demonstration")
    print("=" * 70)

    L = 8
    print(f"Lattice size: {L}x{L} = {L*L} spins")
    print(f"Critical temperature Tc = {TC_ISING:.3f}")

    # Three key temperatures
    key_temps = [1.0, TC_ISING, 4.0]
    key_results = {}

    for T in key_temps:
        label = 'ordered' if T < TC_ISING * 0.8 else ('critical' if abs(T - TC_ISING) < 0.5 else 'disordered')
        print(f"\n--- T={T:.2f} ({label}, T/Tc={T/TC_ISING:.2f}) ---")
        r = analyze_ising_at_temperature(T, L=L)
        key_results[T] = r
        print(f"  Blanket: {r['n_blanket']}/{L*L} vars ({r['blanket_fraction']:.2f})")
        print(f"  Eigengap: {r['eigengap']:.3f}")
        print(f"  Spectral clusters: {r['n_clusters_spectral']}")

    # Three-panel visualization
    fig_main = plot_ising_results(key_results, L)
    save_figure(fig_main, 'ising_three_temps', 'ising_model')

    # Temperature sweep for phase diagram
    print("\n--- Temperature sweep ---")
    sweep_temps = [0.5, 1.0, 1.5, 2.0, TC_ISING, 2.5, 3.0, 3.5, 4.0, 5.0]
    sweep_results = {}

    for T in sweep_temps:
        r = analyze_ising_at_temperature(T, L=L, n_steps=50000, n_burn=15000)
        sweep_results[T] = r
        print(f"  T={T:.2f}: blanket_frac={r['blanket_fraction']:.2f}, "
              f"eigengap={r['eigengap']:.3f}, clusters={r['n_clusters_spectral']}")

    fig_phase = plot_ising_phase_diagram(sweep_results)
    save_figure(fig_phase, 'ising_phase_diagram', 'ising_model')

    # Save results (exclude large arrays)
    metrics = {}
    for T in sweep_temps:
        r = sweep_results[T]
        metrics[str(T)] = {
            'temperature': r['temperature'],
            'T_over_Tc': r['T_over_Tc'],
            'n_blanket': r['n_blanket'],
            'blanket_fraction': r['blanket_fraction'],
            'eigengap': r['eigengap'],
            'n_clusters_spectral': r['n_clusters_spectral'],
            'eigenvalues': r['eigenvalues'],
        }

    config = {
        'lattice_size': L,
        'n_spins': L * L,
        'Tc': float(TC_ISING),
        'J': 1.0,
        'h': 0.0,
        'key_temperatures': key_temps,
        'sweep_temperatures': sweep_temps,
    }

    save_results('ising_model', metrics, config,
                 notes='US-020: 2D Ising model. TB detects domain walls. Phase transition in eigengap.')

    print("\nUS-020 complete.")
    return sweep_results


if __name__ == '__main__':
    run_ising_experiment()
