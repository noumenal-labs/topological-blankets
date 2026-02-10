"""
Wall-Clock Benchmark: Factored vs Monolithic Forward Pass (US-053)
==================================================================

Compares wall-clock latency of two forward-pass strategies:

1. **Monolithic**: A single dense matrix multiply y = W @ x over the
   full state dimension n.  Cost: O(n^2).

2. **Factored (TB-partitioned)**: Given a Topological Blankets partition
   that assigns variables to k objects plus blanket coupling terms,
   perform k smaller intra-object multiplies and a blanket coupling
   pass, then concatenate.  Cost: sum_i O(n_i^2) + blanket overhead,
   which is strictly less than O(n^2) when objects are balanced.

The benchmark sweeps state dimensions {8, 64, 128, 256, 512, 1024},
generates synthetic TB-like partitions (2-3 objects, ~15% blanket),
and records timing statistics over 1000 iterations per dimension
on both CPU (numpy) and GPU (PyTorch, if available).

Outputs:
  - Speedup vs dimension plot
  - Latency histogram at 256D
  - Actual vs theoretical speedup comparison table
  - Real-time feasibility assessment (can factored inference meet <10ms at 256D?)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)
sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# Synthetic TB partition generation
# =========================================================================

def generate_tb_partition(n_dim, n_objects=3, blanket_frac=0.15, rng=None):
    """
    Generate a synthetic TB-like partition for a state space of size n_dim.

    Returns:
        partition: dict with keys:
            - 'objects': list of arrays, each containing variable indices
                         for one object
            - 'blanket': array of blanket variable indices
            - 'assignment': full assignment array (-1 = blanket)
            - 'n_objects': number of objects
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_blanket = max(1, int(round(n_dim * blanket_frac)))
    n_internal = n_dim - n_blanket

    # Divide internal variables roughly equally among objects
    base_size = n_internal // n_objects
    remainder = n_internal % n_objects
    object_sizes = [base_size + (1 if i < remainder else 0)
                    for i in range(n_objects)]

    # Assign indices: objects first, then blanket
    assignment = np.full(n_dim, -1, dtype=int)
    objects = []
    idx = 0
    for obj_id, size in enumerate(object_sizes):
        obj_indices = np.arange(idx, idx + size)
        objects.append(obj_indices)
        assignment[idx:idx + size] = obj_id
        idx += size
    blanket_indices = np.arange(idx, n_dim)

    return {
        'objects': objects,
        'blanket': blanket_indices,
        'assignment': assignment,
        'n_objects': n_objects,
        'object_sizes': object_sizes,
        'n_blanket': n_blanket,
    }


# =========================================================================
# Weight matrix generation (block-structured)
# =========================================================================

def generate_weight_matrix(n_dim, partition, rng=None):
    """
    Generate a dense weight matrix W of shape (n_dim, n_dim) that has
    block structure consistent with the given TB partition.

    Intra-object blocks have strong weights; blanket rows/cols couple
    objects weakly.  The monolithic pass uses W directly.  The factored
    pass uses the extracted sub-blocks.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    W = rng.randn(n_dim, n_dim).astype(np.float64) * 0.01  # weak background

    # Strong intra-object blocks
    for obj_indices in partition['objects']:
        block = rng.randn(len(obj_indices), len(obj_indices)) * 0.5
        W[np.ix_(obj_indices, obj_indices)] = block

    # Moderate blanket coupling
    blanket = partition['blanket']
    if len(blanket) > 0:
        for obj_indices in partition['objects']:
            coupling = rng.randn(len(blanket), len(obj_indices)) * 0.1
            W[np.ix_(blanket, obj_indices)] = coupling
            coupling_t = rng.randn(len(obj_indices), len(blanket)) * 0.1
            W[np.ix_(obj_indices, blanket)] = coupling_t

    return W


# =========================================================================
# Forward pass implementations
# =========================================================================

def monolithic_forward(W, x):
    """Standard dense forward pass: y = W @ x."""
    return W @ x


def factored_forward(W, x, partition):
    """
    Factored forward pass using TB partition structure.

    For each object, compute the intra-object contribution using only the
    object's sub-block of W.  Then compute blanket coupling contributions
    (blanket -> objects and objects -> blanket).  Assemble the full output.

    This avoids the full n x n multiply by performing k smaller multiplies
    plus blanket coupling terms.
    """
    n = len(x)
    y = np.zeros(n, dtype=x.dtype)

    objects = partition['objects']
    blanket = partition['blanket']

    # 1. Intra-object multiplies: y_obj = W[obj, obj] @ x[obj]
    for obj_indices in objects:
        W_block = W[np.ix_(obj_indices, obj_indices)]
        y[obj_indices] = W_block @ x[obj_indices]

    # 2. Blanket self-coupling: y_blanket = W[blanket, blanket] @ x[blanket]
    if len(blanket) > 0:
        W_blanket = W[np.ix_(blanket, blanket)]
        y[blanket] = W_blanket @ x[blanket]

    # 3. Blanket-to-object coupling
    if len(blanket) > 0:
        for obj_indices in objects:
            # Object receives from blanket
            W_obj_blanket = W[np.ix_(obj_indices, blanket)]
            y[obj_indices] += W_obj_blanket @ x[blanket]

            # Blanket receives from object
            W_blanket_obj = W[np.ix_(blanket, obj_indices)]
            y[blanket] += W_blanket_obj @ x[obj_indices]

    return y


# =========================================================================
# GPU (PyTorch) forward pass implementations
# =========================================================================

def _check_torch_cuda():
    """Check if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def monolithic_forward_gpu(W_t, x_t):
    """GPU dense forward pass."""
    return W_t @ x_t


def factored_forward_gpu(W_t, x_t, obj_slices, blanket_slice, torch_module):
    """
    GPU factored forward pass.

    Uses pre-computed index slices to avoid repeated tensor creation.
    """
    torch = torch_module
    n = x_t.shape[0]
    y = torch.zeros(n, device=x_t.device, dtype=x_t.dtype)

    # Intra-object
    for s_row, s_col in obj_slices:
        y[s_row] = W_t[s_row][:, s_col] @ x_t[s_col]

    # Blanket self
    if blanket_slice is not None:
        y[blanket_slice] = W_t[blanket_slice][:, blanket_slice] @ x_t[blanket_slice]

    # Cross-coupling
    if blanket_slice is not None:
        for s_row, s_col in obj_slices:
            y[s_row] += W_t[s_row][:, blanket_slice] @ x_t[blanket_slice]
            y[blanket_slice] += W_t[blanket_slice][:, s_col] @ x_t[s_col]

    return y


# =========================================================================
# Theoretical speedup calculation
# =========================================================================

def theoretical_speedup(n_dim, partition):
    """
    Compute theoretical speedup ratio from US-034 analysis.

    Monolithic cost: n^2
    Factored cost: sum_i(n_i^2) + n_b^2 + 2 * k * n_b * avg(n_i)
      where n_i = object sizes, n_b = blanket size, k = number of objects

    Speedup = monolithic_cost / factored_cost
    """
    n = n_dim
    mono_cost = n * n

    object_sizes = partition['object_sizes']
    n_b = partition['n_blanket']
    k = partition['n_objects']

    # Intra-object cost
    intra_cost = sum(ni * ni for ni in object_sizes)

    # Blanket self-coupling cost
    blanket_self_cost = n_b * n_b

    # Cross-coupling cost: for each object, two rectangular multiplies
    # (blanket x obj) and (obj x blanket), each costs n_b * n_i
    cross_cost = 2 * sum(n_b * ni for ni in object_sizes)

    factored_cost = intra_cost + blanket_self_cost + cross_cost

    speedup = mono_cost / factored_cost if factored_cost > 0 else 1.0

    return {
        'monolithic_flops': mono_cost,
        'factored_flops': factored_cost,
        'theoretical_speedup': speedup,
        'intra_cost': intra_cost,
        'blanket_self_cost': blanket_self_cost,
        'cross_cost': cross_cost,
    }


# =========================================================================
# Benchmarking harness
# =========================================================================

def benchmark_cpu(dims, n_iters=1000, n_objects=3, blanket_frac=0.15):
    """
    Run CPU (numpy) benchmarks across dimensions.

    Returns:
        results: dict mapping dimension to timing statistics.
    """
    rng = np.random.RandomState(42)
    results = {}

    for n_dim in dims:
        print(f"  CPU benchmark: dim={n_dim}, {n_iters} iterations ...")
        partition = generate_tb_partition(n_dim, n_objects=n_objects,
                                          blanket_frac=blanket_frac, rng=rng)
        W = generate_weight_matrix(n_dim, partition, rng=rng)
        x = rng.randn(n_dim)

        # Verify correctness: factored should produce same result as monolithic
        y_mono = monolithic_forward(W, x)
        y_fact = factored_forward(W, x, partition)
        max_err = np.max(np.abs(y_mono - y_fact))

        # Warmup
        for _ in range(50):
            monolithic_forward(W, x)
            factored_forward(W, x, partition)

        # Monolithic timing
        mono_times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            monolithic_forward(W, x)
            t1 = time.perf_counter()
            mono_times.append(t1 - t0)

        # Factored timing
        fact_times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            factored_forward(W, x, partition)
            t1 = time.perf_counter()
            fact_times.append(t1 - t0)

        mono_times = np.array(mono_times)
        fact_times = np.array(fact_times)

        theory = theoretical_speedup(n_dim, partition)
        actual_speedup = np.mean(mono_times) / np.mean(fact_times)

        results[n_dim] = {
            'mono_mean': float(np.mean(mono_times)),
            'mono_std': float(np.std(mono_times)),
            'mono_median': float(np.median(mono_times)),
            'fact_mean': float(np.mean(fact_times)),
            'fact_std': float(np.std(fact_times)),
            'fact_median': float(np.median(fact_times)),
            'actual_speedup': float(actual_speedup),
            'theoretical_speedup': float(theory['theoretical_speedup']),
            'max_error': float(max_err),
            'partition': {
                'n_objects': n_objects,
                'object_sizes': partition['object_sizes'],
                'n_blanket': partition['n_blanket'],
            },
            'theory': {k: float(v) for k, v in theory.items()},
            'mono_times_all': mono_times.tolist(),
            'fact_times_all': fact_times.tolist(),
        }

        print(f"    Monolithic: {np.mean(mono_times)*1e6:.1f} +/- {np.std(mono_times)*1e6:.1f} us")
        print(f"    Factored:   {np.mean(fact_times)*1e6:.1f} +/- {np.std(fact_times)*1e6:.1f} us")
        print(f"    Speedup:    {actual_speedup:.3f}x (theoretical: {theory['theoretical_speedup']:.3f}x)")
        print(f"    Max error:  {max_err:.2e}")

    return results


def benchmark_gpu(dims, n_iters=1000, n_objects=3, blanket_frac=0.15):
    """
    Run GPU (PyTorch CUDA) benchmarks across dimensions.

    Returns None if CUDA is not available.
    """
    if not _check_torch_cuda():
        print("  GPU benchmark: CUDA not available, skipping.")
        return None

    import torch
    rng = np.random.RandomState(42)
    device = torch.device('cuda')
    results = {}

    for n_dim in dims:
        print(f"  GPU benchmark: dim={n_dim}, {n_iters} iterations ...")
        partition = generate_tb_partition(n_dim, n_objects=n_objects,
                                          blanket_frac=blanket_frac, rng=rng)
        W = generate_weight_matrix(n_dim, partition, rng=rng)
        x = rng.randn(n_dim)

        W_t = torch.tensor(W, dtype=torch.float64, device=device)
        x_t = torch.tensor(x, dtype=torch.float64, device=device)

        # Pre-compute index tensors for factored pass
        obj_slices = []
        for obj_indices in partition['objects']:
            idx = torch.tensor(obj_indices, dtype=torch.long, device=device)
            obj_slices.append((idx, idx))

        blanket = partition['blanket']
        blanket_slice = (torch.tensor(blanket, dtype=torch.long, device=device)
                         if len(blanket) > 0 else None)

        # Warmup + sync
        for _ in range(100):
            monolithic_forward_gpu(W_t, x_t)
            torch.cuda.synchronize()

        # Monolithic timing
        mono_times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            monolithic_forward_gpu(W_t, x_t)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            mono_times.append(t1 - t0)

        # Factored timing
        for _ in range(100):
            factored_forward_gpu(W_t, x_t, obj_slices, blanket_slice, torch)
            torch.cuda.synchronize()

        fact_times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            factored_forward_gpu(W_t, x_t, obj_slices, blanket_slice, torch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fact_times.append(t1 - t0)

        mono_times = np.array(mono_times)
        fact_times = np.array(fact_times)

        theory = theoretical_speedup(n_dim, partition)
        actual_speedup = np.mean(mono_times) / np.mean(fact_times)

        results[n_dim] = {
            'mono_mean': float(np.mean(mono_times)),
            'mono_std': float(np.std(mono_times)),
            'fact_mean': float(np.mean(fact_times)),
            'fact_std': float(np.std(fact_times)),
            'actual_speedup': float(actual_speedup),
            'theoretical_speedup': float(theory['theoretical_speedup']),
            'partition': {
                'n_objects': n_objects,
                'object_sizes': partition['object_sizes'],
                'n_blanket': partition['n_blanket'],
            },
        }

        print(f"    Monolithic: {np.mean(mono_times)*1e6:.1f} +/- {np.std(mono_times)*1e6:.1f} us")
        print(f"    Factored:   {np.mean(fact_times)*1e6:.1f} +/- {np.std(fact_times)*1e6:.1f} us")
        print(f"    Speedup:    {actual_speedup:.3f}x")

    return results


# =========================================================================
# Visualization
# =========================================================================

def plot_speedup_vs_dimension(cpu_results, gpu_results=None):
    """
    Plot (a): Speedup ratio vs state dimension for CPU and GPU.
    """
    dims = sorted(cpu_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: absolute latencies
    ax = axes[0]
    mono_means = [cpu_results[d]['mono_mean'] * 1e6 for d in dims]
    fact_means = [cpu_results[d]['fact_mean'] * 1e6 for d in dims]
    mono_stds = [cpu_results[d]['mono_std'] * 1e6 for d in dims]
    fact_stds = [cpu_results[d]['fact_std'] * 1e6 for d in dims]

    ax.errorbar(dims, mono_means, yerr=mono_stds, marker='s', capsize=3,
                label='Monolithic (CPU)', color='#e74c3c', linewidth=2)
    ax.errorbar(dims, fact_means, yerr=fact_stds, marker='o', capsize=3,
                label='Factored (CPU)', color='#2ecc71', linewidth=2)

    if gpu_results:
        gpu_mono = [gpu_results[d]['mono_mean'] * 1e6 for d in dims if d in gpu_results]
        gpu_fact = [gpu_results[d]['fact_mean'] * 1e6 for d in dims if d in gpu_results]
        gpu_dims = [d for d in dims if d in gpu_results]
        ax.plot(gpu_dims, gpu_mono, '--s', label='Monolithic (GPU)',
                color='#e74c3c', alpha=0.5, linewidth=1.5)
        ax.plot(gpu_dims, gpu_fact, '--o', label='Factored (GPU)',
                color='#2ecc71', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('State Dimension', fontsize=11)
    ax.set_ylabel('Latency (microseconds)', fontsize=11)
    ax.set_title('Forward Pass Latency vs Dimension', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Right panel: speedup ratios
    ax = axes[1]
    actual = [cpu_results[d]['actual_speedup'] for d in dims]
    theoretical = [cpu_results[d]['theoretical_speedup'] for d in dims]

    ax.plot(dims, actual, 'o-', label='Actual speedup (CPU)',
            color='#3498db', linewidth=2, markersize=8)
    ax.plot(dims, theoretical, 's--', label='Theoretical speedup',
            color='#9b59b6', linewidth=2, markersize=8)

    if gpu_results:
        gpu_actual = [gpu_results[d]['actual_speedup']
                      for d in dims if d in gpu_results]
        gpu_dims = [d for d in dims if d in gpu_results]
        ax.plot(gpu_dims, gpu_actual, '^-', label='Actual speedup (GPU)',
                color='#f39c12', linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Break-even')
    ax.set_xlabel('State Dimension', fontsize=11)
    ax.set_ylabel('Speedup Ratio (mono / factored)', fontsize=11)
    ax.set_title('Factored Speedup vs Dimension', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_figure(fig, 'speedup_vs_dimension', 'wall_clock_factored')
    return path


def plot_latency_histogram_256d(cpu_results):
    """
    Plot (b): Latency histogram at 256D comparing monolithic vs factored.
    """
    if 256 not in cpu_results:
        print("  No 256D results for histogram, skipping.")
        return None

    data = cpu_results[256]
    mono_times = np.array(data['mono_times_all']) * 1e6  # to microseconds
    fact_times = np.array(data['fact_times_all']) * 1e6

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(
        min(mono_times.min(), fact_times.min()) * 0.9,
        max(mono_times.max(), fact_times.max()) * 1.1,
        60
    )

    ax.hist(mono_times, bins=bins, alpha=0.6, label='Monolithic', color='#e74c3c',
            edgecolor='white', linewidth=0.5)
    ax.hist(fact_times, bins=bins, alpha=0.6, label='Factored', color='#2ecc71',
            edgecolor='white', linewidth=0.5)

    # Mark means
    ax.axvline(np.mean(mono_times), color='#c0392b', linestyle='--', linewidth=2,
               label=f'Mono mean: {np.mean(mono_times):.1f} us')
    ax.axvline(np.mean(fact_times), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Fact mean: {np.mean(fact_times):.1f} us')

    # 10ms line for reference (10000 us)
    if np.max(mono_times) > 5000:
        ax.axvline(10000, color='black', linestyle=':', linewidth=2,
                   label='10ms real-time target')

    ax.set_xlabel('Latency (microseconds)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Latency Distribution at 256D (CPU, 1000 iterations)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = save_figure(fig, 'latency_histogram_256d', 'wall_clock_factored')
    return path


def plot_actual_vs_theoretical_table(cpu_results, gpu_results=None):
    """
    Plot (c): Table comparing actual vs theoretical speedup across dimensions.
    """
    dims = sorted(cpu_results.keys())

    fig, ax = plt.subplots(figsize=(12, 4 + 0.3 * len(dims)))
    ax.axis('off')

    # Build table data
    headers = ['Dim', 'Objects', 'Blanket',
               'Mono (us)', 'Fact (us)',
               'Actual\nSpeedup', 'Theoretical\nSpeedup',
               'Efficiency\n(Act/Theo)']
    if gpu_results:
        headers.extend(['GPU Mono\n(us)', 'GPU Fact\n(us)', 'GPU\nSpeedup'])

    table_data = []
    for d in dims:
        r = cpu_results[d]
        eff = r['actual_speedup'] / r['theoretical_speedup'] if r['theoretical_speedup'] > 0 else 0
        row = [
            str(d),
            f"{r['partition']['n_objects']} ({','.join(str(s) for s in r['partition']['object_sizes'])})",
            str(r['partition']['n_blanket']),
            f"{r['mono_mean']*1e6:.1f}",
            f"{r['fact_mean']*1e6:.1f}",
            f"{r['actual_speedup']:.3f}x",
            f"{r['theoretical_speedup']:.3f}x",
            f"{eff:.1%}",
        ]
        if gpu_results and d in gpu_results:
            gr = gpu_results[d]
            row.extend([
                f"{gr['mono_mean']*1e6:.1f}",
                f"{gr['fact_mean']*1e6:.1f}",
                f"{gr['actual_speedup']:.3f}x",
            ])
        elif gpu_results:
            row.extend(['N/A', 'N/A', 'N/A'])
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    ax.set_title('Actual vs Theoretical Speedup: Factored Forward Pass',
                 fontsize=13, pad=20)

    plt.tight_layout()
    path = save_figure(fig, 'speedup_comparison_table', 'wall_clock_factored')
    return path


# =========================================================================
# Crossover and real-time feasibility analysis
# =========================================================================

def find_crossover_dimension(cpu_results):
    """
    Identify the crossover dimension where factored becomes faster.

    Returns the smallest dimension d where actual_speedup > 1.0,
    or None if factored is never faster.
    """
    dims = sorted(cpu_results.keys())
    for d in dims:
        if cpu_results[d]['actual_speedup'] > 1.0:
            return d
    return None


def assess_realtime_feasibility(cpu_results, target_ms=10.0):
    """
    Check whether factored inference at 256D meets the <10ms target.
    """
    if 256 not in cpu_results:
        return {'feasible': None, 'note': 'No 256D data available.'}

    data = cpu_results[256]
    fact_mean_ms = data['fact_mean'] * 1000.0
    fact_p99_ms = float(np.percentile(data['fact_times_all'], 99)) * 1000.0
    mono_mean_ms = data['mono_mean'] * 1000.0

    return {
        'target_ms': target_ms,
        'factored_mean_ms': fact_mean_ms,
        'factored_p99_ms': fact_p99_ms,
        'monolithic_mean_ms': mono_mean_ms,
        'feasible_mean': fact_mean_ms < target_ms,
        'feasible_p99': fact_p99_ms < target_ms,
        'headroom_factor': target_ms / fact_mean_ms if fact_mean_ms > 0 else float('inf'),
    }


# =========================================================================
# Main entry point
# =========================================================================

def run_wall_clock_benchmark():
    """Execute the full wall-clock benchmark and generate all outputs."""

    print("=" * 70)
    print("US-053: Wall-Clock Benchmark, Factored vs Monolithic Forward Pass")
    print("=" * 70)

    dims = [8, 64, 128, 256, 512, 1024]
    n_iters = 1000
    n_objects = 3
    blanket_frac = 0.15

    config = {
        'dimensions': dims,
        'n_iterations': n_iters,
        'n_objects': n_objects,
        'blanket_fraction': blanket_frac,
    }

    # ── CPU benchmark ────────────────────────────────────────────────
    print("\n--- CPU (numpy) Benchmark ---")
    cpu_results = benchmark_cpu(dims, n_iters=n_iters,
                                 n_objects=n_objects,
                                 blanket_frac=blanket_frac)

    # ── GPU benchmark ────────────────────────────────────────────────
    print("\n--- GPU (PyTorch CUDA) Benchmark ---")
    gpu_results = benchmark_gpu(dims, n_iters=n_iters,
                                 n_objects=n_objects,
                                 blanket_frac=blanket_frac)

    # ── Crossover analysis ───────────────────────────────────────────
    crossover = find_crossover_dimension(cpu_results)
    print(f"\n--- Crossover Analysis ---")
    if crossover is not None:
        print(f"  Crossover dimension (factored becomes faster): {crossover}D")
    else:
        print("  Factored never becomes faster than monolithic in tested range.")
        # Report which dim is closest
        dims_sorted = sorted(cpu_results.keys())
        best_dim = max(dims_sorted, key=lambda d: cpu_results[d]['actual_speedup'])
        print(f"  Best speedup: {cpu_results[best_dim]['actual_speedup']:.3f}x at {best_dim}D")

    # ── Real-time feasibility at 256D ────────────────────────────────
    rt = assess_realtime_feasibility(cpu_results)
    print(f"\n--- Real-Time Feasibility at 256D ---")
    print(f"  Target:          <{rt['target_ms']:.0f} ms")
    print(f"  Factored mean:   {rt['factored_mean_ms']:.4f} ms")
    print(f"  Factored P99:    {rt['factored_p99_ms']:.4f} ms")
    print(f"  Monolithic mean: {rt['monolithic_mean_ms']:.4f} ms")
    print(f"  Feasible (mean): {rt['feasible_mean']}")
    print(f"  Feasible (P99):  {rt['feasible_p99']}")
    print(f"  Headroom factor: {rt['headroom_factor']:.1f}x under budget")

    # ── Summary table to stdout ──────────────────────────────────────
    print(f"\n--- Summary Table ---")
    print(f"{'Dim':>6s} | {'Mono (us)':>12s} | {'Fact (us)':>12s} | "
          f"{'Actual':>8s} | {'Theory':>8s} | {'Efficiency':>10s}")
    print("-" * 70)
    for d in sorted(cpu_results.keys()):
        r = cpu_results[d]
        eff = r['actual_speedup'] / r['theoretical_speedup']
        print(f"{d:>6d} | {r['mono_mean']*1e6:>12.1f} | {r['fact_mean']*1e6:>12.1f} | "
              f"{r['actual_speedup']:>7.3f}x | {r['theoretical_speedup']:>7.3f}x | "
              f"{eff:>9.1%}")

    # ── Generate plots ───────────────────────────────────────────────
    print("\n--- Generating Plots ---")
    path_speedup = plot_speedup_vs_dimension(cpu_results, gpu_results)
    print(f"  Speedup plot: {path_speedup}")

    path_hist = plot_latency_histogram_256d(cpu_results)
    if path_hist:
        print(f"  Histogram:    {path_hist}")

    path_table = plot_actual_vs_theoretical_table(cpu_results, gpu_results)
    print(f"  Table:        {path_table}")

    # ── Save results JSON ────────────────────────────────────────────
    # Strip raw timing arrays for the JSON (keep them small)
    cpu_results_json = {}
    for d, r in cpu_results.items():
        r_copy = {k: v for k, v in r.items()
                  if k not in ('mono_times_all', 'fact_times_all')}
        cpu_results_json[str(d)] = r_copy

    gpu_results_json = None
    if gpu_results:
        gpu_results_json = {str(d): r for d, r in gpu_results.items()}

    metrics = {
        'cpu_results': cpu_results_json,
        'gpu_results': gpu_results_json,
        'crossover_dimension': crossover,
        'realtime_feasibility': rt,
    }

    save_results(
        'wall_clock_factored_benchmark',
        metrics,
        config,
        notes='US-053: Wall-clock benchmark comparing factored vs monolithic '
              'forward pass across dimensions. Factored pass uses TB partition '
              'structure to decompose the full n x n multiply into k smaller '
              'intra-object blocks plus blanket coupling terms.'
    )

    print("\n" + "=" * 70)
    print("US-053 complete.")
    print("=" * 70)

    return {
        'cpu_results': cpu_results,
        'gpu_results': gpu_results,
        'crossover': crossover,
        'realtime': rt,
        'config': config,
    }


if __name__ == '__main__':
    run_wall_clock_benchmark()
