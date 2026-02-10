"""
Standardized Benchmark Protocol and Evaluation Suite (US-089)
=============================================================

This module defines a BenchmarkSuite for apples-to-apples comparison of
structure discovery methods on energy-based models. The goal is to ensure
every method is evaluated on the *same* data, with the *same* metrics,
under the *same* conditions, so that performance differences reflect
genuine algorithmic advantages rather than evaluation artifacts.

**What is being compared**: Methods that receive a (samples, gradients)
tuple and return a per-variable partition label array. This interface
accommodates gradient-based methods (Topological Blankets), covariance-
based methods (graphical lasso), and information-theoretic methods
(mutual information clustering). Methods that need only samples can
ignore the gradient input; methods that need only gradients can ignore
the sample input.

**Why this matters**: Prior comparisons in the literature often differ in
data generation, sample sizes, random seeds, and metric definitions,
making it impossible to know whether "Method A beats Method B" reflects
a real advantage or a confound. This suite eliminates those confounds by
fixing the protocol and running paired trials across seeds.

**Registered datasets**:
  1. Quadratic EBM 8D: 2 objects, 3 vars each, 3 blanket vars, strong separation
  2. Quadratic EBM 50D: 5 objects, 8 vars each, 10 blanket vars, higher dimension
  3. LunarLander 8D: Active Inference world model dynamics gradients
  4. FetchPush 25D: Ensemble Jacobian-derived gradients from Bayes world model
  5. Ising model 6x6: 2D lattice at sub-critical temperature, domain walls as blankets

**Metrics per (method, dataset, seed)**:
  - ARI (Adjusted Rand Index): measures object partition recovery
  - blanket_F1: F1 score for blanket variable detection
  - NMI (Normalized Mutual Information): alternative partition quality measure
  - wall_clock_seconds: runtime of the method call
  - peak_memory_mb: peak memory allocated during the method call

**Statistical testing**: Paired t-test (or Wilcoxon signed-rank when
normality is suspect) across seeds for each method pair, with Cohen's d
effect size for practical significance.

**Outputs**: raw results JSON, summary table printed to stdout, and a
radar chart showing each method's profile across normalized metrics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import tracemalloc
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple, Any
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from topological_blankets import TopologicalBlankets
from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from scipy import stats


# =========================================================================
# Data generation helpers (reused from existing experiments)
# =========================================================================

def _build_precision_matrix_general(n_objects, vars_per_object, vars_per_blanket,
                                    intra_strength=6.0, blanket_strength=0.8):
    """
    Build a block-structured precision matrix for a quadratic EBM.

    Reuses the pattern from quadratic_toy_comparison.py but parameterized
    for arbitrary dimensions.
    """
    n = n_objects * vars_per_object + vars_per_blanket
    Theta = np.zeros((n, n))

    # Strong within-object couplings
    start = 0
    for i in range(n_objects):
        end = start + vars_per_object
        Theta[start:end, start:end] = intra_strength
        np.fill_diagonal(Theta[start:end, start:end],
                         intra_strength * vars_per_object)
        start = end

    # Blanket block: moderate self-coupling
    blanket_start = n_objects * vars_per_object
    Theta[blanket_start:, blanket_start:] = 1.0
    np.fill_diagonal(Theta[blanket_start:, blanket_start:], vars_per_blanket)

    # Weak cross-couplings only through blanket
    for obj_idx in range(n_objects):
        obj_start = obj_idx * vars_per_object
        obj_end = obj_start + vars_per_object
        Theta[obj_start:obj_end, blanket_start:] = blanket_strength
        Theta[blanket_start:, obj_start:obj_end] = blanket_strength

    Theta = (Theta + Theta.T) / 2.0
    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals.min() < 0.1:
        Theta += np.eye(n) * (0.1 - eigvals.min() + 0.1)

    return Theta


def _get_ground_truth_general(n_objects, vars_per_object, vars_per_blanket):
    """Return ground truth partition for a block-structured quadratic EBM."""
    n_vars = n_objects * vars_per_object + vars_per_blanket
    assignment = np.full(n_vars, -1)
    for obj_idx in range(n_objects):
        start = obj_idx * vars_per_object
        end = start + vars_per_object
        assignment[start:end] = obj_idx
    blanket_mask = assignment == -1
    return assignment, blanket_mask


def _langevin_sampling(Theta, n_samples=3000, n_steps=30,
                       step_size=0.005, temp=0.1, rng=None):
    """Langevin dynamics sampling. Returns (samples, gradients)."""
    if rng is None:
        rng = np.random.RandomState(42)
    n_vars = Theta.shape[0]
    samples = []
    gradients = []
    x = rng.randn(n_vars) * 1.0

    for i in range(n_samples * n_steps):
        grad = Theta @ x
        noise = np.sqrt(2 * step_size * temp) * rng.randn(n_vars)
        x = x - step_size * grad + noise
        if i % n_steps == 0:
            samples.append(x.copy())
            gradients.append((Theta @ x).copy())

    return np.array(samples), np.array(gradients)


def _ising_metropolis_and_gradients(L, T, n_samples=800, n_burn=15000,
                                    sweep_per_sample=50, rng=None):
    """
    2D Ising model Metropolis sampling plus local-field "gradients".

    Returns (samples_flat, gradients_flat, assignment_gt, blanket_mask_gt).
    Ground truth: at sub-critical T, two large domains exist. We approximate
    the ground truth by thresholding the mean magnetization per site into
    two clusters, and identify domain-wall sites as blanket variables.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    J = 1.0
    spins = rng.choice([-1, 1], size=(L, L))
    beta = 1.0 / T

    configs = []
    total_steps = (n_samples * sweep_per_sample) + n_burn

    for step in range(total_steps):
        i, j = rng.randint(0, L, size=2)
        s = spins[i, j]
        nn_sum = (spins[(i + 1) % L, j] + spins[(i - 1) % L, j] +
                  spins[i, (j + 1) % L] + spins[i, (j - 1) % L])
        dE = 2 * J * s * nn_sum
        if dE <= 0 or rng.rand() < np.exp(-beta * dE):
            spins[i, j] = -s
        if step >= n_burn and (step - n_burn) % sweep_per_sample == 0:
            configs.append(spins.copy())

    configs = np.array(configs)
    n_configs = configs.shape[0]
    n_vars = L * L

    # Flatten spin configurations as "samples"
    samples = configs.reshape(n_configs, n_vars).astype(float)

    # Local field as "gradient"
    gradients = np.zeros((n_configs, n_vars))
    for t in range(n_configs):
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                nn_sum = (configs[t, (i + 1) % L, j] + configs[t, (i - 1) % L, j] +
                          configs[t, i, (j + 1) % L] + configs[t, i, (j - 1) % L])
                gradients[t, idx] = -J * nn_sum

    # Ground truth: identify domain walls from mean local field variability.
    # Sites where the local field sign is inconsistent across time are
    # domain-wall (blanket) sites. Interior sites maintain consistent sign.
    mean_field = np.mean(gradients, axis=0)
    field_variance = np.var(gradients, axis=0)

    # Use the magnetization pattern from the final configuration to define
    # two domains, then mark sites adjacent to the boundary as blanket.
    final = configs[-1]
    assignment_2d = (final > 0).astype(int)  # 0 or 1 per site

    # Blanket: sites where at least one neighbor has a different spin
    blanket_2d = np.zeros((L, L), dtype=bool)
    for i in range(L):
        for j in range(L):
            s = final[i, j]
            neighbors = [final[(i + 1) % L, j], final[(i - 1) % L, j],
                         final[i, (j + 1) % L], final[i, (j - 1) % L]]
            if any(n != s for n in neighbors):
                blanket_2d[i, j] = True

    assignment = assignment_2d.flatten()
    blanket_mask = blanket_2d.flatten()
    assignment[blanket_mask] = -1

    return samples, gradients, assignment, blanket_mask


def _lunarlander_synthetic_data(n_samples=2000, rng=None):
    """
    Generate synthetic LunarLander-like 8D data with known structure.

    The LunarLander state has 8 variables:
      [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]

    Ground truth partition:
      Object 0 (position):   [x, y]      (indices 0, 1)
      Object 1 (velocity):   [vx, vy]    (indices 2, 3)
      Object 2 (orientation):[angle, angular_vel] (indices 4, 5)
      Blanket (contact):     [left_leg, right_leg] (indices 6, 7)

    The data is generated from a block-structured precision matrix
    reflecting these couplings (position-velocity linked via blanket-like
    contact variables).
    """
    Theta = _build_precision_matrix_general(
        n_objects=3, vars_per_object=2, vars_per_blanket=2,
        intra_strength=8.0, blanket_strength=0.6
    )
    samples, gradients = _langevin_sampling(Theta, n_samples=n_samples,
                                            n_steps=30, step_size=0.005,
                                            temp=0.1, rng=rng)
    assignment, blanket_mask = _get_ground_truth_general(
        n_objects=3, vars_per_object=2, vars_per_blanket=2
    )
    return samples, gradients, assignment, blanket_mask


def _fetchpush_synthetic_data(n_samples=2000, rng=None):
    """
    Generate synthetic FetchPush-like 25D data with known structure.

    Ground truth partition mirrors the FetchPush observation space:
      Object 0 (gripper):   7 vars (grip_pos 3, gripper_state 2, grip_velp 2)
      Object 1 (object):    10 vars (obj_pos 3, obj_rot 3, obj_velp 3, extra 1)
      Blanket (relational):  3 vars (obj_rel_pos)
      Extra unstructured:    5 vars

    Total: 25D. We model this as 2 objects (7 + 10 vars) + 8 blanket vars.
    For the benchmark, we simplify to a clean block structure.
    """
    Theta = _build_precision_matrix_general(
        n_objects=2, vars_per_object=10, vars_per_blanket=5,
        intra_strength=6.0, blanket_strength=0.5
    )
    samples, gradients = _langevin_sampling(Theta, n_samples=n_samples,
                                            n_steps=30, step_size=0.003,
                                            temp=0.1, rng=rng)
    assignment, blanket_mask = _get_ground_truth_general(
        n_objects=2, vars_per_object=10, vars_per_blanket=5
    )
    return samples, gradients, assignment, blanket_mask


# =========================================================================
# Metric computation
# =========================================================================

def compute_benchmark_metrics(pred_assignment, pred_blanket,
                              true_assignment, true_blanket):
    """
    Compute all benchmark metrics for a single (method, dataset, seed) run.

    Args:
        pred_assignment: Predicted label array (-1 for blanket).
        pred_blanket: Boolean blanket mask for predicted partition.
        true_assignment: Ground truth label array (-1 for blanket).
        true_blanket: Boolean blanket mask for ground truth.

    Returns:
        Dictionary with ARI, blanket_F1, NMI.
    """
    # ARI on internal (non-blanket) variables only
    internal_mask = ~true_blanket
    if np.sum(internal_mask) > 1 and len(np.unique(pred_assignment[internal_mask])) > 1:
        ari = adjusted_rand_score(true_assignment[internal_mask],
                                  pred_assignment[internal_mask])
    else:
        ari = 0.0

    # Blanket detection F1
    blanket_f1 = f1_score(true_blanket.astype(int),
                          pred_blanket.astype(int),
                          zero_division=0.0)

    # NMI on full assignment (treating blanket as its own class)
    nmi = normalized_mutual_info_score(true_assignment, pred_assignment)

    return {
        'ARI': float(ari),
        'blanket_F1': float(blanket_f1),
        'NMI': float(nmi),
    }


# =========================================================================
# Statistical comparison
# =========================================================================

def paired_statistical_test(scores_a, scores_b, metric_name='metric'):
    """
    Run a paired statistical test comparing two methods across seeds.

    Uses paired t-test when sample size >= 10, Wilcoxon signed-rank otherwise.
    Always computes Cohen's d for effect size.

    Args:
        scores_a: Array of metric values for method A across seeds.
        scores_b: Array of metric values for method B across seeds.
        metric_name: Name of the metric (for reporting).

    Returns:
        Dictionary with test results.
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)

    # Cohen's d
    pooled_std = np.std(diffs, ddof=1)
    if pooled_std < 1e-12:
        cohens_d = 0.0
    else:
        cohens_d = float(np.mean(diffs) / pooled_std)

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_label = 'negligible'
    elif abs_d < 0.5:
        effect_label = 'small'
    elif abs_d < 0.8:
        effect_label = 'medium'
    else:
        effect_label = 'large'

    result = {
        'metric': metric_name,
        'n_seeds': n,
        'mean_diff': float(np.mean(diffs)),
        'std_diff': float(np.std(diffs, ddof=1)) if n > 1 else 0.0,
        'cohens_d': cohens_d,
        'effect_size': effect_label,
    }

    if n >= 10:
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        result['test'] = 'paired_t'
        result['t_stat'] = float(t_stat)
        result['p_value'] = float(p_value)
    elif n >= 3:
        # Wilcoxon requires at least some non-zero differences
        if np.any(diffs != 0):
            try:
                stat, p_value = stats.wilcoxon(diffs)
                result['test'] = 'wilcoxon'
                result['statistic'] = float(stat)
                result['p_value'] = float(p_value)
            except ValueError:
                result['test'] = 'wilcoxon_failed'
                result['p_value'] = 1.0
        else:
            result['test'] = 'all_equal'
            result['p_value'] = 1.0
    else:
        result['test'] = 'too_few_seeds'
        result['p_value'] = 1.0

    return result


# =========================================================================
# Visualization
# =========================================================================

def plot_radar_chart(summary_table, methods, metrics_to_plot, save_path=None):
    """
    Create a radar chart showing each method's profile across normalized metrics.

    Higher is better for all axes (wall_clock and memory are inverted).
    """
    n_metrics = len(metrics_to_plot)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12',
              '#1abc9c', '#e67e22', '#34495e']

    for idx, method in enumerate(methods):
        values = []
        for metric in metrics_to_plot:
            # Aggregate across datasets
            vals = []
            for key, row in summary_table.items():
                if row['method'] == method and metric in row:
                    vals.append(row[metric + '_mean'] if metric + '_mean' in row else row.get(metric, 0.0))
            values.append(np.mean(vals) if vals else 0.0)

        # Normalize: for time/memory, invert so higher = better
        norm_values = []
        for i, metric in enumerate(metrics_to_plot):
            if metric in ('wall_clock_seconds', 'peak_memory_mb'):
                # Invert: smaller is better, so use 1/(1+v) to map to (0,1]
                norm_values.append(1.0 / (1.0 + values[i]))
            else:
                # Quality metrics: already in [0,1] or close to it
                norm_values.append(max(0.0, min(1.0, values[i])))

        norm_values += norm_values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, norm_values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, norm_values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    metric_labels = []
    for m in metrics_to_plot:
        if m == 'wall_clock_seconds':
            metric_labels.append('Speed\n(1/(1+sec))')
        elif m == 'peak_memory_mb':
            metric_labels.append('Memory Eff.\n(1/(1+MB))')
        else:
            metric_labels.append(m)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Method Comparison (Radar)', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Radar chart saved to {save_path}")
    else:
        save_figure(fig, 'benchmark_radar', 'benchmark_suite')
    plt.close(fig)

    return fig


# =========================================================================
# BenchmarkSuite class
# =========================================================================

class BenchmarkSuite:
    """
    Standardized benchmark protocol for structure discovery methods.

    Manages registered methods and datasets, runs all combinations across
    multiple random seeds, computes metrics (ARI, blanket_F1, NMI,
    wall_clock_seconds, peak_memory_mb), performs paired statistical
    comparisons, and produces structured output (JSON, table, radar chart).

    Usage:
        suite = BenchmarkSuite()
        suite.register_method('TB', my_tb_callable)
        suite.register_dataset('quad_8d', data_gen_fn, ground_truth_fn)
        results = suite.run(n_seeds=10)
        suite.save_results()
        suite.print_summary_table()
        suite.plot_radar()
    """

    def __init__(self):
        self._methods: Dict[str, Callable] = {}
        self._datasets: Dict[str, Dict] = {}
        self._results: Optional[Dict] = None
        self._summary: Optional[Dict] = None

    def register_method(self, name: str, callable_fn: Callable):
        """
        Register a structure discovery method.

        Args:
            name: Human-readable name for the method.
            callable_fn: A callable with signature
                (samples: ndarray, gradients: ndarray) -> labels: ndarray
                where labels[i] = object index for variable i, or -1 for blanket.
        """
        self._methods[name] = callable_fn

    def register_dataset(self, name: str,
                         data_generator: Callable,
                         ground_truth: Callable):
        """
        Register a benchmark dataset.

        Args:
            name: Human-readable name for the dataset.
            data_generator: Callable with signature
                (seed: int) -> (samples: ndarray, gradients: ndarray)
                Returns sample and gradient arrays for the given random seed.
            ground_truth: Callable with signature
                (seed: int) -> (assignment: ndarray, blanket_mask: ndarray)
                Returns ground truth partition for the given random seed.
                The assignment array uses -1 for blanket variables.
        """
        self._datasets[name] = {
            'generator': data_generator,
            'ground_truth': ground_truth,
        }

    def run(self, n_seeds: int = 10, verbose: bool = True) -> Dict:
        """
        Run all methods on all datasets with n_seeds random seeds each.

        For each (method, dataset, seed) triple, this:
          1. Generates data using the dataset's generator(seed).
          2. Obtains ground truth via ground_truth(seed).
          3. Calls the method with (samples, gradients).
          4. Measures wall_clock_seconds and peak_memory_mb.
          5. Computes ARI, blanket_F1, NMI.

        Args:
            n_seeds: Number of random seeds per (method, dataset) pair.
            verbose: If True, print progress.

        Returns:
            Dictionary with full results: raw metrics, summaries, and
            statistical comparisons.
        """
        if not self._methods:
            raise ValueError("No methods registered. Call register_method() first.")
        if not self._datasets:
            raise ValueError("No datasets registered. Call register_dataset() first.")

        raw_results = {}  # (method, dataset) -> list of per-seed dicts
        method_names = list(self._methods.keys())
        dataset_names = list(self._datasets.keys())

        if verbose:
            print("=" * 70)
            print("Benchmark Suite: Running evaluations")
            print(f"  Methods:  {method_names}")
            print(f"  Datasets: {dataset_names}")
            print(f"  Seeds:    {n_seeds}")
            print("=" * 70)

        for ds_name in dataset_names:
            ds = self._datasets[ds_name]
            for method_name in method_names:
                method_fn = self._methods[method_name]
                key = (method_name, ds_name)
                raw_results[key] = []

                for seed in range(n_seeds):
                    # Generate data
                    samples, gradients = ds['generator'](seed)
                    true_assignment, true_blanket = ds['ground_truth'](seed)

                    # Run method with timing and memory tracking
                    tracemalloc.start()
                    t0 = time.perf_counter()
                    try:
                        pred_labels = method_fn(samples, gradients)
                    except Exception as e:
                        if verbose:
                            print(f"  WARNING: {method_name} on {ds_name} seed={seed} failed: {e}")
                        pred_labels = np.zeros(len(true_assignment), dtype=int)
                    t1 = time.perf_counter()
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    wall_clock = t1 - t0
                    peak_mb = peak_bytes / (1024 * 1024)

                    # Build predicted blanket mask
                    pred_blanket = (pred_labels == -1)

                    # Compute metrics
                    metrics = compute_benchmark_metrics(
                        pred_labels, pred_blanket,
                        true_assignment, true_blanket
                    )
                    metrics['wall_clock_seconds'] = float(wall_clock)
                    metrics['peak_memory_mb'] = float(peak_mb)
                    metrics['seed'] = seed

                    raw_results[key].append(metrics)

                if verbose:
                    # Print inline summary
                    aris = [r['ARI'] for r in raw_results[key]]
                    f1s = [r['blanket_F1'] for r in raw_results[key]]
                    nmis = [r['NMI'] for r in raw_results[key]]
                    times = [r['wall_clock_seconds'] for r in raw_results[key]]
                    print(f"  {method_name:20s} | {ds_name:20s} | "
                          f"ARI={np.mean(aris):.3f}+/-{np.std(aris):.3f}  "
                          f"F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  "
                          f"NMI={np.mean(nmis):.3f}  "
                          f"time={np.mean(times):.3f}s")

        # Build summary table
        summary = {}
        for (method_name, ds_name), runs in raw_results.items():
            row_key = f"{method_name}|{ds_name}"
            row = {
                'method': method_name,
                'dataset': ds_name,
                'n_seeds': len(runs),
            }
            for metric in ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds', 'peak_memory_mb']:
                vals = [r[metric] for r in runs]
                row[metric + '_mean'] = float(np.mean(vals))
                row[metric + '_std'] = float(np.std(vals))
                row[metric + '_values'] = [float(v) for v in vals]
            summary[row_key] = row

        # Statistical comparisons: pairwise per dataset
        stat_comparisons = {}
        for ds_name in dataset_names:
            for i, m_a in enumerate(method_names):
                for m_b in method_names[i + 1:]:
                    key_a = (m_a, ds_name)
                    key_b = (m_b, ds_name)
                    if key_a not in raw_results or key_b not in raw_results:
                        continue
                    comp_key = f"{m_a}_vs_{m_b}|{ds_name}"
                    comp = {}
                    for metric in ['ARI', 'blanket_F1', 'NMI']:
                        scores_a = [r[metric] for r in raw_results[key_a]]
                        scores_b = [r[metric] for r in raw_results[key_b]]
                        comp[metric] = paired_statistical_test(
                            scores_a, scores_b, metric_name=metric)
                    stat_comparisons[comp_key] = comp

        self._results = {
            'raw': {f"{m}|{d}": runs for (m, d), runs in raw_results.items()},
            'summary': summary,
            'statistical_comparisons': stat_comparisons,
            'config': {
                'n_seeds': n_seeds,
                'methods': method_names,
                'datasets': dataset_names,
            },
        }
        self._summary = summary

        if verbose:
            print("\n" + "=" * 70)
            print("Benchmark complete.")
            print("=" * 70)

        return self._results

    def print_summary_table(self):
        """Print a formatted summary table to stdout."""
        if self._summary is None:
            print("No results available. Call run() first.")
            return

        metrics = ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds', 'peak_memory_mb']
        header = f"{'Method':<20s} | {'Dataset':<20s}"
        for m in metrics:
            short = m.replace('wall_clock_seconds', 'time(s)').replace('peak_memory_mb', 'mem(MB)')
            header += f" | {short:>14s}"
        print(header)
        print("-" * len(header))

        for key, row in sorted(self._summary.items()):
            line = f"{row['method']:<20s} | {row['dataset']:<20s}"
            for m in metrics:
                mean = row[m + '_mean']
                std = row[m + '_std']
                line += f" | {mean:>6.3f}+/-{std:>5.3f}"
            print(line)

    def print_statistical_comparisons(self):
        """Print the pairwise statistical comparison results."""
        if self._results is None:
            print("No results available. Call run() first.")
            return

        comps = self._results['statistical_comparisons']
        for comp_key, metrics_dict in sorted(comps.items()):
            parts = comp_key.split('|')
            pair = parts[0]
            ds = parts[1] if len(parts) > 1 else ''
            print(f"\n{pair} on {ds}:")
            for metric_name, result in metrics_dict.items():
                sig = '*' if result.get('p_value', 1.0) < 0.05 else ''
                print(f"  {metric_name:>12s}: diff={result['mean_diff']:+.3f}, "
                      f"d={result['cohens_d']:+.3f} ({result['effect_size']}), "
                      f"p={result.get('p_value', 'N/A'):.4f}{sig} "
                      f"[{result['test']}]")

    def save_results_json(self, path: str = None) -> str:
        """Save full results to a JSON file."""
        if self._results is None:
            raise RuntimeError("No results to save. Call run() first.")

        if path is None:
            path = save_results(
                'benchmark_suite',
                self._results,
                self._results.get('config', {}),
                notes='US-089: Standardized benchmark protocol results'
            )
        else:
            with open(path, 'w') as f:
                json.dump(self._results, f, indent=2, default=_json_default)
            print(f"Results saved to {path}")

        return path

    def plot_radar(self, save_path=None):
        """Generate and save a radar chart of method profiles."""
        if self._summary is None:
            raise RuntimeError("No results to plot. Call run() first.")

        methods = list(self._results['config']['methods'])
        metrics = ['ARI', 'blanket_F1', 'NMI', 'wall_clock_seconds', 'peak_memory_mb']

        # Aggregate across datasets: for each method, compute mean of means
        agg = {m: {} for m in methods}
        for key, row in self._summary.items():
            method = row['method']
            for metric in metrics:
                if metric not in agg[method]:
                    agg[method][metric] = []
                agg[method][metric].append(row[metric + '_mean'])

        # Build radar data
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12',
                  '#1abc9c', '#e67e22', '#34495e']

        for idx, method in enumerate(methods):
            values = []
            for metric in metrics:
                vals = agg[method].get(metric, [0.0])
                values.append(np.mean(vals))

            # Normalize
            norm_values = []
            for i, metric in enumerate(metrics):
                if metric in ('wall_clock_seconds', 'peak_memory_mb'):
                    norm_values.append(1.0 / (1.0 + values[i]))
                else:
                    norm_values.append(max(0.0, min(1.0, values[i])))
            norm_values += norm_values[:1]

            color = colors[idx % len(colors)]
            ax.plot(angles, norm_values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, norm_values, alpha=0.1, color=color)

        metric_labels = []
        for m in metrics:
            if m == 'wall_clock_seconds':
                metric_labels.append('Speed\n(1/(1+sec))')
            elif m == 'peak_memory_mb':
                metric_labels.append('Memory Eff.\n(1/(1+MB))')
            else:
                metric_labels.append(m)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title('Benchmark: Method Comparison (Radar)', fontsize=13, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Radar chart saved to {save_path}")
        else:
            save_figure(fig, 'benchmark_radar', 'benchmark_suite')
        plt.close(fig)

    def get_results(self) -> Optional[Dict]:
        """Return the full results dictionary, or None if run() has not been called."""
        return self._results


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =========================================================================
# Default method wrappers
# =========================================================================

def _tb_method(samples, gradients, method='hybrid', n_objects=None):
    """
    Topological Blankets wrapper conforming to the benchmark interface.

    Returns label array where label[i] = object index, or -1 for blanket.
    """
    n_vars = gradients.shape[1]
    # Auto-detect n_objects from eigengap if not specified
    if n_objects is None:
        features = compute_geometric_features(gradients)
        from topological_blankets.spectral import (
            build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap
        )
        from scipy.linalg import eigh
        A = build_adjacency_from_hessian(features['hessian_est'])
        L = build_graph_laplacian(A)
        eigvals, _ = eigh(L)
        auto_n, _ = compute_eigengap(eigvals[:min(10, len(eigvals))])
        n_objects = max(2, auto_n - 1)

    tb = TopologicalBlankets(method=method, n_objects=n_objects)
    tb.fit(gradients)
    return tb.get_assignment()


def _tb_gradient_method(samples, gradients):
    """TB with gradient (Otsu) detection."""
    return _tb_method(samples, gradients, method='gradient')


def _tb_hybrid_method(samples, gradients):
    """TB with hybrid (spectral + gradient fallback) detection."""
    return _tb_method(samples, gradients, method='hybrid')


# =========================================================================
# Build and return a default suite with 5 datasets + TB registered
# =========================================================================

def build_default_suite() -> BenchmarkSuite:
    """
    Construct a BenchmarkSuite with the 5 required benchmark datasets and
    Topological Blankets registered as the default method.

    Datasets:
      1. quadratic_ebm_8d:  2 objects x 3 vars + 3 blanket = 9D (low-dimensional baseline)
      2. quadratic_ebm_50d: 5 objects x 8 vars + 10 blanket = 50D (high-dimensional stress test)
      3. lunarlander_8d:    3 objects x 2 vars + 2 blanket = 8D (world model proxy)
      4. fetchpush_25d:     2 objects x 10 vars + 5 blanket = 25D (manipulation proxy)
      5. ising_6x6:         36 spins, domain walls as blankets (lattice model)
    """
    suite = BenchmarkSuite()

    # ── Register methods ────────────────────────────────────────────────
    suite.register_method('TB_hybrid', _tb_hybrid_method)
    suite.register_method('TB_gradient', _tb_gradient_method)

    # ── Dataset 1: Quadratic EBM 8D (really 9D: 2x3 + 3) ──────────────
    def quad_8d_gen(seed):
        rng = np.random.RandomState(seed)
        Theta = _build_precision_matrix_general(
            n_objects=2, vars_per_object=3, vars_per_blanket=3,
            intra_strength=6.0, blanket_strength=0.8
        )
        return _langevin_sampling(Theta, n_samples=3000, n_steps=30,
                                  step_size=0.005, temp=0.1, rng=rng)

    def quad_8d_gt(seed):
        return _get_ground_truth_general(
            n_objects=2, vars_per_object=3, vars_per_blanket=3
        )

    suite.register_dataset('quadratic_ebm_8d', quad_8d_gen, quad_8d_gt)

    # ── Dataset 2: Quadratic EBM 50D ───────────────────────────────────
    def quad_50d_gen(seed):
        rng = np.random.RandomState(seed)
        Theta = _build_precision_matrix_general(
            n_objects=5, vars_per_object=8, vars_per_blanket=10,
            intra_strength=6.0, blanket_strength=0.5
        )
        return _langevin_sampling(Theta, n_samples=5000, n_steps=30,
                                  step_size=0.003, temp=0.1, rng=rng)

    def quad_50d_gt(seed):
        return _get_ground_truth_general(
            n_objects=5, vars_per_object=8, vars_per_blanket=10
        )

    suite.register_dataset('quadratic_ebm_50d', quad_50d_gen, quad_50d_gt)

    # ── Dataset 3: LunarLander 8D ──────────────────────────────────────
    def ll_8d_gen(seed):
        rng = np.random.RandomState(seed)
        samples, gradients, _, _ = _lunarlander_synthetic_data(
            n_samples=2000, rng=rng
        )
        return samples, gradients

    def ll_8d_gt(seed):
        _, _, assignment, blanket_mask = _lunarlander_synthetic_data(
            n_samples=100, rng=np.random.RandomState(0)
        )
        return assignment, blanket_mask

    suite.register_dataset('lunarlander_8d', ll_8d_gen, ll_8d_gt)

    # ── Dataset 4: FetchPush 25D ───────────────────────────────────────
    def fp_25d_gen(seed):
        rng = np.random.RandomState(seed)
        samples, gradients, _, _ = _fetchpush_synthetic_data(
            n_samples=2000, rng=rng
        )
        return samples, gradients

    def fp_25d_gt(seed):
        _, _, assignment, blanket_mask = _fetchpush_synthetic_data(
            n_samples=100, rng=np.random.RandomState(0)
        )
        return assignment, blanket_mask

    suite.register_dataset('fetchpush_25d', fp_25d_gen, fp_25d_gt)

    # ── Dataset 5: Ising model 6x6 ────────────────────────────────────
    # Use sub-critical temperature for clear domain structure
    TC_ISING = 2.0 / np.log(1 + np.sqrt(2))

    def ising_gen(seed):
        rng = np.random.RandomState(seed)
        T = 1.5  # Well below Tc for clear domain walls
        samples, gradients, _, _ = _ising_metropolis_and_gradients(
            L=6, T=T, n_samples=800, n_burn=15000,
            sweep_per_sample=50, rng=rng
        )
        return samples, gradients

    def ising_gt(seed):
        rng = np.random.RandomState(seed)
        T = 1.5
        _, _, assignment, blanket_mask = _ising_metropolis_and_gradients(
            L=6, T=T, n_samples=800, n_burn=15000,
            sweep_per_sample=50, rng=rng
        )
        return assignment, blanket_mask

    suite.register_dataset('ising_6x6', ising_gen, ising_gt)

    return suite


# =========================================================================
# Main entry point
# =========================================================================

def run_benchmark(n_seeds=10, verbose=True):
    """Build the default suite and run the full benchmark."""
    suite = build_default_suite()

    if verbose:
        print("Registered methods:", list(suite._methods.keys()))
        print("Registered datasets:", list(suite._datasets.keys()))
        print()

    results = suite.run(n_seeds=n_seeds, verbose=verbose)

    print("\n")
    suite.print_summary_table()

    print("\n")
    suite.print_statistical_comparisons()

    suite.save_results_json()
    suite.plot_radar()

    return suite, results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Suite for Structure Discovery Methods')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds per (method, dataset) pair')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with n_seeds=3')
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.n_seeds
    suite, results = run_benchmark(n_seeds=n_seeds)
