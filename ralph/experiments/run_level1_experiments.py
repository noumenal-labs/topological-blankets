"""
Level 1 Validation Experiments Runner
=====================================

Runs all Level 1 experiments from the paper (Section 10.1):
- US-005: Systematic strength sweep (10 trials, 7 strengths)
- US-006: Spectral vs gradient vs hybrid comparison
- US-007: Scaling experiment (n_objects x vars_per_object)
- US-008: Temperature sensitivity sweep

Each experiment saves JSON results and PNG plots to results/.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, topological_blankets, dmbd_style_partition,
    axiom_style_partition, compute_metrics, run_strength_sweep,
    plot_strength_sweep
)
from experiments.spectral_friston_detection import run_spectral_experiment
from experiments.utils.results import save_results, build_registry
from experiments.utils.plotting import save_figure

# v2 imports: packaged pipeline with all 4 detection methods
from topological_blankets.core import topological_blankets as tb_pipeline


# =========================================================================
# US-005: Systematic Strength Sweep (10 trials, 7 strengths)
# =========================================================================

def run_us005():
    """US-005: Core strength sweep, 10 trials, 7 strengths."""
    print("=" * 70)
    print("US-005: Systematic Strength Sweep (10 trials, 7 strengths)")
    print("=" * 70)

    strengths = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    n_trials = 10

    results = run_strength_sweep(strengths=strengths, n_trials=n_trials, verbose=True)

    # Build structured metrics
    metrics = {}
    for strength in strengths:
        s_key = str(strength)
        metrics[s_key] = {}
        for method in ['tc', 'dmbd', 'axiom']:
            trials = results[strength][method]
            aris = [r['object_ari'] for r in trials]
            f1s = [r['blanket_f1'] for r in trials]
            full_aris = [r['full_ari'] for r in trials]
            metrics[s_key][method] = {
                'per_trial': trials,
                'mean_ari': float(np.mean(aris)),
                'std_ari': float(np.std(aris)),
                'mean_f1': float(np.mean(f1s)),
                'std_f1': float(np.std(f1s)),
                'mean_full_ari': float(np.mean(full_aris)),
                'std_full_ari': float(np.std(full_aris)),
            }

    config = {
        'strengths': strengths,
        'n_trials': n_trials,
        'n_objects': 2,
        'vars_per_object': 3,
        'vars_per_blanket': 3,
        'intra_strength': 6.0,
    }

    save_results('strength_sweep_10trials', metrics, config,
                 notes='US-005: Core Level 1 experiment. 10 trials x 7 strengths x 3 methods.')

    # Save separate ARI and F1 plots
    _plot_strength_sweep_separate(results, strengths)

    print("\nUS-005 complete.")
    return results


def _plot_strength_sweep_separate(results, strengths):
    """Save separate ARI and F1 error-bar plots."""
    methods = ['tc', 'dmbd', 'axiom']
    labels = ['Topological Blankets', 'DMBD-style', 'AXIOM-style']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # ARI plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, color in zip(methods, labels, colors):
        means = [np.mean([r['object_ari'] for r in results[s][method]]) for s in strengths]
        stds = [np.std([r['object_ari'] for r in results[s][method]]) for s in strengths]
        ax.errorbar(strengths, means, yerr=stds, label=label,
                    color=color, marker='o', capsize=3, linewidth=2)
    ax.set_xlabel('Blanket Strength (coupling)')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Object Partition Recovery vs Blanket Strength')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'ari_vs_strength', 'strength_sweep')

    # F1 plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, color in zip(methods, labels, colors):
        means = [np.mean([r['blanket_f1'] for r in results[s][method]]) for s in strengths]
        stds = [np.std([r['blanket_f1'] for r in results[s][method]]) for s in strengths]
        ax.errorbar(strengths, means, yerr=stds, label=label,
                    color=color, marker='o', capsize=3, linewidth=2)
    ax.set_xlabel('Blanket Strength (coupling)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Blanket Detection F1 vs Blanket Strength')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'f1_vs_strength', 'strength_sweep')

    # Combined (same as plot_strength_sweep but saved)
    plot_strength_sweep(results)


# =========================================================================
# US-006: Spectral vs Gradient vs Hybrid Comparison
# =========================================================================

def run_us006():
    """US-006: Run spectral experiment (already saves results via refactored script)."""
    print("\n" + "=" * 70)
    print("US-006: Spectral vs Gradient vs Hybrid Comparison")
    print("=" * 70)

    results = run_spectral_experiment()
    print("\nUS-006 complete.")
    return results


# =========================================================================
# US-007: Scaling Experiment
# =========================================================================

def run_us007():
    """US-007: Vary n_objects and vars_per_object, 5 trials each."""
    print("\n" + "=" * 70)
    print("US-007: Scaling Experiment (n_objects x vars_per_object)")
    print("=" * 70)

    n_objects_list = [2, 3, 4]
    vars_per_object_list = [3, 5, 8]
    n_trials = 5

    metrics = {}

    for n_obj in n_objects_list:
        for vpo in vars_per_object_list:
            key = f"{n_obj}obj_{vpo}vpo"
            print(f"\n--- n_objects={n_obj}, vars_per_object={vpo} ---")

            trial_results = []
            total_time = 0

            for trial in range(n_trials):
                cfg = QuadraticEBMConfig(
                    n_objects=n_obj,
                    vars_per_object=vpo,
                    vars_per_blanket=3,
                    intra_strength=6.0,
                    blanket_strength=0.8,
                )
                Theta = build_precision_matrix(cfg)
                truth = get_ground_truth(cfg)
                n_vars = Theta.shape[0]

                np.random.seed(42 + trial)
                t0 = time.time()
                samples, gradients = langevin_sampling(
                    Theta, n_samples=3000, n_steps=30,
                    step_size=0.005, temp=0.1
                )
                tc_result = topological_blankets(gradients, n_objects=n_obj)
                elapsed = time.time() - t0
                total_time += elapsed

                tc_metrics = compute_metrics(tc_result, truth)
                trial_results.append({
                    'object_ari': tc_metrics['object_ari'],
                    'blanket_f1': tc_metrics['blanket_f1'],
                    'full_ari': tc_metrics['full_ari'],
                    'wall_time_s': elapsed,
                })

            aris = [r['object_ari'] for r in trial_results]
            f1s = [r['blanket_f1'] for r in trial_results]
            metrics[key] = {
                'n_objects': n_obj,
                'vars_per_object': vpo,
                'total_dimension': n_obj * vpo + cfg.vars_per_blanket,
                'per_trial': trial_results,
                'mean_ari': float(np.mean(aris)),
                'std_ari': float(np.std(aris)),
                'mean_f1': float(np.mean(f1s)),
                'std_f1': float(np.std(f1s)),
                'mean_wall_time_s': float(total_time / n_trials),
            }
            print(f"  ARI={np.mean(aris):.3f} +/- {np.std(aris):.3f}, "
                  f"F1={np.mean(f1s):.3f}, dim={n_obj * vpo + cfg.vars_per_blanket}, "
                  f"time={total_time / n_trials:.2f}s")

    config = {
        'n_objects_list': n_objects_list,
        'vars_per_object_list': vars_per_object_list,
        'n_trials': n_trials,
        'vars_per_blanket': 3,
        'blanket_strength': 0.8,
    }

    save_results('scaling_experiment', metrics, config,
                 notes='US-007: Scaling over n_objects x vars_per_object.')

    # Heatmap plot
    _plot_scaling_heatmap(metrics, n_objects_list, vars_per_object_list)

    print("\nUS-007 complete.")
    return metrics


def _plot_scaling_heatmap(metrics, n_objects_list, vars_per_object_list):
    """Heatmap of ARI across (n_objects, vars_per_object)."""
    ari_matrix = np.zeros((len(n_objects_list), len(vars_per_object_list)))
    f1_matrix = np.zeros_like(ari_matrix)

    for i, n_obj in enumerate(n_objects_list):
        for j, vpo in enumerate(vars_per_object_list):
            key = f"{n_obj}obj_{vpo}vpo"
            ari_matrix[i, j] = metrics[key]['mean_ari']
            f1_matrix[i, j] = metrics[key]['mean_f1']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ARI heatmap
    ax = axes[0]
    im = ax.imshow(ari_matrix, cmap='RdYlGn', vmin=-0.1, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(vars_per_object_list)))
    ax.set_xticklabels(vars_per_object_list)
    ax.set_yticks(range(len(n_objects_list)))
    ax.set_yticklabels(n_objects_list)
    ax.set_xlabel('vars_per_object')
    ax.set_ylabel('n_objects')
    ax.set_title('Object ARI (TC method)')
    for i in range(len(n_objects_list)):
        for j in range(len(vars_per_object_list)):
            ax.text(j, i, f"{ari_matrix[i, j]:.2f}", ha='center', va='center',
                    color='black', fontsize=11)
    plt.colorbar(im, ax=ax)

    # F1 heatmap
    ax = axes[1]
    im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(vars_per_object_list)))
    ax.set_xticklabels(vars_per_object_list)
    ax.set_yticks(range(len(n_objects_list)))
    ax.set_yticklabels(n_objects_list)
    ax.set_xlabel('vars_per_object')
    ax.set_ylabel('n_objects')
    ax.set_title('Blanket F1 (TC method)')
    for i in range(len(n_objects_list)):
        for j in range(len(vars_per_object_list)):
            ax.text(j, i, f"{f1_matrix[i, j]:.2f}", ha='center', va='center',
                    color='black', fontsize=11)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_figure(fig, 'scaling_heatmap', 'scaling')


# =========================================================================
# US-008: Temperature Sensitivity Experiment
# =========================================================================

def run_us008():
    """US-008: Sweep temperature, fixed blanket_strength=0.8."""
    print("\n" + "=" * 70)
    print("US-008: Temperature Sensitivity Experiment")
    print("=" * 70)

    temperatures = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    n_trials = 5
    blanket_strength = 0.8

    metrics = {}

    for temp in temperatures:
        t_key = str(temp)
        print(f"\n--- Temperature T={temp} ---")
        metrics[t_key] = {}

        for method_name in ['tc', 'dmbd', 'axiom']:
            metrics[t_key][method_name] = {'per_trial': []}

        for trial in range(n_trials):
            cfg = QuadraticEBMConfig(
                n_objects=2,
                vars_per_object=3,
                vars_per_blanket=3,
                intra_strength=6.0,
                blanket_strength=blanket_strength,
            )
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            samples, gradients = langevin_sampling(
                Theta, n_samples=3000, n_steps=30,
                step_size=0.005, temp=temp
            )

            tc_result = topological_blankets(gradients, n_objects=cfg.n_objects)
            tc_m = compute_metrics(tc_result, truth)
            metrics[t_key]['tc']['per_trial'].append(tc_m)

            dmbd_result = dmbd_style_partition(gradients, n_objects=cfg.n_objects)
            dmbd_m = compute_metrics(dmbd_result, truth)
            metrics[t_key]['dmbd']['per_trial'].append(dmbd_m)

            axiom_result = axiom_style_partition(samples, n_objects=cfg.n_objects,
                                                  gradients=gradients)
            axiom_m = compute_metrics(axiom_result, truth)
            metrics[t_key]['axiom']['per_trial'].append(axiom_m)

        for method_name in ['tc', 'dmbd', 'axiom']:
            trials = metrics[t_key][method_name]['per_trial']
            aris = [r['object_ari'] for r in trials]
            f1s = [r['blanket_f1'] for r in trials]
            metrics[t_key][method_name]['mean_ari'] = float(np.mean(aris))
            metrics[t_key][method_name]['std_ari'] = float(np.std(aris))
            metrics[t_key][method_name]['mean_f1'] = float(np.mean(f1s))
            metrics[t_key][method_name]['std_f1'] = float(np.std(f1s))

        print(f"  TC:    ARI={metrics[t_key]['tc']['mean_ari']:.3f}, "
              f"F1={metrics[t_key]['tc']['mean_f1']:.3f}")
        print(f"  DMBD:  ARI={metrics[t_key]['dmbd']['mean_ari']:.3f}, "
              f"F1={metrics[t_key]['dmbd']['mean_f1']:.3f}")
        print(f"  AXIOM: ARI={metrics[t_key]['axiom']['mean_ari']:.3f}, "
              f"F1={metrics[t_key]['axiom']['mean_f1']:.3f}")

    config = {
        'temperatures': temperatures,
        'n_trials': n_trials,
        'blanket_strength': blanket_strength,
        'n_objects': 2,
        'vars_per_object': 3,
    }

    save_results('temperature_sensitivity', metrics, config,
                 notes='US-008: Temperature sweep. Paper Section 6.7 predicts degradation at high T.')

    # Plot ARI vs temperature
    _plot_temperature_sweep(metrics, temperatures)

    print("\nUS-008 complete.")
    return metrics


def _plot_temperature_sweep(metrics, temperatures):
    """Plot ARI vs temperature for all methods with error bars."""
    methods = ['tc', 'dmbd', 'axiom']
    labels = ['Topological Blankets', 'DMBD-style', 'AXIOM-style']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, color in zip(methods, labels, colors):
        means = [metrics[str(t)][method]['mean_ari'] for t in temperatures]
        stds = [metrics[str(t)][method]['std_ari'] for t in temperatures]
        ax.errorbar(temperatures, means, yerr=stds, label=label,
                    color=color, marker='o', capsize=3, linewidth=2)

    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Object Partition Recovery vs Temperature\n(blanket_strength=0.8)')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'ari_vs_temperature', 'temperature_sensitivity')


# =========================================================================
# US-009: Results Registry
# =========================================================================

def run_us009():
    """US-009: Build and print results registry."""
    print("\n" + "=" * 70)
    print("US-009: Building Results Registry")
    print("=" * 70)

    registry = build_registry()

    print(f"\n{'Experiment':<40s} {'Date':<22s} {'Key Metrics'}")
    print("-" * 90)
    for entry in registry['entries']:
        name = entry['experiment_name']
        ts = entry['timestamp'][:19] if entry['timestamp'] else 'N/A'
        ms = entry.get('metrics_summary', {})
        ms_str = ', '.join(f"{k}={v}" for k, v in ms.items()) if ms else '(nested)'
        print(f"{name:<40s} {ts:<22s} {ms_str}")

    print(f"\nTotal experiments: {registry['total_experiments']}")
    print("\nUS-009 complete.")
    return registry


# =========================================================================
# US-016: v2 Experiments (all TC methods + improved baselines)
# =========================================================================

TC_METHODS = ['gradient', 'spectral', 'hybrid', 'coupling']
TC_LABELS = {
    'gradient': 'TC-gradient',
    'spectral': 'TC-spectral',
    'hybrid': 'TC-hybrid',
    'coupling': 'TC-coupling',
}
TC_COLORS = {
    'gradient': '#2ecc71',
    'spectral': '#9b59b6',
    'hybrid': '#f39c12',
    'coupling': '#1abc9c',
}
BASELINE_COLORS = {'dmbd': '#3498db', 'axiom': '#e74c3c'}


def run_us016_strength_sweep():
    """US-016a: Strength sweep with all 4 TC methods + improved baselines."""
    print("=" * 70)
    print("US-016a: v2 Strength Sweep (4 TC methods + improved baselines)")
    print("=" * 70)

    strengths = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    n_trials = 10
    all_methods = TC_METHODS + ['dmbd', 'axiom']

    results = {s: {m: [] for m in all_methods} for s in strengths}

    for strength in strengths:
        print(f"\nBlanket strength = {strength}")
        print("-" * 40)

        for trial in range(n_trials):
            cfg = QuadraticEBMConfig(
                n_objects=2, vars_per_object=3, vars_per_blanket=3,
                intra_strength=6.0, blanket_strength=strength,
            )
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            samples, gradients = langevin_sampling(
                Theta, n_samples=3000, n_steps=30,
                step_size=0.005, temp=0.1
            )

            # 4 TC methods via packaged pipeline
            for method in TC_METHODS:
                tc_result = tb_pipeline(gradients, n_objects=cfg.n_objects,
                                        method=method)
                results[strength][method].append(compute_metrics(tc_result, truth))

            # Baselines
            dmbd_result = dmbd_style_partition(gradients, n_objects=cfg.n_objects)
            results[strength]['dmbd'].append(compute_metrics(dmbd_result, truth))

            axiom_result = axiom_style_partition(samples, n_objects=cfg.n_objects,
                                                  gradients=gradients)
            results[strength]['axiom'].append(compute_metrics(axiom_result, truth))

        for method in all_methods:
            mean_ari = np.mean([r['object_ari'] for r in results[strength][method]])
            mean_f1 = np.mean([r['blanket_f1'] for r in results[strength][method]])
            label = TC_LABELS.get(method, method.upper())
            print(f"  {label:14s}: ARI={mean_ari:.3f}, F1={mean_f1:.3f}")

    # Build metrics dict
    metrics = {}
    for strength in strengths:
        s_key = str(strength)
        metrics[s_key] = {}
        for method in all_methods:
            trials = results[strength][method]
            aris = [r['object_ari'] for r in trials]
            f1s = [r['blanket_f1'] for r in trials]
            full_aris = [r['full_ari'] for r in trials]
            metrics[s_key][method] = {
                'per_trial': trials,
                'mean_ari': float(np.mean(aris)),
                'std_ari': float(np.std(aris)),
                'mean_f1': float(np.mean(f1s)),
                'std_f1': float(np.std(f1s)),
                'mean_full_ari': float(np.mean(full_aris)),
                'std_full_ari': float(np.std(full_aris)),
            }

    config = {
        'version': 'v2',
        'strengths': strengths,
        'n_trials': n_trials,
        'tc_methods': TC_METHODS,
        'n_objects': 2,
        'vars_per_object': 3,
        'vars_per_blanket': 3,
        'intra_strength': 6.0,
    }

    save_results('v2_strength_sweep', metrics, config,
                 notes='US-016: v2 strength sweep. 4 TC methods + improved DMBD/AXIOM baselines.')

    _plot_v2_strength_sweep(results, strengths)

    print("\nUS-016a (strength sweep) complete.")
    return results


def _plot_v2_strength_sweep(results, strengths):
    """v2 strength sweep plots with all methods."""
    all_methods = TC_METHODS + ['dmbd', 'axiom']
    all_labels = {**TC_LABELS, 'dmbd': 'DMBD-style', 'axiom': 'AXIOM-style'}
    all_colors = {**TC_COLORS, **BASELINE_COLORS}

    for metric_key, ylabel, title_suffix in [
        ('object_ari', 'Adjusted Rand Index', 'Object Partition ARI'),
        ('blanket_f1', 'F1 Score', 'Blanket Detection F1'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for method in all_methods:
            means = [np.mean([r[metric_key] for r in results[s][method]])
                     for s in strengths]
            stds = [np.std([r[metric_key] for r in results[s][method]])
                    for s in strengths]
            ls = '--' if method in ('dmbd', 'axiom') else '-'
            ax.errorbar(strengths, means, yerr=stds,
                        label=all_labels[method], color=all_colors[method],
                        marker='o', capsize=3, linewidth=2, linestyle=ls)
        ax.set_xlabel('Blanket Strength (coupling)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'v2: {title_suffix} vs Blanket Strength')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        save_figure(fig, f'v2_{metric_key}_vs_strength', 'v2_strength_sweep')


def run_us016_scaling():
    """US-016b: Scaling experiment with coupling method added."""
    print("\n" + "=" * 70)
    print("US-016b: v2 Scaling Experiment (includes coupling method)")
    print("=" * 70)

    n_objects_list = [2, 3, 4]
    vars_per_object_list = [3, 5, 8]
    n_trials = 5
    methods_to_test = ['gradient', 'coupling']

    metrics = {}

    for n_obj in n_objects_list:
        for vpo in vars_per_object_list:
            key = f"{n_obj}obj_{vpo}vpo"
            print(f"\n--- n_objects={n_obj}, vars_per_object={vpo} ---")

            metrics[key] = {
                'n_objects': n_obj,
                'vars_per_object': vpo,
                'total_dimension': n_obj * vpo + 3,
            }

            for method in methods_to_test:
                trial_results = []
                total_time = 0

                for trial in range(n_trials):
                    cfg = QuadraticEBMConfig(
                        n_objects=n_obj, vars_per_object=vpo,
                        vars_per_blanket=3, intra_strength=6.0,
                        blanket_strength=0.8,
                    )
                    Theta = build_precision_matrix(cfg)
                    truth = get_ground_truth(cfg)

                    np.random.seed(42 + trial)
                    t0 = time.time()
                    samples, gradients = langevin_sampling(
                        Theta, n_samples=3000, n_steps=30,
                        step_size=0.005, temp=0.1
                    )
                    tc_result = tb_pipeline(gradients, n_objects=n_obj, method=method)
                    elapsed = time.time() - t0
                    total_time += elapsed

                    tc_metrics = compute_metrics(tc_result, truth)
                    trial_results.append({
                        'object_ari': tc_metrics['object_ari'],
                        'blanket_f1': tc_metrics['blanket_f1'],
                        'full_ari': tc_metrics['full_ari'],
                        'wall_time_s': elapsed,
                    })

                aris = [r['object_ari'] for r in trial_results]
                f1s = [r['blanket_f1'] for r in trial_results]
                metrics[key][method] = {
                    'per_trial': trial_results,
                    'mean_ari': float(np.mean(aris)),
                    'std_ari': float(np.std(aris)),
                    'mean_f1': float(np.mean(f1s)),
                    'std_f1': float(np.std(f1s)),
                    'mean_wall_time_s': float(total_time / n_trials),
                }
                print(f"  {method:10s}: ARI={np.mean(aris):.3f}+/-{np.std(aris):.3f}, "
                      f"F1={np.mean(f1s):.3f}, time={total_time / n_trials:.2f}s")

    config = {
        'version': 'v2',
        'n_objects_list': n_objects_list,
        'vars_per_object_list': vars_per_object_list,
        'n_trials': n_trials,
        'methods': methods_to_test,
        'vars_per_blanket': 3,
        'blanket_strength': 0.8,
    }

    save_results('v2_scaling_experiment', metrics, config,
                 notes='US-016: v2 scaling experiment with gradient and coupling methods.')

    _plot_v2_scaling_heatmap(metrics, n_objects_list, vars_per_object_list, methods_to_test)

    print("\nUS-016b (scaling) complete.")
    return metrics


def _plot_v2_scaling_heatmap(metrics, n_objects_list, vars_per_object_list, methods):
    """Heatmaps of ARI for each method."""
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        ari_matrix = np.zeros((len(n_objects_list), len(vars_per_object_list)))
        for i, n_obj in enumerate(n_objects_list):
            for j, vpo in enumerate(vars_per_object_list):
                key = f"{n_obj}obj_{vpo}vpo"
                ari_matrix[i, j] = metrics[key][method]['mean_ari']

        im = ax.imshow(ari_matrix, cmap='RdYlGn', vmin=-0.1, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(vars_per_object_list)))
        ax.set_xticklabels(vars_per_object_list)
        ax.set_yticks(range(len(n_objects_list)))
        ax.set_yticklabels(n_objects_list)
        ax.set_xlabel('vars_per_object')
        ax.set_ylabel('n_objects')
        ax.set_title(f'Object ARI ({method})')
        for i in range(len(n_objects_list)):
            for j in range(len(vars_per_object_list)):
                ax.text(j, i, f"{ari_matrix[i, j]:.2f}", ha='center',
                        va='center', color='black', fontsize=11)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_figure(fig, 'v2_scaling_heatmap', 'v2_scaling')


def run_us016_temperature():
    """US-016c: Temperature sensitivity with coupling method added."""
    print("\n" + "=" * 70)
    print("US-016c: v2 Temperature Sensitivity (includes coupling method)")
    print("=" * 70)

    temperatures = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    n_trials = 5
    blanket_strength = 0.8
    all_methods = TC_METHODS + ['dmbd', 'axiom']

    metrics = {}

    for temp in temperatures:
        t_key = str(temp)
        print(f"\n--- Temperature T={temp} ---")
        metrics[t_key] = {m: {'per_trial': []} for m in all_methods}

        for trial in range(n_trials):
            cfg = QuadraticEBMConfig(
                n_objects=2, vars_per_object=3, vars_per_blanket=3,
                intra_strength=6.0, blanket_strength=blanket_strength,
            )
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            samples, gradients = langevin_sampling(
                Theta, n_samples=3000, n_steps=30,
                step_size=0.005, temp=temp
            )

            for method in TC_METHODS:
                tc_result = tb_pipeline(gradients, n_objects=cfg.n_objects,
                                        method=method)
                tc_m = compute_metrics(tc_result, truth)
                metrics[t_key][method]['per_trial'].append(tc_m)

            dmbd_result = dmbd_style_partition(gradients, n_objects=cfg.n_objects)
            metrics[t_key]['dmbd']['per_trial'].append(
                compute_metrics(dmbd_result, truth))

            axiom_result = axiom_style_partition(samples, n_objects=cfg.n_objects,
                                                  gradients=gradients)
            metrics[t_key]['axiom']['per_trial'].append(
                compute_metrics(axiom_result, truth))

        for method in all_methods:
            trials = metrics[t_key][method]['per_trial']
            aris = [r['object_ari'] for r in trials]
            f1s = [r['blanket_f1'] for r in trials]
            metrics[t_key][method]['mean_ari'] = float(np.mean(aris))
            metrics[t_key][method]['std_ari'] = float(np.std(aris))
            metrics[t_key][method]['mean_f1'] = float(np.mean(f1s))
            metrics[t_key][method]['std_f1'] = float(np.std(f1s))

        for method in all_methods:
            label = TC_LABELS.get(method, method.upper())
            print(f"  {label:14s}: ARI={metrics[t_key][method]['mean_ari']:.3f}, "
                  f"F1={metrics[t_key][method]['mean_f1']:.3f}")

    config = {
        'version': 'v2',
        'temperatures': temperatures,
        'n_trials': n_trials,
        'blanket_strength': blanket_strength,
        'tc_methods': TC_METHODS,
        'n_objects': 2,
        'vars_per_object': 3,
    }

    save_results('v2_temperature_sensitivity', metrics, config,
                 notes='US-016: v2 temperature sweep with all TC methods + improved baselines.')

    _plot_v2_temperature_sweep(metrics, temperatures)

    print("\nUS-016c (temperature) complete.")
    return metrics


def _plot_v2_temperature_sweep(metrics, temperatures):
    """Temperature sweep plot with all methods."""
    all_methods = TC_METHODS + ['dmbd', 'axiom']
    all_labels = {**TC_LABELS, 'dmbd': 'DMBD-style', 'axiom': 'AXIOM-style'}
    all_colors = {**TC_COLORS, **BASELINE_COLORS}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in all_methods:
        means = [metrics[str(t)][method]['mean_ari'] for t in temperatures]
        stds = [metrics[str(t)][method]['std_ari'] for t in temperatures]
        ls = '--' if method in ('dmbd', 'axiom') else '-'
        ax.errorbar(temperatures, means, yerr=stds,
                    label=all_labels[method], color=all_colors[method],
                    marker='o', capsize=3, linewidth=2, linestyle=ls)
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('v2: Object Partition Recovery vs Temperature')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'v2_ari_vs_temperature', 'v2_temperature_sensitivity')


def run_us017():
    """US-017: Ablation study on detection methods."""
    print("\n" + "=" * 70)
    print("US-017: Ablation Study on Detection Methods")
    print("=" * 70)

    r1 = _ablation_hessian_estimation()
    r2 = _ablation_blanket_detection()
    r3 = _ablation_object_clustering()
    r4 = _ablation_sample_efficiency()

    # Combine all ablation results
    all_metrics = {
        'hessian_estimation': r1,
        'blanket_detection': r2,
        'object_clustering': r3,
        'sample_efficiency': r4,
    }

    config = {
        'n_objects': 2,
        'vars_per_object': 3,
        'vars_per_blanket': 3,
        'blanket_strength': 0.8,
        'intra_strength': 6.0,
        'n_trials': 5,
    }

    save_results('v2_ablation_study', all_metrics, config,
                 notes='US-017: 4-factor ablation study on TB pipeline components.')

    _plot_ablation_summary(all_metrics)
    _plot_sample_efficiency(r4)

    # Print key finding
    _print_ablation_findings(all_metrics)

    print("\nUS-017 complete.")
    return all_metrics


def _make_standard_config():
    """Standard quadratic EBM config for ablation."""
    return QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )


def _ablation_hessian_estimation():
    """Factor A: gradient covariance vs true Hessian."""
    from topological_blankets.features import compute_geometric_features
    from topological_blankets.detection import detect_blankets_otsu
    from topological_blankets.clustering import cluster_internals

    print("\n--- Factor A: Hessian Estimation ---")
    n_trials = 5
    results = {'estimated': [], 'true': []}

    for trial in range(n_trials):
        cfg = _make_standard_config()
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=3000, n_steps=30,
            step_size=0.005, temp=0.1
        )

        # Estimated Hessian (default pipeline)
        features_est = compute_geometric_features(gradients)
        is_blanket_est, _ = detect_blankets_otsu(features_est)
        assign_est = cluster_internals(features_est, is_blanket_est, n_clusters=2)
        m_est = compute_metrics({'assignment': assign_est, 'is_blanket': is_blanket_est}, truth)
        results['estimated'].append(m_est)

        # True Hessian (use Theta directly)
        features_true = dict(features_est)  # copy base features
        features_true['hessian_est'] = Theta.copy()
        D = np.sqrt(np.abs(np.diag(Theta))) + 1e-8
        coupling_true = np.abs(Theta) / np.outer(D, D)
        np.fill_diagonal(coupling_true, 0)
        features_true['coupling'] = coupling_true
        is_blanket_true, _ = detect_blankets_otsu(features_true)
        assign_true = cluster_internals(features_true, is_blanket_true, n_clusters=2)
        m_true = compute_metrics({'assignment': assign_true, 'is_blanket': is_blanket_true}, truth)
        results['true'].append(m_true)

    for key in results:
        aris = [r['object_ari'] for r in results[key]]
        f1s = [r['blanket_f1'] for r in results[key]]
        results[key] = {
            'per_trial': results[key],
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
        }
        print(f"  {key:12s}: ARI={results[key]['mean_ari']:.3f}+/-{results[key]['std_ari']:.3f}, "
              f"F1={results[key]['mean_f1']:.3f}")

    return results


def _ablation_blanket_detection():
    """Factor B: blanket detection method (Otsu vs spectral vs coupling)."""
    from topological_blankets.features import compute_geometric_features
    from topological_blankets.detection import (
        detect_blankets_otsu, detect_blankets_spectral, detect_blankets_coupling
    )
    from topological_blankets.clustering import cluster_internals

    print("\n--- Factor B: Blanket Detection Method ---")
    n_trials = 5
    detection_methods = {
        'otsu': lambda feat: detect_blankets_otsu(feat),
        'spectral': lambda feat: (
            detect_blankets_spectral(feat['hessian_est'])['is_blanket'], 0.0
        ),
        'coupling': lambda feat: (
            detect_blankets_coupling(feat['hessian_est'], feat['coupling'], n_objects=2), 0.0
        ),
    }
    results = {m: [] for m in detection_methods}

    for trial in range(n_trials):
        cfg = _make_standard_config()
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=3000, n_steps=30,
            step_size=0.005, temp=0.1
        )
        features = compute_geometric_features(gradients)

        for method_name, detect_fn in detection_methods.items():
            is_blanket, _ = detect_fn(features)
            assignment = cluster_internals(features, is_blanket, n_clusters=2)
            m = compute_metrics({'assignment': assignment, 'is_blanket': is_blanket}, truth)
            results[method_name].append(m)

    for key in results:
        aris = [r['object_ari'] for r in results[key]]
        f1s = [r['blanket_f1'] for r in results[key]]
        results[key] = {
            'per_trial': results[key],
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
        }
        print(f"  {key:12s}: ARI={results[key]['mean_ari']:.3f}+/-{results[key]['std_ari']:.3f}, "
              f"F1={results[key]['mean_f1']:.3f}")

    return results


def _ablation_object_clustering():
    """Factor C: object clustering method (spectral vs kmeans vs agglomerative)."""
    from topological_blankets.features import compute_geometric_features
    from topological_blankets.detection import detect_blankets_otsu
    from topological_blankets.clustering import cluster_internals

    print("\n--- Factor C: Object Clustering Method ---")
    n_trials = 5
    clustering_methods = ['spectral', 'kmeans', 'agglomerative']
    results = {m: [] for m in clustering_methods}

    for trial in range(n_trials):
        cfg = _make_standard_config()
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=3000, n_steps=30,
            step_size=0.005, temp=0.1
        )
        features = compute_geometric_features(gradients)
        is_blanket, _ = detect_blankets_otsu(features)

        for method in clustering_methods:
            assignment = cluster_internals(features, is_blanket, n_clusters=2,
                                            method=method)
            m = compute_metrics({'assignment': assignment, 'is_blanket': is_blanket}, truth)
            results[method].append(m)

    for key in results:
        aris = [r['object_ari'] for r in results[key]]
        f1s = [r['blanket_f1'] for r in results[key]]
        results[key] = {
            'per_trial': results[key],
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
        }
        print(f"  {key:14s}: ARI={results[key]['mean_ari']:.3f}+/-{results[key]['std_ari']:.3f}, "
              f"F1={results[key]['mean_f1']:.3f}")

    return results


def _ablation_sample_efficiency():
    """Factor D: number of Langevin samples."""
    print("\n--- Factor D: Sample Efficiency ---")
    n_trials = 5
    sample_counts = [100, 500, 1000, 3000, 5000]
    results = {str(n): [] for n in sample_counts}

    for n_samples in sample_counts:
        for trial in range(n_trials):
            cfg = _make_standard_config()
            Theta = build_precision_matrix(cfg)
            truth = get_ground_truth(cfg)

            np.random.seed(42 + trial)
            samples, gradients = langevin_sampling(
                Theta, n_samples=n_samples, n_steps=30,
                step_size=0.005, temp=0.1
            )

            tc_result = tb_pipeline(gradients, n_objects=cfg.n_objects, method='gradient')
            m = compute_metrics(tc_result, truth)
            results[str(n_samples)].append(m)

    for key in results:
        aris = [r['object_ari'] for r in results[key]]
        f1s = [r['blanket_f1'] for r in results[key]]
        results[key] = {
            'per_trial': results[key],
            'mean_ari': float(np.mean(aris)),
            'std_ari': float(np.std(aris)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
        }
        print(f"  N={key:>5s}: ARI={results[key]['mean_ari']:.3f}+/-{results[key]['std_ari']:.3f}, "
              f"F1={results[key]['mean_f1']:.3f}")

    return results


def _plot_ablation_summary(all_metrics):
    """Summary bar chart for each ablation factor."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    factor_names = ['Hessian Estimation', 'Blanket Detection',
                    'Object Clustering', 'Sample Count']

    for ax, (factor, name) in zip(axes, zip(
        ['hessian_estimation', 'blanket_detection', 'object_clustering', 'sample_efficiency'],
        factor_names
    )):
        data = all_metrics[factor]
        labels = list(data.keys())
        aris = [data[k]['mean_ari'] for k in labels]
        stds = [data[k]['std_ari'] for k in labels]

        bars = ax.bar(range(len(labels)), aris, yerr=stds, capsize=3,
                      color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'][:len(labels)],
                      alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('ARI')
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, ari in zip(bars, aris):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f'{ari:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    save_figure(fig, 'v2_ablation_summary', 'v2_ablation')


def _plot_sample_efficiency(sample_data):
    """ARI vs number of Langevin samples curve."""
    sample_counts = sorted([int(k) for k in sample_data.keys()])
    means = [sample_data[str(n)]['mean_ari'] for n in sample_counts]
    stds = [sample_data[str(n)]['std_ari'] for n in sample_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sample_counts, means, yerr=stds, marker='o', capsize=3,
                linewidth=2, color='#2ecc71')
    ax.set_xlabel('Number of Langevin Samples')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Sample Efficiency: ARI vs Number of Gradient Samples')
    ax.set_ylim(0, 1.1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='ARI=0.95 threshold')
    ax.legend()
    save_figure(fig, 'v2_sample_efficiency', 'v2_ablation')


def _print_ablation_findings(all_metrics):
    """Print key findings from the ablation study."""
    print("\n" + "=" * 60)
    print("KEY ABLATION FINDINGS")
    print("=" * 60)

    # Factor A: Hessian
    est_ari = all_metrics['hessian_estimation']['estimated']['mean_ari']
    true_ari = all_metrics['hessian_estimation']['true']['mean_ari']
    print(f"\n1. Hessian estimation: estimated={est_ari:.3f}, true={true_ari:.3f}")
    if abs(est_ari - true_ari) < 0.05:
        print("   -> Gradient covariance is sufficient; true Hessian provides minimal benefit.")
    else:
        print(f"   -> True Hessian {'improves' if true_ari > est_ari else 'degrades'} ARI by {abs(true_ari - est_ari):.3f}.")

    # Factor B: Detection
    det_data = all_metrics['blanket_detection']
    best_det = max(det_data, key=lambda k: det_data[k]['mean_ari'])
    print(f"\n2. Blanket detection: best method = {best_det} (ARI={det_data[best_det]['mean_ari']:.3f})")
    for k, v in det_data.items():
        if k != best_det:
            print(f"   {k}: ARI={v['mean_ari']:.3f}")

    # Factor C: Clustering
    clust_data = all_metrics['object_clustering']
    best_clust = max(clust_data, key=lambda k: clust_data[k]['mean_ari'])
    print(f"\n3. Object clustering: best method = {best_clust} (ARI={clust_data[best_clust]['mean_ari']:.3f})")
    for k, v in clust_data.items():
        if k != best_clust:
            print(f"   {k}: ARI={v['mean_ari']:.3f}")

    # Factor D: Samples
    sample_data = all_metrics['sample_efficiency']
    sample_counts = sorted([int(k) for k in sample_data.keys()])
    min_sufficient = None
    for n in sample_counts:
        if sample_data[str(n)]['mean_ari'] >= 0.95:
            min_sufficient = n
            break
    if min_sufficient:
        print(f"\n4. Sample efficiency: ARI >= 0.95 achieved at N={min_sufficient} samples")
    else:
        best_n = max(sample_counts, key=lambda n: sample_data[str(n)]['mean_ari'])
        print(f"\n4. Sample efficiency: best ARI={sample_data[str(best_n)]['mean_ari']:.3f} at N={best_n}")

    # Overall finding
    factors = {
        'hessian': abs(max(est_ari, true_ari) - min(est_ari, true_ari)),
        'detection': max(v['mean_ari'] for v in det_data.values()) - min(v['mean_ari'] for v in det_data.values()),
        'clustering': max(v['mean_ari'] for v in clust_data.values()) - min(v['mean_ari'] for v in clust_data.values()),
        'samples': sample_data[str(sample_counts[-1])]['mean_ari'] - sample_data[str(sample_counts[0])]['mean_ari'],
    }
    most_important = max(factors, key=factors.get)
    print(f"\nMost impactful factor: {most_important} (ARI spread = {factors[most_important]:.3f})")


def run_us016():
    """US-016: Full v2 Level 1 experiment suite."""
    print("\n" + "=" * 70)
    print("US-016: LEVEL 1 v2 EXPERIMENTS")
    print("=" * 70)

    r1 = run_us016_strength_sweep()
    r2 = run_us016_scaling()
    r3 = run_us016_temperature()

    # Rebuild registry with new v2 results
    build_registry()

    print("\n" + "=" * 70)
    print("US-016 COMPLETE: All v2 experiments saved")
    print("=" * 70)
    return r1, r2, r3


# =========================================================================
# Main: Run all experiments
# =========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Level 1 Validation Experiments')
    parser.add_argument('--v2-only', action='store_true',
                        help='Run only US-016 v2 experiments')
    parser.add_argument('--ablation', action='store_true',
                        help='Run only US-017 ablation study')
    parser.add_argument('--all', action='store_true',
                        help='Run both v1 and v2 experiments')
    args = parser.parse_args()

    if args.v2_only:
        run_us016()
    elif args.ablation:
        run_us017()
    elif args.all:
        print("Level 1 Validation: Running all experiments (v1 + v2)")
        print("=" * 70)
        run_us005()
        run_us006()
        run_us007()
        run_us008()
        run_us016()
        run_us017()
        run_us009()
    else:
        print("Level 1 Validation: Running v1 experiments")
        print("=" * 70)
        run_us005()
        run_us006()
        run_us007()
        run_us008()
        run_us009()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("Results saved to results/")
