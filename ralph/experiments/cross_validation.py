"""
US-023: Cross-Validation and Confidence Intervals
==================================================

100-trial run on standard quadratic config.
Bootstrap 95% CI for ARI and F1.
Wilcoxon signed-rank test: TB vs DMBD.
Distribution plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets.core import topological_blankets as tb_pipeline
from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    langevin_sampling, dmbd_style_partition, axiom_style_partition,
    compute_metrics
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(data)
    bootstraps = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bootstraps[b] = np.mean(data[idx])
    alpha = (1 - ci) / 2
    lo = np.percentile(bootstraps, 100 * alpha)
    hi = np.percentile(bootstraps, 100 * (1 - alpha))
    return float(lo), float(hi), float(np.mean(bootstraps))


def run_cross_validation():
    """Run the full cross-validation experiment."""
    print("=" * 70)
    print("US-023: Cross-Validation and Confidence Intervals")
    print("=" * 70)

    n_trials = 100
    cfg = QuadraticEBMConfig(
        n_objects=2, vars_per_object=3, vars_per_blanket=3,
        intra_strength=6.0, blanket_strength=0.8,
    )

    tc_aris, tc_f1s = [], []
    dmbd_aris, dmbd_f1s = [], []
    axiom_aris, axiom_f1s = [], []

    print(f"\nRunning {n_trials} trials...")
    for trial in range(n_trials):
        Theta = build_precision_matrix(cfg)
        truth = get_ground_truth(cfg)

        np.random.seed(42 + trial)
        samples, gradients = langevin_sampling(
            Theta, n_samples=3000, n_steps=30,
            step_size=0.005, temp=0.1
        )

        # TB
        tc_result = tb_pipeline(gradients, n_objects=cfg.n_objects, method='gradient')
        tc_m = compute_metrics(tc_result, truth)
        tc_aris.append(tc_m['object_ari'])
        tc_f1s.append(tc_m['blanket_f1'])

        # DMBD
        dmbd_result = dmbd_style_partition(gradients, n_objects=cfg.n_objects)
        dmbd_m = compute_metrics(dmbd_result, truth)
        dmbd_aris.append(dmbd_m['object_ari'])
        dmbd_f1s.append(dmbd_m['blanket_f1'])

        # AXIOM
        axiom_result = axiom_style_partition(samples, n_objects=cfg.n_objects,
                                              gradients=gradients)
        axiom_m = compute_metrics(axiom_result, truth)
        axiom_aris.append(axiom_m['object_ari'])
        axiom_f1s.append(axiom_m['blanket_f1'])

        if (trial + 1) % 20 == 0:
            print(f"  Trial {trial + 1}/{n_trials}: "
                  f"TC ARI={np.mean(tc_aris):.3f}, DMBD ARI={np.mean(dmbd_aris):.3f}")

    tc_aris = np.array(tc_aris)
    tc_f1s = np.array(tc_f1s)
    dmbd_aris = np.array(dmbd_aris)
    dmbd_f1s = np.array(dmbd_f1s)
    axiom_aris = np.array(axiom_aris)
    axiom_f1s = np.array(axiom_f1s)

    # Bootstrap CIs
    print("\n--- Bootstrap 95% Confidence Intervals ---")
    tc_ari_ci = bootstrap_ci(tc_aris)
    tc_f1_ci = bootstrap_ci(tc_f1s)
    dmbd_ari_ci = bootstrap_ci(dmbd_aris)
    dmbd_f1_ci = bootstrap_ci(dmbd_f1s)
    axiom_ari_ci = bootstrap_ci(axiom_aris)
    axiom_f1_ci = bootstrap_ci(axiom_f1s)

    print(f"  TC   ARI: {tc_ari_ci[2]:.3f} [{tc_ari_ci[0]:.3f}, {tc_ari_ci[1]:.3f}]")
    print(f"  TC   F1:  {tc_f1_ci[2]:.3f} [{tc_f1_ci[0]:.3f}, {tc_f1_ci[1]:.3f}]")
    print(f"  DMBD ARI: {dmbd_ari_ci[2]:.3f} [{dmbd_ari_ci[0]:.3f}, {dmbd_ari_ci[1]:.3f}]")
    print(f"  DMBD F1:  {dmbd_f1_ci[2]:.3f} [{dmbd_f1_ci[0]:.3f}, {dmbd_f1_ci[1]:.3f}]")
    print(f"  AXIOM ARI: {axiom_ari_ci[2]:.3f} [{axiom_ari_ci[0]:.3f}, {axiom_ari_ci[1]:.3f}]")
    print(f"  AXIOM F1:  {axiom_f1_ci[2]:.3f} [{axiom_f1_ci[0]:.3f}, {axiom_f1_ci[1]:.3f}]")

    # CI width check
    tc_ari_width = tc_ari_ci[1] - tc_ari_ci[0]
    tc_f1_width = tc_f1_ci[1] - tc_f1_ci[0]
    print(f"\n  TC ARI CI width: {tc_ari_width:.3f} (target < 0.1)")
    print(f"  TC F1 CI width: {tc_f1_width:.3f} (target < 0.1)")

    # Wilcoxon signed-rank test: TC vs DMBD
    print("\n--- Statistical Tests ---")
    # For ARI
    if not np.all(tc_aris == dmbd_aris):
        stat_ari, p_ari = stats.wilcoxon(tc_aris, dmbd_aris, alternative='greater')
    else:
        stat_ari, p_ari = 0.0, 1.0
    print(f"  Wilcoxon (TC > DMBD, ARI): stat={stat_ari:.1f}, p={p_ari:.4f}")

    # For F1
    if not np.all(tc_f1s == dmbd_f1s):
        stat_f1, p_f1 = stats.wilcoxon(tc_f1s, dmbd_f1s)
    else:
        stat_f1, p_f1 = 0.0, 1.0
    print(f"  Wilcoxon (TC vs DMBD, F1): stat={stat_f1:.1f}, p={p_f1:.4f}")

    # TC vs AXIOM
    if not np.all(tc_aris == axiom_aris):
        stat_axiom, p_axiom = stats.wilcoxon(tc_aris, axiom_aris, alternative='greater')
    else:
        stat_axiom, p_axiom = 0.0, 1.0
    print(f"  Wilcoxon (TC > AXIOM, ARI): stat={stat_axiom:.1f}, p={p_axiom:.4f}")

    # Build metrics
    metrics = {
        'tc': {
            'ari_trials': tc_aris.tolist(),
            'f1_trials': tc_f1s.tolist(),
            'ari_mean': float(np.mean(tc_aris)),
            'ari_std': float(np.std(tc_aris)),
            'ari_ci_95': list(tc_ari_ci[:2]),
            'f1_mean': float(np.mean(tc_f1s)),
            'f1_std': float(np.std(tc_f1s)),
            'f1_ci_95': list(tc_f1_ci[:2]),
        },
        'dmbd': {
            'ari_trials': dmbd_aris.tolist(),
            'f1_trials': dmbd_f1s.tolist(),
            'ari_mean': float(np.mean(dmbd_aris)),
            'ari_std': float(np.std(dmbd_aris)),
            'ari_ci_95': list(dmbd_ari_ci[:2]),
            'f1_mean': float(np.mean(dmbd_f1s)),
            'f1_std': float(np.std(dmbd_f1s)),
            'f1_ci_95': list(dmbd_f1_ci[:2]),
        },
        'axiom': {
            'ari_trials': axiom_aris.tolist(),
            'f1_trials': axiom_f1s.tolist(),
            'ari_mean': float(np.mean(axiom_aris)),
            'ari_std': float(np.std(axiom_aris)),
            'ari_ci_95': list(axiom_ari_ci[:2]),
            'f1_mean': float(np.mean(axiom_f1s)),
            'f1_std': float(np.std(axiom_f1s)),
            'f1_ci_95': list(axiom_f1_ci[:2]),
        },
        'tests': {
            'wilcoxon_tc_vs_dmbd_ari': {'statistic': float(stat_ari), 'p_value': float(p_ari)},
            'wilcoxon_tc_vs_dmbd_f1': {'statistic': float(stat_f1), 'p_value': float(p_f1)},
            'wilcoxon_tc_vs_axiom_ari': {'statistic': float(stat_axiom), 'p_value': float(p_axiom)},
        },
    }

    config = {
        'n_trials': n_trials,
        'n_bootstrap': 1000,
        'n_objects': 2,
        'vars_per_object': 3,
        'blanket_strength': 0.8,
    }

    save_results('cross_validation_100trials', metrics, config,
                 notes='US-023: 100-trial cross-validation with bootstrap CI and Wilcoxon tests.')

    _plot_distributions(tc_aris, tc_f1s, dmbd_aris, dmbd_f1s, axiom_aris, axiom_f1s)

    print("\nUS-023 complete.")
    return metrics


def _plot_distributions(tc_aris, tc_f1s, dmbd_aris, dmbd_f1s, axiom_aris, axiom_f1s):
    """Histogram of ARI and F1 across 100 trials."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ARI distribution
    ax = axes[0]
    bins = np.linspace(-0.1, 1.05, 25)
    ax.hist(tc_aris, bins=bins, alpha=0.6, label='TC', color='#2ecc71')
    ax.hist(dmbd_aris, bins=bins, alpha=0.6, label='DMBD', color='#3498db')
    ax.hist(axiom_aris, bins=bins, alpha=0.6, label='AXIOM', color='#e74c3c')
    ax.set_xlabel('Object ARI')
    ax.set_ylabel('Count')
    ax.set_title(f'ARI Distribution (100 trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 distribution
    ax = axes[1]
    ax.hist(tc_f1s, bins=bins, alpha=0.6, label='TC', color='#2ecc71')
    ax.hist(dmbd_f1s, bins=bins, alpha=0.6, label='DMBD', color='#3498db')
    ax.hist(axiom_f1s, bins=bins, alpha=0.6, label='AXIOM', color='#e74c3c')
    ax.set_xlabel('Blanket F1')
    ax.set_ylabel('Count')
    ax.set_title(f'F1 Distribution (100 trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'cv_distributions', 'cross_validation')


if __name__ == '__main__':
    run_cross_validation()
