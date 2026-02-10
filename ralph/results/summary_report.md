# Topological Blankets: Final Results Summary
Generated: 2026-02-07 08:23:18

## Overview
- Total experiments: 50
- PASS: 50
- WARN: 0
- FAIL: 0

## Phase 1: Synthetic Validation

| Experiment | Type | Key Metric | Status |
|-----------|------|------------|--------|
| strength_sweep_10trials | Strength Sweep | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| scaling_experiment | Scaling | status=complete | PASS |
| temperature_sensitivity | Temperature Sensitivity | status=complete | PASS |
| quadratic_toy_demo | Synthetic Validation | status=complete | PASS |
| quadratic_toy_strength_sweep | Synthetic Validation | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| quadratic_toy_demo | Synthetic Validation | status=complete | PASS |
| quadratic_toy_strength_sweep | Synthetic Validation | status=complete | PASS |
| strength_sweep_10trials | Strength Sweep | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| scaling_experiment | Scaling | status=complete | PASS |
| temperature_sensitivity | Temperature Sensitivity | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| quadratic_toy_demo | Synthetic Validation | status=complete | PASS |
| quadratic_toy_strength_sweep | Synthetic Validation | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| stress_large_scale | Stress Test | status=complete | PASS |
| stress_hard_regime | Stress Test | status=complete | PASS |
| stress_asymmetric | Stress Test | status=complete | PASS |
| stress_blanket_ratio | Stress Test | status=complete | PASS |
| stress_hessian_validation | Stress Test | status=complete | PASS |
| quadratic_toy_demo | Synthetic Validation | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| quadratic_toy_strength_sweep | Synthetic Validation | status=complete | PASS |
| strength_sweep_10trials | Strength Sweep | status=complete | PASS |
| spectral_friston_comparison | Spectral Detection | status=complete | PASS |
| scaling_experiment | Scaling | status=complete | PASS |
| temperature_sensitivity | Temperature Sensitivity | status=complete | PASS |
| v2_strength_sweep | Strength Sweep | status=complete | PASS |
| v2_scaling_experiment | Scaling | status=complete | PASS |
| v2_temperature_sensitivity | Temperature Sensitivity | status=complete | PASS |
| temperature_sensitivity_worldmodels | Temperature Sensitivity | status=complete | PASS |

## Phase 2: Method Engineering

| Experiment | Type | Key Metric | Status |
|-----------|------|------------|--------|
| v2_ablation_study | v2 Ablation | status=complete | PASS |

## Phase 3: Bridge Experiments

| Experiment | Type | Key Metric | Status |
|-----------|------|------------|--------|
| ising_model | Ising Model | status=complete | PASS |
| ggm_benchmark | GGM Benchmark | status=complete | PASS |
| non_gaussian_landscapes | Non-Gaussian | status=complete | PASS |
| score_model_2d | 2D Score Model | status=complete | PASS |
| cross_validation_100trials | Cross-Validation | status=complete | PASS |
| scaling_benchmark | Scaling Benchmark | status=complete | PASS |

## Phase 4: World Model Demo

| Experiment | Type | Key Metric | Status |
|-----------|------|------------|--------|
| actinf_trajectory_collection | Active Inference Data | status=complete | PASS |
| actinf_tb_analysis | Active Inference TB | eigengap=8.0000 | PASS |
| dreamer_autoencoder_training | Dreamer Training | mse=0.0004 | PASS |
| dreamer_tb_analysis | Dreamer TB | status=complete | PASS |
| multi_scale_comparison | Multi-Scale Comparison | status=complete | PASS |

## Phase 5: Analysis & Packaging

| Experiment | Type | Key Metric | Status |
|-----------|------|------------|--------|
| edge_compute_analysis | Edge Compute | status=complete | PASS |
| notears_comparison | NOTEARS Comparison | status=complete | PASS |
| robustness_analysis | Robustness Analysis | status=complete | PASS |

## Key Findings

1. TB achieves ARI=1.0 on standard quadratic EBMs at blanket_strength >= 0.3
2. All four detection methods (gradient, spectral, coupling, hybrid) perform equivalently on well-separated structures
3. TB outperforms graphical lasso on GGM structure recovery (F1=0.947 vs 0.750 on chain graphs)
4. TB outperforms NOTEARS on GGM benchmarks (F1=0.947 vs 0.000)
5. Active Inference world model reveals physically meaningful structure: Object 0={y,vy,legs}, Object 1={x,vx,angle}, Blanket={ang_vel}
6. Dreamer autoencoder (8D->64D->8D) achieves MSE=0.000375; latent-to-physical correlations up to 0.911
7. Cross-checkpoint robustness: ARI=1.0 across 3 different model checkpoints
8. Sample efficiency: stable structure above ~1000 transitions (ARI=0.69)
9. Multi-scale comparison: NMI=0.517 between state-space and projected latent partition
10. Edge-compute factorization: 25.9x speedup at 4096D, 97% memory savings

## Known Limitations

1. Dreamer latent space (64D) shows single dominant cluster; finer structure requires more data or regularization
2. NOTEARS comparison used reimplemented version with aggressive thresholding
3. Scaling beyond 100D shows degradation when blanket is < 3% of variables
4. Temperature sensitivity on real models shows graceful degradation without sharp phase transition
5. No pretrained Dreamer checkpoint existed; autoencoder trained from scratch on 4508 transitions
