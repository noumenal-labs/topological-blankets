# Code Review Map for Alec
February 11, 2026

## Quick Start

Three repositories, one system:

```
topological-blankets/   -- TB algorithm library + experiments + ralph agent
pandas/                 -- JAX/Equinox ensemble world model + CEM planner
lunar-lander/           -- PyTorch active inference baseline (reference)
```

---

## 1. Bayesian Ensemble Dynamics Model (the sample efficiency source)

All in `pandas/panda/`:

| File | What it does |
|------|-------------|
| `model.py` | `DynamicsMember` class (JAX/Equinox). MLP with shared trunk, separate heads for obs/achieved_goal prediction. Normalizer integration. |
| `planner.py` | CEM planner. `CEMConfig`, reward weighting, epistemic uncertainty bonus from ensemble disagreement. |
| `tb_discovery.py` | `discover_or_fallback()`: collects frames, computes Jacobians from ensemble, runs TB pipeline. Fallback to ground-truth partition. |
| `learned_planner.py` | `TBGuidedPlanner` (US-084): decomposes goals into per-object subgoals using TB-discovered partition. Task-agnostic. |
| `tb_replay.py` | TB-informed replay buffer management. |

Training entry point: `pandas/train.py`
Eval: `pandas/eval.py`

---

## 2. Topological Blankets Library (the structure discovery engine)

All in `topological-blankets/topological_blankets/`:

| File | What it does |
|------|-------------|
| `core.py` | Main pipeline. `TopologicalBlankets` class, `topological_blankets()` convenience function. |
| `detection.py` | Blanket detection methods: Otsu gradient, spectral (Friston eigenvector variance), hybrid, coupling-based, persistence-based, PCCA. |
| `features.py` | Geometric feature computation from energy landscape gradients. Covariance estimation (Pearson, rank-based Spearman). |
| `spectral.py` | Graph Laplacian eigenmodes, L1 sparsification (soft thresholding, BIC/CV lambda selection, stability selection), recursive spectral detection. |
| `clustering.py` | Object clustering from blanket-partitioned coupling. Spectral, K-means, agglomerative. |
| `pcca.py` | PCCA+ soft blanket membership without hard thresholding. |
| `extractors.py` | Gradient extraction adapters: PyTorch, JAX, plain Python, score-based models. Finite differences. |

---

## 3. Key Experiments (ralph/experiments/)

| File | US | What it does |
|------|----|-------------|
| `bayesformer_tb.py` | US-112 | TB-structured Bayesian Transformer. Weight-space QKV masking from TB coupling. Eigengap regularizer. MC Dropout. |
| `pandas_ensemble_analysis.py` | US-076 | TB applied to Bayes ensemble Jacobians on FetchPush. Coupling matrix visualization. |
| `pandas_structure_emergence.py` | US-080 | Structure emergence during training. TB at interpolated checkpoints. Filmstrip coupling matrices. |
| `train_ddpg_her_fetchpush.py` | US-105 | DDPG+HER baseline for sample efficiency comparison. |

Lambda emergence script: `ralph/results/lambda_emergence/train_with_tb_checkpoints.py`

---

## 4. For the Sample Efficiency Claim Specifically

The comparison is:

**Ours** (model-based):
- `pandas/train.py` trains the 5-member ensemble on FetchPush-v4
- 5,000 random steps -> train model -> CEM plans through model
- First successful push at 7,500 env steps
- Training log: `ralph/results/lambda_emergence/training_full.log`
- Trained model: `ralph/results/lambda_models/fetchpush/`

**Baseline** (model-free):
- `ralph/experiments/train_ddpg_her_fetchpush.py` (architecture-matched DDPG+HER)
- Alec's wandb run: reaches 95% success at ~500k steps
- Baseline plot: `sharing/alec_2026-02-11/wandb_compare_fetchpush_v1.png`

**Comparison plot**: `sharing/alec_2026-02-11/fetchpush_sample_efficiency_comparison.png`
**Plot data**: `ralph/results/fetchpush_sample_efficiency_data.json`
**Plot script**: `ralph/results/generate_push_comparison.py`

---

## 5. Results Artifacts

All pulled from Lambda (GH200 480GB):

```
ralph/results/lambda_models/
    fetchpush/          model.eqx, eval_latest.gif, train_summary.json
    fetchreach/         "
    fetchpickandplace/  "
    fetchslide/         "
    push_demo/          (Wednesday demo model)

ralph/results/lambda_emergence/
    structure_emergence_results.json   (TB metrics at 11 checkpoints)
    structure_emergence_curves.png
    coupling_matrix_timelapse.png
    checkpoints/iter_0000..0100/       (11 saved models)
```
