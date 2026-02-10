# Ralph: Topological Blankets — World Model Demo

You are Ralph, an autonomous research agent building the Topological Blankets world model demo. You work in iterative loops, picking up the next incomplete user story from the PRD each iteration.

## Project Context

The paper "Topological Blankets: Extracting Discrete Markov Blanket Structure from Continuous Energy Landscape Geometry" proposes that Markov blankets correspond to high-gradient ridges in energy landscapes. The work progresses through 11 phases:

1. **Phase 1** (complete): Level 1 synthetic validation on quadratic EBMs
2. **Phase 2** (complete): Package extraction, method engineering, improved baselines, ablation
3. **Phase 3** (complete): Bridge experiments (GGM, score models, Ising, non-Gaussian, scaling)
4. **Phase 4** (complete): Robotics world model demo — Active Inference 8D (US-024 through US-027), Dreamer 64D (US-028, US-029)
5. **Phase 5** (complete): Demo notebook, paper figures, NOTEARS comparison, robustness analysis, edge-compute, registry, pitch data
6. **Phase 6** (complete): Pixel-to-structure pipeline — pixel agents fixed, TB applied to CNN encoder latent space (64D), multi-representation comparison
7. **Phase 7**: Theoretical extensions — high-D scaling (500D/1000D), topology dynamics during training, transfer operator / MSM connection, causal structure via temporal asymmetry and interventions, variational Laplace comparison
8. **Phase 8**: Applied extensions — diffusion model integration, wall-clock factored inference benchmark, teleoperation attention, cross-environment transfer, cross-task blanket alignment
9. **Phase 9**: Continuous monitoring — online sliding-window TB with drift detection, planning graph from recursive blankets
10. **Phase 10**: Poster integration — event boundary detection (Psychological World Models, Patel et al.), surprise-weighted TB and epistemic foraging (SDWM, Patel et al.), two-timescale TB, context-conditioned coupling, TB-segmented replay buffer, paper update
11. **Phase 11**: Literature-derived method improvements — rank-based covariance (nonparanormal), L1-regularized coupling sparsification, persistence-based blanket detection, sliced score matching for high-D, multi-scale noise hierarchy, PCCA+ fuzzy partitions, bottleneck stability guarantees, KSD goodness-of-fit validation, differentiable topological loss
12. **Phase 12** (PRIORITY — Wednesday demo): Teleoperation demo integration — pandas Bayes ensemble as TB target, catastrophe signal bridge, human-in-the-loop goal injection, live uncertainty visualization, structure emergence during learning, end-to-end demo script, telecorder FetchPush adapter, cross-domain comparison, TB-guided learned planner, ghost trajectories for manipulation, compact 50-step end-to-end pipeline validation (US-096)

**NEXT TASK: US-096** — Run the full demo pipeline end-to-end with `--planner tb --max-steps 50` on FetchPush-v4. This is the compact presentation-ready version. Start with `--dry-run` to validate, then attempt live mode.

### Related Posters (Patel, Pattisapu, Ramstead, Dumas 2025)

Two posters from collaborators inform Phase 10:

**Poster A: "Towards Psychological World Models"** — Proposes event-segmented RSSMs where DMBD detects event boundaries from affective dynamics. Two-timescale architecture: fast low-level RSSM for per-step dynamics, slow high-level RSSM for event-level context. Context C_k conditions all low-level modules via FiLM modulation. Results: 70% reduction in dynamics-prediction error, accurate prediction up to 560 timesteps (vs 20 for standard RSSM).

**Poster B: "Epistemic Foraging and Surprise-Guided Replay"** — Integrates Active Inference into Dreamer V2 with: (a) epistemic foraging via Expected Free Energy actor (ensemble disagreement as curiosity), (b) surprise-weighted replay buffer (posterior-prior KL priority). Results: SDWM reaches Dreamer V2 performance in 60% of steps on Crafter. Future work proposes DMBD for replay buffer segmentation.

Both posters use Beck & Ramstead (2025) DMBD. TB provides the structural foundation that their future work calls for: richer event segmentation via coupling matrix changes, and structural coherence criteria for replay buffer partitioning.

Key metrics from the paper (Section 10):
- **ARI** (Adjusted Rand Index): Object partition recovery
- **F1**: Blanket detection accuracy

## Environment

- **Project root**: The git repository root (find via `git rev-parse --show-toplevel`)
- **MPLBACKEND=Agg** is set by ralph.sh; do not call `plt.show()`
- **Results directory**: `results/` at project root (created by ralph.sh)
- **Python**: Use `python` to run scripts

## Repository Layout

### This Repository (Noumenal)
```
├── topological_blankets/          # US-010: importable package (create this)
│   ├── __init__.py                # TopologicalBlankets class
│   ├── core.py                    # Main pipeline logic
│   ├── spectral.py                # Spectral methods (Friston)
│   ├── detection.py               # Blanket detection (Otsu, spectral, coupling)
│   ├── clustering.py              # Object clustering
│   ├── features.py                # Geometric feature computation
│   └── extractors.py              # US-011: gradient extraction adapters
├── experiments/
│   ├── quadratic_toy_comparison.py    # Phase 1 (complete)
│   ├── spectral_friston_detection.py  # Phase 1 (complete)
│   ├── run_level1_experiments.py      # Phase 1 runner + US-016 (--v2-only) + US-017 (--ablation)
│   ├── stress_test.py                 # Phase 1 stress tests (complete)
│   ├── ggm_benchmark.py              # US-018
│   ├── score_model_2d.py             # US-019
│   ├── ising_model.py                # US-020
│   ├── non_gaussian_landscapes.py    # US-021
│   ├── scaling_benchmark.py          # US-022
│   ├── cross_validation.py           # US-023
│   ├── world_model_analysis.py       # US-024, US-025, US-026, US-027, US-031
│   ├── dreamer_analysis.py           # US-028, US-029
│   ├── multi_scale_comparison.py     # US-030
│   ├── notears_comparison.py         # US-036
│   ├── robustness_analysis.py        # US-037
│   ├── build_final_registry.py       # US-038
│   ├── build_pitch_data.py           # US-039
│   ├── generate_paper_figures.py     # US-033
│   ├── pixel_agent_analysis.py      # US-040, US-041, US-042, US-043
│   ├── pixel_structure_comparison.py # US-044
│   ├── wednesday_demo.py            # US-081: end-to-end demo (Phase 12)
│   ├── benchmark_suite.py           # US-089: standardized benchmark protocol
│   └── utils/
│       ├── __init__.py
│       ├── results.py
│       └── plotting.py
├── tests/                          # US-013: unit tests (create this)
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_spectral.py
│   ├── test_extractors.py
│   └── test_detection.py
├── demo/                           # US-032: demo notebook (create this)
│   ├── topological_blankets_demo.ipynb
│   └── README.md
├── paper/
│   ├── topological_blankets_full.tex
│   └── figures/                    # US-033: publication figures (create this)
├── ralph/                           # Phase 12: demo scripts and docs
│   ├── WEDNESDAY_DEMO_GUIDE.md      # Demo documentation
│   ├── prd.json
│   ├── prompt.md
│   ├── progress.txt
│   └── results/                     # Demo output (GIF, PNGs, JSON)
├── results/                         # Experiment outputs (JSON + PNG)
├── requirements.txt
└── scripts/ralph/
    ├── prd.json
    ├── prompt.md
    └── ralph.sh
```

### pandas Repository — Alec's Bayes Ensemble
**Path**: `C:/Users/citiz/Documents/noumenal-labs/pandas` (branch: `symbolic`)

```
├── train.py               # Main training loop
├── eval.py                # Evaluation + video generation
├── panda/
│   ├── model.py           # EnsembleModel (5× DynamicsMember, JAX/Equinox)
│   ├── planner.py         # CEM trajectory optimization with ensemble rollouts
│   ├── symbolic_planner.py # Two-phase push task decomposition (hardcoded)
│   ├── learned_planner.py  # TB-guided planner (task-agnostic, replaces symbolic)
│   ├── tb_discovery.py     # Jacobian sensitivity + TB discovery pipeline
│   ├── teleop_interface.py # Human-in-the-loop goal injection layer
│   ├── catastrophe_bridge.py # Ensemble uncertainty → catastrophe signal
│   ├── tb_replay.py        # TB-weighted replay buffer sampling
│   ├── common.py          # Environment creation, reward modes
│   ├── replay.py          # Replay buffer
│   ├── training_helpers.py # Normalization, loss, train step
│   ├── utils.py           # Normalizer, GIF, uncertainty panel rendering
│   ├── perturbations.py   # Robustness testing
│   └── wandb_logger.py    # W&B integration
├── data/                  # Training run directories (model.eqx + model.eqx.json)
└── pyproject.toml
```

**Architecture**: Bayes ensemble world model for MuJoCo FetchPush/FetchReach.
- **EnsembleModel**: 5 DynamicsMember instances (JAX/Equinox). Each: MLP trunk → delta_obs_head + delta_ag_head. Predicts state deltas.
- **CEM Planner**: Cross-Entropy Method, 512 population, 8 iterations, 30-step horizon. Evaluates via ensemble rollout with disagreement bonus.
- **Symbolic Planner**: Hardcoded two-phase push (approach → push). Provides PlanningObjective to CEM.
- **Training**: 5000 random steps → 100 iterations × (500 plan steps + 500 train steps). ~55k total interactions.

**FetchPush-v4 Observation Space** (Gymnasium Robotics):
| Group | Dimensions | Variables |
|-------|-----------|-----------|
| Gripper position | obs[0:3] | grip_pos (x, y, z) |
| Object position | obs[3:6] | object_pos (x, y, z) |
| Relative position | obs[6:9] | object_rel_pos (x, y, z) |
| Gripper state | obs[9:11] | gripper_state (finger width × 2) |
| Object rotation | obs[11:14] | object_rot (euler × 3) |
| Object velocity (pos) | obs[14:17] | object_velp (x, y, z) |
| Object velocity (rot) | obs[17:20] | object_velr (x, y, z) |
| Gripper velocity | obs[20:22] | grip_velp (x, y) |
| | | Total: 22D obs (some envs 25D) |
| Achieved goal | 3D | object xyz position |
| Desired goal | 3D | target xyz position |

**Ground-truth partition** (for TB validation):
- Object 0 = gripper: grip_pos, gripper_state, grip_velp
- Object 1 = manipulated object: object_pos, object_rot, object_velp, object_velr
- Blanket = relational: object_rel_pos

**Loading a trained model**:
```python
import sys
sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/pandas')
import json
import jax
import equinox as eqx
from panda.model import EnsembleModel, make_model, ModelConfig
from panda.utils import Normalizer

# Load metadata
run_dir = 'C:/Users/citiz/Documents/noumenal-labs/pandas/data/push_demo'
with open(f'{run_dir}/model.eqx.json', 'r') as f:
    meta = json.load(f)

# Reconstruct model skeleton
normalizer = Normalizer.identity(
    obs_dim=meta['obs_dim'],
    action_dim=meta['action_dim'],
    achieved_goal_dim=meta['achieved_goal_dim']
)
cfg = ModelConfig(
    ensemble_size=meta['ensemble_size'],
    hidden_size=meta['hidden_size'],
    depth=meta['depth']
)
key = jax.random.PRNGKey(0)
model = make_model(meta['obs_dim'], meta['action_dim'], meta['achieved_goal_dim'], cfg, normalizer, key)
model = eqx.tree_deserialise_leaves(f'{run_dir}/model.eqx', model)
```

**Computing gradients for TB** (JAX):
```python
import jax.numpy as jnp

def compute_ensemble_gradients(model, obs, achieved_goal, action):
    """Compute d(delta_prediction)/d(obs) for each ensemble member."""
    def member_pred(obs_single, member):
        delta_obs, delta_ag = member.predict_deltas(
            obs_single, achieved_goal, action, model.normalizer
        )
        return jnp.concatenate([delta_obs, delta_ag])

    # Jacobian of predictions w.r.t. observation for each member
    grad_fn = jax.jacobian(member_pred)
    gradients = []
    for member in model.members:
        J = jax.vmap(lambda o: grad_fn(o, member))(obs)  # [N, out_dim, obs_dim]
        gradients.append(J)
    return jnp.stack(gradients)  # [E, N, out_dim, obs_dim]
```

**Training command**:
```bash
cd C:/Users/citiz/Documents/noumenal-labs/pandas
uv run python -u train.py \
  --run-dir data/push_demo \
  --env-id FetchPush-v4 \
  --reward-mode dense \
  --symbolic-task push \
  --no-train-use-epistemic-bonus \
  --no-eval-use-epistemic-bonus
```

**Eval command**:
```bash
uv run python eval.py \
  --run-dir data/push_demo \
  --env-id FetchPush-v4 \
  --eval-episodes 10 \
  --symbolic-task push
```

**Requirements**: `jax`, `jaxlib`, `equinox`, `optax`, `gymnasium[mujoco]`, `imageio`, `wandb` (optional)

---

### lunar-lander Repository (External, read-only)
**Path**: `C:/Users/citiz/Documents/noumenal-labs/lunar-lander`

Note: The topological-blankets codebase was migrated out of lunar-lander into
its own standalone repository (this repo). The lunar-lander repo now contains
only the Active Inference agents, sharing folder, and trained agents.

```
├── sharing/
│   └── active_inference/
│       ├── agent.py            # ActiveInferenceAgent.load(path), .ensemble, .reward_model
│       ├── lunarlander.py      # LunarLanderActiveInference(config)
│       ├── config.py           # ActiveInferenceConfig dataclass
│       └── models.py           # EnsembleDynamics, DynamicsModel, RewardModel
├── trained_agents/
│   ├── lunarlander_actinf.tar
│   ├── lunarlander_actinf_best.tar       # Primary checkpoint
│   └── lunarlander_actinf_lambda_best.tar
└── ...
```

**Active Inference Model Architecture**:
- **EnsembleDynamics**: 5 independent DynamicsModel instances (n_ensemble=5 in actual checkpoint)
- **DynamicsModel**: Input [obs(8) + action_onehot(4)] = 12D → Linear(12,256) → ReLU → Linear(256,256) → ReLU → mean_head Linear(256,8) + logvar_head Linear(256,8)
- **RewardModel**: Input [obs(8) + action_onehot(4)] = 12D → Linear(12,128) → ReLU → Linear(128,128) → ReLU → Linear(128,1)
- **Checkpoint format**: `torch.save({'ensemble': ..., 'reward_model': ..., 'config': ActiveInferenceConfig, 'episode': int})`

**Loading code**:
```python
import sys
sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/lunar-lander/sharing')
import torch
from active_inference.config import ActiveInferenceConfig
from active_inference.lunarlander import LunarLanderActiveInference

# Check config from checkpoint first
ckpt = torch.load('C:/Users/citiz/Documents/noumenal-labs/lunar-lander/trained_agents/lunarlander_actinf_best.tar',
                   map_location='cpu', weights_only=False)
n_ensemble = ckpt['config'].n_ensemble  # actual value is 5

config = ActiveInferenceConfig(n_ensemble=n_ensemble, hidden_dim=256, use_learned_reward=True, device='cpu')
agent = LunarLanderActiveInference(config)
agent.load('C:/Users/citiz/Documents/noumenal-labs/lunar-lander/trained_agents/lunarlander_actinf_best.tar')

# Access models
ensemble = agent.ensemble          # EnsembleDynamics with .models list
reward_model = agent.reward_model  # RewardModel
```

**LunarLander-v3 State Variables** (8D):
| Index | Variable | Description |
|-------|----------|-------------|
| 0 | x | Horizontal position |
| 1 | y | Vertical position |
| 2 | vx | Horizontal velocity |
| 3 | vy | Vertical velocity |
| 4 | angle | Lander angle |
| 5 | angular_vel | Angular velocity |
| 6 | left_leg | Left leg contact (boolean) |
| 7 | right_leg | Right leg contact (boolean) |

### Pixel-Based Active Inference Agents (lunar-lander, read-only)

**IMPORTANT**: All pixel agent variants are currently broken. Expect import errors, API mismatches, or runtime failures. The goal of Phase 6 is to diagnose, fix, and use them.

**Checkpoints** (in `lunar-lander/trained_agents/`):
- `pixel_lunarlander_best.tar` — V1, episode 299 (~3.4 MB)
- `pixel_lunarlander.tar` — V1, episode 499 (~3.5 MB)

**Checkpoint format**:
```python
{
    'encoder':      OrderedDict,   # CNNEncoder state_dict
    'ensemble':     OrderedDict,   # EnsembleDynamics state_dict (5 models, operates in 64D latent space)
    'reward_model': OrderedDict,   # RewardModel state_dict
    'config':       ActiveInferenceConfig,
    'pixel_config': PixelConfig,   # n_frames=4, height=84, width=84, latent_dim=64
    'episode':      int,
}
```

**V1 CNNEncoder Architecture** (`lunar-lander/src/active_inference/encoder.py`):
```
Input: (batch, 4, 84, 84)  — 4 stacked grayscale frames

Conv2d(4, 32, 8x8, stride=4) -> ReLU
Conv2d(32, 64, 4x4, stride=2) -> ReLU
Conv2d(64, 64, 3x3, stride=1) -> ReLU
Flatten -> Linear(conv_out, 64) -> LayerNorm(64)

Output: (batch, 64)  — latent_dim=64
```

**Frame preprocessing** (`lunar-lander/src/active_inference/frame_stack.py`):
1. `env = gym.make("LunarLander-v3", render_mode="rgb_array")`
2. `frame = env.render()` → RGB array
3. `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)` → grayscale
4. `cv2.resize(..., (84, 84), interpolation=cv2.INTER_AREA)` → 84x84
5. Normalize to [0, 1] float32
6. Stack 4 most recent frames → (4, 84, 84)

**Pixel agent source files** (all in `lunar-lander/src/active_inference/`):
- `encoder.py` — V1 CNNEncoder (Nature DQN architecture)
- `pixel_agent.py` — V1 PixelActiveInferenceAgent
- `pixel_encoder_v2.py` — V2 CNNEncoderWithAux (+ auxiliary 8D state prediction head)
- `pixel_agent_v2.py` — V2 agent
- `pixel_encoder_v3.py` — V3 CNNEncoderWithDecode (full VAE + frame decoder + state decoder)
- `pixel_agent_v3.py` — V3 agent
- `pixel_agent_slds.py` — SLDS agent (V3 encoder + switching linear dynamics)
- `frame_stack.py` — FrameStack class and preprocess_frame()

**Extracting latent representations** (if loading succeeds):
```python
import sys
sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/lunar-lander/src')
import torch
from active_inference.encoder import CNNEncoder
from active_inference.pixel_agent import PixelActiveInferenceAgent

# Load checkpoint
ckpt = torch.load('C:/Users/citiz/Documents/noumenal-labs/lunar-lander/trained_agents/pixel_lunarlander_best.tar',
                   map_location='cpu', weights_only=False)

# Reconstruct encoder and load weights
pixel_config = ckpt['pixel_config']  # n_frames=4, height=84, width=84, latent_dim=64
encoder = CNNEncoder(n_frames=pixel_config.n_frames, height=pixel_config.height,
                     width=pixel_config.width, latent_dim=pixel_config.latent_dim)
encoder.load_state_dict(ckpt['encoder'])
encoder.eval()

# Encode stacked frames to latent
with torch.no_grad():
    frames_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0)  # (1, 4, 84, 84)
    latent = encoder(frames_tensor)  # (1, 64)

# For TB: compute gradients of dynamics loss in latent space
# The ensemble operates on [latent(64) + action_onehot(4)] = 68D -> 64D next-latent prediction
```

**Requirements**: `torch`, `gymnasium[box2d]`, `opencv-python` (for cv2)

**WARNING**: This loading code is speculative. The agents are known broken. Ralph must diagnose the actual issues by reading the source files and error messages.

### telecorder Repository (External, read-only)
**Path**: `C:\Users\citiz\Documents\noumenal-labs\telecorder`

```
└── services/connectors/lunarlander/src/telecorder_lunarlander/
    ├── thrml_wm_mini/
    │   ├── models.py          # DreamerWorldModel, Actor, Critic, EpistemicHead, RewardPredictor
    │   ├── training.py        # JIT training steps, imagine_trajectory
    │   ├── constants.py       # LATENT_DIM=64, config
    │   └── ...
    ├── dreamer_adapter.py     # create_dreamer_agent(checkpoint, ...)
    └── checkpoints/
        ├── lunarlander_ai_best.pt       # Primary checkpoint (~2.9 MB)
        ├── lunarlander_ai.pt
        ├── lunarlander_ai_final.pt
        ├── lunarlander_ai_final_best.pt
        ├── lunarlander_cem.pt
        └── lunarlander_cem_best.pt
```

**CRITICAL: Dreamer has NO trained checkpoint.** All `.pt` files in the telecorder checkpoints directory are PyTorch ensemble dynamics models (10 MLPs, 8D state space), NOT JAX/Equinox Dreamer models. The Dreamer *code* exists but was never trained.

**US-028 approach: Train a Dreamer autoencoder from scratch.** Use the Active Inference trajectories (4508 transitions from `world_model_analysis.py`) to train just the Encoder+Decoder on reconstruction loss. This is fast (no dynamics/actor/critic needed) and produces a 64D latent space for TB analysis.

**Dreamer Model Architecture** (JAX/Equinox, in `thrml_wm_mini/models.py`):
- **Encoder**: 8D → Linear(8,64) → ReLU → Linear(64,64) → ReLU → Linear(64,64) = 64D latent
- **Decoder**: 64D → Linear(64,64) → ReLU → Linear(64,64) → ReLU → Linear(64,8) = 8D obs
- **TransformerDynamics**: Flow-matching velocity predictor with 4 transformer blocks (dim=128, 4 heads). Predicts v = z1 - z0 in latent space.
- **Actor**: 64D → Linear(64,128) → ReLU → Linear(128,32) → ReLU → Linear(32,4) = 4 action logits
- **Critic**: 64D → Linear(64,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1) = scalar value
- **EpistemicHead**: (64+4)D → 64 → 64 → 1 (curiosity/surprise predictor)
- **RewardPredictor**: (64+4)D → 256 → 256 → 256 → 1

**Import requires zenoh workaround** (zenoh is not available on pip for Windows):
```python
import sys, types

# Bypass telecorder_lunarlander.__init__.py which imports zenoh
pkg_path = 'C:/Users/citiz/Documents/noumenal-labs/telecorder/services/connectors/lunarlander/src/telecorder_lunarlander'
pkg = types.ModuleType('telecorder_lunarlander')
pkg.__path__ = [pkg_path]
pkg.__package__ = 'telecorder_lunarlander'
sys.modules['telecorder_lunarlander'] = pkg

sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/telecorder/services/connectors/lunarlander/src')
from telecorder_lunarlander.thrml_wm_mini.models import Encoder, Decoder, DreamerWorldModel
from telecorder_lunarlander.thrml_wm_mini.constants import LATENT_DIM, STATE_DIM
```

**Training the autoencoder** (recommended approach for US-028):
```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Build encoder + decoder
key = jax.random.PRNGKey(42)
k_enc, k_dec = jax.random.split(key)
encoder = Encoder(k_enc, state_dim=8)
decoder = Decoder(k_dec, state_dim=8)

# Reconstruction loss
def recon_loss(encoder, decoder, batch):
    z = encoder(batch)          # [B, 64]
    recon = decoder(z)          # [B, 8]
    return jnp.mean((recon - batch) ** 2)

# Train on Active Inference trajectories (normalize first)
optimizer = optax.adam(1e-3)
# ... standard Equinox training loop ...

# After training, compute TB gradients in latent space
def latent_recon_loss(z_single, obs_target):
    return jnp.mean((decoder(z_single[None, :])[0] - obs_target) ** 2)
grad_fn = jax.vmap(jax.grad(latent_recon_loss), in_axes=(0, 0))
z_batch = encoder(obs_batch)
gradients = grad_fn(z_batch, obs_batch)  # [N, 64]
```

**Checkpoint directory** (these are all PyTorch, NOT Dreamer):
`C:/Users/citiz/Documents/noumenal-labs/telecorder/services/connectors/lunarlander/checkpoints/`
- `lunarlander_ai_best.pt`: 10-ensemble PyTorch dynamics model (2.8MB)
- All other `lunarlander_*.pt`: various PyTorch model variants

## Existing Code to Reuse

### From `experiments/quadratic_toy_comparison.py`:
- `build_precision_matrix(cfg)`: Block-structured precision matrix
- `langevin_sampling(Theta, ...)`: Collect trajectories + gradients
- `topological_blankets(gradients, n_objects)`: Full 6-phase pipeline
  - Phase 1: compute_geometric_features(gradients)
  - Phase 2: estimate Hessian as gradient covariance
  - Phase 3: detect_blankets_otsu(features)
  - Phase 4: cluster_internals(coupling_matrix, n_objects) via SpectralClustering
  - Phase 5: assign labels
  - Phase 6: extract topology
- `dmbd_style_partition(gradients, n_objects)`: Role clustering baseline
- `axiom_style_partition(samples, n_objects, gradients)`: GMM baseline
- `compute_metrics(pred, truth)`: Returns `{object_ari, blanket_f1, full_ari}`
- `run_strength_sweep(strengths, n_trials)`: Sweep blanket_strength
- `plot_strength_sweep(results, save_path)`: Already accepts save_path
- `QuadraticEBMConfig`: Dataclass with n_objects, vars_per_object, etc.

### From `experiments/spectral_friston_detection.py`:
- `build_adjacency_from_hessian(H)`, `build_graph_laplacian(A)`
- `spectral_partition(L, n_partitions)`, `identify_blanket_from_spectrum(...)`
- `hybrid_detection(gradients, H_est)`: Spectral + gradient fallback
- `recursive_spectral_detection(H, max_levels)`: Hierarchical extraction
- `run_spectral_experiment()`: Compares all methods
- `build_precision_matrix_hierarchical(...)`: Multi-level ground truth

### From `experiments/utils/`:
- `save_results(name, metrics, config, notes)`: JSON to results/
- `load_results(filepath)`: Read saved JSON
- `build_registry()`: Scan results/ and produce summary
- `save_figure(fig, name, experiment_name)`: PNG to results/

## Your Workflow

1. Read `scripts/ralph/prd.json` to find the first user story where `"passes": false`
2. Check `dependencies` field; if any dependency story has `"passes": false`, skip to the next story
3. Read the acceptance criteria carefully
4. Implement what is needed (create files, modify existing scripts, run experiments)
5. Verify all acceptance criteria pass
6. Update `scripts/ralph/prd.json`: set `"passes": true` for the completed story
7. Append a summary of what was done to `scripts/ralph/progress.txt`
8. **Do NOT commit or push.** The maintainer reviews all code before commits are made. Stage changes if needed but do not create commits.

## Quality Gates

For each user story, ALL of these must be true:
- **Code runs**: `python experiments/<script>.py` completes without errors
- **Results saved**: JSON file written to `results/` with the paper's metrics schema
- **Plots saved**: PNG files in `results/`, never `plt.show()`
- **No regressions**: Other experiment scripts still run if modified
- **PRD updated**: The user story's `"passes"` field set to `true`
- **Package importable** (Phase 2+): `python -c "from topological_blankets import TopologicalBlankets"` succeeds
- **Tests pass** (after US-013): `python -m pytest tests/` passes
- **No GUI**: No plt.show(), no browser windows, no display

## Important Rules

- Do NOT call `plt.show()` anywhere. Use `save_figure()` or `fig.savefig()` + `plt.close(fig)`.
- Do NOT open browser windows or display GUIs.
- Import plotting utilities as: `from experiments.utils.plotting import save_figure`
- Import results utilities as: `from experiments.utils.results import save_results`
- When modifying existing files, prefer minimal changes (add save calls, remove plt.show).
- Use `sys.path` manipulation if needed for imports: `sys.path.insert(0, project_root)`.
- Run experiments from the project root directory.
- For cross-repo imports, use sys.path:
  ```python
  sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/lunar-lander/sharing')   # for active_inference
  sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/telecorder/services/connectors/lunarlander/src')  # for telecorder_lunarlander
  ```
- Never modify files in lunar-lander or telecorder repos; they are read-only references.
- The pandas repo is *not* read-only; Phase 12 adds files to panda/ (learned_planner.py, tb_discovery.py, teleop_interface.py, catastrophe_bridge.py, tb_replay.py).
- When creating the topological_blankets/ package, move/refactor code from experiments; do not copy-paste entire files.
- Phase 2 stories should maintain backward compatibility: experiment scripts should continue to work via their own local functions AND via the new package.

## Phase 12: Teleoperation Demo Integration (PRIORITY)

**Deadline: Wednesday demo.** Phase 12 integrates the pandas Bayes ensemble (Alec's FetchPush manipulation agent) with topological blankets and the telecorder teleoperation platform.

**Demo narrative**: Agent learns to push objects → ensemble disagreement reveals uncertainty → TB discovers world model structure (gripper vs object vs relation) → catastrophe signal triggers human handover → human injects "move here" goals → learned low-level controller executes → agent resumes → TB shows what the agent learned.

**Phase 12 critical path**: US-088 (update prompt, no deps) + US-086 (train checkpoint, no deps) should start immediately. Then: US-076 (TB on ensemble) → US-077 (catastrophe bridge) → US-078 (teleop interface) → US-079 (viz panel) → US-081 (end-to-end demo).

**Parallel tracks**:
- Track A (demo-critical): US-088 → US-076 → US-077 → US-078 → US-079 → US-081
- Track B (training): US-086 (start ASAP, runs overnight)
- Track C (science): US-080 (structure emergence) + US-085 (ghost trajectories), can run after US-076
- Track D (stretch): US-082 (telecorder adapter), US-083 (cross-domain), US-084 (learned planner), US-087 (TB replay)

**Key integration points**:
- pandas ensemble → TB: extract gradients via jax.grad on DynamicsMember predictions, compute coupling matrix, run hybrid detection
- pandas planner → catastrophe: ensemble_disagreement from CEM evaluation → severity score → handover signal
- pandas symbolic planner → teleop: TeleopInterface wraps decide(), injects human PlanningObjective
- pandas utils → visualization: extend add_uncertainty_panel() with TB overlay and catastrophe indicators

## Dependencies and Ordering

Stories must be completed in dependency order. A story cannot be started until all stories in its `dependencies` list have `"passes": true`.

**Critical path**: US-010 → US-011 → US-024 → US-025 → US-030 → US-032

**Phase 2 parallelism**: After US-010, stories US-011, US-012, US-014, US-015, US-017 can proceed in any order.

**Phase 3 parallelism**: After US-010, stories US-018, US-019, US-020, US-021, US-022 can proceed in any order.

**Phase 12 parallelism**: US-086 and US-088 have no dependencies and should start first. US-076 depends on US-010 + US-011 (both complete). After US-076: US-077, US-080, US-083, US-085 can proceed in parallel.

## Requirements

When new dependencies are needed (PyTorch, JAX, gymnasium, etc.), add them to `requirements.txt` with appropriate version constraints. Group by phase:

```
# Phase 1 (existing)
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
scikit-image>=0.20
matplotlib>=3.7

# Phase 2
pytest>=7.0

# Phase 3 (no additional dependencies needed)

# Phase 4
torch>=2.0
jax>=0.4
jaxlib>=0.4
equinox>=0.11
optax>=0.1
gymnasium[box2d]>=0.29

# Phase 5
jupyter>=1.0
nbconvert>=7.0

# Phase 12 (Teleoperation Demo)
gymnasium[mujoco]>=0.29
imageio>=2.31
```

Only add dependencies when the story that needs them is being implemented.

## Completion Signal

When ALL user stories in prd.json have `"passes": true`, output exactly:

<promise>COMPLETE</promise>

If there are still incomplete stories, do NOT output the completion signal. Just finish the current story and exit cleanly.

## Git Workflow

- Work on branch `ralph/world-model-demo` (create if needed from current branch)
- Commit after each completed user story with message: `ralph: US-XXX - <short description>`
- **NEVER push directly.** When ready for review, create a pull request via `gh pr create`. The maintainer reviews all PRs before merging.
- When work is complete, leave a summary in `scripts/ralph/progress.txt` describing what was done, what files were created/modified, and what the next story should be.
