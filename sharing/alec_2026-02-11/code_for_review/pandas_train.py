from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import time
from typing import Callable
from collections import deque
from collections.abc import Mapping

import equinox as eqx
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro  # type: ignore[import-not-found]

from panda.common import (
    ObsDict,
    action_bounds,
    distance_threshold,
    make_env,
    normalize_reward_mode,
    reset_env,
    resolve_run_dir,
    save_json,
    save_run_config,
)
from panda.model import EnsembleModel, ModelConfig, make_model
from panda.planner import (
    CEMConfig,
    cem_action_entropy,
    cem_plan,
    cem_plan_spread,
    rollout_ensemble_disagreement,
)
from panda.symbolic_planner import (
    FetchPushSymbolicPlanner,
    SymbolicPlannerConfig,
    make_symbolic_planner,
)
from panda.replay import (
    CollectSummary,
    ReplayBuffer,
    ReplayData,
    build_collect_summary,
    episode_success_stats,
    stable_train_test_masks,
    train_test_masks,
)
from panda.training_helpers import (
    compute_stats,
    loss_fn,
    make_train_step,
    prepare_training_data,
)
from panda.utils import render_rgb_frame, save_gif
from panda.wandb_logger import WandbLogger


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "FetchReach-v4"
    reward_mode: str = "dense"
    run_dir: str | None = None
    seed: int = 0
    max_episode_steps: int = 200
    obs_key: str = "observation"
    achieved_goal_key: str = "achieved_goal"
    desired_goal_key: str = "desired_goal"

    initial_random_steps: int = 5000
    iterations: int = 100
    plan_steps: int = 500

    train_steps: int = 500
    train_log_every: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    train_ratio: float = 0.9
    shuffle: bool = True
    stable_episode_split: bool = True
    split_seed: int = 0

    ensemble_size: int = 5
    hidden_size: int = 256
    depth: int = 2

    horizon: int = 30
    population: int = 512
    elite_frac: float = 0.1
    cem_iters: int = 8
    init_std: float = 0.6
    action_penalty: float = 3e-4
    reward_weight: float = 1.0

    train_use_epistemic_bonus: bool = False
    train_epistemic_bonus_weight: float = 0.2
    eval_use_epistemic_bonus: bool = False
    eval_epistemic_bonus_weight: float = 0.2

    eval_every: int = 5
    eval_episodes: int = 1
    eval_horizon: int = 200
    eval_log_video: bool = True
    eval_video_fps: int = 20
    eval_video_frame_stride: int = 1
    eval_video_max_frames: int = 0
    render: bool = False

    planner_diagnostics_every: int = 10
    planner_spread_samples: int = 32
    train_loss_ema_beta: float = 0.98
    train_loss_window: int = 50
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    train_sample_from_cem: bool = True
    train_cem_sampling_anneal_fraction: float = 0.2
    train_cem_sampling_start_prob: float = 1.0
    train_cem_sampling_end_prob: float = 0.0

    symbolic_task: str = "none"
    symbolic_gripper_indices: tuple[int, int, int] = (0, 1, 2)
    symbolic_push_approach_offset: float = 0.08
    symbolic_push_approach_height_offset: float = 0.0
    symbolic_push_approach_threshold: float = 0.04
    symbolic_push_goal_threshold: float | None = None
    symbolic_phase1_object_goal_weight: float = 0.1
    symbolic_phase1_gripper_target_weight: float = 1.0
    symbolic_phase2_object_goal_weight: float = 1.0
    symbolic_phase2_gripper_target_weight: float = 0.25

    use_wandb: bool = True
    wandb_project: str = "panda-fetchreach"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_mode: str | None = None
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlannerStepDiagnostics:
    entropy: float | None = None
    plan_spread: float | None = None
    disagreement: float | None = None
    objective_name: str = "goal_tracking"
    symbolic_phase_name: str | None = None
    symbolic_phase_index: int | None = None
    symbolic_phase_total: int | None = None
    symbolic_reward: float | None = None
    object_reward: float | None = None
    gripper_penalty: float | None = None


@dataclass(frozen=True)
class EvalSummary:
    avg_return: float
    success_rate: float
    avg_steps: float
    elapsed: float
    episodes: int
    min_return: float
    max_return: float
    mean_plan_spread: float
    mean_disagreement: float


@dataclass(frozen=True)
class TrainSummary:
    train_transitions: int
    test_transitions: int
    train_episodes: int
    train_successful_episodes: int
    test_episodes: int
    test_successful_episodes: int
    last_train_loss: float
    last_test_loss: float | None


PlanActionFn = Callable[
    [ObsDict, int, int], tuple[np.ndarray, PlannerStepDiagnostics | None]
]
MetricValue = float | int | str | bool
MetricsLike = Mapping[str, MetricValue]


def _save_model(
    model: EnsembleModel,
    path: Path,
    obs_dim: int,
    action_dim: int,
    achieved_goal_dim: int,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)

    meta = {
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "achieved_goal_dim": int(achieved_goal_dim),
        "ensemble_size": int(cfg.ensemble_size),
        "hidden_size": int(cfg.hidden_size),
        "depth": int(cfg.depth),
        "obs_key": cfg.obs_key,
        "achieved_goal_key": cfg.achieved_goal_key,
        "desired_goal_key": cfg.desired_goal_key,
        "env_id": cfg.env_id,
        "reward_mode": cfg.reward_mode,
        "max_episode_steps": int(cfg.max_episode_steps),
    }
    save_json(path.with_suffix(path.suffix + ".json"), meta)


def _make_cem_cfg(
    cfg: TrainConfig,
    reward_mode: str,
    epistemic_bonus_weight: float,
    seed: int | None,
) -> CEMConfig:
    return CEMConfig(
        horizon=cfg.horizon,
        population=cfg.population,
        elite_frac=cfg.elite_frac,
        cem_iters=cfg.cem_iters,
        init_std=cfg.init_std,
        action_penalty=cfg.action_penalty,
        reward_weight=cfg.reward_weight,
        reward_mode=reward_mode,
        epistemic_bonus_weight=epistemic_bonus_weight,
        seed=seed,
    )


def _goal_reward_scalar(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    reward_mode: str,
    distance_threshold: float,
) -> float:
    dist = float(np.linalg.norm(achieved_goal - desired_goal))
    if reward_mode == "dense":
        return -dist
    if reward_mode == "sparse":
        return -1.0 if dist > distance_threshold else 0.0
    if reward_mode == "none":
        return 0.0
    raise ValueError(f"Unsupported reward_mode: {reward_mode!r}.")


def _success_from_info(info: object) -> bool:
    if not isinstance(info, dict):
        return False
    raw = info.get("is_success")
    if raw is None:
        return False
    arr = np.asarray(raw)
    if arr.size == 0:
        return False
    value = float(arr.reshape(-1)[0])
    if np.isnan(value):
        return False
    return value >= 0.5


def _success_from_obs(
    obs: ObsDict,
    achieved_goal_key: str,
    desired_goal_key: str,
    goal_distance_threshold: float,
) -> bool:
    achieved_goal = np.asarray(obs[achieved_goal_key], dtype=np.float32)
    desired_goal = np.asarray(obs[desired_goal_key], dtype=np.float32)
    return (
        float(np.linalg.norm(achieved_goal - desired_goal)) <= goal_distance_threshold
    )


def _plan_action(
    model: EnsembleModel,
    obs: ObsDict,
    episode_step: int,
    cfg: TrainConfig,
    cem_cfg: CEMConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
    goal_distance_threshold: float,
    rng: np.random.Generator,
    collect_step: int,
    sample_action_prob: float = 0.0,
    symbolic_planner: FetchPushSymbolicPlanner | None = None,
) -> tuple[np.ndarray, PlannerStepDiagnostics | None]:
    obs_vec = np.asarray(obs[cfg.obs_key], dtype=np.float32)
    achieved_goal = np.asarray(obs[cfg.achieved_goal_key], dtype=np.float32)
    desired_goal = np.asarray(obs[cfg.desired_goal_key], dtype=np.float32)
    objective = None
    symbolic_status = None
    if symbolic_planner is not None:
        if episode_step <= 0:
            symbolic_planner.reset()
        symbolic_decision = symbolic_planner.decide(
            obs_vec, achieved_goal, desired_goal
        )
        objective = symbolic_decision.objective
        symbolic_status = symbolic_decision.status

    objective_name = objective.name if objective is not None else "goal_tracking"
    symbolic_reward = None
    object_reward = None
    gripper_penalty = None
    if objective is not None:
        desired_goal_obj = (
            np.asarray(objective.desired_goal, dtype=np.float32)
            if objective.desired_goal is not None
            else desired_goal
        )
        object_reward = _goal_reward_scalar(
            achieved_goal,
            desired_goal_obj,
            cfg.reward_mode,
            goal_distance_threshold,
        )
        gripper_penalty = 0.0
        if objective.gripper_target is not None and objective.gripper_indices:
            gripper_pos = np.asarray(
                obs_vec[list(objective.gripper_indices)], dtype=np.float32
            )
            gripper_target = np.asarray(objective.gripper_target, dtype=np.float32)
            gripper_penalty = float(objective.gripper_target_weight) * float(
                np.linalg.norm(gripper_pos - gripper_target)
            )
        symbolic_reward = float(objective.object_goal_weight) * float(
            object_reward
        ) - float(gripper_penalty)
    remaining = cfg.max_episode_steps - episode_step
    valid_horizon = min(cem_cfg.horizon, remaining)
    step_seed = int(rng.integers(0, 2**32 - 1))
    step_cfg = replace(cem_cfg, seed=step_seed)

    log_diag = cfg.planner_diagnostics_every > 0 and (
        collect_step % cfg.planner_diagnostics_every == 0
    )
    plan_actions, plan_mean, plan_std, plan_rollout = cem_plan(
        model,
        obs_vec,
        achieved_goal,
        desired_goal,
        action_low,
        action_high,
        step_cfg,
        goal_distance_threshold,
        valid_horizon=valid_horizon,
        return_rollout=log_diag,
        objective=objective,
    )

    diagnostics = PlannerStepDiagnostics(
        objective_name=objective_name,
        symbolic_phase_name=(
            symbolic_status.phase_name if symbolic_status is not None else None
        ),
        symbolic_phase_index=(
            symbolic_status.phase_index if symbolic_status is not None else None
        ),
        symbolic_phase_total=(
            symbolic_status.phase_total if symbolic_status is not None else None
        ),
        symbolic_reward=symbolic_reward,
        object_reward=object_reward,
        gripper_penalty=gripper_penalty,
    )
    if log_diag:
        plan_spread = cem_plan_spread(
            model,
            obs_vec,
            achieved_goal,
            plan_mean,
            plan_std,
            action_low,
            action_high,
            cfg.planner_spread_samples,
            rng,
        )
        if plan_rollout is not None:
            disagreement = float(np.mean(plan_rollout.ensemble_disagreement))
        else:
            disagreement = float(
                np.mean(
                    np.asarray(
                        rollout_ensemble_disagreement(
                            model,
                            jnp.asarray(obs_vec),
                            jnp.asarray(achieved_goal),
                            jnp.asarray(plan_actions),
                        )
                    )
                )
            )
        diagnostics = replace(
            diagnostics,
            entropy=cem_action_entropy(plan_std),
            plan_spread=plan_spread,
            disagreement=disagreement,
        )

    action = plan_actions[0]
    if sample_action_prob > 0.0 and rng.random() < sample_action_prob:
        sampled = plan_mean[0] + plan_std[0] * rng.standard_normal(
            size=plan_mean.shape[1]
        ).astype(np.float32)
        action = np.clip(sampled, action_low, action_high)
    return np.asarray(action, dtype=np.float32), diagnostics


def _cem_sample_probability(cfg: TrainConfig, iteration_idx: int) -> float:
    if not cfg.train_sample_from_cem:
        return 0.0
    start = float(np.clip(cfg.train_cem_sampling_start_prob, 0.0, 1.0))
    end = float(np.clip(cfg.train_cem_sampling_end_prob, 0.0, 1.0))
    frac = float(np.clip(cfg.train_cem_sampling_anneal_fraction, 0.0, 1.0))
    if cfg.iterations <= 0:
        return end
    warm_iters = max(1, int(np.ceil(cfg.iterations * frac)))
    if warm_iters <= 1:
        return start if iteration_idx <= 0 else end
    if iteration_idx >= warm_iters:
        return end
    t = float(iteration_idx) / float(warm_iters - 1)
    return (1.0 - t) * start + t * end


def _collect_steps(
    env: gym.Env,
    buffer: ReplayBuffer,
    num_steps: int,
    obs: ObsDict,
    episode_id: int,
    episode_step: int,
    action_fn: PlanActionFn,
    rng: np.random.Generator,
    reward_mode: str,
    achieved_goal_key: str,
    desired_goal_key: str,
    goal_distance_threshold: float,
    render: bool,
    max_episode_steps: int,
    log_every: int,
    log_prefix: str,
) -> tuple[ObsDict, int, int, CollectSummary]:
    rewards: list[float] = []
    dones: list[bool] = []
    successes: list[bool] = []
    episode_ids: list[int] = []
    planner_entropy: list[float] = []
    planner_spread: list[float] = []
    planner_disagreement: list[float] = []

    for idx in range(num_steps):
        action, diagnostics = action_fn(obs, episode_step, idx)
        next_obs, reward_raw, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        success_from_info = _success_from_info(info)
        success_from_distance = _success_from_obs(
            next_obs,
            achieved_goal_key=achieved_goal_key,
            desired_goal_key=desired_goal_key,
            goal_distance_threshold=goal_distance_threshold,
        )
        success = success_from_distance
        true_reward = float(reward_raw)
        reward = 0.0 if reward_mode == "none" else float(reward_raw)

        buffer.add(obs, action, reward, next_obs, done, success, episode_id)

        if render:
            env.render()

        rewards.append(float(reward))
        dones.append(done)
        successes.append(success)
        episode_ids.append(episode_id)

        if diagnostics is not None:
            if diagnostics.entropy is not None:
                planner_entropy.append(diagnostics.entropy)
            if diagnostics.plan_spread is not None:
                planner_spread.append(diagnostics.plan_spread)
            if diagnostics.disagreement is not None:
                planner_disagreement.append(diagnostics.disagreement)

        episode_step += 1
        if log_every > 0 and (idx + 1) % log_every == 0:
            log_line = (
                f"{log_prefix} step {idx + 1:05d}/{num_steps} | "
                f"ep {episode_id} step {episode_step} | "
                f"reward {reward:.4f} | true_reward {true_reward:.4f} | "
                f"done {done} success {success} (info={success_from_info})"
            )
            if diagnostics is not None and (
                diagnostics.symbolic_phase_name is not None
                or diagnostics.symbolic_reward is not None
            ):
                phase_name = diagnostics.symbolic_phase_name
                phase_idx = diagnostics.symbolic_phase_index
                phase_total = diagnostics.symbolic_phase_total
                if phase_name:
                    if phase_idx is not None and phase_total is not None:
                        phase = f"{phase_idx}/{phase_total}:{phase_name}"
                    else:
                        phase = phase_name
                else:
                    phase = diagnostics.objective_name
                sym_reward_text = (
                    f"{diagnostics.symbolic_reward:.4f}"
                    if diagnostics.symbolic_reward is not None
                    else "n/a"
                )
                log_line += f" | sym_phase {phase} | sym_reward {sym_reward_text}"
            print(log_line)

        if done or success or episode_step >= max_episode_steps:
            obs = reset_env(env, int(rng.integers(0, 2**32 - 1)))
            episode_id += 1
            episode_step = 0
        else:
            obs = next_obs

    summary = build_collect_summary(
        rewards,
        dones,
        successes,
        episode_ids,
        planner_entropy,
        planner_spread,
        planner_disagreement,
    )
    return obs, episode_id, episode_step, summary


def _train_model(
    data: ReplayData,
    cfg: TrainConfig,
    rng: np.random.Generator,
    model: EnsembleModel | None,
    opt_state: optax.OptState | None,
    optimizer: optax.GradientTransformation,
    log_fn: Callable[[MetricsLike], None] | None,
    update_step_offset: int,
    env_steps: int,
) -> tuple[EnsembleModel, optax.OptState, TrainSummary]:
    obs = data.obs[cfg.obs_key]
    achieved_goal = data.obs[cfg.achieved_goal_key]
    next_obs = data.next_obs[cfg.obs_key]
    next_achieved_goal = data.next_obs[cfg.achieved_goal_key]
    actions = data.actions

    if cfg.stable_episode_split:
        train_mask, test_mask = stable_train_test_masks(
            data.episode_ids,
            cfg.train_ratio,
            split_seed=cfg.split_seed,
        )
    else:
        train_mask, test_mask = train_test_masks(
            data.episode_ids, cfg.train_ratio, cfg.shuffle, rng
        )

    train_obs = obs[train_mask]
    train_achieved = achieved_goal[train_mask]
    train_actions = actions[train_mask]
    train_next_obs = next_obs[train_mask]
    train_next_achieved = next_achieved_goal[train_mask]

    if train_obs.shape[0] == 0:
        raise ValueError("No transitions available for training.")

    test_obs = obs[test_mask]
    test_achieved = achieved_goal[test_mask]
    test_actions = actions[test_mask]
    test_next_obs = next_obs[test_mask]
    test_next_achieved = next_achieved_goal[test_mask]

    normalizer = compute_stats(
        train_obs,
        train_actions,
        train_next_obs,
        train_achieved,
        train_next_achieved,
    )
    obs_j, ag_j, actions_j, delta_obs_j, delta_ag_j = prepare_training_data(
        train_obs,
        train_achieved,
        train_actions,
        train_next_obs,
        train_next_achieved,
        normalizer,
    )

    test_batch = None
    if test_obs.shape[0] > 0:
        test_batch = prepare_training_data(
            test_obs,
            test_achieved,
            test_actions,
            test_next_obs,
            test_next_achieved,
            normalizer,
        )

    obs_dim = train_obs.shape[1]
    action_dim = train_actions.shape[1]
    achieved_goal_dim = train_achieved.shape[1]

    model_seed = int(rng.integers(0, 2**32 - 1))
    key = jax.random.PRNGKey(model_seed)

    if model is None:
        key, init_key = jax.random.split(key)
        model = make_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            achieved_goal_dim=achieved_goal_dim,
            config=ModelConfig(cfg.ensemble_size, cfg.hidden_size, cfg.depth),
            normalizer=normalizer,
            key=init_key,
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    else:
        model = eqx.tree_at(lambda m: m.normalizer, model, normalizer)
        if opt_state is None:
            opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    assert model is not None
    assert opt_state is not None

    train_step = make_train_step(optimizer)

    @eqx.filter_jit
    def eval_loss(eval_model: EnsembleModel, eval_batch):
        return loss_fn(eval_model, eval_batch)

    dataset_size = obs_j.shape[0]
    last_train_loss = 0.0
    last_test_loss: float | None = None
    ema_beta = float(np.clip(cfg.train_loss_ema_beta, 0.0, 0.9999))
    loss_ema: float | None = None
    loss_window = deque(maxlen=max(1, int(cfg.train_loss_window)))
    loss_window_mean = 0.0

    for local_step in range(cfg.train_steps):
        key, batch_key = jax.random.split(key)
        idx = jax.random.randint(
            batch_key,
            (cfg.ensemble_size, cfg.batch_size),
            0,
            dataset_size,
        )
        batch = (
            obs_j[idx],
            ag_j[idx],
            actions_j[idx],
            delta_obs_j[idx],
            delta_ag_j[idx],
        )
        model, opt_state, loss = train_step(model, opt_state, batch)
        last_train_loss = float(loss)
        if loss_ema is None:
            loss_ema = last_train_loss
        else:
            loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * last_train_loss
        loss_window.append(last_train_loss)
        loss_window_mean = float(np.mean(loss_window))

        should_log = (
            local_step % cfg.train_log_every == 0 or local_step == cfg.train_steps - 1
        )
        if should_log:
            if test_batch is not None:
                last_test_loss = float(eval_loss(model, test_batch))
                print(
                    f"train step {local_step:05d} | "
                    f"train_raw {last_train_loss:.6f} | "
                    f"train_ema {float(loss_ema):.6f} | "
                    f"test {last_test_loss:.6f}"
                )
            else:
                print(
                    f"train step {local_step:05d} | "
                    f"train_raw {last_train_loss:.6f} | "
                    f"train_ema {float(loss_ema):.6f}"
                )

            if log_fn is not None:
                metrics: dict[str, MetricValue] = {
                    "train/loss": (
                        float(loss_ema) if loss_ema is not None else last_train_loss
                    ),
                    "train/loss_raw": last_train_loss,
                    "train/loss_window_mean": loss_window_mean,
                    "train/update_step": float(update_step_offset + local_step),
                    "data/env_steps": float(env_steps),
                }
                if last_test_loss is not None:
                    metrics["train/test_loss"] = last_test_loss
                log_fn(metrics)

    train_episodes, train_success_episodes = episode_success_stats(
        data.episode_ids, data.successes, train_mask
    )
    test_episodes, test_success_episodes = episode_success_stats(
        data.episode_ids, data.successes, test_mask
    )

    summary = TrainSummary(
        train_transitions=int(train_obs.shape[0]),
        test_transitions=int(test_obs.shape[0]),
        train_episodes=train_episodes,
        train_successful_episodes=train_success_episodes,
        test_episodes=test_episodes,
        test_successful_episodes=test_success_episodes,
        last_train_loss=last_train_loss,
        last_test_loss=last_test_loss,
    )
    return model, opt_state, summary


def _evaluate_policy(
    env: gym.Env,
    model: EnsembleModel,
    cfg: TrainConfig,
    cem_cfg: CEMConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
    goal_distance_threshold: float,
    rng: np.random.Generator,
    symbolic_planner: FetchPushSymbolicPlanner | None = None,
    video_gif_path: Path | None = None,
    video_fps: int = 20,
    video_frame_stride: int = 3,
    video_max_frames: int = 120,
    logger: WandbLogger | None = None,
) -> EvalSummary:
    returns = []
    successes = []
    lengths = []
    spreads: list[float] = []
    disagreements: list[float] = []
    capture_video = video_gif_path is not None
    frame_stride = max(1, int(video_frame_stride))
    max_frames = int(video_max_frames)
    gif_fps = max(1, int(video_fps))
    video_frames: list[np.ndarray] = []
    video_target_shape: tuple[int, int] | None = None

    horizon = cfg.eval_horizon if cfg.eval_horizon > 0 else cfg.max_episode_steps
    start_time = time.perf_counter()

    def maybe_capture_frame(goal: np.ndarray | None = None) -> None:
        nonlocal video_target_shape
        if not capture_video:
            return
        if max_frames > 0 and len(video_frames) >= max_frames:
            return
        rendered = render_rgb_frame(env, video_target_shape, desired_goal=goal)
        if rendered is None:
            return
        frame, video_target_shape = rendered
        video_frames.append(frame)

    for _episode_idx in range(cfg.eval_episodes):
        if symbolic_planner is not None:
            symbolic_planner.reset()
        obs = reset_env(env, int(rng.integers(0, 2**32 - 1)))
        episode_return = 0.0
        success = False
        steps_taken = 0
        capture_episode = capture_video and _episode_idx == 0
        if capture_episode:
            maybe_capture_frame(np.asarray(obs[cfg.desired_goal_key], dtype=np.float32))

        for step in range(horizon):
            action, diagnostics = _plan_action(
                model,
                obs,
                step,
                cfg,
                cem_cfg,
                action_low,
                action_high,
                goal_distance_threshold,
                rng,
                collect_step=step,
                symbolic_planner=symbolic_planner,
            )
            obs, reward_raw, terminated, truncated, info = env.step(action)
            reward = 0.0 if cfg.reward_mode == "none" else float(reward_raw)
            episode_return += reward
            if cfg.render and not capture_video:
                env.render()
            if capture_episode and (step + 1) % frame_stride == 0:
                maybe_capture_frame(np.asarray(obs[cfg.desired_goal_key], dtype=np.float32))

            if diagnostics is not None:
                if diagnostics.plan_spread is not None:
                    spreads.append(diagnostics.plan_spread)
                if diagnostics.disagreement is not None:
                    disagreements.append(diagnostics.disagreement)

            success_from_distance = _success_from_obs(
                obs,
                achieved_goal_key=cfg.achieved_goal_key,
                desired_goal_key=cfg.desired_goal_key,
                goal_distance_threshold=goal_distance_threshold,
            )
            success = success or success_from_distance
            steps_taken = step + 1
            if terminated or truncated or success:
                break

        returns.append(episode_return)
        successes.append(success)
        lengths.append(steps_taken)
        if capture_episode:
            maybe_capture_frame(np.asarray(obs[cfg.desired_goal_key], dtype=np.float32))

    if capture_video and video_frames:
        assert video_gif_path is not None
        save_gif(video_gif_path, video_frames, gif_fps)
        if logger is not None:
            logger.log_gif("eval/video", video_gif_path, gif_fps)
    elif capture_video:
        print("eval video skipped: no frames captured")

    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_steps = float(np.mean(lengths)) if lengths else 0.0
    min_return = float(np.min(returns)) if returns else 0.0
    max_return = float(np.max(returns)) if returns else 0.0
    elapsed = time.perf_counter() - start_time
    mean_plan_spread = float(np.mean(spreads)) if spreads else 0.0
    mean_disagreement = float(np.mean(disagreements)) if disagreements else 0.0

    print(
        f"eval | avg return {avg_return:.3f} | success {success_rate:.2%} "
        f"({sum(successes)}/{len(successes)}) | avg steps {avg_steps:.1f} | "
        f"range [{min_return:.3f}, {max_return:.3f}] | "
        f"spread {mean_plan_spread:.4f} | disagreement {mean_disagreement:.4f} | "
        f"time {elapsed:.1f}s"
    )

    return EvalSummary(
        avg_return=avg_return,
        success_rate=success_rate,
        avg_steps=avg_steps,
        elapsed=elapsed,
        episodes=len(returns),
        min_return=min_return,
        max_return=max_return,
        mean_plan_spread=mean_plan_spread,
        mean_disagreement=mean_disagreement,
    )


def train(cfg: TrainConfig) -> None:
    reward_mode = normalize_reward_mode(cfg.reward_mode)
    cfg = replace(cfg, reward_mode=reward_mode)

    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())

    run_dir = resolve_run_dir(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, run_dir)

    rng = np.random.default_rng(cfg.seed)
    render_mode = "human" if cfg.render else None
    env = make_env(
        env_id=cfg.env_id,
        max_episode_steps=cfg.max_episode_steps,
        reward_mode=cfg.reward_mode,
        render_mode=render_mode,
    )
    eval_render_mode = "rgb_array" if cfg.eval_log_video else render_mode
    eval_env = make_env(
        env_id=cfg.env_id,
        max_episode_steps=cfg.max_episode_steps,
        reward_mode=cfg.reward_mode,
        render_mode=eval_render_mode,
    )

    logger = WandbLogger(
        enabled=cfg.use_wandb,
        project=cfg.wandb_project,
        run_dir=run_dir,
        config=cfg,
        entity=cfg.wandb_entity,
        name=cfg.wandb_name,
        group=cfg.wandb_group,
        mode=cfg.wandb_mode,
        tags=cfg.wandb_tags,
    )

    if cfg.grad_clip_norm > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
        )
    else:
        optimizer = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)

    try:
        print(f"run dir: {run_dir}")
        print(
            f"env: {cfg.env_id} | reward_mode: {cfg.reward_mode} | "
            f"seed {cfg.seed} | max steps {cfg.max_episode_steps}"
        )

        obs = reset_env(env, cfg.seed)
        buffer = ReplayBuffer(list(obs.keys()))
        episode_id = 0
        episode_step = 0

        action_low, action_high = action_bounds(env)
        goal_distance_threshold = distance_threshold(env)
        symbolic_cfg = SymbolicPlannerConfig(
            task=cfg.symbolic_task,
            gripper_indices=tuple(cfg.symbolic_gripper_indices),
            push_approach_offset=cfg.symbolic_push_approach_offset,
            push_approach_height_offset=cfg.symbolic_push_approach_height_offset,
            push_approach_threshold=cfg.symbolic_push_approach_threshold,
            push_goal_threshold=cfg.symbolic_push_goal_threshold,
            phase1_object_goal_weight=cfg.symbolic_phase1_object_goal_weight,
            phase1_gripper_target_weight=cfg.symbolic_phase1_gripper_target_weight,
            phase2_object_goal_weight=cfg.symbolic_phase2_object_goal_weight,
            phase2_gripper_target_weight=cfg.symbolic_phase2_gripper_target_weight,
        )
        symbolic_planner = make_symbolic_planner(
            symbolic_cfg,
            env_id=cfg.env_id,
            default_goal_threshold=goal_distance_threshold,
        )
        print(
            "symbolic planner: "
            f"{cfg.symbolic_task} "
            f"({'on' if symbolic_planner is not None else 'off'})"
        )

        train_epistemic_weight = (
            cfg.train_epistemic_bonus_weight if cfg.train_use_epistemic_bonus else 0.0
        )
        eval_epistemic_weight = (
            cfg.eval_epistemic_bonus_weight if cfg.eval_use_epistemic_bonus else 0.0
        )

        train_cem_cfg = _make_cem_cfg(
            cfg,
            reward_mode=cfg.reward_mode,
            epistemic_bonus_weight=train_epistemic_weight,
            seed=None,
        )
        eval_cem_cfg = _make_cem_cfg(
            cfg,
            reward_mode=cfg.reward_mode,
            epistemic_bonus_weight=eval_epistemic_weight,
            seed=None,
        )
        eval_video_path = run_dir / "eval_latest.gif" if cfg.eval_log_video else None

        print(f"collecting {cfg.initial_random_steps} random steps...")
        collect_start = time.perf_counter()
        obs, episode_id, episode_step, random_summary = _collect_steps(
            env=env,
            buffer=buffer,
            num_steps=cfg.initial_random_steps,
            obs=obs,
            episode_id=episode_id,
            episode_step=episode_step,
            action_fn=lambda _obs, _ep_step, _idx: (env.action_space.sample(), None),
            rng=rng,
            reward_mode=cfg.reward_mode,
            achieved_goal_key=cfg.achieved_goal_key,
            desired_goal_key=cfg.desired_goal_key,
            goal_distance_threshold=goal_distance_threshold,
            render=cfg.render,
            max_episode_steps=cfg.max_episode_steps,
            log_every=50,
            log_prefix="random",
        )
        collect_elapsed = time.perf_counter() - collect_start
        print(
            "random collect stats: "
            f"episodes {random_summary.episodes_touched} | "
            f"successful episodes {random_summary.successful_episodes} | "
            f"mean reward {random_summary.reward_mean:.4f} | "
            f"time {collect_elapsed:.1f}s"
        )

        total_env_steps = len(buffer)
        logger.log(
            {
                "collect/random_steps": float(random_summary.num_steps),
                "collect/random_reward_mean": random_summary.reward_mean,
                "collect/random_reward_std": random_summary.reward_std,
                "collect/random_success_episode_rate": (
                    random_summary.successful_episodes / random_summary.episodes_touched
                    if random_summary.episodes_touched > 0
                    else 0.0
                ),
                "collect/random_success_transition_rate": random_summary.success_transition_rate,
                "data/env_steps": float(total_env_steps),
            }
        )

        data = buffer.stack()
        train_start = time.perf_counter()
        update_step_offset = 0
        model, opt_state, train_summary = _train_model(
            data=data,
            cfg=cfg,
            rng=rng,
            model=None,
            opt_state=None,
            optimizer=optimizer,
            log_fn=logger.log,
            update_step_offset=update_step_offset,
            env_steps=total_env_steps,
        )
        update_step_offset += cfg.train_steps
        train_elapsed = time.perf_counter() - train_start

        logger.log(
            {
                "train/phase_time_sec": train_elapsed,
                "train/transitions": float(train_summary.train_transitions),
                "train/test_transitions": float(train_summary.test_transitions),
                "train/data_episode_success_rate": (
                    train_summary.train_successful_episodes
                    / train_summary.train_episodes
                    if train_summary.train_episodes > 0
                    else 0.0
                ),
                "test/data_episode_success_rate": (
                    train_summary.test_successful_episodes / train_summary.test_episodes
                    if train_summary.test_episodes > 0
                    else 0.0
                ),
                "data/env_steps": float(total_env_steps),
            }
        )

        _save_model(
            model,
            run_dir / "model.eqx",
            data.obs[cfg.obs_key].shape[1],
            data.actions.shape[1],
            data.obs[cfg.achieved_goal_key].shape[1],
            cfg,
        )

        for iteration in range(cfg.iterations):
            sample_action_prob = _cem_sample_probability(cfg, iteration)
            print(
                f"iteration {iteration + 1}/{cfg.iterations} | "
                f"cem sample prob {sample_action_prob:.3f}"
            )
            collect_start = time.perf_counter()
            if symbolic_planner is not None:
                symbolic_planner.reset()
            obs, episode_id, episode_step, planned_summary = _collect_steps(
                env=env,
                buffer=buffer,
                num_steps=cfg.plan_steps,
                obs=obs,
                episode_id=episode_id,
                episode_step=episode_step,
                action_fn=lambda obs_t, step_t, idx_t: _plan_action(
                    model,
                    obs_t,
                    step_t,
                    cfg,
                    train_cem_cfg,
                    action_low,
                    action_high,
                    goal_distance_threshold,
                    rng,
                    idx_t,
                    sample_action_prob=sample_action_prob,
                    symbolic_planner=symbolic_planner,
                ),
                rng=rng,
                reward_mode=cfg.reward_mode,
                achieved_goal_key=cfg.achieved_goal_key,
                desired_goal_key=cfg.desired_goal_key,
                goal_distance_threshold=goal_distance_threshold,
                render=cfg.render,
                max_episode_steps=cfg.max_episode_steps,
                log_every=50,
                log_prefix="planned",
            )
            collect_elapsed = time.perf_counter() - collect_start
            total_env_steps = len(buffer)

            print(
                "planned collect stats: "
                f"episodes {planned_summary.episodes_touched} | "
                f"successful episodes {planned_summary.successful_episodes} | "
                f"mean reward {planned_summary.reward_mean:.4f} | "
                f"diag spread {planned_summary.planner_spread_mean:.4f} | "
                f"diag disagreement {planned_summary.planner_disagreement_mean:.4f} | "
                f"time {collect_elapsed:.1f}s"
            )

            logger.log(
                {
                    "collect/planned_steps": float(planned_summary.num_steps),
                    "collect/planned_reward_mean": planned_summary.reward_mean,
                    "collect/planned_reward_std": planned_summary.reward_std,
                    "collect/planned_success_episode_rate": (
                        planned_summary.successful_episodes
                        / planned_summary.episodes_touched
                        if planned_summary.episodes_touched > 0
                        else 0.0
                    ),
                    "collect/planned_success_transition_rate": planned_summary.success_transition_rate,
                    "collect/planner_entropy_mean": planned_summary.planner_entropy_mean,
                    "collect/planner_spread_mean": planned_summary.planner_spread_mean,
                    "collect/planner_disagreement_mean": planned_summary.planner_disagreement_mean,
                    "collect/cem_sample_prob": sample_action_prob,
                    "collect/phase_time_sec": collect_elapsed,
                    "data/env_steps": float(total_env_steps),
                    "data/buffer_size": float(total_env_steps),
                }
            )

            data = buffer.stack()
            train_start = time.perf_counter()
            model, opt_state, train_summary = _train_model(
                data=data,
                cfg=cfg,
                rng=rng,
                model=model,
                opt_state=opt_state,
                optimizer=optimizer,
                log_fn=logger.log,
                update_step_offset=update_step_offset,
                env_steps=total_env_steps,
            )
            update_step_offset += cfg.train_steps
            train_elapsed = time.perf_counter() - train_start

            logger.log(
                {
                    "train/phase_time_sec": train_elapsed,
                    "train/transitions": float(train_summary.train_transitions),
                    "train/test_transitions": float(train_summary.test_transitions),
                    "train/data_episode_success_rate": (
                        train_summary.train_successful_episodes
                        / train_summary.train_episodes
                        if train_summary.train_episodes > 0
                        else 0.0
                    ),
                    "test/data_episode_success_rate": (
                        train_summary.test_successful_episodes
                        / train_summary.test_episodes
                        if train_summary.test_episodes > 0
                        else 0.0
                    ),
                    "data/env_steps": float(total_env_steps),
                }
            )

            _save_model(
                model,
                run_dir / "model.eqx",
                data.obs[cfg.obs_key].shape[1],
                data.actions.shape[1],
                data.obs[cfg.achieved_goal_key].shape[1],
                cfg,
            )

            if cfg.eval_every > 0 and (iteration + 1) % cfg.eval_every == 0:
                summary = _evaluate_policy(
                    env=eval_env,
                    model=model,
                    cfg=cfg,
                    cem_cfg=eval_cem_cfg,
                    action_low=action_low,
                    action_high=action_high,
                    goal_distance_threshold=goal_distance_threshold,
                    rng=rng,
                    symbolic_planner=symbolic_planner,
                    video_gif_path=eval_video_path,
                    video_fps=cfg.eval_video_fps,
                    video_frame_stride=cfg.eval_video_frame_stride,
                    video_max_frames=cfg.eval_video_max_frames,
                    logger=logger,
                )
                logger.log(
                    {
                        "eval/avg_return": summary.avg_return,
                        "eval/min_return": summary.min_return,
                        "eval/max_return": summary.max_return,
                        "eval/success_rate": summary.success_rate,
                        "eval/avg_steps": summary.avg_steps,
                        "eval/episodes": float(summary.episodes),
                        "eval/plan_spread_mean": summary.mean_plan_spread,
                        "eval/disagreement_mean": summary.mean_disagreement,
                        "eval/elapsed_sec": summary.elapsed,
                        "data/env_steps": float(total_env_steps),
                    }
                )

        metadata = {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_env_steps": total_env_steps,
            "total_train_updates": update_step_offset,
            "reward_mode": cfg.reward_mode,
            "train_use_epistemic_bonus": cfg.train_use_epistemic_bonus,
            "eval_use_epistemic_bonus": cfg.eval_use_epistemic_bonus,
            "symbolic_task": cfg.symbolic_task,
        }
        (run_dir / "train_summary.json").write_text(json.dumps(metadata, indent=2))
        print(f"training complete | run dir: {run_dir}")
    finally:
        env.close()
        eval_env.close()
        logger.finish()


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
