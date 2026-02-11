from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .model import EnsembleModel


@dataclass(frozen=True)
class CEMConfig:
    horizon: int
    population: int
    elite_frac: float
    cem_iters: int
    init_std: float
    action_penalty: float
    reward_weight: float
    reward_mode: str
    epistemic_bonus_weight: float
    seed: int | None = None


@dataclass(frozen=True)
class PlanningObjective:
    name: str = "goal_tracking"
    desired_goal: np.ndarray | None = None
    object_goal_weight: float = 1.0
    gripper_target: np.ndarray | None = None
    gripper_target_weight: float = 0.0
    gripper_indices: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class CEMPlanRollout:
    reward_members: np.ndarray
    goal_distance_members: np.ndarray
    ensemble_disagreement: np.ndarray
    gripper_target_distance_members: np.ndarray | None = None
    objective_name: str = "goal_tracking"


@dataclass
class PlannerHistory:
    steps: list[int] = field(default_factory=list)
    cem_action_entropy: list[float] = field(default_factory=list)
    cem_plan_spread: list[float] = field(default_factory=list)
    ensemble_disagreement: list[float] = field(default_factory=list)
    true_goal_distance: list[float] = field(default_factory=list)
    symbolic_phase_name: list[str] = field(default_factory=list)
    symbolic_phase_index: list[int] = field(default_factory=list)
    symbolic_phase_total: list[int] = field(default_factory=list)
    symbolic_subgoal_distance: list[float] = field(default_factory=list)
    symbolic_subgoal_threshold: list[float] = field(default_factory=list)
    symbolic_object_goal_distance: list[float] = field(default_factory=list)
    symbolic_status_text: list[str] = field(default_factory=list)

    def record(
        self,
        step: int,
        cem_std: np.ndarray,
        cem_plan_spread: float,
        ensemble_disagreement: float,
        true_goal_distance: float | None = None,
        symbolic_phase_name: str | None = None,
        symbolic_phase_index: int | None = None,
        symbolic_phase_total: int | None = None,
        symbolic_subgoal_distance: float | None = None,
        symbolic_subgoal_threshold: float | None = None,
        symbolic_object_goal_distance: float | None = None,
        symbolic_status_text: str | None = None,
    ) -> None:
        self.steps.append(step)
        self.cem_action_entropy.append(cem_action_entropy(cem_std))
        self.cem_plan_spread.append(cem_plan_spread)
        self.ensemble_disagreement.append(ensemble_disagreement)
        if true_goal_distance is not None:
            self.true_goal_distance.append(float(true_goal_distance))
        self.symbolic_phase_name.append(symbolic_phase_name or "")
        self.symbolic_phase_index.append(
            int(symbolic_phase_index) if symbolic_phase_index is not None else 0
        )
        self.symbolic_phase_total.append(
            int(symbolic_phase_total) if symbolic_phase_total is not None else 0
        )
        self.symbolic_subgoal_distance.append(
            float(symbolic_subgoal_distance)
            if symbolic_subgoal_distance is not None
            else float("nan")
        )
        self.symbolic_subgoal_threshold.append(
            float(symbolic_subgoal_threshold)
            if symbolic_subgoal_threshold is not None
            else float("nan")
        )
        self.symbolic_object_goal_distance.append(
            float(symbolic_object_goal_distance)
            if symbolic_object_goal_distance is not None
            else float("nan")
        )
        self.symbolic_status_text.append(symbolic_status_text or "")


def cem_action_entropy(action_std: np.ndarray) -> float:
    return float(np.mean(np.log(action_std + 1e-6)))


@eqx.filter_jit
def rollout_model_batch(
    model: EnsembleModel,
    obs0: jnp.ndarray,
    achieved_goal0: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    actions_t = jnp.swapaxes(actions, 0, 1)

    def step(carry, action_t):
        obs, achieved_goal = carry
        delta_obs, delta_ag = model.predict_deltas(obs, achieved_goal, action_t)
        delta_obs_mean = jnp.mean(delta_obs, axis=0)
        delta_ag_mean = jnp.mean(delta_ag, axis=0)
        obs_next = obs + delta_obs_mean
        achieved_goal_next = achieved_goal + delta_ag_mean
        return (obs_next, achieved_goal_next), achieved_goal_next

    _, ag_seq = jax.lax.scan(step, (obs0, achieved_goal0), actions_t)
    return ag_seq


def _predict_members(
    model: EnsembleModel,
    obs_members: jnp.ndarray,
    achieved_goal_members: jnp.ndarray,
    action: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    delta_obs = []
    delta_ag = []
    for idx, member in enumerate(model.members):
        delta_obs_norm, delta_ag_norm = member(
            obs_members[idx],
            achieved_goal_members[idx],
            action,
            model.normalizer,
        )
        delta_obs.append(model.normalizer.denormalize_delta_obs(delta_obs_norm))
        delta_ag.append(model.normalizer.denormalize_delta_ag(delta_ag_norm))
    return jnp.stack(delta_obs, axis=0), jnp.stack(delta_ag, axis=0)


def _goal_reward(
    achieved_goal: jnp.ndarray,
    desired_goal: jnp.ndarray,
    reward_mode: str,
    distance_threshold: float,
) -> jnp.ndarray:
    dist = jnp.linalg.norm(achieved_goal - desired_goal, axis=-1)
    if reward_mode == "dense":
        return -dist
    if reward_mode == "sparse":
        return -(dist > distance_threshold).astype(jnp.float32)
    if reward_mode == "none":
        return jnp.zeros_like(dist)
    raise ValueError(f"Unsupported reward_mode: {reward_mode!r}.")


@eqx.filter_jit
def rollout_ensemble_objective_metrics(
    model: EnsembleModel,
    obs0: jnp.ndarray,
    achieved_goal0: jnp.ndarray,
    actions: jnp.ndarray,
    desired_goal: jnp.ndarray,
    reward_mode: str,
    distance_threshold: float,
    object_goal_weight: float,
    gripper_target: jnp.ndarray,
    gripper_target_weight: float,
    gripper_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    actions_t = jnp.swapaxes(actions, 0, 1)
    ensemble_size = len(model.members)
    obs_members = jnp.repeat(obs0[None, ...], ensemble_size, axis=0)
    achieved_goal_members = jnp.repeat(achieved_goal0[None, ...], ensemble_size, axis=0)

    def step(carry, action_t):
        obs_members_t, achieved_goal_members_t = carry
        delta_obs, delta_ag = _predict_members(
            model, obs_members_t, achieved_goal_members_t, action_t
        )
        obs_members_t = obs_members_t + delta_obs
        achieved_goal_members_t = achieved_goal_members_t + delta_ag

        object_rewards = _goal_reward(
            achieved_goal_members_t, desired_goal, reward_mode, distance_threshold
        )
        gripper_pos = jnp.take(obs_members_t, gripper_indices, axis=-1)
        gripper_dist = jnp.linalg.norm(gripper_pos - gripper_target, axis=-1)
        rewards = (
            object_goal_weight * object_rewards
            - gripper_target_weight * gripper_dist
        )
        reward_mean = jnp.mean(rewards, axis=0)
        ensemble_disagreement = jnp.mean(
            jnp.var(achieved_goal_members_t, axis=0), axis=-1
        )
        return (
            obs_members_t,
            achieved_goal_members_t,
        ), (reward_mean, ensemble_disagreement)

    _, (reward_seq, ensemble_disagreement_seq) = jax.lax.scan(
        step, (obs_members, achieved_goal_members), actions_t
    )
    return reward_seq, ensemble_disagreement_seq


@eqx.filter_jit
def rollout_ensemble_metrics(
    model: EnsembleModel,
    obs0: jnp.ndarray,
    achieved_goal0: jnp.ndarray,
    actions: jnp.ndarray,
    desired_goal: jnp.ndarray,
    reward_mode: str,
    distance_threshold: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    actions_t = jnp.swapaxes(actions, 0, 1)
    ensemble_size = len(model.members)
    obs_members = jnp.repeat(obs0[None, ...], ensemble_size, axis=0)
    achieved_goal_members = jnp.repeat(achieved_goal0[None, ...], ensemble_size, axis=0)

    def step(carry, action_t):
        obs_members_t, achieved_goal_members_t = carry
        delta_obs, delta_ag = _predict_members(
            model, obs_members_t, achieved_goal_members_t, action_t
        )
        obs_members_t = obs_members_t + delta_obs
        achieved_goal_members_t = achieved_goal_members_t + delta_ag
        rewards = _goal_reward(
            achieved_goal_members_t, desired_goal, reward_mode, distance_threshold
        )
        reward_mean = jnp.mean(rewards, axis=0)
        ensemble_disagreement = jnp.mean(
            jnp.var(achieved_goal_members_t, axis=0), axis=-1
        )
        return (
            obs_members_t,
            achieved_goal_members_t,
        ), (reward_mean, ensemble_disagreement)

    _, (reward_seq, ensemble_disagreement_seq) = jax.lax.scan(
        step, (obs_members, achieved_goal_members), actions_t
    )
    return reward_seq, ensemble_disagreement_seq


@eqx.filter_jit
def rollout_ensemble_disagreement(
    model: EnsembleModel,
    obs0: jnp.ndarray,
    achieved_goal0: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    actions_t = actions
    ensemble_size = len(model.members)
    obs_members = jnp.repeat(obs0[None, ...], ensemble_size, axis=0)
    achieved_goal_members = jnp.repeat(achieved_goal0[None, ...], ensemble_size, axis=0)

    def step(carry, action_t):
        obs_members_t, achieved_goal_members_t = carry
        delta_obs, delta_ag = _predict_members(
            model, obs_members_t, achieved_goal_members_t, action_t
        )
        obs_members_t = obs_members_t + delta_obs
        achieved_goal_members_t = achieved_goal_members_t + delta_ag
        ensemble_disagreement = jnp.mean(
            jnp.var(achieved_goal_members_t, axis=0), axis=-1
        )
        return (
            obs_members_t,
            achieved_goal_members_t,
        ), ensemble_disagreement

    _, ensemble_disagreement_seq = jax.lax.scan(
        step, (obs_members, achieved_goal_members), actions_t
    )
    return ensemble_disagreement_seq


def evaluate_sequences(
    model: EnsembleModel,
    obs0: np.ndarray,
    achieved_goal0: np.ndarray,
    actions: np.ndarray,
    desired_goal: np.ndarray,
    action_penalty: float,
    reward_weight: float,
    epistemic_bonus_weight: float,
    reward_mode: str,
    distance_threshold: float,
    valid_horizon: int | None = None,
    objective: PlanningObjective | None = None,
) -> np.ndarray:
    obs0_j = jnp.asarray(obs0)
    achieved_goal0_j = jnp.asarray(achieved_goal0)
    actions_j = jnp.asarray(actions)
    pop = actions_j.shape[0]
    horizon = actions_j.shape[1]
    obs0_batch = jnp.repeat(obs0_j[None, :], pop, axis=0)
    achieved_goal0_batch = jnp.repeat(achieved_goal0_j[None, :], pop, axis=0)
    desired_goal_source = (
        objective.desired_goal
        if objective is not None and objective.desired_goal is not None
        else desired_goal
    )
    desired_goal_j = jnp.asarray(desired_goal_source)
    if desired_goal_j.ndim == 1:
        desired_goal_batch = jnp.repeat(desired_goal_j[None, :], pop, axis=0)
    else:
        desired_goal_batch = desired_goal_j
    if objective is None:
        reward_seq, ensemble_disagreement_seq = rollout_ensemble_metrics(
            model,
            obs0_batch,
            achieved_goal0_batch,
            actions_j,
            desired_goal_batch,
            reward_mode,
            distance_threshold,
        )
    else:
        gripper_indices = tuple(objective.gripper_indices)
        if not gripper_indices:
            raise ValueError("objective.gripper_indices must be non-empty.")
        target_source = objective.gripper_target
        if target_source is None:
            target_source = np.zeros((len(gripper_indices),), dtype=np.float32)
        gripper_target_j = jnp.asarray(target_source)
        if gripper_target_j.ndim == 1:
            gripper_target_batch = jnp.repeat(gripper_target_j[None, :], pop, axis=0)
        else:
            gripper_target_batch = gripper_target_j
        reward_seq, ensemble_disagreement_seq = rollout_ensemble_objective_metrics(
            model,
            obs0_batch,
            achieved_goal0_batch,
            actions_j,
            desired_goal_batch,
            reward_mode,
            distance_threshold,
            float(objective.object_goal_weight),
            gripper_target_batch,
            float(objective.gripper_target_weight),
            jnp.asarray(gripper_indices, dtype=jnp.int32),
        )

    if valid_horizon is None:
        valid_horizon = horizon
    valid_horizon = int(min(max(valid_horizon, 1), horizon))

    mask = (jnp.arange(horizon) < valid_horizon).astype(reward_seq.dtype)
    mask = mask[:, None]
    reward_obj = jnp.sum(reward_seq * mask, axis=0)
    ensemble_disagreement_obj = jnp.sum(ensemble_disagreement_seq * mask, axis=0)

    score = reward_weight * reward_obj
    if epistemic_bonus_weight != 0.0:
        score = score + epistemic_bonus_weight * ensemble_disagreement_obj

    cost = -score
    if action_penalty > 0.0:
        cost = cost + action_penalty * jnp.mean(actions_j**2, axis=(1, 2))

    return np.asarray(cost)


def _build_plan_rollout(
    model: EnsembleModel,
    obs0: np.ndarray,
    achieved_goal0: np.ndarray,
    actions: np.ndarray,
    desired_goal: np.ndarray,
    reward_mode: str,
    distance_threshold: float,
    objective: PlanningObjective | None = None,
) -> CEMPlanRollout:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to have shape [H, A], got {actions.shape}.")

    horizon = actions.shape[0]
    obs_members = jnp.repeat(jnp.asarray(obs0)[None, :], len(model.members), axis=0)
    ag_members = jnp.repeat(
        jnp.asarray(achieved_goal0)[None, :], len(model.members), axis=0
    )
    obs_seq = []
    ag_seq = []

    for t in range(horizon):
        action_t = jnp.asarray(actions[t])
        delta_obs, delta_ag = _predict_members(model, obs_members, ag_members, action_t)
        obs_members = obs_members + delta_obs
        ag_members = ag_members + delta_ag
        obs_seq.append(obs_members)
        ag_seq.append(ag_members)

    ag_seq_arr = jnp.stack(ag_seq, axis=1) if ag_seq else ag_members[:, None, :]
    obs_seq_arr = jnp.stack(obs_seq, axis=1) if obs_seq else obs_members[:, None, :]
    desired_goal_source = (
        objective.desired_goal
        if objective is not None and objective.desired_goal is not None
        else desired_goal
    )
    desired_goal_j = jnp.asarray(desired_goal_source)
    if desired_goal_j.ndim == 1:
        desired_goal_seq = jnp.repeat(desired_goal_j[None, :], horizon, axis=0)
    else:
        desired_goal_seq = desired_goal_j[:horizon]
    rewards = _goal_reward(
        ag_seq_arr, desired_goal_seq, reward_mode, distance_threshold
    )
    goal_dists = jnp.linalg.norm(ag_seq_arr - desired_goal_seq[None, :, :], axis=-1)
    disagreement = jnp.mean(jnp.var(ag_seq_arr, axis=0), axis=-1)
    gripper_target_dists = None
    if objective is not None and objective.gripper_target is not None:
        gripper_indices = tuple(objective.gripper_indices)
        if gripper_indices:
            gripper_seq = jnp.take(
                obs_seq_arr, jnp.asarray(gripper_indices, dtype=jnp.int32), axis=-1
            )
            gripper_target_j = jnp.asarray(objective.gripper_target)
            if gripper_target_j.ndim == 1:
                gripper_target_seq = jnp.repeat(gripper_target_j[None, :], horizon, axis=0)
            else:
                gripper_target_seq = gripper_target_j[:horizon]
            gripper_target_dists = np.asarray(
                jnp.linalg.norm(gripper_seq - gripper_target_seq[None, :, :], axis=-1),
                dtype=np.float32,
            )

    return CEMPlanRollout(
        reward_members=np.asarray(rewards, dtype=np.float32),
        goal_distance_members=np.asarray(goal_dists, dtype=np.float32),
        ensemble_disagreement=np.asarray(disagreement, dtype=np.float32),
        gripper_target_distance_members=gripper_target_dists,
        objective_name=objective.name if objective is not None else "goal_tracking",
    )


def cem_plan(
    model: EnsembleModel,
    obs0: np.ndarray,
    achieved_goal0: np.ndarray,
    desired_goal: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    cfg: CEMConfig,
    distance_threshold: float,
    horizon: int | None = None,
    valid_horizon: int | None = None,
    return_rollout: bool = False,
    objective: PlanningObjective | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CEMPlanRollout | None]:
    rng = np.random.default_rng(cfg.seed)
    act_dim = action_low.shape[0]
    plan_horizon = cfg.horizon if horizon is None else horizon
    mean = np.zeros((plan_horizon, act_dim), dtype=np.float32)
    std = np.ones((plan_horizon, act_dim), dtype=np.float32) * cfg.init_std
    elite_count = max(1, int(cfg.population * cfg.elite_frac))

    for _ in range(cfg.cem_iters):
        noise = rng.standard_normal(
            size=(cfg.population, plan_horizon, act_dim)
        ).astype(np.float32)
        actions = mean[None, ...] + std[None, ...] * noise
        actions = np.clip(actions, action_low, action_high)

        costs = evaluate_sequences(
            model,
            obs0,
            achieved_goal0,
            actions,
            desired_goal,
            cfg.action_penalty,
            cfg.reward_weight,
            cfg.epistemic_bonus_weight,
            cfg.reward_mode,
            distance_threshold,
            valid_horizon=valid_horizon,
            objective=objective,
        )
        elite_idx = np.argpartition(costs, elite_count - 1)[:elite_count]
        elite_actions = actions[elite_idx]
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0) + 1e-6

    best_idx = int(np.argmin(costs))
    best_actions = actions[best_idx]
    rollout = None
    if return_rollout:
        rollout = _build_plan_rollout(
            model,
            obs0,
            achieved_goal0,
            best_actions,
            desired_goal,
            cfg.reward_mode,
            distance_threshold,
            objective=objective,
        )
    return best_actions, mean, std, rollout


def cem_plan_spread(
    model: EnsembleModel,
    obs0: np.ndarray,
    achieved_goal0: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    act_dim = action_low.shape[0]
    horizon = action_mean.shape[0]
    noise = rng.standard_normal(size=(samples, horizon, act_dim)).astype(np.float32)
    actions = action_mean[None, ...] + action_std[None, ...] * noise
    actions = np.clip(actions, action_low, action_high)

    ag_seq = rollout_model_batch(
        model,
        jnp.repeat(jnp.asarray(obs0)[None, :], samples, axis=0),
        jnp.repeat(jnp.asarray(achieved_goal0)[None, :], samples, axis=0),
        jnp.asarray(actions),
    )
    ag_seq = jnp.swapaxes(ag_seq, 0, 1)
    std_across_samples = jnp.std(ag_seq, axis=0)
    return float(jnp.mean(std_across_samples))
