from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .utils import Normalizer


class DynamicsMember(eqx.Module):
    trunk: eqx.nn.MLP
    delta_obs_head: eqx.nn.Linear
    delta_ag_head: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        depth: int,
        obs_dim: int,
        achieved_goal_dim: int,
        key: jax.Array,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.MLP(
            in_size=in_dim,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.relu,
            key=k1,
        )
        self.delta_obs_head = eqx.nn.Linear(hidden_size, obs_dim, key=k2)
        self.delta_ag_head = eqx.nn.Linear(hidden_size, achieved_goal_dim, key=k3)

    def __call__(
        self,
        obs: jnp.ndarray,
        achieved_goal: jnp.ndarray,
        action: jnp.ndarray,
        normalizer: Normalizer,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def _forward_single(
            obs_single: jnp.ndarray,
            achieved_goal_single: jnp.ndarray,
            action_single: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            obs_norm = normalizer.normalize_obs(obs_single)
            action_norm = normalizer.normalize_action(action_single)
            achieved_goal_norm = normalizer.normalize_achieved_goal(
                achieved_goal_single
            )
            x = jnp.concatenate([obs_norm, achieved_goal_norm, action_norm], axis=-1)
            h = self.trunk(x)
            delta_obs = self.delta_obs_head(h)
            delta_ag = self.delta_ag_head(h)
            return delta_obs, delta_ag

        if obs.ndim == 1:
            return _forward_single(obs, achieved_goal, action)
        if obs.ndim == 2:
            return jax.vmap(_forward_single)(obs, achieved_goal, action)
        raise ValueError(f"Expected obs to have ndim 1 or 2, got shape {obs.shape}.")

    def predict_deltas(
        self,
        obs: jnp.ndarray,
        achieved_goal: jnp.ndarray,
        action: jnp.ndarray,
        normalizer: Normalizer,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_obs_norm, delta_ag_norm = self(obs, achieved_goal, action, normalizer)
        delta_obs = normalizer.denormalize_delta_obs(delta_obs_norm)
        delta_ag = normalizer.denormalize_delta_ag(delta_ag_norm)
        return delta_obs, delta_ag


class EnsembleModel(eqx.Module):
    members: tuple[DynamicsMember, ...]
    normalizer: Normalizer

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        achieved_goal_dim: int,
        ensemble_size: int,
        hidden_size: int,
        depth: int,
        normalizer: Normalizer,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, ensemble_size)
        in_dim = obs_dim + achieved_goal_dim + action_dim
        self.members = tuple(
            DynamicsMember(in_dim, hidden_size, depth, obs_dim, achieved_goal_dim, k)
            for k in keys
        )
        self.normalizer = normalizer

    def forward_normalized(
        self,
        obs: jnp.ndarray,
        achieved_goal: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_obs = []
        delta_ag = []
        if obs.ndim == 3:
            for idx, member in enumerate(self.members):
                delta_obs_member, delta_ag_member = member(
                    obs[idx],
                    achieved_goal[idx],
                    action[idx],
                    self.normalizer,
                )
                delta_obs.append(delta_obs_member)
                delta_ag.append(delta_ag_member)
        else:
            for member in self.members:
                delta_obs_member, delta_ag_member = member(
                    obs, achieved_goal, action, self.normalizer
                )
                delta_obs.append(delta_obs_member)
                delta_ag.append(delta_ag_member)
        return jnp.stack(delta_obs, axis=0), jnp.stack(delta_ag, axis=0)

    def predict_deltas(
        self, obs: jnp.ndarray, achieved_goal: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_obs_norm, delta_ag_norm = self.forward_normalized(
            obs, achieved_goal, action
        )
        delta_obs = self.normalizer.denormalize_delta_obs(delta_obs_norm)
        delta_ag = self.normalizer.denormalize_delta_ag(delta_ag_norm)
        return delta_obs, delta_ag


@dataclass(frozen=True)
class ModelConfig:
    ensemble_size: int
    hidden_size: int
    depth: int


def make_model(
    obs_dim: int,
    action_dim: int,
    achieved_goal_dim: int,
    config: ModelConfig,
    normalizer: Normalizer,
    key: jax.Array,
) -> EnsembleModel:
    return EnsembleModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        achieved_goal_dim=achieved_goal_dim,
        ensemble_size=config.ensemble_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        normalizer=normalizer,
        key=key,
    )
