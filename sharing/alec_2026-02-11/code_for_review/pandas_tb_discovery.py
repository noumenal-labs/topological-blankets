"""Topological Blanket discovery from ensemble world models.

Collects transition data from a FetchPush environment, computes
Jacobian sensitivity profiles from an ensemble model, and runs
the TopologicalBlankets pipeline to discover variable partitions.

Falls back gracefully to ground-truth partitions when the TB
package or a trained ensemble model is unavailable.

Usage::

    from panda.tb_discovery import discover_or_fallback
    partition = discover_or_fallback(model, env)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .learned_planner import (
    TBPartition,
    make_fetchpush_ground_truth_partition,
    partition_from_tb_result,
)


def collect_random_frames(
    env,
    n_frames: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect observation/action frames by running random actions.

    Parameters
    ----------
    env:
        A Gymnasium FetchPush-style environment (dict obs space).
    n_frames:
        Number of transition frames to collect.
    seed:
        Random seed for action sampling.

    Returns
    -------
    (observations, achieved_goals, actions)
        Each is an ndarray of shape (n_frames, dim).
    """
    rng = np.random.default_rng(seed)
    action_space = env.action_space

    obs_list = []
    ag_list = []
    act_list = []

    obs, _ = env.reset(seed=seed)
    collected = 0

    while collected < n_frames:
        action = rng.uniform(
            action_space.low, action_space.high
        ).astype(np.float32)

        obs_vec = np.asarray(obs["observation"], dtype=np.float32)
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)

        obs_list.append(obs_vec)
        ag_list.append(achieved)
        act_list.append(action)
        collected += 1

        next_obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(ag_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
    )


def compute_gradients_from_ensemble(
    model,
    observations: np.ndarray,
    achieved_goals: np.ndarray,
    actions: np.ndarray,
    max_samples: int = 500,
) -> np.ndarray:
    """Compute Jacobian sensitivity profiles from an ensemble model.

    For each ensemble member and each sample, computes the Jacobian
    d(delta_pred)/d(obs) and extracts per-variable sensitivity as
    column norms. The result is a (E * N, obs_dim) array of gradient
    samples suitable for TopologicalBlankets.fit().

    Parameters
    ----------
    model:
        An EnsembleModel with .members and .normalizer attributes.
    observations:
        Observation vectors, shape (N, obs_dim).
    achieved_goals:
        Achieved goal vectors, shape (N, goal_dim).
    actions:
        Action vectors, shape (N, act_dim).
    max_samples:
        Cap on the number of samples to use (for speed).

    Returns
    -------
    gradients:
        Shape (E * min(N, max_samples), obs_dim).
    """
    import jax
    import jax.numpy as jnp

    n_samples = min(len(observations), max_samples)
    obs_batch = jnp.asarray(observations[:n_samples])
    ag_batch = jnp.asarray(achieved_goals[:n_samples])
    act_batch = jnp.asarray(actions[:n_samples])

    all_sensitivities = []

    for member in model.members:
        def member_forward(obs_single, ag_single, act_single):
            delta_obs, delta_ag = member(
                obs_single, ag_single, act_single, model.normalizer
            )
            return jnp.concatenate([delta_obs, delta_ag])

        jac_fn = jax.jacobian(member_forward, argnums=0)

        sensitivities = []
        chunk_size = 64
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            for i in range(start, end):
                J = jac_fn(obs_batch[i], ag_batch[i], act_batch[i])
                # Column norms: sensitivity of output to each input variable
                sens = jnp.sqrt(jnp.sum(J ** 2, axis=0))
                sensitivities.append(np.asarray(sens))

        all_sensitivities.append(np.stack(sensitivities, axis=0))

    # Stack all members: (E, N, obs_dim) -> (E*N, obs_dim)
    stacked = np.concatenate(all_sensitivities, axis=0)
    return stacked


def run_tb_discovery(
    gradients: np.ndarray,
    n_objects: int = 2,
    method: str = "hybrid",
) -> TBPartition:
    """Run the TopologicalBlankets pipeline on gradient samples.

    Parameters
    ----------
    gradients:
        Shape (n_samples, obs_dim) sensitivity profiles.
    n_objects:
        Expected number of objects (e.g. 2 for gripper + manipulated).
    method:
        TB detection method (default: 'hybrid').

    Returns
    -------
    TBPartition
        Discovered variable partition.
    """
    from topological_blankets import TopologicalBlankets

    tb = TopologicalBlankets(method=method, n_objects=n_objects)
    tb.fit(gradients)

    assignment = tb.get_assignment()
    coupling = tb.get_coupling_matrix()

    return partition_from_tb_result(
        labels=assignment,
        coupling_matrix=coupling,
    )


def discover_or_fallback(
    model=None,
    env=None,
    n_frames: int = 10_000,
    n_objects: int = 2,
    seed: int = 42,
    obs_dim: int = 25,
) -> TBPartition:
    """End-to-end TB discovery with graceful fallback.

    Attempts the full pipeline: collect frames, compute Jacobian
    sensitivities, run TB discovery. Falls back to ground-truth
    partition if any component is unavailable (no trained model,
    no TB package, etc.).

    Parameters
    ----------
    model:
        Optional EnsembleModel. If None, falls back immediately.
    env:
        Optional Gymnasium environment. If None, falls back immediately.
    n_frames:
        Number of transition frames to collect.
    n_objects:
        Expected number of objects.
    seed:
        Random seed.
    obs_dim:
        Observation dimensionality (used for ground-truth fallback).

    Returns
    -------
    TBPartition
    """
    if model is None or env is None:
        print("[tb_discovery] No model or env provided; using ground-truth partition.")
        return make_fetchpush_ground_truth_partition(obs_dim)

    try:
        from topological_blankets import TopologicalBlankets
    except ImportError:
        print("[tb_discovery] topological_blankets package not available; "
              "using ground-truth partition.")
        return make_fetchpush_ground_truth_partition(obs_dim)

    try:
        print(f"[tb_discovery] Collecting {n_frames} random frames...")
        obs, ag, acts = collect_random_frames(env, n_frames=n_frames, seed=seed)

        print(f"[tb_discovery] Computing Jacobian sensitivities...")
        gradients = compute_gradients_from_ensemble(model, obs, ag, acts)

        print(f"[tb_discovery] Running TB discovery on {gradients.shape[0]} "
              f"gradient samples ({gradients.shape[1]}D)...")
        partition = run_tb_discovery(gradients, n_objects=n_objects)

        n_discovered = len(partition.objects)
        n_blanket = len(partition.blanket)
        print(f"[tb_discovery] Discovered {n_discovered} objects, "
              f"{n_blanket} blanket variables.")
        return partition

    except Exception as e:
        print(f"[tb_discovery] Discovery failed ({e}); "
              "using ground-truth partition.")
        return make_fetchpush_ground_truth_partition(obs_dim)
