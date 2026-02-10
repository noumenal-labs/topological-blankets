"""Teleop push demo: pre-scripted human-in-the-loop goal injection for FetchPush.

Runs FetchPush-v4 with a TeleopInterface that wraps the symbolic planner.
A pre-scripted sequence of goal injections simulates a human operator who
overrides the planner at specific timesteps, then releases control.

The script operates in two modes:

  1. *Full mode* (when a trained ensemble checkpoint is available): loads the
     model, runs CEM planning, and collects real episode trajectories.
  2. *Dry-run mode* (no checkpoint): exercises the TeleopInterface decision
     logic using the *environment's own dynamics* with random actions so that
     all interface bookkeeping and logging paths are validated without needing
     a trained model.

Output is saved to ``results/`` as JSON.  No ``plt.show()`` calls.

Usage::

    python experiments/teleop_push_demo.py                  # dry-run
    python experiments/teleop_push_demo.py --run-dir data/push_demo  # full mode
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("MPLBACKEND", "Agg")

_headless = sys.platform.startswith("linux") and os.environ.get("DISPLAY", "") == ""
if _headless:
    os.environ.setdefault("MUJOCO_GL", "egl")

# Ensure the pandas package is importable.
_pandas_root = str(Path(__file__).resolve().parents[4] / "pandas")
if _pandas_root not in sys.path:
    sys.path.insert(0, _pandas_root)

# Ensure the lunar-lander ralph directory is importable for utils.
_ralph_root = str(Path(__file__).resolve().parents[1])
if _ralph_root not in sys.path:
    sys.path.insert(0, _ralph_root)

import numpy as np

# Register Gymnasium Robotics environments before any gym.make() call.
import gymnasium  # noqa: E402
import gymnasium_robotics  # noqa: E402

gymnasium_robotics.register_robotics_envs()

from panda.common import (  # noqa: E402
    action_bounds,
    distance_threshold,
    make_env,
    reset_env,
)
from panda.symbolic_planner import (  # noqa: E402
    SymbolicPlannerConfig,
    make_symbolic_planner,
)
from panda.teleop_interface import TeleopInterface, TeleopMode  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TeleopDemoConfig:
    """Configuration for the pre-scripted teleop demo."""

    env_id: str = "FetchPush-v4"
    reward_mode: str = "dense"
    max_episode_steps: int = 50
    seed: int = 42
    episodes: int = 3

    # Symbolic planner defaults
    symbolic_task: str = "push"
    gripper_indices: tuple[int, ...] = (0, 1, 2)

    # Optional trained model for full mode
    run_dir: str | None = None

    # Results output
    results_dir: str = str(
        Path(__file__).resolve().parents[1] / "results"
    )


# ---------------------------------------------------------------------------
# Pre-scripted intervention schedule
# ---------------------------------------------------------------------------

@dataclass
class ScriptedInjection:
    """A single pre-scripted goal injection."""
    inject_step: int
    target_xyz: tuple[float, float, float]
    release_step: int


# Three episodes with distinct intervention patterns.
SCRIPTED_SCHEDULE: list[list[ScriptedInjection]] = [
    # Episode 0: single early intervention; guide gripper to a position
    # slightly above the table centre, then release.
    [
        ScriptedInjection(inject_step=5, target_xyz=(1.30, 0.75, 0.45), release_step=15),
    ],
    # Episode 1: two interventions; first corrects approach, second nudges.
    [
        ScriptedInjection(inject_step=3, target_xyz=(1.25, 0.80, 0.43), release_step=10),
        ScriptedInjection(inject_step=25, target_xyz=(1.35, 0.70, 0.42), release_step=35),
    ],
    # Episode 2: no interventions (pure agent control), for comparison.
    [],
]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: gymnasium.Env,
    teleop: TeleopInterface,
    schedule: list[ScriptedInjection],
    max_steps: int,
    seed: int | None,
    obs_key: str = "observation",
    ag_key: str = "achieved_goal",
    dg_key: str = "desired_goal",
    use_model: bool = False,
    model=None,
    cem_cfg=None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    goal_thr: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run a single FetchPush episode with the teleop interface.

    Returns a dict of per-episode metrics and logs.
    """
    obs = reset_env(env, seed)
    # Snapshot intervention count *before* reset so that the per-episode
    # delta can be computed at the end.
    interventions_before = teleop.interventions
    human_steps_before = teleop._human_steps
    total_steps_before = teleop._total_steps
    teleop.reset()

    step_logs: list[dict] = []
    total_return = 0.0
    success = False

    # Build a step->injection lookup for quick scheduling.
    inject_map: dict[int, ScriptedInjection] = {}
    release_set: set[int] = set()
    for inj in schedule:
        inject_map[inj.inject_step] = inj
        release_set.add(inj.release_step)

    for step in range(max_steps):
        obs_vec = np.asarray(obs[obs_key], dtype=np.float32)
        achieved_goal = np.asarray(obs[ag_key], dtype=np.float32)
        desired_goal = np.asarray(obs[dg_key], dtype=np.float32)

        # --- scripted injection / release ---
        if step in inject_map:
            inj = inject_map[step]
            teleop.inject_goal(np.array(inj.target_xyz, dtype=np.float32))
        if step in release_set and teleop.mode == TeleopMode.HUMAN:
            teleop.release()

        # --- decide ---
        decision = teleop.decide(obs_vec, achieved_goal, desired_goal)
        status = teleop.get_status()

        # --- select action ---
        if use_model and model is not None and cem_cfg is not None:
            # Full CEM planning (requires a trained model)
            from dataclasses import replace as _replace
            from panda.planner import cem_plan

            step_seed = int(rng.integers(0, 2**32 - 1)) if rng is not None else None
            step_cfg = _replace(cem_cfg, seed=step_seed)
            remaining = max_steps - step
            valid_horizon = min(cem_cfg.horizon, remaining)
            plan_actions, _, _, _ = cem_plan(
                model,
                obs_vec,
                achieved_goal,
                desired_goal,
                action_low,
                action_high,
                step_cfg,
                goal_thr,
                valid_horizon=valid_horizon,
                objective=decision.objective,
            )
            action = plan_actions[0]
        else:
            # Dry-run: use a small random action so the episode is not
            # completely static, but keep it bounded.
            if action_low is not None and action_high is not None:
                action = (rng or np.random.default_rng()).uniform(
                    action_low * 0.3, action_high * 0.3
                ).astype(np.float32)
            else:
                action = np.zeros(4, dtype=np.float32)

        # --- step environment ---
        next_obs, reward_raw, terminated, truncated, info = env.step(action)
        reward = float(reward_raw)
        total_return += reward

        info_success = bool(info.get("is_success", False)) if isinstance(info, dict) else False
        success = success or info_success

        goal_dist = float(np.linalg.norm(achieved_goal - desired_goal))

        step_logs.append({
            "step": step,
            "mode": status.mode.value,
            "active_goal": status.active_goal.tolist() if status.active_goal is not None else None,
            "progress": round(status.progress, 5),
            "goal_distance": round(goal_dist, 5),
            "phase_name": decision.status.phase_name,
            "objective_name": decision.objective.name,
            "reward": round(reward, 5),
        })

        obs = next_obs
        if terminated or truncated or info_success:
            break

    final_status = teleop.get_status()
    ep_interventions = final_status.interventions - interventions_before
    ep_human_steps = final_status.human_steps - human_steps_before
    ep_total_steps = final_status.total_steps - total_steps_before
    ep_autonomy = (
        100.0 * (1.0 - ep_human_steps / ep_total_steps)
        if ep_total_steps > 0
        else 100.0
    )
    return {
        "steps": len(step_logs),
        "total_return": round(total_return, 5),
        "success": success,
        "interventions": ep_interventions,
        "human_steps": ep_human_steps,
        "total_steps": ep_total_steps,
        "autonomy_pct": round(ep_autonomy, 2),
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Teleop push demo (pre-scripted)")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to trained model run directory")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    args = parser.parse_args()

    cfg = TeleopDemoConfig(
        run_dir=args.run_dir,
        episodes=args.episodes,
        seed=args.seed,
        max_episode_steps=args.max_steps,
    )
    if args.results_dir:
        cfg.results_dir = args.results_dir

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Try to load model (full mode) or fall back to dry-run ----
    model = None
    cem_cfg = None
    use_model = False

    if cfg.run_dir is not None:
        run_dir = Path(cfg.run_dir)
        model_path = run_dir / "model.eqx"
        if model_path.exists():
            try:
                import equinox as eqx
                import jax
                from panda.common import load_json
                from panda.model import ModelConfig, make_model
                from panda.planner import CEMConfig
                from panda.utils import Normalizer
                import jax.numpy as jnp

                model_meta = load_json(model_path.with_suffix(model_path.suffix + ".json"))
                run_cfg_data = load_json(run_dir / "run_config.json")

                def _get(key, *dicts, default=None):
                    for d in dicts:
                        if key in d:
                            return d[key]
                    return default

                obs_dim = int(_get("obs_dim", model_meta, default=25))
                action_dim = int(_get("action_dim", model_meta, default=4))
                ag_dim = int(_get("achieved_goal_dim", model_meta, default=3))

                normalizer = Normalizer.from_stats(
                    obs_mean=jnp.zeros(obs_dim), obs_std=jnp.ones(obs_dim),
                    achieved_goal_mean=jnp.zeros(ag_dim), achieved_goal_std=jnp.ones(ag_dim),
                    action_mean=jnp.zeros(action_dim), action_std=jnp.ones(action_dim),
                    delta_obs_mean=jnp.zeros(obs_dim), delta_obs_std=jnp.ones(obs_dim),
                    delta_ag_mean=jnp.zeros(ag_dim), delta_ag_std=jnp.ones(ag_dim),
                )
                mcfg = ModelConfig(
                    ensemble_size=int(_get("ensemble_size", model_meta, run_cfg_data, default=5)),
                    hidden_size=int(_get("hidden_size", model_meta, run_cfg_data, default=128)),
                    depth=int(_get("depth", model_meta, run_cfg_data, default=2)),
                )
                key = jax.random.PRNGKey(0)
                model = make_model(obs_dim, action_dim, ag_dim, mcfg, normalizer, key)
                model = eqx.tree_deserialise_leaves(str(model_path), model)

                cem_cfg = CEMConfig(
                    horizon=int(_get("horizon", run_cfg_data, default=30)),
                    population=int(_get("population", run_cfg_data, default=512)),
                    elite_frac=float(_get("elite_frac", run_cfg_data, default=0.1)),
                    cem_iters=int(_get("cem_iters", run_cfg_data, default=8)),
                    init_std=float(_get("init_std", run_cfg_data, default=0.6)),
                    action_penalty=float(_get("action_penalty", run_cfg_data, default=1e-3)),
                    reward_weight=float(_get("reward_weight", run_cfg_data, default=1.0)),
                    reward_mode=cfg.reward_mode,
                    epistemic_bonus_weight=0.0,
                    seed=cfg.seed,
                )
                use_model = True
                print(f"[teleop_demo] Loaded model from {model_path}")
            except Exception as exc:
                print(f"[teleop_demo] Could not load model: {exc}")
                print("[teleop_demo] Falling back to dry-run mode.")
        else:
            print(f"[teleop_demo] Model not found at {model_path}; using dry-run mode.")
    else:
        print("[teleop_demo] No --run-dir provided; using dry-run mode.")

    # ---- Create environment ----
    env = make_env(
        env_id=cfg.env_id,
        max_episode_steps=cfg.max_episode_steps,
        reward_mode=cfg.reward_mode,
        render_mode=None,  # headless, no rendering needed
    )
    action_low, action_high = action_bounds(env)
    goal_thr = distance_threshold(env)

    # ---- Create symbolic planner + teleop interface ----
    symbolic_cfg = SymbolicPlannerConfig(
        task=cfg.symbolic_task,
        gripper_indices=cfg.gripper_indices,
    )
    symbolic_planner = make_symbolic_planner(
        symbolic_cfg,
        env_id=cfg.env_id,
        default_goal_threshold=goal_thr,
    )
    teleop = TeleopInterface(
        symbolic_planner=symbolic_planner,
        gripper_indices=cfg.gripper_indices,
        goal_reached_threshold=0.04,
    )

    rng = np.random.default_rng(cfg.seed)

    print(f"[teleop_demo] env={cfg.env_id}  episodes={cfg.episodes}  "
          f"mode={'full' if use_model else 'dry-run'}")
    print(f"[teleop_demo] scripted schedule has {len(SCRIPTED_SCHEDULE)} episode plans")

    # ---- Run episodes ----
    all_episodes: list[dict] = []
    t0 = time.time()

    for ep_idx in range(cfg.episodes):
        schedule = SCRIPTED_SCHEDULE[ep_idx % len(SCRIPTED_SCHEDULE)]
        ep_seed = int(rng.integers(0, 2**32 - 1))

        ep_result = run_episode(
            env=env,
            teleop=teleop,
            schedule=schedule,
            max_steps=cfg.max_episode_steps,
            seed=ep_seed,
            use_model=use_model,
            model=model,
            cem_cfg=cem_cfg,
            action_low=action_low,
            action_high=action_high,
            goal_thr=goal_thr,
            rng=rng,
        )
        ep_result["episode"] = ep_idx
        ep_result["schedule"] = [
            {
                "inject_step": s.inject_step,
                "target_xyz": list(s.target_xyz),
                "release_step": s.release_step,
            }
            for s in schedule
        ]
        all_episodes.append(ep_result)

        print(
            f"  episode {ep_idx + 1}/{cfg.episodes} | "
            f"steps={ep_result['steps']} | "
            f"return={ep_result['total_return']:.3f} | "
            f"success={ep_result['success']} | "
            f"interventions={ep_result['interventions']} | "
            f"autonomy={ep_result['autonomy_pct']:.1f}%"
        )

    elapsed = time.time() - t0
    env.close()

    # ---- Aggregate metrics ----
    total_steps_all = sum(e["total_steps"] for e in all_episodes)
    human_steps_all = sum(e["human_steps"] for e in all_episodes)
    interventions_all = sum(e["interventions"] for e in all_episodes)
    autonomy_all = (
        100.0 * (1.0 - human_steps_all / total_steps_all) if total_steps_all > 0 else 100.0
    )
    success_count = sum(1 for e in all_episodes if e["success"])

    summary = {
        "experiment": "teleop_push_demo",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "full" if use_model else "dry-run",
        "config": {
            "env_id": cfg.env_id,
            "episodes": cfg.episodes,
            "max_episode_steps": cfg.max_episode_steps,
            "seed": cfg.seed,
            "run_dir": cfg.run_dir,
        },
        "aggregate": {
            "total_episodes": cfg.episodes,
            "total_steps": total_steps_all,
            "human_steps": human_steps_all,
            "interventions": interventions_all,
            "autonomy_pct": round(autonomy_all, 2),
            "success_count": success_count,
            "success_rate": round(success_count / max(cfg.episodes, 1) * 100.0, 2),
            "elapsed_seconds": round(elapsed, 2),
        },
        "episodes": [
            {
                "episode": e["episode"],
                "steps": e["steps"],
                "total_return": e["total_return"],
                "success": e["success"],
                "interventions": e["interventions"],
                "human_steps": e["human_steps"],
                "autonomy_pct": e["autonomy_pct"],
                "schedule": e["schedule"],
            }
            for e in all_episodes
        ],
    }

    # ---- Save results ----
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = results_dir / f"{stamp}_teleop_push_demo.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[teleop_demo] Results saved to {out_path}")

    # ---- Print summary ----
    print("\n=== Teleop Push Demo Summary ===")
    print(f"  Mode:            {'full (with model)' if use_model else 'dry-run (no model)'}")
    print(f"  Episodes:        {cfg.episodes}")
    print(f"  Total steps:     {total_steps_all}")
    print(f"  Human steps:     {human_steps_all}")
    print(f"  Interventions:   {interventions_all}")
    print(f"  Autonomy:        {autonomy_all:.1f}%")
    print(f"  Successes:       {success_count}/{cfg.episodes}")
    print(f"  Elapsed:         {elapsed:.1f}s")


if __name__ == "__main__":
    main()
