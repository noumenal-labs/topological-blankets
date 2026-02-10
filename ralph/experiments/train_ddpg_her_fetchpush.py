"""
US-105: Train DDPG+HER baseline on FetchPush-v4 for comparison.

Architecture-matched comparison: trains a standard model-free DDPG agent
with Hindsight Experience Replay on the same FetchPush-v4 task (50-step
episodes) that the Bayes ensemble uses. This provides the non-active-inference
baseline for benchmarking.

The trained policy and replay buffer can also be analyzed post-hoc with TB
to answer: does TB discover meaningful structure in model-free representations?

Usage:
    python train_ddpg_her_fetchpush.py [--total-timesteps 100000] [--out-dir ./data/ddpg_her_fetchpush]
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class MetricsCallback(BaseCallback):
    """Log training metrics to a JSON-lines file."""

    def __init__(self, log_path: str, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_successes = []
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "is_success" in info:
                self.episode_successes.append(float(info["is_success"]))
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % 5000 == 0 and len(self.episode_successes) > 0:
            elapsed = time.time() - self.start_time
            recent_success = np.mean(self.episode_successes[-100:])
            recent_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            entry = {
                "timestep": self.num_timesteps,
                "elapsed_s": round(elapsed, 1),
                "success_rate_100ep": round(recent_success, 4),
                "mean_reward_100ep": round(recent_reward, 4),
                "total_episodes": len(self.episode_successes),
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            if self.verbose:
                print(f"  step {self.num_timesteps}: success={recent_success:.3f}, "
                      f"reward={recent_reward:.3f}, episodes={len(self.episode_successes)}")
        return True


def make_env(max_episode_steps=50):
    env = gym.make("FetchPush-v4", max_episode_steps=max_episode_steps)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train DDPG+HER on FetchPush-v4")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100k)")
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="./data/ddpg_her_fetchpush")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training DDPG+HER on FetchPush-v4")
    print(f"  timesteps: {args.total_timesteps}")
    print(f"  max_episode_steps: {args.max_episode_steps}")
    print(f"  output: {out_dir}")
    print(f"  seed: {args.seed}")

    # Create environments
    env = make_env(args.max_episode_steps)
    eval_env = make_env(args.max_episode_steps)

    # Action noise (standard for DDPG on Fetch)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Architecture-matched to Bayes ensemble: 256 hidden, 2 layers
    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        seed=args.seed,
        verbose=0,
    )

    metrics_log = str(out_dir / "training_metrics.jsonl")
    metrics_cb = MetricsCallback(metrics_log, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    print("\nStarting training...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[metrics_cb, eval_cb],
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Save final model
    model.save(str(out_dir / "final_model"))

    # Final evaluation
    print(f"\nFinal evaluation ({args.n_eval_episodes} episodes)...")
    successes = []
    rewards = []
    for ep in range(args.n_eval_episodes):
        obs, info = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        successes.append(float(info.get("is_success", False)))
        rewards.append(total_reward)

    success_rate = np.mean(successes)
    mean_reward = np.mean(rewards)
    print(f"  Success rate: {success_rate:.3f}")
    print(f"  Mean reward: {mean_reward:.3f}")

    # Save summary
    summary = {
        "algorithm": "DDPG+HER",
        "env_id": "FetchPush-v4",
        "max_episode_steps": args.max_episode_steps,
        "total_timesteps": args.total_timesteps,
        "training_time_s": round(elapsed, 1),
        "seed": args.seed,
        "policy_kwargs": {"net_arch": [256, 256]},
        "final_eval": {
            "n_episodes": args.n_eval_episodes,
            "success_rate": round(success_rate, 4),
            "mean_reward": round(mean_reward, 4),
        },
        "hyperparameters": {
            "learning_rate": 1e-3,
            "batch_size": 256,
            "gamma": 0.95,
            "tau": 0.05,
            "action_noise_sigma": 0.1,
            "her_n_sampled_goal": 4,
            "her_goal_strategy": "future",
        },
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_dir / 'training_summary.json'}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
