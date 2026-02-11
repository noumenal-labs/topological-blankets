"""FetchPush benchmark suite: compare agent configurations.

Runs multiple training configurations on FetchPush-v4 and collects
eval results for comparison plotting. Each config varies the planner
type and whether epistemic bonus (info gain) is used.

Configurations:
  1. hardcoded_no_ig    : symbolic push planner, no info gain  [DONE]
  2. hardcoded_ig       : symbolic push planner, with info gain
  3. tb_guided_no_ig    : TB-guided planner (ground-truth), no info gain  [RUNNING]
  4. tb_guided_ig       : TB-guided planner (ground-truth), with info gain
  5. no_planner         : pure CEM, no symbolic planner, no info gain
  6. no_planner_ig      : pure CEM, no symbolic planner, with info gain
  7. ddpg_her           : DDPG+HER baseline (Alec's data, already available)

Usage:
  python benchmark_fetchpush.py --configs hardcoded_ig no_planner
  python benchmark_fetchpush.py --configs all
  python benchmark_fetchpush.py --list
  python benchmark_fetchpush.py --results
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PANDAS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "pandas"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "benchmark_fetchpush"


@dataclass
class BenchmarkConfig:
    name: str
    symbolic_task: str
    use_epistemic_bonus: bool
    epistemic_bonus_weight: float = 0.2
    description: str = ""
    status: str = "pending"
    run_dir: str = ""

    def __post_init__(self):
        if not self.run_dir:
            self.run_dir = f"data/benchmark_{self.name}"


CONFIGS: dict[str, BenchmarkConfig] = {
    "hardcoded_no_ig": BenchmarkConfig(
        name="hardcoded_no_ig",
        symbolic_task="push",
        use_epistemic_bonus=False,
        description="Hardcoded symbolic push planner, no info gain",
        status="done",
        run_dir="data/fetchpush_50step",
    ),
    "hardcoded_ig": BenchmarkConfig(
        name="hardcoded_ig",
        symbolic_task="push",
        use_epistemic_bonus=True,
        description="Hardcoded symbolic push planner, with info gain (epistemic bonus 0.2)",
    ),
    "tb_guided_no_ig": BenchmarkConfig(
        name="tb_guided_no_ig",
        symbolic_task="tb-discover",
        use_epistemic_bonus=False,
        description="TB-guided planner (ground-truth partition), no info gain",
        status="running",
        run_dir="data/push_tb_discover",
    ),
    "tb_guided_ig": BenchmarkConfig(
        name="tb_guided_ig",
        symbolic_task="tb-discover",
        use_epistemic_bonus=True,
        description="TB-guided planner (ground-truth partition), with info gain",
    ),
    "no_planner": BenchmarkConfig(
        name="no_planner",
        symbolic_task="none",
        use_epistemic_bonus=False,
        description="Pure CEM, no symbolic planner, no info gain",
    ),
    "no_planner_ig": BenchmarkConfig(
        name="no_planner_ig",
        symbolic_task="none",
        use_epistemic_bonus=True,
        description="Pure CEM, no symbolic planner, with info gain",
    ),
}


COMMON_ARGS = [
    "--env-id", "FetchPush-v4",
    "--reward-mode", "dense",
    "--ensemble-size", "5",
    "--iterations", "100",
    "--plan-steps", "500",
    "--initial-random-steps", "5000",
    "--max-episode-steps", "50",
    "--eval-every", "5",
    "--eval-episodes", "1",
    "--no-use-wandb",
    "--seed", "0",
]


def build_train_command(cfg: BenchmarkConfig) -> list[str]:
    """Build the train.py command for a benchmark config."""
    cmd = [
        sys.executable, "-u",
        str(PANDAS_DIR / "train.py"),
        "--run-dir", cfg.run_dir,
        "--symbolic-task", cfg.symbolic_task,
    ]
    cmd.extend(COMMON_ARGS)

    if cfg.use_epistemic_bonus:
        cmd.extend([
            "--train-use-epistemic-bonus",
            "--eval-use-epistemic-bonus",
            "--train-epistemic-bonus-weight", str(cfg.epistemic_bonus_weight),
            "--eval-epistemic-bonus-weight", str(cfg.epistemic_bonus_weight),
        ])
    else:
        cmd.extend([
            "--no-train-use-epistemic-bonus",
            "--no-eval-use-epistemic-bonus",
        ])

    return cmd


def parse_eval_results(log_path: Path) -> dict:
    """Parse eval results from a training log or stdout capture."""
    if not log_path.exists():
        return {"env_steps": [], "success_rate": [], "avg_steps": [], "avg_return": []}

    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    current_iter = 0
    results = {"env_steps": [], "success_rate": [], "avg_steps": [], "avg_return": []}

    iter_re = re.compile(r"iteration (\d+)/(\d+)")
    eval_re = re.compile(
        r"eval \| avg return ([-\d.]+) \| success ([\d.]+)% \((\d+)/(\d+)\) "
        r"\| avg steps ([\d.]+)"
    )

    for line in lines:
        m_iter = iter_re.search(line)
        if m_iter:
            current_iter = int(m_iter.group(1))

        m_eval = eval_re.search(line)
        if m_eval:
            avg_return = float(m_eval.group(1))
            success_pct = float(m_eval.group(2))
            avg_steps = float(m_eval.group(5))
            env_steps = 5000 + current_iter * 500

            results["env_steps"].append(env_steps)
            results["success_rate"].append(success_pct / 100.0)
            results["avg_steps"].append(avg_steps)
            results["avg_return"].append(avg_return)

    return results


def find_log_file(cfg: BenchmarkConfig) -> Path | None:
    """Find the log file for a given config."""
    run_dir = PANDAS_DIR / cfg.run_dir
    # Check for training.log first
    log = run_dir / "training.log"
    if log.exists():
        return log
    # Check for stdout capture
    log = run_dir / "stdout.log"
    if log.exists():
        return log
    return None


def run_config(cfg: BenchmarkConfig, background: bool = True) -> subprocess.Popen | None:
    """Launch a training run for the given config."""
    if cfg.status == "done":
        print(f"  [{cfg.name}] Already complete, skipping.")
        return None

    run_dir = PANDAS_DIR / cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_train_command(cfg)
    log_file = run_dir / "stdout.log"

    print(f"  [{cfg.name}] Launching: {cfg.description}")
    print(f"    Run dir: {run_dir}")
    print(f"    Log: {log_file}")
    print(f"    Command: {' '.join(cmd[:6])}...")

    if background:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=str(PANDAS_DIR),
            )
        print(f"    PID: {proc.pid}")
        return proc
    else:
        with open(log_file, "w") as f:
            proc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=str(PANDAS_DIR),
            )
        return None


def list_configs():
    """Print all benchmark configurations."""
    print("FetchPush Benchmark Configurations")
    print("=" * 70)
    for name, cfg in CONFIGS.items():
        status_icon = {"done": "+", "running": "~", "pending": " "}.get(cfg.status, "?")
        ig_str = "yes" if cfg.use_epistemic_bonus else "no"
        print(
            f"  [{status_icon}] {name:20s}  planner={cfg.symbolic_task:12s}  "
            f"info_gain={ig_str:3s}  {cfg.description}"
        )
    print()
    print("  + = done, ~ = running, (space) = pending")
    print("  Also: ddpg_her baseline (Alec's data, 22 data points)")


def show_results():
    """Show current results for all configs."""
    print("FetchPush Benchmark Results")
    print("=" * 70)

    # Hardcoded (known data)
    print("\n  hardcoded_no_ig (DONE, 20 evals):")
    print("    Success: 19/20 = 95% (one 0% outlier at iter 15, n=1 per eval)")
    print("    First 100% eval: 7.5k env steps")

    # Parse running/completed configs
    for name, cfg in CONFIGS.items():
        if name == "hardcoded_no_ig":
            continue

        log = find_log_file(cfg)
        if log is None:
            # Check the background task output
            for suffix in ["stdout.log", "training.log"]:
                candidate = PANDAS_DIR / cfg.run_dir / suffix
                if candidate.exists():
                    log = candidate
                    break

        if log is None:
            print(f"\n  {name} (no log found)")
            continue

        results = parse_eval_results(log)
        n_evals = len(results["env_steps"])
        if n_evals == 0:
            print(f"\n  {name} (running, no evals yet)")
            continue

        successes = sum(1 for s in results["success_rate"] if s > 0.5)
        print(f"\n  {name} ({n_evals}/20 evals):")
        print(f"    Success: {successes}/{n_evals} evals passed")
        print(f"    Rates: {[f'{s:.0%}' for s in results['success_rate']]}")
        if results["avg_steps"]:
            print(f"    Avg steps: {[f'{s:.0f}' for s in results['avg_steps']]}")

    # DDPG+HER
    print("\n  ddpg_her (Alec's data, 22 points):")
    print("    95% success at ~400k-500k env steps")
    print("    First nonzero success at ~30k steps")


def main():
    parser = argparse.ArgumentParser(description="FetchPush benchmark suite")
    parser.add_argument("--list", action="store_true", help="List all configs")
    parser.add_argument("--results", action="store_true", help="Show current results")
    parser.add_argument(
        "--configs", nargs="+",
        help="Config names to run (or 'all' for all pending)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run configs sequentially instead of in parallel",
    )
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    if args.results:
        show_results()
        return

    if not args.configs:
        parser.print_help()
        return

    configs_to_run = []
    if "all" in args.configs:
        configs_to_run = [c for c in CONFIGS.values() if c.status == "pending"]
    else:
        for name in args.configs:
            if name not in CONFIGS:
                print(f"Unknown config: {name}")
                print(f"Available: {list(CONFIGS.keys())}")
                return
            configs_to_run.append(CONFIGS[name])

    if not configs_to_run:
        print("No configs to run (all done or running).")
        return

    print(f"Launching {len(configs_to_run)} benchmark runs...")
    processes = []
    for cfg in configs_to_run:
        proc = run_config(cfg, background=not args.sequential)
        if proc is not None:
            processes.append((cfg.name, proc))

    if processes and not args.sequential:
        print(f"\n{len(processes)} runs launched in background.")
        print("Monitor with: python benchmark_fetchpush.py --results")
        print("PIDs:", {name: proc.pid for name, proc in processes})


if __name__ == "__main__":
    main()
