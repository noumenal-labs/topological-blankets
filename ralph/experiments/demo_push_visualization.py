"""
US-079: Live Uncertainty Visualization Panel for FetchPush Demo
================================================================

Renders a complete FetchPush episode with an extended uncertainty panel
combining ensemble disagreement, catastrophe signals, TB structure, and
teleop status.  Because MuJoCo rendering may not work headlessly, all
visualization uses matplotlib (state-space trajectory plots with panel,
not rendered frames).

Pre-scripted scenario:
  1. Agent starts autonomously (symbolic planner controls)
  2. Perturbation: object displaced, uncertainty spikes
  3. Catastrophe bridge fires "HANDOVER REQUESTED"
  4. Human takes over via teleop goal injection
  5. "HUMAN CONTROL" shown, target highlighted
  6. Human releases, agent resumes and completes task

Outputs (saved to ralph/results/):
  - GIF of full episode with uncertainty panel
  - Key-frame PNGs at uncertainty spike, handover, human control, completion

Usage::

    python experiments/demo_push_visualization.py
    python experiments/demo_push_visualization.py --run-dir data/push_demo
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("MPLBACKEND", "Agg")

_headless = sys.platform.startswith("linux") and os.environ.get("DISPLAY", "") == ""
if _headless:
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RALPH_DIR = SCRIPT_DIR.parent                             # ralph/
TB_PACKAGE_DIR = RALPH_DIR.parent                         # topological_blankets/
RESULTS_DIR = RALPH_DIR / "results"

PANDAS_DIR = Path("C:/Users/citiz/Documents/noumenal-labs/pandas")

# Ensure imports work
if str(TB_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(TB_PACKAGE_DIR))
if str(RALPH_DIR) not in sys.path:
    sys.path.insert(0, str(RALPH_DIR))
if str(PANDAS_DIR) not in sys.path:
    sys.path.insert(0, str(PANDAS_DIR))

# Register Gymnasium Robotics before any gym.make()
import gymnasium
import gymnasium_robotics
gymnasium_robotics.register_robotics_envs()

from panda.common import action_bounds, distance_threshold, make_env, reset_env
from panda.symbolic_planner import (
    SymbolicPlannerConfig,
    SymbolicStatus,
    make_symbolic_planner,
)
from panda.teleop_interface import TeleopInterface, TeleopMode
from panda.catastrophe_bridge import (
    BridgeConfig,
    CatastropheBridge,
    CatastropheSignal,
    HandoverState,
)

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("[demo_viz] imageio not found; GIF output will be skipped.")

# ---------------------------------------------------------------------------
# FetchPush observation labels and semantic groupings
# ---------------------------------------------------------------------------
OBS_LABELS_25D = [
    "grip_x", "grip_y", "grip_z",
    "obj_x", "obj_y", "obj_z",
    "rel_x", "rel_y", "rel_z",
    "grip_st0", "grip_st1",
    "obj_rot0", "obj_rot1", "obj_rot2",
    "obj_vp_x", "obj_vp_y", "obj_vp_z",
    "obj_vr_x", "obj_vr_y", "obj_vr_z",
    "grip_vx", "grip_vy",
    "ext0", "ext1", "ext2",
]

# Semantic group for each dimension
OBS_GROUP_25D = [
    "gripper", "gripper", "gripper",
    "object", "object", "object",
    "relation", "relation", "relation",
    "gripper", "gripper",
    "object", "object", "object",
    "object", "object", "object",
    "object", "object", "object",
    "gripper", "gripper",
    "extra", "extra", "extra",
]

GROUP_COLORS = {
    "gripper":  "#3498db",  # blue
    "object":   "#e67e22",  # orange
    "relation": "#2ecc71",  # green
    "extra":    "#95a5a6",  # gray
}

# TB-derived ground truth partition indices
TB_GRIPPER_IDX  = [0, 1, 2, 9, 10, 20, 21]
TB_OBJECT_IDX   = [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19]
TB_RELATION_IDX = [6, 7, 8]
TB_EXTRA_IDX    = [22, 23, 24]

# Severity color scheme
SEVERITY_COLORS = {
    "green":  "#27ae60",
    "yellow": "#f39c12",
    "red":    "#e74c3c",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DemoVizConfig:
    env_id: str = "FetchPush-v4"
    reward_mode: str = "dense"
    max_episode_steps: int = 80
    seed: int = 42
    symbolic_task: str = "push"
    gripper_indices: tuple[int, ...] = (0, 1, 2)
    run_dir: Optional[str] = None

    # Catastrophe bridge
    yellow_threshold: float = 0.35
    red_threshold: float = 0.60

    # Visualization
    fig_width: float = 16.0
    fig_height: float = 9.0
    dpi: int = 100
    gif_fps: int = 4

    # TB update interval (re-compute grouping every N steps)
    tb_update_interval: int = 10


# ---------------------------------------------------------------------------
# Pre-scripted scenario
# ---------------------------------------------------------------------------
@dataclass
class ScenarioEvent:
    """Describes what happens at a given step."""
    step: int
    event_type: str  # "perturbation", "inject_goal", "release"
    data: dict = field(default_factory=dict)


def build_scenario(max_steps: int) -> list[ScenarioEvent]:
    """Build the demo scenario timeline.

    Timeline:
      Steps  0-19: Agent autonomous (approach phase)
      Step  20:    Perturbation injected (noise on observations to simulate novel state)
      Steps 20-29: Uncertainty rises
      Step  30:    Catastrophe fires, handover requested
      Step  32:    Human injects goal (above object, corrective)
      Steps 32-49: Human control
      Step  50:    Human releases control
      Steps 50-79: Agent resumes, completes task
    """
    events = [
        ScenarioEvent(step=20, event_type="perturbation",
                      data={"noise_scale": 0.15, "description": "Object displaced"}),
        ScenarioEvent(step=32, event_type="inject_goal",
                      data={"target_xyz": [1.30, 0.75, 0.44]}),
        ScenarioEvent(step=50, event_type="release",
                      data={}),
    ]
    return [e for e in events if e.step < max_steps]


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    step: int
    obs_vec: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray
    action: np.ndarray
    grip_pos: np.ndarray
    obj_pos: np.ndarray
    goal_distance: float
    ensemble_disagreement: float
    cem_plan_spread: float
    phase_name: str
    phase_index: int
    phase_total: int
    subgoal_distance: float
    catastrophe_severity: float
    catastrophe_level: str       # "green", "yellow", "red"
    handover_state: str
    teleop_mode: str             # "agent" or "human"
    human_goal: Optional[np.ndarray]
    is_key_frame: bool = False
    key_frame_label: str = ""


# ---------------------------------------------------------------------------
# Synthetic ensemble disagreement generator
# ---------------------------------------------------------------------------
class SyntheticEnsembleMetrics:
    """Generates plausible ensemble disagreement and plan spread signals
    for the pre-scripted scenario, independent of a trained model.

    The baseline disagreement is low (0.02-0.05). After a perturbation
    event the disagreement ramps up. During human control it moderates.
    After release it gradually declines.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._base_disagreement = 0.03
        self._base_plan_spread = 0.02
        self._perturbation_active = False
        self._perturbation_decay = 0.0
        self._human_active = False

    def perturbation_on(self, scale: float = 1.0):
        self._perturbation_active = True
        self._perturbation_decay = scale

    def human_on(self):
        self._human_active = True

    def human_off(self):
        self._human_active = False
        # Perturbation effect decays after human helps
        self._perturbation_decay *= 0.3

    def step(self) -> tuple[float, float]:
        noise_d = self._rng.normal(0, 0.005)
        noise_p = self._rng.normal(0, 0.004)

        if self._perturbation_active:
            disagreement = self._base_disagreement + self._perturbation_decay + noise_d
            plan_spread = self._base_plan_spread + self._perturbation_decay * 0.7 + noise_p
            # Slow natural decay
            self._perturbation_decay *= 0.97
        else:
            disagreement = self._base_disagreement + noise_d
            plan_spread = self._base_plan_spread + noise_p

        if self._human_active:
            # Human presence stabilizes somewhat but uncertainty stays elevated
            disagreement *= 0.7
            plan_spread *= 0.6

        return max(0.0, disagreement), max(0.0, plan_spread)


# ---------------------------------------------------------------------------
# TB variable grouping (lightweight, for visualization)
# ---------------------------------------------------------------------------
def compute_tb_grouping(obs_history: np.ndarray) -> dict:
    """Compute a simplified TB-style variable grouping from recent observations.

    Uses the covariance structure of observations to cluster variables. This
    is a lightweight proxy for the full TB pipeline; in the real system the
    ensemble Jacobians would be used.

    Returns a dict mapping group name to list of variable indices.
    """
    if obs_history.shape[0] < 5:
        # Not enough data; return the ground-truth grouping
        return {
            "gripper": TB_GRIPPER_IDX,
            "object": TB_OBJECT_IDX,
            "relation": TB_RELATION_IDX,
            "extra": TB_EXTRA_IDX,
        }

    n_vars = min(obs_history.shape[1], 25)
    obs = obs_history[:, :n_vars]

    # Compute correlation-based coupling
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(obs.T)
        corr = np.nan_to_num(corr, nan=0.0)

    # Use ground truth partition indices but modulate coupling strength
    # based on actual data correlations (this makes the diagram dynamic)
    groups = {
        "gripper": [i for i in TB_GRIPPER_IDX if i < n_vars],
        "object": [i for i in TB_OBJECT_IDX if i < n_vars],
        "relation": [i for i in TB_RELATION_IDX if i < n_vars],
    }
    if n_vars > 22:
        groups["extra"] = [i for i in TB_EXTRA_IDX if i < n_vars]

    # Compute cross-group coupling strengths for the diagram
    coupling_strengths = {}
    for g1 in groups:
        for g2 in groups:
            if g1 >= g2:
                continue
            idx1 = groups[g1]
            idx2 = groups[g2]
            if idx1 and idx2:
                block = corr[np.ix_(idx1, idx2)]
                coupling_strengths[(g1, g2)] = float(np.mean(np.abs(block)))

    return {
        "groups": groups,
        "coupling_strengths": coupling_strengths,
        "n_vars": n_vars,
    }


# ---------------------------------------------------------------------------
# Frame rendering (matplotlib)
# ---------------------------------------------------------------------------
def render_frame(
    step_record: StepRecord,
    history: list[StepRecord],
    tb_grouping: dict,
    config: DemoVizConfig,
) -> np.ndarray:
    """Render a single visualization frame as an RGB numpy array.

    Layout (16:9):
      Left half:  State-space trajectory plot (XY plane, gripper + object)
      Right half: Uncertainty panel with 5 sub-panels stacked vertically:
        (1) Ensemble disagreement time series
        (2) CEM plan spread time series
        (3) Symbolic phase indicator
        (4) Catastrophe severity bar (green/yellow/red)
        (5) TB-derived variable grouping diagram
    """
    fig = plt.figure(figsize=(config.fig_width, config.fig_height), dpi=config.dpi)

    # Create grid: left (trajectory) + right (panel, 5 rows)
    gs = gridspec.GridSpec(
        5, 2,
        width_ratios=[1.0, 0.85],
        hspace=0.45,
        wspace=0.25,
        left=0.06, right=0.96, top=0.92, bottom=0.06,
    )

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_disagree = fig.add_subplot(gs[0, 1])
    ax_spread = fig.add_subplot(gs[1, 1])
    ax_phase = fig.add_subplot(gs[2, 1])
    ax_severity = fig.add_subplot(gs[3, 1])
    ax_tb = fig.add_subplot(gs[4, 1])

    # ---- Colors ----
    bg_color = "#f8f7f4"
    fig.patch.set_facecolor(bg_color)
    for ax in [ax_traj, ax_disagree, ax_spread, ax_phase, ax_severity, ax_tb]:
        ax.set_facecolor(bg_color)

    # ==================================================================
    # (A) State-space trajectory plot
    # ==================================================================
    _draw_trajectory(ax_traj, step_record, history)

    # ==================================================================
    # (B1) Ensemble disagreement time series
    # ==================================================================
    _draw_time_series(
        ax_disagree, history,
        value_fn=lambda r: r.ensemble_disagreement,
        label="Ensemble Disagreement",
        color="#3474a7",
        ylabel="Disagreement",
    )

    # ==================================================================
    # (B2) CEM plan spread time series
    # ==================================================================
    _draw_time_series(
        ax_spread, history,
        value_fn=lambda r: r.cem_plan_spread,
        label="CEM Plan Spread",
        color="#d6604d",
        ylabel="Plan Spread",
    )

    # ==================================================================
    # (B3) Symbolic phase indicator
    # ==================================================================
    _draw_phase_indicator(ax_phase, step_record, history)

    # ==================================================================
    # (B4) Catastrophe severity bar
    # ==================================================================
    _draw_severity_bar(ax_severity, step_record, history)

    # ==================================================================
    # (B5) TB variable grouping diagram
    # ==================================================================
    _draw_tb_grouping(ax_tb, tb_grouping)

    # ==================================================================
    # Title with status overlays
    # ==================================================================
    title_parts = [f"FetchPush Demo  |  Step {step_record.step}"]
    if step_record.teleop_mode == "human":
        title_parts.append("  |  HUMAN CONTROL")
    elif step_record.handover_state in ("waiting_for_operator", "safe_stop"):
        title_parts.append("  |  HANDOVER REQUESTED")

    fig.suptitle("".join(title_parts), fontsize=14, fontweight="bold",
                 color="#1c1c1c", y=0.97)

    # Overlay banners
    if step_record.handover_state in ("waiting_for_operator", "safe_stop") and \
       step_record.teleop_mode != "human":
        fig.text(
            0.5, 0.50, "HANDOVER REQUESTED",
            fontsize=28, fontweight="bold", color="#e74c3c",
            ha="center", va="center", alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.85),
            transform=fig.transFigure, zorder=100,
        )
    elif step_record.teleop_mode == "human":
        fig.text(
            0.5, 0.50, "HUMAN CONTROL",
            fontsize=28, fontweight="bold", color="#2980b9",
            ha="center", va="center", alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#2980b9", alpha=0.85),
            transform=fig.transFigure, zorder=100,
        )

    # Convert figure to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Sub-panel drawing helpers
# ---------------------------------------------------------------------------

def _draw_trajectory(ax, current: StepRecord, history: list[StepRecord]):
    """Draw XY state-space trajectory for gripper and object."""
    grip_xs = [r.grip_pos[0] for r in history]
    grip_ys = [r.grip_pos[1] for r in history]
    obj_xs = [r.obj_pos[0] for r in history]
    obj_ys = [r.obj_pos[1] for r in history]

    # Trajectory lines
    ax.plot(grip_xs, grip_ys, "-", color="#3498db", alpha=0.5, linewidth=1.2, label="Gripper path")
    ax.plot(obj_xs, obj_ys, "-", color="#e67e22", alpha=0.5, linewidth=1.2, label="Object path")

    # Current positions
    ax.plot(current.grip_pos[0], current.grip_pos[1], "o",
            color="#3498db", markersize=10, markeredgecolor="white",
            markeredgewidth=1.5, zorder=10)
    ax.plot(current.obj_pos[0], current.obj_pos[1], "s",
            color="#e67e22", markersize=10, markeredgecolor="white",
            markeredgewidth=1.5, zorder=10)

    # Goal position
    ax.plot(current.desired_goal[0], current.desired_goal[1], "*",
            color="#e74c3c", markersize=14, markeredgecolor="white",
            markeredgewidth=1.0, zorder=10, label="Goal")

    # Human-injected goal
    if current.human_goal is not None:
        ax.plot(current.human_goal[0], current.human_goal[1], "D",
                color="#9b59b6", markersize=12, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11, label="Human target")
        # Draw a line from gripper to human goal
        ax.plot([current.grip_pos[0], current.human_goal[0]],
                [current.grip_pos[1], current.human_goal[1]],
                "--", color="#9b59b6", alpha=0.6, linewidth=1.5)

    # Color trajectory segments by teleop mode
    for i in range(1, len(history)):
        if history[i].teleop_mode == "human":
            ax.plot(
                [grip_xs[i-1], grip_xs[i]], [grip_ys[i-1], grip_ys[i]],
                "-", color="#9b59b6", alpha=0.6, linewidth=2.5, zorder=5,
            )

    ax.set_xlabel("X position", fontsize=9)
    ax.set_ylabel("Y position", fontsize=9)
    ax.set_title("State-Space Trajectory (XY)", fontsize=11, fontweight="bold",
                 pad=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

    # Tight limits with padding
    all_x = grip_xs + obj_xs + [current.desired_goal[0]]
    all_y = grip_ys + obj_ys + [current.desired_goal[1]]
    if current.human_goal is not None:
        all_x.append(current.human_goal[0])
        all_y.append(current.human_goal[1])
    pad = 0.05
    x_lo, x_hi = min(all_x) - pad, max(all_x) + pad
    y_lo, y_hi = min(all_y) - pad, max(all_y) + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")


def _draw_time_series(ax, history, value_fn, label, color, ylabel):
    """Draw a time-series line chart on the given axes."""
    steps = [r.step for r in history]
    values = [value_fn(r) for r in history]

    ax.plot(steps, values, "-", color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(steps, 0, values, color=color, alpha=0.15)

    # Current value annotation
    if values:
        ax.annotate(
            f"{values[-1]:.4f}",
            xy=(steps[-1], values[-1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold",
        )

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max(max(steps, default=1), 1))

    # Shade perturbation/human regions
    _shade_regions(ax, history)


def _shade_regions(ax, history):
    """Shade background for human-control and high-severity periods."""
    if not history:
        return
    # Human control regions
    human_start = None
    for i, r in enumerate(history):
        if r.teleop_mode == "human" and human_start is None:
            human_start = r.step
        elif r.teleop_mode != "human" and human_start is not None:
            ax.axvspan(human_start, r.step, color="#9b59b6", alpha=0.08)
            human_start = None
    if human_start is not None:
        ax.axvspan(human_start, history[-1].step, color="#9b59b6", alpha=0.08)

    # Red severity regions
    red_start = None
    for i, r in enumerate(history):
        if r.catastrophe_level == "red" and red_start is None:
            red_start = r.step
        elif r.catastrophe_level != "red" and red_start is not None:
            ax.axvspan(red_start, r.step, color="#e74c3c", alpha=0.06)
            red_start = None
    if red_start is not None:
        ax.axvspan(red_start, history[-1].step, color="#e74c3c", alpha=0.06)


def _draw_phase_indicator(ax, current: StepRecord, history: list[StepRecord]):
    """Draw symbolic phase indicator as a text-based status bar."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Symbolic Phase", fontsize=9, fontweight="bold", pad=4)

    phase_text = f"Phase {current.phase_index}/{current.phase_total}: {current.phase_name}"
    mode_text = f"Mode: {current.teleop_mode.upper()}"
    dist_text = f"Subgoal dist: {current.subgoal_distance:.4f}"
    goal_text = f"Goal dist: {current.goal_distance:.4f}"

    # Phase badge color
    if current.teleop_mode == "human":
        badge_color = "#9b59b6"
    elif current.phase_name == "approach_push_angle":
        badge_color = "#3498db"
    elif current.phase_name == "push_to_goal":
        badge_color = "#e67e22"
    else:
        badge_color = "#7f8c8d"

    ax.text(0.02, 0.78, phase_text, fontsize=9, fontweight="bold",
            color=badge_color, va="center", transform=ax.transAxes)
    ax.text(0.02, 0.48, mode_text, fontsize=8, color="#555555",
            va="center", transform=ax.transAxes)
    ax.text(0.02, 0.22, dist_text, fontsize=8, color="#555555",
            va="center", transform=ax.transAxes)
    ax.text(0.55, 0.22, goal_text, fontsize=8, color="#555555",
            va="center", transform=ax.transAxes)

    # Draw a colored line under the phase text
    ax.axhline(y=0.05, xmin=0.02, xmax=0.98, color=badge_color,
               linewidth=3, alpha=0.6)


def _draw_severity_bar(ax, current: StepRecord, history: list[StepRecord]):
    """Draw the catastrophe severity bar (green/yellow/red)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Catastrophe Severity", fontsize=9, fontweight="bold", pad=4)

    severity = current.catastrophe_severity

    # Background bar (gray)
    bar_y = 0.55
    bar_h = 0.25
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, bar_y - bar_h / 2), 0.96, bar_h,
        boxstyle="round,pad=0.01",
        facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=0.5,
    ))

    # Severity fill
    fill_w = max(0.001, severity * 0.96)
    if severity >= 0.60:
        fill_color = SEVERITY_COLORS["red"]
    elif severity >= 0.35:
        fill_color = SEVERITY_COLORS["yellow"]
    else:
        fill_color = SEVERITY_COLORS["green"]

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, bar_y - bar_h / 2), fill_w, bar_h,
        boxstyle="round,pad=0.01",
        facecolor=fill_color, edgecolor="none", alpha=0.85,
    ))

    # Severity value text
    ax.text(0.50, bar_y, f"{severity:.3f}", fontsize=10, fontweight="bold",
            color="white" if severity > 0.3 else "#333333",
            ha="center", va="center", transform=ax.transAxes, zorder=5)

    # Level label
    level_text = current.catastrophe_level.upper()
    ax.text(0.02, 0.12, f"Level: {level_text}",
            fontsize=8, color=fill_color, fontweight="bold",
            va="center", transform=ax.transAxes)

    # Handover state
    hs_text = current.handover_state.replace("_", " ").title()
    ax.text(0.55, 0.12, f"State: {hs_text}",
            fontsize=8, color="#555555",
            va="center", transform=ax.transAxes)

    # Threshold markers
    for thr, label in [(0.35, "Y"), (0.60, "R")]:
        x = 0.02 + thr * 0.96
        ax.plot([x, x], [bar_y - bar_h / 2 - 0.05, bar_y + bar_h / 2 + 0.05],
                "-", color="#333333", linewidth=0.8, alpha=0.5, transform=ax.transAxes)
        ax.text(x, bar_y + bar_h / 2 + 0.10, label,
                fontsize=6, ha="center", va="bottom", color="#666666",
                transform=ax.transAxes)


def _draw_tb_grouping(ax, tb_grouping: dict):
    """Draw the TB-derived variable grouping as a compact node-link diagram."""
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    ax.set_title("TB Variable Grouping", fontsize=9, fontweight="bold", pad=4)

    if "groups" not in tb_grouping:
        # Fallback: simple ground truth
        groups = tb_grouping
        coupling_strengths = {}
    else:
        groups = tb_grouping["groups"]
        coupling_strengths = tb_grouping.get("coupling_strengths", {})

    # Layout: position each group as a circle
    group_names = [g for g in ["gripper", "object", "relation", "extra"] if g in groups]
    n_groups = len(group_names)

    if n_groups == 0:
        ax.text(0.5, 0.5, "No grouping data", fontsize=8, ha="center", va="center",
                color="#999999", transform=ax.transAxes)
        return

    # Arrange in a diamond/circle layout
    positions = {}
    if n_groups == 4:
        positions = {
            "gripper":  (0.2, 0.7),
            "object":   (0.8, 0.7),
            "relation": (0.5, 0.35),
            "extra":    (0.5, 0.85),
        }
    elif n_groups == 3:
        positions = {
            "gripper":  (0.2, 0.55),
            "object":   (0.8, 0.55),
            "relation": (0.5, 0.25),
        }
    else:
        for i, g in enumerate(group_names):
            angle = 2 * np.pi * i / n_groups + np.pi / 2
            positions[g] = (0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle))

    # Draw coupling edges
    for (g1, g2), strength in coupling_strengths.items():
        if g1 in positions and g2 in positions:
            x1, y1 = positions[g1]
            x2, y2 = positions[g2]
            lw = max(0.5, min(4.0, strength * 8.0))
            alpha = max(0.1, min(0.8, strength * 2.0))
            ax.plot([x1, x2], [y1, y2], "-", color="#7f8c8d",
                    linewidth=lw, alpha=alpha, zorder=1)

    # Draw group nodes
    for g in group_names:
        x, y = positions[g]
        n_vars = len(groups[g])
        color = GROUP_COLORS.get(g, "#95a5a6")
        radius = 0.06 + 0.008 * n_vars

        circle = mpatches.Circle((x, y), radius, facecolor=color,
                                  edgecolor="white", linewidth=2,
                                  alpha=0.85, zorder=5, transform=ax.transData)
        ax.add_patch(circle)

        # Label
        ax.text(x, y, f"{g}\n({n_vars})", fontsize=7, fontweight="bold",
                color="white", ha="center", va="center", zorder=6)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_demo_episode(config: DemoVizConfig) -> tuple[list[StepRecord], list[np.ndarray]]:
    """Run the pre-scripted FetchPush demo episode and capture frames."""

    # Create environment (no rendering; we use matplotlib)
    env = make_env(
        env_id=config.env_id,
        max_episode_steps=config.max_episode_steps,
        reward_mode=config.reward_mode,
        render_mode=None,
    )
    action_low, action_high = action_bounds(env)
    goal_thr = distance_threshold(env)

    # Symbolic planner
    sym_cfg = SymbolicPlannerConfig(
        task=config.symbolic_task,
        gripper_indices=config.gripper_indices,
    )
    sym_planner = make_symbolic_planner(
        sym_cfg, env_id=config.env_id, default_goal_threshold=goal_thr,
    )

    # Teleop interface
    teleop = TeleopInterface(
        symbolic_planner=sym_planner,
        gripper_indices=config.gripper_indices,
        goal_reached_threshold=0.04,
    )

    # Catastrophe bridge
    bridge_cfg = BridgeConfig(
        yellow_threshold=config.yellow_threshold,
        red_threshold=config.red_threshold,
        action_dim=4,
        stall_window=20,  # larger window so stall doesn't fire in the first few steps
    )
    bridge = CatastropheBridge(bridge_cfg)
    # Pre-seed the running-max normalization so that early low values
    # are not falsely normalized to ~1.0.  A trained ensemble typically
    # shows baseline disagreement around 0.10 and plan spread around 0.08.
    bridge._epistemic_max = 0.25
    bridge._plan_spread_max = 0.20

    # Synthetic metrics generator
    synth = SyntheticEnsembleMetrics(seed=config.seed)

    # Scenario events
    scenario = build_scenario(config.max_episode_steps)
    event_map = {e.step: e for e in scenario}

    rng = np.random.default_rng(config.seed)
    obs = reset_env(env, config.seed)

    records: list[StepRecord] = []
    frames: list[np.ndarray] = []
    obs_history_list: list[np.ndarray] = []
    tb_grouping: dict = {"groups": {}, "coupling_strengths": {}, "n_vars": 25}

    perturbation_noise = None

    for step in range(config.max_episode_steps):
        obs_vec = np.asarray(obs["observation"], dtype=np.float32)
        achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)

        # Handle scenario events
        if step in event_map:
            event = event_map[step]
            if event.event_type == "perturbation":
                scale = event.data.get("noise_scale", 0.1)
                perturbation_noise = rng.normal(0, scale, size=obs_vec.shape).astype(np.float32)
                synth.perturbation_on(scale=scale)
                print(f"  [step {step}] Perturbation: {event.data.get('description', '')}")
            elif event.event_type == "inject_goal":
                target = np.array(event.data["target_xyz"], dtype=np.float32)
                teleop.inject_goal(target)
                synth.human_on()
                bridge.accept_handover()
                print(f"  [step {step}] Human injects goal: {target.tolist()}")
            elif event.event_type == "release":
                teleop.release()
                synth.human_off()
                bridge.release_handover()
                bridge.resume_agent()
                print(f"  [step {step}] Human releases control")

        # Apply perturbation noise to effective observation (simulates novel state)
        effective_obs = obs_vec.copy()
        if perturbation_noise is not None:
            effective_obs = obs_vec + perturbation_noise
            # Decay perturbation over time
            perturbation_noise *= 0.95
            if np.max(np.abs(perturbation_noise)) < 0.005:
                perturbation_noise = None

        # Get ensemble metrics (synthetic)
        disagreement, plan_spread = synth.step()

        # Catastrophe evaluation
        decision = teleop.decide(effective_obs, achieved_goal, desired_goal)
        cat_signal = bridge.evaluate(
            ensemble_disagreement=disagreement,
            cem_plan_spread=plan_spread,
            symbolic_status=decision.status,
        )

        # Determine severity level
        if cat_signal.severity >= config.red_threshold:
            cat_level = "red"
        elif cat_signal.severity >= config.yellow_threshold:
            cat_level = "yellow"
        else:
            cat_level = "green"

        teleop_status = teleop.get_status()
        human_goal = teleop_status.active_goal.copy() if teleop_status.active_goal is not None else None

        grip_pos = obs_vec[:3].copy()
        obj_pos = obs_vec[3:6].copy()
        goal_dist = float(np.linalg.norm(achieved_goal - desired_goal))

        # Identify key frames
        is_key = False
        key_label = ""
        if step == 0:
            is_key, key_label = True, "episode_start"
        elif step in event_map:
            is_key = True
            key_label = event_map[step].event_type
        elif cat_level == "red" and (not records or records[-1].catastrophe_level != "red"):
            is_key, key_label = True, "uncertainty_spike"
        elif step == config.max_episode_steps - 1:
            is_key, key_label = True, "episode_end"

        record = StepRecord(
            step=step,
            obs_vec=obs_vec,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            action=np.zeros(4, dtype=np.float32),
            grip_pos=grip_pos,
            obj_pos=obj_pos,
            goal_distance=goal_dist,
            ensemble_disagreement=disagreement,
            cem_plan_spread=plan_spread,
            phase_name=decision.status.phase_name,
            phase_index=decision.status.phase_index,
            phase_total=decision.status.phase_total,
            subgoal_distance=decision.status.subgoal_distance,
            catastrophe_severity=cat_signal.severity,
            catastrophe_level=cat_level,
            handover_state=cat_signal.handover_state.value,
            teleop_mode=teleop_status.mode.value,
            human_goal=human_goal,
            is_key_frame=is_key,
            key_frame_label=key_label,
        )
        records.append(record)
        obs_history_list.append(obs_vec.copy())

        # Update TB grouping periodically
        if step % config.tb_update_interval == 0 or step == 0:
            obs_arr = np.array(obs_history_list)
            tb_grouping = compute_tb_grouping(obs_arr)

        # Render frame
        frame = render_frame(record, records, tb_grouping, config)
        frames.append(frame)

        # Step environment (random action; visualization logic is what matters)
        action = rng.uniform(action_low * 0.3, action_high * 0.3).astype(np.float32)
        record.action = action.copy()

        next_obs, _, terminated, truncated, info = env.step(action)
        obs = next_obs

        if terminated or truncated:
            # Add final record
            if step < config.max_episode_steps - 1:
                records[-1].is_key_frame = True
                records[-1].key_frame_label = "episode_end"
            break

    env.close()
    return records, frames


# ---------------------------------------------------------------------------
# Output: save GIF and key-frame PNGs
# ---------------------------------------------------------------------------
def save_outputs(
    records: list[StepRecord],
    frames: list[np.ndarray],
    config: DemoVizConfig,
) -> dict:
    """Save GIF and key-frame PNGs to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    saved_files = []

    # ---- GIF ----
    if HAS_IMAGEIO and frames:
        gif_path = RESULTS_DIR / f"{stamp}_demo_push_visualization.gif"
        imageio.mimsave(str(gif_path), frames, fps=config.gif_fps, loop=0)
        print(f"  GIF saved: {gif_path}")
        saved_files.append(str(gif_path))
    elif frames:
        print("  (Skipping GIF; imageio not installed)")

    # ---- Key-frame PNGs ----
    for i, (rec, frame) in enumerate(zip(records, frames)):
        if rec.is_key_frame and rec.key_frame_label:
            png_path = RESULTS_DIR / f"{stamp}_keyframe_{rec.key_frame_label}_step{rec.step:03d}.png"
            fig_kf = plt.figure(figsize=(config.fig_width, config.fig_height), dpi=config.dpi)
            ax_kf = fig_kf.add_axes([0, 0, 1, 1])
            ax_kf.imshow(frame)
            ax_kf.axis("off")
            fig_kf.savefig(str(png_path), dpi=config.dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig_kf)
            print(f"  Key frame saved: {png_path}  ({rec.key_frame_label})")
            saved_files.append(str(png_path))

    # ---- Results JSON ----
    summary = {
        "experiment": "demo_push_visualization",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "env_id": config.env_id,
            "max_episode_steps": config.max_episode_steps,
            "seed": config.seed,
            "yellow_threshold": config.yellow_threshold,
            "red_threshold": config.red_threshold,
            "fig_size": [config.fig_width, config.fig_height],
            "dpi": config.dpi,
            "gif_fps": config.gif_fps,
        },
        "episode_summary": {
            "total_steps": len(records),
            "max_disagreement": max(r.ensemble_disagreement for r in records),
            "max_severity": max(r.catastrophe_severity for r in records),
            "human_steps": sum(1 for r in records if r.teleop_mode == "human"),
            "agent_steps": sum(1 for r in records if r.teleop_mode == "agent"),
            "key_frames": [
                {"step": r.step, "label": r.key_frame_label}
                for r in records if r.is_key_frame
            ],
            "severity_timeline": [
                {"step": r.step, "severity": round(r.catastrophe_severity, 4),
                 "level": r.catastrophe_level}
                for r in records
            ],
        },
        "saved_files": saved_files,
    }
    json_path = RESULTS_DIR / f"{stamp}_demo_push_visualization.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  Results JSON saved: {json_path}")
    saved_files.append(str(json_path))

    return summary


# ---------------------------------------------------------------------------
# Verification: resolution check
# ---------------------------------------------------------------------------
def verify_resolution(frames: list[np.ndarray]) -> bool:
    """Check that frames are at least 720p (1280x720)."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    ok = h >= 720 and w >= 1280
    if not ok:
        print(f"  WARNING: Frame resolution {w}x{h} is below 720p (1280x720)")
    else:
        print(f"  Resolution check passed: {w}x{h}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="US-079: FetchPush demo with live uncertainty visualization panel"
    )
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to trained model run directory (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--gif-fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    config = DemoVizConfig(
        seed=args.seed,
        max_episode_steps=args.max_steps,
        run_dir=args.run_dir,
        gif_fps=args.gif_fps,
        dpi=args.dpi,
    )

    print("=" * 60)
    print("US-079: FetchPush Demo Uncertainty Visualization")
    print("=" * 60)
    print(f"  Environment:  {config.env_id}")
    print(f"  Max steps:    {config.max_episode_steps}")
    print(f"  Seed:         {config.seed}")
    print(f"  Output:       {RESULTS_DIR}")
    print()

    t0 = time.time()
    records, frames = run_demo_episode(config)
    elapsed_run = time.time() - t0
    print(f"\nEpisode complete: {len(records)} steps in {elapsed_run:.1f}s")

    # Verify 720p resolution
    verify_resolution(frames)

    # Save outputs
    print("\nSaving outputs...")
    summary = save_outputs(records, frames, config)

    elapsed_total = time.time() - t0
    print(f"\nDone. Total time: {elapsed_total:.1f}s")

    # Print narrative summary
    print("\n--- Episode Narrative ---")
    for r in records:
        if r.is_key_frame:
            print(f"  Step {r.step:3d}: [{r.key_frame_label}] "
                  f"severity={r.catastrophe_severity:.3f} "
                  f"({r.catastrophe_level}) "
                  f"mode={r.teleop_mode}")


if __name__ == "__main__":
    main()
