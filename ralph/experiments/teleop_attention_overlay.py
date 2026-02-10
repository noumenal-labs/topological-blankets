"""
US-054: Structure-Aware Teleoperation Attention Overlay
=======================================================

Combines TB-discovered blanket structure with ensemble disagreement to produce
a per-state 'structural attention' score.  The overlay highlights which state
variables a teleoperator should focus on: brighter colour indicates higher
uncertainty *within a structurally significant cluster*.

The structural attention score for each variable combines:
  (a) blanket membership (1.0 for blanket vars, 0.0 for interiors),
  (b) ensemble disagreement magnitude per variable,
  (c) distance to the nearest blanket in coupling space.

Three visualization modes:
  1. LunarLander episode frames with per-variable attention bar overlays
  2. FetchPush observation group uncertainty (gripper / object / relation)
  3. Comparison: TB structural attention vs raw disagreement alone

Loads pre-computed results from:
  - US-025: actinf_tb_analysis JSON (LunarLander TB partition + disagreement)
  - US-076: pandas_ensemble_analysis JSON (FetchPush TB partition + disagreement)

Outputs (saved to ralph/results/):
  - Per-episode multi-panel PNGs showing attention overlays
  - FetchPush group uncertainty PNG
  - Structural vs raw comparison PNG
  - Animated GIF assembling 3+ episode attention sequences
  - Results JSON with structural attention metrics

Usage::

    python experiments/teleop_attention_overlay.py
"""

from __future__ import annotations

import json
import glob
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Matplotlib Agg backend (headless)
# ---------------------------------------------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RALPH_DIR = SCRIPT_DIR.parent
TB_PACKAGE_DIR = RALPH_DIR.parent
RESULTS_DIR = RALPH_DIR / "results"

sys.path.insert(0, str(TB_PACKAGE_DIR))
sys.path.insert(0, str(RALPH_DIR))

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

EXPERIMENT_NAME = "teleop_attention_overlay"

# ---------------------------------------------------------------------------
# LunarLander constants
# ---------------------------------------------------------------------------
LL_STATE_LABELS = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"]
LL_STATE_DIM = 8

# Physical interpretation of flight phases
PHASE_LABELS = {
    "ascent": "Ascending (y increasing, vy > 0)",
    "descent": "Descending (vy < 0, high altitude)",
    "approach": "Final approach (low altitude, vy < 0)",
    "landed": "Landed (leg contact)",
    "tilted": "Tilted flight (large angle)",
}

# FetchPush semantic groups
FP_GROUPS = {
    "gripper_pos":   list(range(0, 3)),
    "object_pos":    list(range(3, 6)),
    "relative_pos":  list(range(6, 9)),
    "gripper_state": list(range(9, 11)),
    "object_rot":    list(range(11, 14)),
    "object_velp":   list(range(14, 17)),
    "object_velr":   list(range(17, 20)),
    "gripper_velp":  list(range(20, 22)),
    "extra":         list(range(22, 25)),
}

FP_OBS_LABELS = [
    "grip_x", "grip_y", "grip_z",
    "obj_x", "obj_y", "obj_z",
    "rel_x", "rel_y", "rel_z",
    "grip_state_0", "grip_state_1",
    "obj_rot_0", "obj_rot_1", "obj_rot_2",
    "obj_velp_x", "obj_velp_y", "obj_velp_z",
    "obj_velr_x", "obj_velr_y", "obj_velr_z",
    "grip_velp_x", "grip_velp_y",
    "extra_0", "extra_1", "extra_2",
]

# Colour palette
PALETTE = {
    "attention_high": "#DC2626",
    "attention_low":  "#10B981",
    "blanket":        "#F59E0B",
    "interior":       "#3B82F6",
    "bg":             "#F8FAFC",
    "text":           "#1E293B",
    "grid":           "#E2E8F0",
}

ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "attention", ["#10B981", "#FEF3C7", "#F59E0B", "#DC2626", "#7C2D12"]
)
DISAGREE_CMAP = LinearSegmentedColormap.from_list(
    "disagree", ["#ECFDF5", "#6EE7B7", "#10B981", "#065F46"]
)


# =============================================================================
# Data loading
# =============================================================================

def find_latest_result(pattern: str) -> str | None:
    """Find the most recent results JSON matching a filename pattern."""
    candidates = sorted(glob.glob(str(RESULTS_DIR / f"*{pattern}*.json")))
    if not candidates:
        return None
    return candidates[-1]


def load_lunarlander_tb() -> dict:
    """Load LunarLander TB analysis (US-025)."""
    path = find_latest_result("actinf_tb_analysis")
    if path is None:
        raise FileNotFoundError(
            "No actinf_tb_analysis JSON in results/. Run world_model_analysis.py first."
        )
    print(f"Loading LunarLander TB: {os.path.basename(path)}")
    with open(path) as f:
        return json.load(f)


def load_fetchpush_tb() -> dict:
    """Load FetchPush TB analysis (US-076)."""
    path = find_latest_result("pandas_ensemble_analysis")
    if path is None:
        raise FileNotFoundError(
            "No pandas_ensemble_analysis JSON in results/. Run pandas_ensemble_analysis.py first."
        )
    print(f"Loading FetchPush TB: {os.path.basename(path)}")
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Trajectory collection (LunarLander, random policy)
# =============================================================================

def collect_lunarlander_episodes(n_episodes: int = 5, seed: int = 54) -> list[dict]:
    """
    Collect individual episode trajectories from LunarLander-v3 using random
    actions. Each episode dict contains states, rewards, and per-step metadata.
    """
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        states, rewards, actions = [obs.copy()], [], []

        while True:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            states.append(obs.copy())
            rewards.append(reward)
            actions.append(action)
            if term or trunc:
                break

        episodes.append({
            "states": np.array(states),
            "rewards": np.array(rewards),
            "actions": np.array(actions),
            "length": len(rewards),
            "total_reward": float(np.sum(rewards)),
        })
        print(f"  Episode {ep}: len={len(rewards)}, return={np.sum(rewards):.1f}")

    env.close()
    return episodes


# =============================================================================
# Structural attention score computation
# =============================================================================

def compute_structural_attention(
    coupling: np.ndarray,
    assignment: np.ndarray,
    is_blanket: np.ndarray,
    disagreement_grad_mag: np.ndarray,
) -> dict:
    """
    Compute the per-variable structural attention score.

    The attention score combines three components:
      (a) blanket_score: 1.0 for blanket variables, 0.0 for interiors
      (b) disagreement_score: normalised ensemble disagreement gradient magnitude
      (c) blanket_proximity: distance to nearest blanket in coupling space

    Final attention = (blanket_score + disagreement_score + blanket_proximity) / 3

    Parameters
    ----------
    coupling : (D, D) coupling matrix from TB analysis
    assignment : (D,) object assignment (-1 = blanket)
    is_blanket : (D,) boolean blanket mask
    disagreement_grad_mag : (D,) gradient magnitude of disagreement per variable

    Returns
    -------
    dict with per-variable scores and composite attention
    """
    n_vars = len(assignment)
    coupling = np.array(coupling)
    assignment = np.array(assignment)
    is_blanket = np.array(is_blanket, dtype=bool)
    disagreement_grad_mag = np.array(disagreement_grad_mag)

    # (a) Blanket membership score: 1 for blanket, 0 for interior
    blanket_score = is_blanket.astype(float)

    # (b) Disagreement score: normalised to [0, 1]
    dmin, dmax = disagreement_grad_mag.min(), disagreement_grad_mag.max()
    if dmax > dmin:
        disagreement_score = (disagreement_grad_mag - dmin) / (dmax - dmin)
    else:
        disagreement_score = np.zeros(n_vars)

    # (c) Blanket proximity: for each variable, minimum coupling distance to
    #     any blanket variable. Blanket vars themselves get score 1.0.
    blanket_indices = np.where(is_blanket)[0]
    blanket_proximity = np.zeros(n_vars)

    if len(blanket_indices) > 0:
        for i in range(n_vars):
            if is_blanket[i]:
                blanket_proximity[i] = 1.0
            else:
                # Coupling to blanket variables (higher coupling = closer)
                couplings_to_blanket = coupling[i, blanket_indices]
                blanket_proximity[i] = np.max(couplings_to_blanket)

        # Normalise proximity
        pmin, pmax = blanket_proximity.min(), blanket_proximity.max()
        if pmax > pmin:
            blanket_proximity = (blanket_proximity - pmin) / (pmax - pmin)

    # Composite structural attention
    structural_attention = (blanket_score + disagreement_score + blanket_proximity) / 3.0

    return {
        "blanket_score": blanket_score,
        "disagreement_score": disagreement_score,
        "blanket_proximity": blanket_proximity,
        "structural_attention": structural_attention,
    }


def compute_per_timestep_attention(
    episode_states: np.ndarray,
    structural_attention: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    """
    Compute a per-timestep, per-variable attention score.

    Modulates the static structural attention by the instantaneous deviation
    of each variable from its running mean (normalised). Variables that are
    both structurally important *and* changing rapidly get the highest score.

    Parameters
    ----------
    episode_states : (T, D) states over one episode
    structural_attention : (D,) static structural attention
    coupling : (D, D) coupling matrix

    Returns
    -------
    (T, D) per-timestep attention matrix
    """
    T, D = episode_states.shape

    # Running deviation: |x_t - mean(x_{0:t})| normalised per variable
    running_dev = np.zeros((T, D))
    for t in range(T):
        if t == 0:
            running_dev[t] = 0.0
        else:
            window = episode_states[: t + 1]
            running_mean = window.mean(axis=0)
            running_std = window.std(axis=0) + 1e-8
            running_dev[t] = np.abs(episode_states[t] - running_mean) / running_std

    # Normalise running_dev to [0, 1] per variable across time
    for d in range(D):
        col = running_dev[:, d]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            running_dev[:, d] = (col - cmin) / (cmax - cmin)

    # Modulated attention: structural * (0.5 + 0.5 * deviation)
    # This ensures structurally important vars always have some baseline
    attention = structural_attention[None, :] * (0.5 + 0.5 * running_dev)

    return attention


def classify_flight_phase(state: np.ndarray) -> str:
    """
    Classify the current flight phase from a LunarLander state vector.

    state: [x, y, vx, vy, angle, ang_vel, left_leg, right_leg]
    """
    y = state[1]
    vy = state[3]
    angle = state[4]
    left_leg = state[6]
    right_leg = state[7]

    if left_leg > 0.5 or right_leg > 0.5:
        return "landed"
    if abs(angle) > 0.3:
        return "tilted"
    if vy > 0.05:
        return "ascent"
    if y > 0.5:
        return "descent"
    return "approach"


# =============================================================================
# Visualization: LunarLander episode attention overlay
# =============================================================================

def plot_episode_attention_overlay(
    episode: dict,
    structural_attention: np.ndarray,
    coupling: np.ndarray,
    is_blanket: np.ndarray,
    episode_idx: int,
) -> plt.Figure:
    """
    Create a multi-panel figure for a single LunarLander episode showing:
      Top: state trajectory with phase annotations
      Middle: per-variable attention heatmap over time
      Bottom: attention peaks labelled with physical interpretation
    """
    states = episode["states"]
    T = states.shape[0]

    # Per-timestep attention
    attn = compute_per_timestep_attention(states, structural_attention, coupling)

    # Flight phases
    phases = [classify_flight_phase(states[t]) for t in range(T)]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 0.5, 3, 1.5], hspace=0.35)

    # -- Panel 1: state trajectory -------
    ax_traj = fig.add_subplot(gs[0])
    time_axis = np.arange(T)
    colours = ["#3B82F6", "#EF4444", "#10B981", "#8B5CF6",
               "#F59E0B", "#EC4899", "#6B7280", "#6B7280"]
    for d in range(LL_STATE_DIM):
        ax_traj.plot(time_axis, states[:, d], color=colours[d],
                     alpha=0.7, linewidth=1.2, label=LL_STATE_LABELS[d])
    ax_traj.set_ylabel("State value")
    ax_traj.set_title(
        f"Episode {episode_idx}: LunarLander Trajectory "
        f"(return = {episode['total_reward']:.0f})",
        fontsize=12, fontweight="bold",
    )
    ax_traj.legend(ncol=4, fontsize=7, loc="upper right")
    ax_traj.set_xlim(0, T - 1)
    ax_traj.grid(True, alpha=0.2)

    # -- Panel 2: flight phase strip -----
    ax_phase = fig.add_subplot(gs[1], sharex=ax_traj)
    phase_colours = {
        "ascent": "#60A5FA", "descent": "#FBBF24", "approach": "#F87171",
        "landed": "#34D399", "tilted": "#C084FC",
    }
    for t in range(T):
        ax_phase.axvspan(t - 0.5, t + 0.5, color=phase_colours.get(phases[t], "#D1D5DB"),
                         alpha=0.6)
    ax_phase.set_yticks([])
    ax_phase.set_ylabel("Phase", fontsize=8)

    # Phase legend
    phase_patches = [mpatches.Patch(color=c, label=p, alpha=0.6)
                     for p, c in phase_colours.items()]
    ax_phase.legend(handles=phase_patches, ncol=5, fontsize=6, loc="center")

    # -- Panel 3: attention heatmap ------
    ax_heat = fig.add_subplot(gs[2], sharex=ax_traj)
    im = ax_heat.imshow(
        attn.T, aspect="auto", cmap=ATTENTION_CMAP,
        extent=[0, T, LL_STATE_DIM - 0.5, -0.5],
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(LL_STATE_DIM))
    ax_heat.set_yticklabels(LL_STATE_LABELS, fontsize=9)
    ax_heat.set_ylabel("State variable")
    ax_heat.set_title("Structural Attention Overlay (brighter = higher uncertainty)", fontsize=10)

    # Mark blanket variables with a star label
    blanket_mask = np.array(is_blanket, dtype=bool)
    for d in range(LL_STATE_DIM):
        if blanket_mask[d]:
            ax_heat.text(-1.5, d, "*B*", fontsize=8, color=PALETTE["blanket"],
                         ha="center", va="center", fontweight="bold")

    plt.colorbar(im, ax=ax_heat, label="Attention", shrink=0.8, pad=0.02)

    # -- Panel 4: attention peaks with interpretations -----
    ax_peaks = fig.add_subplot(gs[3], sharex=ax_traj)
    total_attn = attn.sum(axis=1)  # (T,)
    total_attn_norm = total_attn / total_attn.max() if total_attn.max() > 0 else total_attn
    ax_peaks.fill_between(time_axis, total_attn_norm, alpha=0.3, color=PALETTE["attention_high"])
    ax_peaks.plot(time_axis, total_attn_norm, color=PALETTE["attention_high"], linewidth=1.5)
    ax_peaks.set_ylabel("Total attention")
    ax_peaks.set_xlabel("Timestep")
    ax_peaks.set_ylim(0, 1.15)
    ax_peaks.grid(True, alpha=0.2)

    # Identify and annotate attention peaks
    peak_annotations = _find_attention_peaks(total_attn_norm, phases)
    for pa in peak_annotations:
        ax_peaks.annotate(
            pa["label"],
            xy=(pa["t"], pa["value"]),
            xytext=(pa["t"], min(pa["value"] + 0.15, 1.1)),
            fontsize=7, color=PALETTE["text"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=0.8),
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["grid"], alpha=0.9),
        )

    fig.suptitle(
        "US-054: Structure-Aware Teleoperation Attention Overlay",
        fontsize=13, fontweight="bold", y=0.98,
    )

    return fig


def _find_attention_peaks(
    total_attn: np.ndarray, phases: list[str], min_prominence: float = 0.15
) -> list[dict]:
    """Find peaks in total attention and assign physical interpretations."""
    T = len(total_attn)
    peaks = []

    # Simple peak detection: local maxima above threshold
    for t in range(2, T - 2):
        if (total_attn[t] > total_attn[t - 1] and
                total_attn[t] > total_attn[t + 1] and
                total_attn[t] > 0.5):
            # Check prominence
            left_min = total_attn[max(0, t - 10) : t].min()
            right_min = total_attn[t + 1 : min(T, t + 11)].min()
            prominence = total_attn[t] - max(left_min, right_min)
            if prominence >= min_prominence:
                peaks.append({"t": t, "value": total_attn[t], "phase": phases[t]})

    # Limit to top 4 peaks
    peaks = sorted(peaks, key=lambda p: p["value"], reverse=True)[:4]

    # Assign labels
    interpretation_map = {
        "ascent": "Ascent uncertainty",
        "descent": "Descent transition",
        "approach": "Landing approach (critical)",
        "landed": "Ground contact event",
        "tilted": "Tilt instability",
    }
    for p in peaks:
        p["label"] = interpretation_map.get(p["phase"], "Phase transition")

    return peaks


# =============================================================================
# Visualization: FetchPush group uncertainty
# =============================================================================

def plot_fetchpush_group_attention(fp_data: dict) -> plt.Figure:
    """
    Visualize per-group uncertainty for FetchPush, showing which observation
    groups (gripper, object, relation) carry the most ensemble disagreement
    and structural attention.
    """
    metrics = fp_data["metrics"]
    disagreement = np.array(metrics["disagreement_per_var"])
    assignment = np.array(metrics["primary_assignment"])
    blankets = metrics["primary_blankets"]
    is_blanket = np.zeros(len(assignment), dtype=bool)
    for b in blankets:
        if b < len(is_blanket):
            is_blanket[b] = True

    # Compute per-group mean disagreement
    group_names = list(FP_GROUPS.keys())
    group_disagree = []
    group_blanket_frac = []
    group_indices_list = []
    for gname in group_names:
        idx = FP_GROUPS[gname]
        idx_valid = [i for i in idx if i < len(disagreement)]
        group_indices_list.append(idx_valid)
        group_disagree.append(np.mean(disagreement[idx_valid]) if idx_valid else 0)
        group_blanket_frac.append(np.mean(is_blanket[idx_valid]) if idx_valid else 0)

    group_disagree = np.array(group_disagree)
    group_blanket_frac = np.array(group_blanket_frac)

    # Structural attention per group: product of disagreement and blanket fraction
    raw_structural = group_disagree * (0.3 + 0.7 * group_blanket_frac)
    if raw_structural.max() > 0:
        structural_norm = raw_structural / raw_structural.max()
    else:
        structural_norm = raw_structural

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Per-variable disagreement coloured by group
    ax = axes[0]
    group_colours = {
        "gripper_pos": "#3B82F6", "object_pos": "#EF4444",
        "relative_pos": "#10B981", "gripper_state": "#60A5FA",
        "object_rot": "#F87171", "object_velp": "#DC2626",
        "object_velr": "#B91C1C", "gripper_velp": "#93C5FD",
        "extra": "#9CA3AF",
    }
    bar_colours = []
    for gname, idx_list in zip(group_names, group_indices_list):
        for _ in idx_list:
            bar_colours.append(group_colours.get(gname, "#9CA3AF"))

    valid_labels = [FP_OBS_LABELS[i] for i in range(min(len(disagreement), len(FP_OBS_LABELS)))]
    bars = ax.barh(range(len(disagreement)), disagreement, color=bar_colours[:len(disagreement)])
    ax.set_yticks(range(len(disagreement)))
    ax.set_yticklabels(valid_labels[:len(disagreement)], fontsize=6)
    ax.set_xlabel("Ensemble Disagreement")
    ax.set_title("Per-Variable Disagreement", fontsize=10, fontweight="bold")
    ax.invert_yaxis()

    # Mark blanket variables
    for i in range(len(disagreement)):
        if is_blanket[i]:
            ax.plot(disagreement[i] + 0.1, i, marker="*", color=PALETTE["blanket"],
                    markersize=10, zorder=5)

    # Panel 2: Group-level structural attention
    ax = axes[1]
    bar_cols = [group_colours.get(g, "#9CA3AF") for g in group_names]
    x_pos = np.arange(len(group_names))
    ax.bar(x_pos, structural_norm, color=bar_cols, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.replace("_", "\n") for g in group_names], fontsize=7, rotation=0)
    ax.set_ylabel("Structural Attention (normalised)")
    ax.set_title("Group-Level Structural Attention", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)

    # Annotate blanket fraction
    for i, (gn, bf) in enumerate(zip(group_names, group_blanket_frac)):
        if bf > 0:
            ax.text(i, structural_norm[i] + 0.03, f"BF={bf:.0%}", ha="center",
                    fontsize=6, color=PALETTE["text"])

    # Panel 3: Heatmap of coupling within/across groups
    ax = axes[2]
    n_groups = len(group_names)
    group_coupling = np.zeros((n_groups, n_groups))

    # Use sensitivity_per_var as a proxy for coupling if available
    sens = np.array(metrics.get("mean_sensitivity_per_var", disagreement))
    for gi, g_idx_i in enumerate(group_indices_list):
        for gj, g_idx_j in enumerate(group_indices_list):
            if g_idx_i and g_idx_j:
                # Cross-group mean sensitivity product
                si = np.mean(sens[g_idx_i])
                sj = np.mean(sens[g_idx_j])
                group_coupling[gi, gj] = np.sqrt(si * sj)

    im = ax.imshow(group_coupling, cmap=ATTENTION_CMAP, aspect="auto")
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g.replace("_", "\n") for g in group_names], fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([g.replace("_", "\n") for g in group_names], fontsize=6)
    ax.set_title("Group Coupling Strength", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "FetchPush: Observation Group Uncertainty for Teleoperation",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Visualization: structural vs raw disagreement comparison
# =============================================================================

def plot_structural_vs_raw_comparison(
    structural_attention: np.ndarray,
    disagreement_score: np.ndarray,
    blanket_score: np.ndarray,
    blanket_proximity: np.ndarray,
    labels: list[str],
    domain: str = "LunarLander",
) -> plt.Figure:
    """
    Side-by-side comparison of TB-structural attention vs raw disagreement
    alone, demonstrating the value added by structure awareness.
    """
    n_vars = len(labels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(n_vars)
    width = 0.35

    # Panel 1: Raw disagreement vs structural attention
    ax = axes[0]
    ax.bar(x - width / 2, disagreement_score, width, label="Raw Disagreement",
           color="#6EE7B7", edgecolor="white")
    ax.bar(x + width / 2, structural_attention, width, label="Structural Attention",
           color="#F59E0B", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Score (normalised)")
    ax.set_title(f"{domain}: Raw Disagreement vs Structural Attention",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)

    # Panel 2: Component decomposition (stacked)
    ax = axes[1]
    ax.bar(x, blanket_score / 3, label="Blanket Membership (1/3)", color="#F59E0B")
    ax.bar(x, disagreement_score / 3, bottom=blanket_score / 3,
           label="Disagreement (1/3)", color="#10B981")
    ax.bar(x, blanket_proximity / 3, bottom=(blanket_score + disagreement_score) / 3,
           label="Blanket Proximity (1/3)", color="#3B82F6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Composite Score")
    ax.set_title("Attention Score Decomposition", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 1.15)

    # Panel 3: Scatter plot showing correlation
    ax = axes[2]
    scatter_colors = ["#F59E0B" if b else "#3B82F6" for b in blanket_score > 0.5]
    ax.scatter(disagreement_score, structural_attention, c=scatter_colors,
               s=80, edgecolors="white", linewidth=0.5, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (disagreement_score[i], structural_attention[i]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))

    # Reference line: y = x (raw = structural)
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1, label="y = x")
    ax.set_xlabel("Raw Disagreement Score")
    ax.set_ylabel("Structural Attention Score")
    ax.set_title("Structure Amplifies Blanket Uncertainty", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Mark blanket vs interior
    blanket_patch = mpatches.Patch(color="#F59E0B", label="Blanket variable")
    interior_patch = mpatches.Patch(color="#3B82F6", label="Interior variable")
    ax.legend(handles=[blanket_patch, interior_patch,
              plt.Line2D([0], [0], linestyle="--", color="#9CA3AF", label="y = x")],
              fontsize=7, loc="lower right")

    fig.suptitle(
        "US-054: TB Structure vs Raw Disagreement Comparison",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Summary card: which flight phases trigger high structural attention
# =============================================================================

def plot_phase_attention_summary(episodes: list[dict], structural_attention: np.ndarray,
                                  coupling: np.ndarray) -> plt.Figure:
    """
    Summarise which LunarLander flight phases trigger the highest structural
    attention across all episodes.
    """
    phase_attention = {p: [] for p in PHASE_LABELS}

    for ep in episodes:
        states = ep["states"]
        attn = compute_per_timestep_attention(states, structural_attention, coupling)
        total_attn = attn.sum(axis=1)

        for t in range(states.shape[0]):
            phase = classify_flight_phase(states[t])
            if phase in phase_attention:
                phase_attention[phase].append(total_attn[t])

    # Compute statistics
    phase_names = []
    phase_means = []
    phase_stds = []
    phase_counts = []
    for p in ["ascent", "descent", "approach", "landed", "tilted"]:
        vals = phase_attention.get(p, [])
        if vals:
            phase_names.append(p.capitalize())
            phase_means.append(np.mean(vals))
            phase_stds.append(np.std(vals))
            phase_counts.append(len(vals))
        else:
            phase_names.append(p.capitalize())
            phase_means.append(0)
            phase_stds.append(0)
            phase_counts.append(0)

    phase_means = np.array(phase_means)
    phase_stds = np.array(phase_stds)

    fig, ax = plt.subplots(figsize=(10, 5))
    phase_colors = ["#60A5FA", "#FBBF24", "#F87171", "#34D399", "#C084FC"]
    x = np.arange(len(phase_names))
    bars = ax.bar(x, phase_means, yerr=phase_stds, capsize=4,
                  color=phase_colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, fontsize=10)
    ax.set_ylabel("Mean Structural Attention", fontsize=10)
    ax.set_title("Flight Phase vs Structural Attention (aggregated across episodes)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)

    # Annotate counts
    for i, (m, c) in enumerate(zip(phase_means, phase_counts)):
        ax.text(i, m + phase_stds[i] + 0.02 * phase_means.max(),
                f"n={c}", ha="center", fontsize=8, color=PALETTE["text"])

    fig.tight_layout()
    return fig


# =============================================================================
# GIF assembly
# =============================================================================

def make_demo_gif(frame_paths: list[str], output_path: str, duration_ms: int = 1500):
    """Assemble PNGs into an animated GIF using PIL."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available; skipping GIF creation.")
        return None

    if not frame_paths:
        print("No frames to assemble into GIF.")
        return None

    frames = []
    for p in frame_paths:
        img = Image.open(p)
        # Convert RGBA to RGB if needed
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        frames.append(img)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF saved to {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("US-054: Structure-Aware Teleoperation Attention Overlay")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []
    all_metrics = {}

    # ------------------------------------------------------------------
    # 1. Load pre-computed TB data
    # ------------------------------------------------------------------
    print("\n--- Loading pre-computed TB results ---")
    ll_data = load_lunarlander_tb()
    fp_data = load_fetchpush_tb()

    # Extract LunarLander TB partition
    ll_metrics = ll_data["metrics"]
    ll_dynamics = ll_metrics["dynamics"]
    ll_coupling = np.array(ll_dynamics["coupling"])
    ll_hybrid = ll_dynamics["hybrid_method"]
    ll_assignment = np.array(ll_hybrid["assignment"])
    ll_is_blanket = np.array(ll_hybrid["is_blanket"])

    # Use disagreement gradient magnitudes from the disagreement analysis
    if "disagreement" in ll_metrics:
        ll_disagree_grad = np.array(ll_metrics["disagreement"]["grad_magnitude"])
        print(f"  LunarLander disagreement grad magnitudes: {ll_disagree_grad.round(4)}")
    else:
        # Fallback: use dynamics gradient magnitudes
        ll_disagree_grad = np.array(ll_dynamics["grad_magnitude"])
        print("  (Using dynamics grad magnitudes as disagreement proxy)")

    print(f"  LunarLander: {LL_STATE_DIM} vars, "
          f"blanket={list(np.where(ll_is_blanket)[0])}, "
          f"objects={{0: {list(np.where(ll_assignment == 0)[0])}, "
          f"1: {list(np.where(ll_assignment == 1)[0])}}}")

    # ------------------------------------------------------------------
    # 2. Compute structural attention (LunarLander)
    # ------------------------------------------------------------------
    print("\n--- Computing structural attention (LunarLander) ---")
    ll_attention = compute_structural_attention(
        ll_coupling, ll_assignment, ll_is_blanket, ll_disagree_grad,
    )
    ll_sa = ll_attention["structural_attention"]
    print("  Per-variable structural attention:")
    for i, (lbl, sa) in enumerate(zip(LL_STATE_LABELS, ll_sa)):
        tag = " [BLANKET]" if ll_is_blanket[i] else ""
        print(f"    {lbl:>10s}: {sa:.3f}{tag}")

    all_metrics["lunarlander_structural_attention"] = {
        lbl: float(sa) for lbl, sa in zip(LL_STATE_LABELS, ll_sa)
    }
    all_metrics["lunarlander_blanket_vars"] = [
        LL_STATE_LABELS[i] for i in range(LL_STATE_DIM) if ll_is_blanket[i]
    ]
    all_metrics["lunarlander_attention_components"] = {
        "blanket_score": ll_attention["blanket_score"].tolist(),
        "disagreement_score": ll_attention["disagreement_score"].tolist(),
        "blanket_proximity": ll_attention["blanket_proximity"].tolist(),
    }

    # ------------------------------------------------------------------
    # 3. Collect LunarLander episodes
    # ------------------------------------------------------------------
    print("\n--- Collecting LunarLander episodes ---")
    episodes = collect_lunarlander_episodes(n_episodes=5, seed=54)

    episode_summaries = []
    for i, ep in enumerate(episodes):
        episode_summaries.append({
            "episode": i,
            "length": ep["length"],
            "total_reward": ep["total_reward"],
        })
    all_metrics["episode_summaries"] = episode_summaries

    # ------------------------------------------------------------------
    # 4. Visualize episode attention overlays (LunarLander)
    # ------------------------------------------------------------------
    print("\n--- Generating episode attention overlays ---")
    episode_png_paths = []
    for i, ep in enumerate(episodes[:3]):
        fig = plot_episode_attention_overlay(
            ep, ll_sa, ll_coupling, ll_is_blanket, episode_idx=i,
        )
        path = save_figure(fig, f"episode_{i}_attention", EXPERIMENT_NAME)
        episode_png_paths.append(path)
        saved_files.append(path)

    # ------------------------------------------------------------------
    # 5. Structural vs raw comparison (LunarLander)
    # ------------------------------------------------------------------
    print("\n--- Generating structural vs raw comparison ---")
    fig_cmp = plot_structural_vs_raw_comparison(
        ll_sa,
        ll_attention["disagreement_score"],
        ll_attention["blanket_score"],
        ll_attention["blanket_proximity"],
        LL_STATE_LABELS,
        domain="LunarLander",
    )
    path = save_figure(fig_cmp, "structural_vs_raw_ll", EXPERIMENT_NAME)
    saved_files.append(path)

    # ------------------------------------------------------------------
    # 6. FetchPush group uncertainty
    # ------------------------------------------------------------------
    print("\n--- Generating FetchPush group uncertainty ---")
    fig_fp = plot_fetchpush_group_attention(fp_data)
    path = save_figure(fig_fp, "fetchpush_group_attention", EXPERIMENT_NAME)
    saved_files.append(path)

    # FetchPush structural attention (from pre-computed data)
    fp_metrics = fp_data["metrics"]
    fp_disagree = np.array(fp_metrics["disagreement_per_var"])
    fp_assignment = np.array(fp_metrics["primary_assignment"])
    fp_blankets = fp_metrics["primary_blankets"]
    fp_is_blanket = np.zeros(len(fp_assignment), dtype=bool)
    for b in fp_blankets:
        if b < len(fp_is_blanket):
            fp_is_blanket[b] = True

    # Group-level attention summary for FetchPush
    fp_group_attention = {}
    for gname, gidx in FP_GROUPS.items():
        valid = [i for i in gidx if i < len(fp_disagree)]
        if valid:
            mean_d = float(np.mean(fp_disagree[valid]))
            blanket_frac = float(np.mean(fp_is_blanket[valid]))
            fp_group_attention[gname] = {
                "mean_disagreement": mean_d,
                "blanket_fraction": blanket_frac,
                "structural_attention": mean_d * (0.3 + 0.7 * blanket_frac),
            }
    all_metrics["fetchpush_group_attention"] = fp_group_attention

    # FetchPush structural vs raw comparison
    # Build a per-variable coupling matrix proxy from the sensitivity data
    fp_sens = np.array(fp_metrics.get("mean_sensitivity_per_var", fp_disagree))
    fp_n = len(fp_disagree)
    fp_coupling_proxy = np.zeros((fp_n, fp_n))
    for i in range(fp_n):
        for j in range(fp_n):
            fp_coupling_proxy[i, j] = np.sqrt(fp_sens[i] * fp_sens[j])
    np.fill_diagonal(fp_coupling_proxy, 0)
    # Normalise
    cp_max = fp_coupling_proxy.max()
    if cp_max > 0:
        fp_coupling_proxy /= cp_max

    fp_attention = compute_structural_attention(
        fp_coupling_proxy, fp_assignment, fp_is_blanket, fp_disagree,
    )

    valid_fp_labels = FP_OBS_LABELS[:fp_n]
    fig_fp_cmp = plot_structural_vs_raw_comparison(
        fp_attention["structural_attention"],
        fp_attention["disagreement_score"],
        fp_attention["blanket_score"],
        fp_attention["blanket_proximity"],
        valid_fp_labels,
        domain="FetchPush",
    )
    path = save_figure(fig_fp_cmp, "structural_vs_raw_fp", EXPERIMENT_NAME)
    saved_files.append(path)

    all_metrics["fetchpush_structural_attention"] = {
        lbl: float(v) for lbl, v in zip(valid_fp_labels, fp_attention["structural_attention"])
    }

    # ------------------------------------------------------------------
    # 7. Phase attention summary
    # ------------------------------------------------------------------
    print("\n--- Generating phase attention summary ---")
    fig_phase = plot_phase_attention_summary(episodes, ll_sa, ll_coupling)
    path = save_figure(fig_phase, "phase_attention_summary", EXPERIMENT_NAME)
    saved_files.append(path)

    # Compute phase statistics for results
    phase_stats = {}
    for ep in episodes:
        states = ep["states"]
        attn = compute_per_timestep_attention(states, ll_sa, ll_coupling)
        total_attn = attn.sum(axis=1)
        for t in range(states.shape[0]):
            phase = classify_flight_phase(states[t])
            if phase not in phase_stats:
                phase_stats[phase] = []
            phase_stats[phase].append(float(total_attn[t]))

    all_metrics["phase_attention_summary"] = {
        phase: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "count": len(vals),
        }
        for phase, vals in phase_stats.items() if vals
    }

    # Identify which phases trigger highest attention
    phase_ranking = sorted(
        all_metrics["phase_attention_summary"].items(),
        key=lambda kv: kv[1]["mean"],
        reverse=True,
    )
    print("\n  Flight phase attention ranking:")
    for phase, stats in phase_ranking:
        print(f"    {phase:>10s}: mean={stats['mean']:.3f} +/- {stats['std']:.3f} (n={stats['count']})")

    all_metrics["highest_attention_phase"] = phase_ranking[0][0] if phase_ranking else "unknown"
    all_metrics["attention_interpretation"] = (
        f"Highest structural attention during '{phase_ranking[0][0]}' phase. "
        f"Blanket variables ({', '.join(all_metrics['lunarlander_blanket_vars'])}) "
        f"amplify uncertainty at structural boundaries, especially during flight "
        f"phase transitions where the world model is most uncertain."
    ) if phase_ranking else "No data."

    # ------------------------------------------------------------------
    # 8. Animated GIF from episode overlays
    # ------------------------------------------------------------------
    print("\n--- Assembling demo GIF ---")
    gif_path = str(RESULTS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{EXPERIMENT_NAME}_demo.gif")
    gif_result = make_demo_gif(episode_png_paths, gif_path, duration_ms=2000)
    if gif_result:
        saved_files.append(gif_result)
        all_metrics["gif_path"] = gif_result

    # ------------------------------------------------------------------
    # 9. Save results JSON
    # ------------------------------------------------------------------
    print("\n--- Saving results ---")
    all_metrics["n_episodes_visualized"] = len(episode_png_paths)
    all_metrics["saved_files"] = [os.path.basename(f) for f in saved_files]

    results_path = save_results(
        EXPERIMENT_NAME,
        metrics=all_metrics,
        config={
            "n_episodes": 5,
            "n_episodes_visualized": 3,
            "attention_formula": "(blanket_score + disagreement_score + blanket_proximity) / 3",
            "domains": ["LunarLander-v3", "FetchPush-v4"],
        },
        notes=(
            "US-054: Structure-aware teleoperation attention overlay. "
            "Combines TB blanket membership, ensemble disagreement, and "
            "blanket proximity in coupling space to produce per-variable "
            "structural attention scores. Visualized on LunarLander episodes "
            "with phase annotations and on FetchPush observation groups. "
            f"Highest attention phase: {all_metrics.get('highest_attention_phase', 'unknown')}."
        ),
    )
    saved_files.append(results_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("US-054 COMPLETE")
    print("=" * 70)
    print(f"  Saved {len(saved_files)} files to results/")
    for f in saved_files:
        print(f"    {os.path.basename(f)}")
    print(f"\n  Highest attention phase: {all_metrics.get('highest_attention_phase', 'unknown')}")
    print(f"  LunarLander blanket vars: {all_metrics['lunarlander_blanket_vars']}")
    print(f"  Interpretation: {all_metrics.get('attention_interpretation', 'N/A')}")

    return all_metrics


if __name__ == "__main__":
    main()
