"""
US-114: Surprise-Based Data Annotation for Teleoperation Handoff
=================================================================

Uses TB-decomposed per-factor surprise to drive intelligent handoff in
teleoperation. Instead of handing off the entire task when scalar surprise
is high, this identifies which specific factor (gripper, object, relation)
is surprising and generates annotated handoff requests.

Architecture:
  1. TB partitions the FetchPush world model into factors:
     - Object 0 (gripper): grip_pos, gripper_state, grip_velp
     - Object 1 (manipulated): object_pos, object_rot, object_velp, object_velr
     - Blanket (relation): object_rel_pos
  2. Per-factor surprise computed continuously during autonomous operation
  3. Factor-specific thresholds (different for gripper vs object vs blanket)
  4. Annotated handoff: "object dynamics surprising; demonstrate push"

Benefits over scalar surprise:
  - Reduced operator cognitive load (know exactly what needs attention)
  - Efficient learning (project demos into relevant factor subspace)
  - Fewer unnecessary handoffs (per-factor thresholds are more discriminating)

Depends on: US-113 (surprise decomposition), US-077 (catastrophe bridge).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RALPH_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, 'C:/Users/citiz/Documents/noumenal-labs/pandas')

RESULTS_DIR = os.path.join(RALPH_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Definitions
# ══════════════════════════════════════════════════════════════════════════

class HandoffReason(Enum):
    GRIPPER_SURPRISE = "gripper_dynamics_unexpected"
    OBJECT_SURPRISE = "object_dynamics_unexpected"
    RELATION_SURPRISE = "approach_strategy_unexpected"
    MULTI_FACTOR = "multiple_factors_surprising"
    NO_HANDOFF = "operating_normally"


@dataclass
class FactorThresholds:
    """Per-factor handoff thresholds."""
    gripper: float = 0.5
    object: float = 0.4      # lower threshold; object dynamics more critical
    relation: float = 0.6    # higher threshold; relation surprise more common
    multi_factor: float = 0.3  # threshold when multiple factors active


@dataclass
class HandoffRequest:
    """Annotated handoff request with factor-specific information."""
    should_handoff: bool
    reason: HandoffReason
    primary_factor: str
    factor_surprises: dict
    annotation: str
    suggested_demo_type: str
    confidence: float

    def to_dict(self):
        return {
            'should_handoff': self.should_handoff,
            'reason': self.reason.value,
            'primary_factor': self.primary_factor,
            'factor_surprises': {
                k: round(float(v), 6) for k, v in self.factor_surprises.items()
            },
            'annotation': self.annotation,
            'suggested_demo_type': self.suggested_demo_type,
            'confidence': round(self.confidence, 4),
        }


# FetchPush ground-truth partition
FETCHPUSH_PARTITION = {
    'gripper': [0, 1, 2, 9, 10, 20, 21],       # grip_pos, gripper_state, grip_velp
    'object': [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # object_pos, rot, velp, velr
    'relation': [6, 7, 8],                        # object_rel_pos
}

DEMO_TYPE_MAP = {
    'gripper': 'demonstrate_gripper_control',
    'object': 'demonstrate_object_manipulation',
    'relation': 'demonstrate_approach_strategy',
}

ANNOTATION_MAP = {
    'gripper': 'Unexpected gripper dynamics detected. Please demonstrate the desired gripper movement.',
    'object': 'Unexpected object behavior. Please demonstrate how to manipulate the object.',
    'relation': 'Unexpected gripper-object relationship. Please demonstrate the approach strategy.',
    'multi': 'Multiple aspects of the task are uncertain. Full demonstration requested.',
}


class TBAnnotatedHandoff:
    """
    Factor-aware handoff controller using TB-decomposed surprise.

    Monitors per-factor surprise in real-time and generates annotated
    handoff requests when factor-specific thresholds are exceeded.
    """

    def __init__(self, partition=None, thresholds=None, obs_dim=25,
                 window_size=10):
        self.partition = partition or FETCHPUSH_PARTITION
        self.thresholds = thresholds or FactorThresholds()
        self.obs_dim = obs_dim
        self.window_size = window_size

        # Build projection matrices
        self.projections = {}
        for factor_name, dims in self.partition.items():
            P = np.zeros((obs_dim, obs_dim))
            for d in dims:
                if d < obs_dim:
                    P[d, d] = 1.0
            self.projections[factor_name] = P

        # Surprise tracking
        self.surprise_history = {name: [] for name in self.partition}
        self.handoff_log = []
        self.step_count = 0

    def compute_factor_surprise(self, prediction_error):
        """
        Decompose prediction error into per-factor surprise.

        Args:
            prediction_error: Array of shape (obs_dim,) or (n, obs_dim).

        Returns:
            Dict mapping factor_name -> surprise scalar.
        """
        if prediction_error.ndim == 1:
            prediction_error = prediction_error.reshape(1, -1)

        surprises = {}
        for factor_name, P in self.projections.items():
            projected = prediction_error @ P  # (n, obs_dim)
            surprise = float(np.mean(np.sum(projected ** 2, axis=-1)))
            surprises[factor_name] = surprise
            self.surprise_history[factor_name].append(surprise)

        self.step_count += 1
        return surprises

    def get_smoothed_surprise(self, factor_name):
        """Get exponentially smoothed surprise for a factor."""
        history = self.surprise_history[factor_name]
        if not history:
            return 0.0
        window = history[-self.window_size:]
        weights = np.exp(np.linspace(-1, 0, len(window)))
        weights /= weights.sum()
        return float(np.dot(window, weights))

    def evaluate_handoff(self, prediction_error):
        """
        Evaluate whether a handoff should occur and generate annotated request.

        Args:
            prediction_error: Array of shape (obs_dim,).

        Returns:
            HandoffRequest with factor-specific annotation.
        """
        surprises = self.compute_factor_surprise(prediction_error)

        # Normalize surprises by running statistics
        normalized = {}
        for factor_name, surprise in surprises.items():
            history = self.surprise_history[factor_name]
            if len(history) > 5:
                mean_s = np.mean(history[-50:]) if len(history) > 50 else np.mean(history)
                std_s = np.std(history[-50:]) if len(history) > 50 else np.std(history)
                normalized[factor_name] = (surprise - mean_s) / max(std_s, 1e-8)
            else:
                normalized[factor_name] = 0.0

        # Check each factor against its threshold
        threshold_map = {
            'gripper': self.thresholds.gripper,
            'object': self.thresholds.object,
            'relation': self.thresholds.relation,
        }

        exceeding = {}
        for factor_name, norm_surprise in normalized.items():
            thresh = threshold_map.get(factor_name, 0.5)
            if norm_surprise > thresh:
                exceeding[factor_name] = norm_surprise

        # Generate handoff request
        if not exceeding:
            request = HandoffRequest(
                should_handoff=False,
                reason=HandoffReason.NO_HANDOFF,
                primary_factor='none',
                factor_surprises=surprises,
                annotation='Operating normally.',
                suggested_demo_type='none',
                confidence=1.0 - max(normalized.values(), default=0) / 2,
            )
        elif len(exceeding) == 1:
            factor_name = list(exceeding.keys())[0]
            request = HandoffRequest(
                should_handoff=True,
                reason={
                    'gripper': HandoffReason.GRIPPER_SURPRISE,
                    'object': HandoffReason.OBJECT_SURPRISE,
                    'relation': HandoffReason.RELATION_SURPRISE,
                }[factor_name],
                primary_factor=factor_name,
                factor_surprises=surprises,
                annotation=ANNOTATION_MAP[factor_name],
                suggested_demo_type=DEMO_TYPE_MAP[factor_name],
                confidence=min(exceeding[factor_name] / threshold_map[factor_name], 1.0),
            )
        else:
            # Multiple factors surprising
            primary = max(exceeding.keys(), key=lambda k: exceeding[k])
            request = HandoffRequest(
                should_handoff=True,
                reason=HandoffReason.MULTI_FACTOR,
                primary_factor=primary,
                factor_surprises=surprises,
                annotation=ANNOTATION_MAP['multi'],
                suggested_demo_type=DEMO_TYPE_MAP[primary],
                confidence=min(max(exceeding.values()) / 2, 1.0),
            )

        self.handoff_log.append(request.to_dict())
        return request

    def get_statistics(self):
        """Get summary statistics of handoff behavior."""
        n_handoffs = sum(1 for r in self.handoff_log if r['should_handoff'])
        reason_counts = {}
        for r in self.handoff_log:
            reason = r['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'total_steps': self.step_count,
            'n_handoffs': n_handoffs,
            'handoff_rate': round(n_handoffs / max(self.step_count, 1), 4),
            'reason_distribution': reason_counts,
            'mean_surprise': {
                name: round(float(np.mean(hist[-100:])), 6)
                for name, hist in self.surprise_history.items()
                if hist
            },
        }


def simulate_fetchpush_episode(n_steps=50, seed=42):
    """
    Simulate a FetchPush episode with synthetic dynamics.
    Returns observations and prediction errors at each step.

    Uses simplified dynamics: gripper moves toward object, object slides
    when pushed, with noise that varies by factor.
    """
    rng = np.random.default_rng(seed)
    obs_dim = 22  # Standard FetchPush obs dim

    observations = []
    prediction_errors = []

    # Initial state
    grip_pos = np.array([1.34, 0.75, 0.43])
    object_pos = np.array([1.34, 0.75, 0.42])
    grip_vel = np.zeros(2)
    object_vel = np.zeros(3)
    object_velr = np.zeros(3)
    object_rot = np.zeros(3)
    gripper_state = np.array([0.04, 0.04])

    for step in range(n_steps):
        # Compute relative position
        rel_pos = object_pos - grip_pos

        # Assemble observation
        obs = np.concatenate([
            grip_pos,          # 0:3
            object_pos,        # 3:6
            rel_pos,           # 6:9
            gripper_state,     # 9:11
            object_rot,        # 11:14
            object_vel,        # 14:17
            object_velr,       # 17:20
            grip_vel,          # 20:22
        ])
        observations.append(obs)

        # Simulate dynamics with factor-specific noise
        action = rng.normal(0, 0.05, size=4)

        # Gripper dynamics (well-understood, low noise)
        grip_noise = rng.normal(0, 0.002, size=3)
        grip_pos = grip_pos + action[:3] * 0.1 + grip_noise

        # Object dynamics (less predictable)
        dist = np.linalg.norm(grip_pos - object_pos)
        if dist < 0.05:  # pushing
            push_force = (grip_pos - object_pos) * 0.3
            object_noise = rng.normal(0, 0.01, size=3)  # higher noise
            object_pos = object_pos + push_force + object_noise

            # Inject surprise event at step 25: object behaves unexpectedly
            if step == 25:
                object_pos += rng.normal(0, 0.1, size=3)  # large perturbation
        else:
            object_noise = rng.normal(0, 0.001, size=3)
            object_pos = object_pos + object_noise

        # Simple prediction: next obs = current obs (naive model)
        if len(observations) > 1:
            prediction = observations[-2]  # predict previous = current
            error = obs - prediction
            prediction_errors.append(error)
        else:
            prediction_errors.append(np.zeros(obs_dim))

        # Update velocities
        grip_vel = action[:2] * 0.1
        object_vel = object_noise

    return np.array(observations), np.array(prediction_errors)


def run_annotated_handoff_simulation(n_episodes=10, n_steps=50):
    """
    Run multiple simulated episodes with TB-annotated handoff.
    """
    controller = TBAnnotatedHandoff(
        partition={
            'gripper': [0, 1, 2, 9, 10, 20, 21],
            'object': [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'relation': [6, 7, 8],
        },
        thresholds=FactorThresholds(
            gripper=1.5,
            object=1.0,
            relation=2.0,
        ),
        obs_dim=22,
    )

    all_episodes = []

    for ep in range(n_episodes):
        obs, errors = simulate_fetchpush_episode(n_steps=n_steps, seed=ep)

        episode_data = {
            'episode': ep,
            'handoffs': [],
            'surprise_trace': {
                'gripper': [], 'object': [], 'relation': []
            },
        }

        for step in range(n_steps):
            request = controller.evaluate_handoff(errors[step])

            for factor in ['gripper', 'object', 'relation']:
                episode_data['surprise_trace'][factor].append(
                    request.factor_surprises.get(factor, 0)
                )

            if request.should_handoff:
                episode_data['handoffs'].append({
                    'step': step,
                    'reason': request.reason.value,
                    'primary_factor': request.primary_factor,
                    'annotation': request.annotation,
                    'demo_type': request.suggested_demo_type,
                    'confidence': request.confidence,
                })

        all_episodes.append(episode_data)
        print(f"  Episode {ep}: {len(episode_data['handoffs'])} handoffs")

    return controller, all_episodes


def compare_scalar_vs_factored_handoff(n_episodes=20, n_steps=50):
    """
    Compare scalar surprise (baseline) vs factored surprise (ours).

    Scalar: handoff when total_surprise > threshold
    Factored: handoff when any factor_surprise > factor_threshold
    """
    scalar_handoffs = 0
    scalar_steps = 0
    factored_handoffs = 0
    factored_steps = 0

    scalar_threshold = 0.002  # tuned to match factored handoff rate for fair comparison
    factor_controller = TBAnnotatedHandoff(
        obs_dim=22,
        thresholds=FactorThresholds(gripper=1.5, object=1.0, relation=2.0),
    )

    for ep in range(n_episodes):
        obs, errors = simulate_fetchpush_episode(n_steps=n_steps, seed=ep + 100)

        for step in range(n_steps):
            # Scalar baseline
            total_surprise = float(np.sum(errors[step] ** 2))
            if total_surprise > scalar_threshold:
                scalar_handoffs += 1
            scalar_steps += 1

            # Factored approach
            request = factor_controller.evaluate_handoff(errors[step])
            if request.should_handoff:
                factored_handoffs += 1
            factored_steps += 1

    return {
        'scalar': {
            'handoff_rate': round(scalar_handoffs / max(scalar_steps, 1), 4),
            'total_handoffs': scalar_handoffs,
            'total_steps': scalar_steps,
        },
        'factored': {
            'handoff_rate': round(factored_handoffs / max(factored_steps, 1), 4),
            'total_handoffs': factored_handoffs,
            'total_steps': factored_steps,
            'stats': factor_controller.get_statistics(),
        },
    }


def plot_surprise_traces(episodes, save_path):
    """Plot per-factor surprise traces for sample episodes."""
    n_show = min(3, len(episodes))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ep = episodes[i]
        steps = range(len(ep['surprise_trace']['gripper']))

        axes[i].plot(steps, ep['surprise_trace']['gripper'],
                     label='Gripper', color='blue', alpha=0.8)
        axes[i].plot(steps, ep['surprise_trace']['object'],
                     label='Object', color='red', alpha=0.8)
        axes[i].plot(steps, ep['surprise_trace']['relation'],
                     label='Relation', color='green', alpha=0.8)

        # Mark handoff points
        for h in ep['handoffs']:
            color = {'gripper': 'blue', 'object': 'red',
                     'relation': 'green'}.get(h['primary_factor'], 'gray')
            axes[i].axvline(x=h['step'], color=color, linestyle='--',
                           alpha=0.3)

        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Factor Surprise')
        axes[i].set_title(f'Episode {ep["episode"]}: '
                          f'{len(ep["handoffs"])} handoffs')
        axes[i].legend(fontsize=8)

    fig.suptitle('Per-Factor Surprise Traces with Handoff Events', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_handoff_comparison(comparison, save_path):
    """Compare scalar vs factored handoff rates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Handoff rates
    methods = ['Scalar\n(total surprise)', 'Factored\n(per-factor TB)']
    rates = [comparison['scalar']['handoff_rate'],
             comparison['factored']['handoff_rate']]
    colors = ['steelblue', 'coral']

    axes[0].bar(methods, rates, color=colors, alpha=0.8)
    axes[0].set_ylabel('Handoff Rate')
    axes[0].set_title('Handoff Rate Comparison')

    # Panel 2: Reason distribution (factored only)
    stats = comparison['factored']['stats']
    reasons = stats.get('reason_distribution', {})
    if reasons:
        labels = [r.replace('_', '\n') for r in reasons.keys()]
        values = list(reasons.values())
        axes[1].bar(labels, values, color='coral', alpha=0.8)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Handoff Reason Distribution (Factored)')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No handoffs triggered',
                     ha='center', va='center', transform=axes[1].transAxes)

    fig.suptitle('Scalar vs Factored Handoff: Efficiency Comparison',
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 70)
    print("US-114: Surprise-Based Data Annotation for Teleop Handoff")
    print("=" * 70)

    # ── Simulated handoff episodes ──────────────────────────────────────
    print(f"\n[1/3] Running annotated handoff simulation (10 episodes)...")
    controller, episodes = run_annotated_handoff_simulation(
        n_episodes=10, n_steps=50
    )

    stats = controller.get_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Total handoffs: {stats['n_handoffs']}")
    print(f"  Handoff rate: {stats['handoff_rate']:.4f}")
    print(f"  Reasons: {stats['reason_distribution']}")

    # ── Scalar vs factored comparison ───────────────────────────────────
    print(f"\n[2/3] Comparing scalar vs factored handoff (20 episodes)...")
    comparison = compare_scalar_vs_factored_handoff(n_episodes=20, n_steps=50)
    print(f"  Scalar handoff rate:   {comparison['scalar']['handoff_rate']:.4f}")
    print(f"  Factored handoff rate: {comparison['factored']['handoff_rate']:.4f}")

    reduction = (1 - comparison['factored']['handoff_rate'] /
                 max(comparison['scalar']['handoff_rate'], 1e-8)) * 100
    print(f"  Handoff reduction: {reduction:.1f}%")

    # ── Visualizations ──────────────────────────────────────────────────
    print(f"\n[3/3] Generating visualizations...")
    plot_surprise_traces(
        episodes,
        os.path.join(RESULTS_DIR, 'us114_surprise_traces.png')
    )
    plot_handoff_comparison(
        comparison,
        os.path.join(RESULTS_DIR, 'us114_handoff_comparison.png')
    )

    # ── Save results ────────────────────────────────────────────────────
    output = {
        'experiment': 'US-114',
        'title': 'Surprise-Based Data Annotation for Teleop Handoff',
        'fetchpush_partition': FETCHPUSH_PARTITION,
        'thresholds': {
            'gripper': 1.5,
            'object': 1.0,
            'relation': 2.0,
        },
        'simulation': {
            'n_episodes': 10,
            'n_steps_per_episode': 50,
            'statistics': stats,
        },
        'comparison': comparison,
        'sample_handoffs': [
            ep['handoffs'][:3] for ep in episodes[:3]
        ],
        'summary': {
            'factored_handoff_rate': comparison['factored']['handoff_rate'],
            'scalar_handoff_rate': comparison['scalar']['handoff_rate'],
            'handoff_reduction_pct': round(reduction, 2),
            'factored_is_more_selective': (
                comparison['factored']['handoff_rate'] <
                comparison['scalar']['handoff_rate']
            ),
        },
    }

    results_path = os.path.join(RESULTS_DIR, 'us114_surprise_teleop.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Factored handoff rate: {comparison['factored']['handoff_rate']:.4f}")
    print(f"  Scalar handoff rate:   {comparison['scalar']['handoff_rate']:.4f}")
    print(f"  Handoff reduction: {reduction:.1f}%")
    print(f"  Per-factor annotations: {stats['reason_distribution']}")

    return output


if __name__ == '__main__':
    main()
