"""
US-084: TB-Guided Learned Symbolic Planner -- Validation and Ablation
=====================================================================

Tests that the TB-guided planner (panda/learned_planner.py) produces
the same phase decomposition as the hardcoded symbolic planner on
FetchPush, a single-phase plan on FetchReach, and compares ablation
performance across three conditions: hardcoded, TB-guided, and flat
(no planner).

Tests are synthetic (no MuJoCo required): we simulate observation
sequences that exercise the planner logic and verify subgoal ordering,
phase transitions, and PlanningObjective contents.

Results are saved to ralph/results/ as JSON and comparison plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
from datetime import datetime

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)
PANDAS_DIR = 'C:/Users/citiz/Documents/noumenal-labs/pandas'

sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)
sys.path.insert(0, PANDAS_DIR)

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

from panda.learned_planner import (
    TBPartition,
    TBGuidedPlanner,
    TBGuidedPlannerConfig,
    make_tb_guided_planner,
    partition_from_tb_result,
    make_fetchpush_ground_truth_partition,
    make_fetchreach_partition,
    infer_causal_order,
    classify_objects,
)
from panda.symbolic_planner import (
    FetchPushSymbolicPlanner,
    SymbolicPlannerConfig,
    SymbolicDecision,
)
from panda.planner import PlanningObjective


RESULTS_DIR = os.path.join(RALPH_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Test helpers
# ======================================================================

def make_obs_vec(
    grip_pos: np.ndarray,
    obj_pos: np.ndarray,
    obs_dim: int = 25,
) -> np.ndarray:
    """Build a FetchPush-style observation vector for testing."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    obs[0:3] = grip_pos
    obs[3:6] = obj_pos
    obs[6:9] = obj_pos - grip_pos  # relative position
    obs[9:11] = [0.04, 0.04]       # gripper state (open)
    # Everything else stays zero (rotations, velocities)
    return obs


def simulate_push_episode(
    planner,
    n_steps: int = 80,
    grip_start: np.ndarray = None,
    obj_start: np.ndarray = None,
    desired_goal: np.ndarray = None,
) -> dict:
    """Simulate a FetchPush episode with synthetic dynamics.

    The gripper moves toward its current subgoal target at a fixed
    speed.  When close enough, pushes the object toward the goal.
    Returns a log of phases, distances, and subgoals.
    """
    if grip_start is None:
        grip_start = np.array([1.1, 0.7, 0.45], dtype=np.float32)
    if obj_start is None:
        obj_start = np.array([1.3, 0.7, 0.42], dtype=np.float32)
    if desired_goal is None:
        desired_goal = np.array([1.5, 0.8, 0.42], dtype=np.float32)

    grip_pos = grip_start.copy()
    obj_pos = obj_start.copy()
    speed = 0.01

    planner.reset()
    log = {
        'phases': [],
        'grip_positions': [],
        'obj_positions': [],
        'subgoal_distances': [],
        'object_goal_distances': [],
        'objective_names': [],
    }

    for step in range(n_steps):
        obs_vec = make_obs_vec(grip_pos, obj_pos)
        achieved_goal = obj_pos.copy()

        decision = planner.decide(obs_vec, achieved_goal, desired_goal)

        log['phases'].append(decision.status.phase_index)
        log['grip_positions'].append(grip_pos.copy())
        log['obj_positions'].append(obj_pos.copy())
        log['subgoal_distances'].append(decision.status.subgoal_distance)
        log['object_goal_distances'].append(decision.status.object_goal_distance)
        log['objective_names'].append(decision.objective.name)

        if decision.status.done:
            break

        # Synthetic dynamics: move gripper toward its target
        if decision.objective.gripper_target is not None:
            target = decision.objective.gripper_target
            delta = target - grip_pos
            dist = float(np.linalg.norm(delta))
            if dist > speed:
                grip_pos = grip_pos + (delta / dist) * speed
            else:
                grip_pos = target.copy()

        # If gripper is touching the object, push it
        grip_obj_dist = float(np.linalg.norm(grip_pos - obj_pos))
        if grip_obj_dist < 0.05:
            push_dir = desired_goal - obj_pos
            push_norm = float(np.linalg.norm(push_dir))
            if push_norm > 1e-6:
                obj_pos = obj_pos + (push_dir / push_norm) * speed * 0.5

    return log


def simulate_reach_episode(
    planner,
    n_steps: int = 60,
    grip_start: np.ndarray = None,
    desired_goal: np.ndarray = None,
) -> dict:
    """Simulate a FetchReach episode (gripper only, no object)."""
    if grip_start is None:
        grip_start = np.array([1.1, 0.7, 0.45], dtype=np.float32)
    if desired_goal is None:
        desired_goal = np.array([1.3, 0.8, 0.50], dtype=np.float32)

    grip_pos = grip_start.copy()
    speed = 0.01

    planner.reset()
    log = {
        'phases': [],
        'grip_positions': [],
        'subgoal_distances': [],
        'objective_names': [],
    }

    for step in range(n_steps):
        obs_dim = 10
        obs_vec = np.zeros(obs_dim, dtype=np.float32)
        obs_vec[0:3] = grip_pos
        achieved_goal = grip_pos.copy()

        decision = planner.decide(obs_vec, achieved_goal, desired_goal)

        log['phases'].append(decision.status.phase_index)
        log['grip_positions'].append(grip_pos.copy())
        log['subgoal_distances'].append(decision.status.subgoal_distance)
        log['objective_names'].append(decision.objective.name)

        if decision.status.done:
            break

        # Move gripper toward target
        if decision.objective.gripper_target is not None:
            target = decision.objective.gripper_target
            delta = target - grip_pos
            dist = float(np.linalg.norm(delta))
            if dist > speed:
                grip_pos = grip_pos + (delta / dist) * speed
            else:
                grip_pos = target.copy()

    return log


# ======================================================================
# Test 1: FetchPush decomposition equivalence
# ======================================================================

def test_fetchpush_decomposition():
    """Verify TB-guided planner produces the same phases as hardcoded."""
    print("=" * 70)
    print("Test 1: FetchPush decomposition equivalence")
    print("=" * 70)

    # Ground-truth partition (what TB should discover)
    partition = make_fetchpush_ground_truth_partition(obs_dim=25)

    # TB-guided planner
    tb_planner = make_tb_guided_planner(
        partition=partition,
        gripper_object_id=0,
        gripper_pos_indices=(0, 1, 2),
        default_goal_threshold=0.05,
    )

    # Hardcoded planner (for comparison)
    hc_cfg = SymbolicPlannerConfig(
        task="push",
        gripper_indices=(0, 1, 2),
    )
    hc_planner = FetchPushSymbolicPlanner(hc_cfg, default_goal_threshold=0.05)

    # Test scenario: gripper far from object, object far from goal
    grip_pos = np.array([1.1, 0.7, 0.45], dtype=np.float32)
    obj_pos = np.array([1.3, 0.7, 0.42], dtype=np.float32)
    desired_goal = np.array([1.5, 0.8, 0.42], dtype=np.float32)

    obs_vec = make_obs_vec(grip_pos, obj_pos)
    achieved_goal = obj_pos.copy()

    # Both planners should start in approach phase
    tb_decision = tb_planner.decide(obs_vec, achieved_goal, desired_goal)
    hc_decision = hc_planner.decide(obs_vec, achieved_goal, desired_goal)

    results = {}

    # Check phase structure
    assert tb_planner.n_phases == 2, f"Expected 2 phases, got {tb_planner.n_phases}"
    assert tb_planner.has_target_object, "Expected target object present"
    results['n_phases'] = tb_planner.n_phases
    results['has_target_object'] = True

    # Check phase 1 (approach): both should focus on gripper positioning
    assert tb_decision.status.phase_index == 1, (
        f"TB planner phase should be 1, got {tb_decision.status.phase_index}"
    )
    assert hc_decision.status.phase_index == 1, (
        f"HC planner phase should be 1, got {hc_decision.status.phase_index}"
    )
    print(f"  Phase 1 match: TB={tb_decision.status.phase_index}, "
          f"HC={hc_decision.status.phase_index}")

    # Both should have low object-goal weight in approach phase
    assert tb_decision.objective.object_goal_weight < 0.5, (
        f"Approach phase should have low object weight, "
        f"got {tb_decision.objective.object_goal_weight}"
    )
    assert tb_decision.objective.gripper_target_weight > 0.5, (
        f"Approach phase should have high gripper weight, "
        f"got {tb_decision.objective.gripper_target_weight}"
    )
    print(f"  Approach weights: TB obj={tb_decision.objective.object_goal_weight:.2f}, "
          f"grip={tb_decision.objective.gripper_target_weight:.2f}")
    print(f"  Approach weights: HC obj={hc_decision.objective.object_goal_weight:.2f}, "
          f"grip={hc_decision.objective.gripper_target_weight:.2f}")

    # Approach targets should be similar (behind object along push direction)
    tb_approach = tb_decision.objective.gripper_target
    hc_approach = hc_decision.objective.gripper_target
    approach_diff = float(np.linalg.norm(tb_approach - hc_approach))
    print(f"  Approach target difference: {approach_diff:.6f}")
    assert approach_diff < 0.01, (
        f"Approach targets differ by {approach_diff:.6f} (should be < 0.01)"
    )
    results['approach_target_difference'] = approach_diff

    # Simulate full episodes with both planners
    tb_log = simulate_push_episode(tb_planner)
    hc_log = simulate_push_episode(hc_planner)

    # Both should transition to push phase
    tb_phases = set(tb_log['phases'])
    hc_phases = set(hc_log['phases'])
    print(f"  TB phases observed: {sorted(tb_phases)}")
    print(f"  HC phases observed: {sorted(hc_phases)}")

    # Both should have at least 2 distinct phases
    assert len(tb_phases) >= 2, f"TB planner should show 2 phases, got {tb_phases}"
    assert len(hc_phases) >= 2, f"HC planner should show 2 phases, got {hc_phases}"
    results['tb_phases'] = sorted(list(tb_phases))
    results['hc_phases'] = sorted(list(hc_phases))

    # Phase transition should happen at similar steps
    tb_transition = None
    for i, p in enumerate(tb_log['phases']):
        if p == 2:
            tb_transition = i
            break
    hc_transition = None
    for i, p in enumerate(hc_log['phases']):
        if p == 2:
            hc_transition = i
            break

    if tb_transition is not None and hc_transition is not None:
        transition_diff = abs(tb_transition - hc_transition)
        print(f"  Phase transition: TB at step {tb_transition}, "
              f"HC at step {hc_transition} (diff={transition_diff})")
        results['tb_transition_step'] = tb_transition
        results['hc_transition_step'] = hc_transition
        results['transition_step_difference'] = transition_diff
    else:
        print(f"  Phase transition: TB={tb_transition}, HC={hc_transition}")
        results['tb_transition_step'] = tb_transition
        results['hc_transition_step'] = hc_transition

    # Final object-goal distance should be similar
    tb_final_dist = tb_log['object_goal_distances'][-1]
    hc_final_dist = hc_log['object_goal_distances'][-1]
    print(f"  Final object-goal distance: TB={tb_final_dist:.4f}, "
          f"HC={hc_final_dist:.4f}")
    results['tb_final_obj_dist'] = tb_final_dist
    results['hc_final_obj_dist'] = hc_final_dist

    print("  PASSED: TB-guided planner produces same decomposition as hardcoded")
    results['test_passed'] = True
    return results


# ======================================================================
# Test 2: FetchReach single-phase plan
# ======================================================================

def test_fetchreach_single_phase():
    """Verify TB-guided planner produces single-phase plan for FetchReach."""
    print("\n" + "=" * 70)
    print("Test 2: FetchReach single-phase plan")
    print("=" * 70)

    partition = make_fetchreach_partition(obs_dim=10)

    tb_planner = make_tb_guided_planner(
        partition=partition,
        gripper_object_id=0,
        gripper_pos_indices=(0, 1, 2),
        default_goal_threshold=0.05,
    )

    results = {}

    # Should have 1 phase only
    assert tb_planner.n_phases == 1, f"Expected 1 phase, got {tb_planner.n_phases}"
    assert not tb_planner.has_target_object, "Should have no target object"
    results['n_phases'] = tb_planner.n_phases
    results['has_target_object'] = False
    print(f"  Phases: {tb_planner.n_phases} (expected 1)")
    print(f"  Has target object: {tb_planner.has_target_object} (expected False)")

    # Simulate episode
    log = simulate_reach_episode(tb_planner)

    # All steps should be in the same phase
    phases = set(log['phases'])
    assert len(phases) == 1, f"Expected single phase, got {phases}"
    print(f"  Phases observed: {sorted(phases)} (expected single phase)")
    results['phases_observed'] = sorted(list(phases))

    # Objective should be reach-type
    assert all('reach' in name for name in log['objective_names']), (
        f"All objectives should be reach-type, got {set(log['objective_names'])}"
    )
    print(f"  Objective type: {log['objective_names'][0]}")
    results['objective_type'] = log['objective_names'][0]

    # Should converge toward goal
    first_dist = log['subgoal_distances'][0]
    last_dist = log['subgoal_distances'][-1]
    print(f"  Distance: {first_dist:.4f} -> {last_dist:.4f}")
    assert last_dist < first_dist, "Should move closer to goal"
    results['initial_distance'] = first_dist
    results['final_distance'] = last_dist

    print("  PASSED: Single-phase reach plan")
    results['test_passed'] = True
    return results


# ======================================================================
# Test 3: Causal ordering inference
# ======================================================================

def test_causal_ordering():
    """Test that causal ordering from coupling matrix is correct."""
    print("\n" + "=" * 70)
    print("Test 3: Causal ordering from coupling matrix")
    print("=" * 70)

    results = {}

    # Build a synthetic coupling matrix where Object 0 (gripper) has
    # stronger coupling to blanket than Object 1 (manipulated object)
    obs_dim = 25
    C = np.random.rand(obs_dim, obs_dim).astype(np.float32) * 0.1
    C = (C + C.T) / 2  # symmetrize

    gripper_vars = [0, 1, 2, 9, 10, 20, 21]
    object_vars = [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    blanket_vars = [6, 7, 8]

    # Make gripper->blanket coupling stronger
    for i in gripper_vars:
        for j in blanket_vars:
            C[i, j] = 2.0
            C[j, i] = 2.0

    # Object->blanket coupling weaker
    for i in object_vars:
        for j in blanket_vars:
            C[i, j] = 0.5
            C[j, i] = 0.5

    partition = TBPartition(
        objects={0: gripper_vars, 1: object_vars},
        blanket=blanket_vars,
        coupling_matrix=C,
    )

    order = infer_causal_order(partition)
    print(f"  Causal order (should be [0, 1]): {order}")
    assert order == [0, 1], f"Expected [0, 1], got {order}"
    results['causal_order'] = order

    gripper_id, target_id = classify_objects(partition)
    print(f"  Gripper: {gripper_id}, Target: {target_id}")
    assert gripper_id == 0, f"Expected gripper=0, got {gripper_id}"
    assert target_id == 1, f"Expected target=1, got {target_id}"
    results['gripper_id'] = gripper_id
    results['target_id'] = target_id

    # Test reversed coupling: make object->blanket stronger
    C2 = C.copy()
    for i in gripper_vars:
        for j in blanket_vars:
            C2[i, j] = 0.3
            C2[j, i] = 0.3
    for i in object_vars:
        for j in blanket_vars:
            C2[i, j] = 3.0
            C2[j, i] = 3.0

    partition2 = TBPartition(
        objects={0: gripper_vars, 1: object_vars},
        blanket=blanket_vars,
        coupling_matrix=C2,
    )
    order2 = infer_causal_order(partition2)
    print(f"  Reversed coupling order (should be [1, 0]): {order2}")
    assert order2 == [1, 0], f"Expected [1, 0], got {order2}"
    results['reversed_order'] = order2

    # Test without coupling matrix: falls back to size heuristic
    partition3 = TBPartition(
        objects={0: gripper_vars, 1: object_vars},
        blanket=blanket_vars,
    )
    order3 = infer_causal_order(partition3)
    print(f"  No coupling (size heuristic, should be [0, 1]): {order3}")
    # Gripper has 7 vars, object has 12; smaller = more upstream
    assert order3[0] == 0, f"Expected gripper (smaller set) first, got {order3}"
    results['size_heuristic_order'] = order3

    print("  PASSED: Causal ordering correct in all cases")
    results['test_passed'] = True
    return results


# ======================================================================
# Test 4: partition_from_tb_result
# ======================================================================

def test_partition_from_tb_result():
    """Test conversion from TB labels to TBPartition."""
    print("\n" + "=" * 70)
    print("Test 4: partition_from_tb_result conversion")
    print("=" * 70)

    results = {}

    # Simulate TB output labels: -1 = blanket, 0 = gripper, 1 = object
    labels = np.array([
        0, 0, 0,     # grip_pos -> object 0
        1, 1, 1,     # obj_pos -> object 1
        -1, -1, -1,  # rel_pos -> blanket
        0, 0,        # gripper_state -> object 0
        1, 1, 1,     # obj_rot -> object 1
        1, 1, 1,     # obj_velp -> object 1
        1, 1, 1,     # obj_velr -> object 1
        0, 0,        # grip_velp -> object 0
        -1, -1, -1,  # extras -> blanket
    ])

    partition = partition_from_tb_result(labels)

    assert 0 in partition.objects, "Object 0 should be present"
    assert 1 in partition.objects, "Object 1 should be present"
    assert len(partition.blanket) == 6, f"Expected 6 blanket vars, got {len(partition.blanket)}"
    assert len(partition.objects[0]) == 7, f"Expected 7 gripper vars, got {len(partition.objects[0])}"
    assert len(partition.objects[1]) == 12, f"Expected 12 object vars, got {len(partition.objects[1])}"

    results['n_blanket'] = len(partition.blanket)
    results['n_object_0'] = len(partition.objects[0])
    results['n_object_1'] = len(partition.objects[1])
    print(f"  Object 0: {len(partition.objects[0])} vars, "
          f"Object 1: {len(partition.objects[1])} vars, "
          f"Blanket: {len(partition.blanket)} vars")

    # Verify it can make a planner
    planner = make_tb_guided_planner(partition)
    assert planner.n_phases == 2
    print(f"  Planner phases: {planner.n_phases}")
    results['planner_phases'] = planner.n_phases

    print("  PASSED: TB result conversion")
    results['test_passed'] = True
    return results


# ======================================================================
# Test 5: Ablation comparison (synthetic)
# ======================================================================

def run_ablation_comparison(n_episodes: int = 50):
    """Compare task performance: hardcoded vs TB-guided vs flat CEM.

    This is a synthetic ablation that measures how quickly each planner
    strategy reduces the object-goal distance.  Instead of running real
    MuJoCo + CEM (which requires GPU and trained models), we simulate
    planner-guided gripper trajectories.

    Conditions:
      1. Hardcoded: FetchPushSymbolicPlanner (approach then push)
      2. TB-guided: TBGuidedPlanner with ground-truth partition
      3. Flat: no planner (gripper moves directly toward object goal)
    """
    print("\n" + "=" * 70)
    print(f"Test 5: Ablation comparison ({n_episodes} episodes)")
    print("=" * 70)

    max_steps = 80
    goal_threshold = 0.05
    speed = 0.01

    # Build planners
    partition = make_fetchpush_ground_truth_partition(obs_dim=25)
    tb_planner = make_tb_guided_planner(
        partition=partition,
        gripper_object_id=0,
        gripper_pos_indices=(0, 1, 2),
        default_goal_threshold=goal_threshold,
    )
    hc_cfg = SymbolicPlannerConfig(task="push", gripper_indices=(0, 1, 2))
    hc_planner = FetchPushSymbolicPlanner(hc_cfg, default_goal_threshold=goal_threshold)

    rng = np.random.default_rng(42)

    conditions = {
        'hardcoded': {'successes': 0, 'final_dists': [], 'steps_to_done': []},
        'tb_guided': {'successes': 0, 'final_dists': [], 'steps_to_done': []},
        'flat_cem':  {'successes': 0, 'final_dists': [], 'steps_to_done': []},
    }

    for ep in range(n_episodes):
        # Random initial positions
        grip_start = np.array([
            1.05 + rng.uniform(0, 0.15),
            0.65 + rng.uniform(0, 0.1),
            0.42 + rng.uniform(0, 0.06),
        ], dtype=np.float32)
        obj_start = np.array([
            1.25 + rng.uniform(0, 0.1),
            0.70 + rng.uniform(0, 0.1),
            0.42,
        ], dtype=np.float32)
        desired_goal = np.array([
            1.35 + rng.uniform(0, 0.15),
            0.80 + rng.uniform(0, 0.1),
            0.42,
        ], dtype=np.float32)

        for cond_name, planner_or_none in [
            ('hardcoded', hc_planner),
            ('tb_guided', tb_planner),
            ('flat_cem', None),
        ]:
            grip_pos = grip_start.copy()
            obj_pos = obj_start.copy()

            if planner_or_none is not None:
                planner_or_none.reset()

            done = False
            for step in range(max_steps):
                obs_vec = make_obs_vec(grip_pos, obj_pos)
                achieved_goal = obj_pos.copy()

                if planner_or_none is not None:
                    decision = planner_or_none.decide(
                        obs_vec, achieved_goal, desired_goal
                    )
                    target = decision.objective.gripper_target
                    if target is None:
                        target = desired_goal.copy()
                else:
                    # Flat CEM: gripper moves directly toward desired goal
                    target = desired_goal.copy()

                # Move gripper
                delta = target - grip_pos
                dist = float(np.linalg.norm(delta))
                if dist > speed:
                    grip_pos = grip_pos + (delta / dist) * speed
                else:
                    grip_pos = target.copy()

                # Push physics
                grip_obj_dist = float(np.linalg.norm(grip_pos - obj_pos))
                if grip_obj_dist < 0.05:
                    push_dir = desired_goal - obj_pos
                    push_norm = float(np.linalg.norm(push_dir))
                    if push_norm > 1e-6:
                        obj_pos = obj_pos + (push_dir / push_norm) * speed * 0.5

                obj_goal_dist = float(np.linalg.norm(obj_pos - desired_goal))
                if obj_goal_dist <= goal_threshold:
                    done = True
                    conditions[cond_name]['steps_to_done'].append(step + 1)
                    break

            final_dist = float(np.linalg.norm(obj_pos - desired_goal))
            conditions[cond_name]['final_dists'].append(final_dist)
            if done:
                conditions[cond_name]['successes'] += 1

    # Summarize
    results = {}
    for cond_name, data in conditions.items():
        success_rate = data['successes'] / n_episodes
        mean_final_dist = float(np.mean(data['final_dists']))
        mean_steps = (
            float(np.mean(data['steps_to_done']))
            if data['steps_to_done'] else float('nan')
        )
        results[cond_name] = {
            'success_rate': success_rate,
            'mean_final_distance': mean_final_dist,
            'std_final_distance': float(np.std(data['final_dists'])),
            'mean_steps_to_done': mean_steps,
            'n_successes': data['successes'],
            'n_episodes': n_episodes,
        }
        print(f"  {cond_name:12s}: success={success_rate:.2%}, "
              f"mean_dist={mean_final_dist:.4f}, "
              f"mean_steps={mean_steps:.1f}")

    return results, conditions


# ======================================================================
# Visualization
# ======================================================================

def plot_ablation_comparison(
    conditions: dict,
    n_episodes: int,
) -> plt.Figure:
    """Create comparison bar chart and distance distribution plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cond_names = ['hardcoded', 'tb_guided', 'flat_cem']
    display_names = ['Hardcoded\n(symbolic)', 'TB-Guided\n(learned)', 'Flat CEM\n(no planner)']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    # Bar 1: Success rate
    ax = axes[0]
    rates = [conditions[c]['successes'] / n_episodes for c in cond_names]
    ax.bar(display_names, rates, color=colors, alpha=0.8)
    ax.set_ylabel('Success Rate')
    ax.set_title('Task Success Rate')
    ax.set_ylim(0, 1.1)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f'{r:.0%}', ha='center', fontsize=11)

    # Bar 2: Mean final distance
    ax = axes[1]
    dists = [float(np.mean(conditions[c]['final_dists'])) for c in cond_names]
    stds = [float(np.std(conditions[c]['final_dists'])) for c in cond_names]
    ax.bar(display_names, dists, yerr=stds, color=colors, alpha=0.8, capsize=4)
    ax.set_ylabel('Final Object-Goal Distance')
    ax.set_title('Final Distance (lower = better)')
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='threshold')
    ax.legend(fontsize=9)

    # Box plot 3: Steps to completion
    ax = axes[2]
    step_data = []
    labels_for_box = []
    for c, dn in zip(cond_names, display_names):
        steps = conditions[c]['steps_to_done']
        if steps:
            step_data.append(steps)
            labels_for_box.append(dn)
    if step_data:
        bp = ax.boxplot(step_data, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(step_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel('Steps to Completion')
    ax.set_title('Efficiency (successful episodes)')

    fig.suptitle('US-084: TB-Guided Planner Ablation', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_phase_comparison(
    tb_log: dict,
    hc_log: dict,
) -> plt.Figure:
    """Plot phase trajectories for TB-guided vs hardcoded planner."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (log, label) in enumerate([(hc_log, 'Hardcoded'), (tb_log, 'TB-Guided')]):
        # Phase over time
        ax = axes[0, idx]
        ax.plot(log['phases'], 'o-', markersize=3, alpha=0.7)
        ax.set_ylabel('Phase Index')
        ax.set_xlabel('Step')
        ax.set_title(f'{label}: Phase Sequence')
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Approach', 'Push'])

        # Distances over time
        ax = axes[1, idx]
        ax.plot(log['subgoal_distances'], label='Subgoal dist', alpha=0.8)
        ax.plot(log['object_goal_distances'], label='Object-goal dist', alpha=0.8)
        ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='threshold')
        ax.set_ylabel('Distance')
        ax.set_xlabel('Step')
        ax.set_title(f'{label}: Distances')
        ax.legend(fontsize=8)

    fig.suptitle('US-084: Phase Decomposition Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================

def main():
    print("US-084: TB-Guided Learned Symbolic Planner -- Validation")
    print("=" * 70)

    all_results = {}

    # Test 1: FetchPush decomposition
    push_results = test_fetchpush_decomposition()
    all_results['fetchpush_decomposition'] = push_results

    # Test 2: FetchReach single-phase
    reach_results = test_fetchreach_single_phase()
    all_results['fetchreach_single_phase'] = reach_results

    # Test 3: Causal ordering
    causal_results = test_causal_ordering()
    all_results['causal_ordering'] = causal_results

    # Test 4: partition_from_tb_result
    conversion_results = test_partition_from_tb_result()
    all_results['partition_conversion'] = conversion_results

    # Test 5: Ablation comparison
    ablation_results, ablation_conditions = run_ablation_comparison(n_episodes=50)
    all_results['ablation'] = ablation_results

    # Generate phase comparison plot
    partition = make_fetchpush_ground_truth_partition(obs_dim=25)
    tb_planner = make_tb_guided_planner(
        partition=partition,
        gripper_object_id=0,
        gripper_pos_indices=(0, 1, 2),
        default_goal_threshold=0.05,
    )
    hc_cfg = SymbolicPlannerConfig(task="push", gripper_indices=(0, 1, 2))
    hc_planner = FetchPushSymbolicPlanner(hc_cfg, default_goal_threshold=0.05)

    tb_log = simulate_push_episode(tb_planner)
    hc_log = simulate_push_episode(hc_planner)

    fig_phases = plot_phase_comparison(tb_log, hc_log)
    save_figure(fig_phases, 'tb_learned_planner_phase_comparison', 'tb_learned_planner')
    plt.close(fig_phases)

    fig_ablation = plot_ablation_comparison(ablation_conditions, n_episodes=50)
    save_figure(fig_ablation, 'tb_learned_planner_ablation', 'tb_learned_planner')
    plt.close(fig_ablation)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = all(
        all_results[k].get('test_passed', True)
        for k in ['fetchpush_decomposition', 'fetchreach_single_phase',
                   'causal_ordering', 'partition_conversion']
    )

    print(f"  FetchPush decomposition: {'PASS' if push_results['test_passed'] else 'FAIL'}")
    print(f"  FetchReach single-phase: {'PASS' if reach_results['test_passed'] else 'FAIL'}")
    print(f"  Causal ordering:         {'PASS' if causal_results['test_passed'] else 'FAIL'}")
    print(f"  Partition conversion:    {'PASS' if conversion_results['test_passed'] else 'FAIL'}")
    print(f"  Ablation (50 episodes):")
    for cond, data in ablation_results.items():
        print(f"    {cond:12s}: {data['success_rate']:.0%} success, "
              f"dist={data['mean_final_distance']:.4f}")
    print(f"\n  All tests passed: {all_passed}")

    # Save results
    all_results['all_tests_passed'] = all_passed
    save_results(
        experiment_name='tb_learned_planner',
        metrics=all_results,
        config={
            'n_ablation_episodes': 50,
            'goal_threshold': 0.05,
            'approach_offset': 0.08,
            'obs_dim': 25,
        },
        notes=(
            'US-084: TB-guided learned planner validation. Tests that '
            'TBGuidedPlanner produces the same decomposition as the '
            'hardcoded FetchPushSymbolicPlanner on FetchPush (approach '
            'then push), a single-phase plan on FetchReach, and '
            'compares ablation performance (hardcoded vs TB-guided vs '
            'flat CEM) over 50 synthetic episodes.'
        ),
    )
    print(f"\nResults saved to {RESULTS_DIR}/")

    return all_passed


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
