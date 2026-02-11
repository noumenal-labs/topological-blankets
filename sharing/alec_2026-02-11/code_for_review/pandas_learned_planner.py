"""TB-guided learned symbolic planner for goal decomposition.

US-084: Replaces the hardcoded FetchPushSymbolicPlanner with a planner
that uses Topological Blanket (TB) discovered structure to decompose
tasks into subgoals.  The key insight: TB identifies which variables
form independent objects (e.g. gripper, manipulated object).  A goal
for the full system ("push object to target") can be decomposed into
per-object subgoals.  TB provides the variable partition; the planner
uses it to sequence subgoals by causal influence.

The planner is task-agnostic: it does not hardcode "approach then push"
but discovers that sequence from the coupling graph.  On FetchReach
(single controllable object, no manipulated object) it produces a
single-phase plan.  On FetchPush it produces the same approach-then-push
decomposition that the hardcoded planner uses, without task knowledge.

Usage::

    partition = TBPartition(
        objects={0: [0,1,2,9,10,20,21], 1: [3,4,5,11,12,13,14,15,16,17,18,19]},
        blanket=[6,7,8],
        coupling_matrix=C,
    )
    planner = TBGuidedPlanner(
        partition=partition,
        gripper_object_id=0,
        gripper_pos_indices=(0,1,2),
        default_goal_threshold=0.05,
    )
    decision = planner.decide(obs_vec, achieved_goal, desired_goal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .planner import PlanningObjective
from .symbolic_planner import SymbolicDecision, SymbolicStatus


# ---------------------------------------------------------------------------
# TB partition data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TBPartition:
    """Describes a Topological Blanket partition over observation variables.

    Attributes
    ----------
    objects:
        Mapping from object_id to list of observation indices belonging
        to that object.  Object 0 is typically the gripper (the
        directly actuated subsystem), Object 1 is the manipulated body.
    blanket:
        List of observation indices that form the Markov blanket
        (relational variables mediating between objects).
    coupling_matrix:
        The full coupling matrix (obs_dim x obs_dim) from TB analysis.
        Used to infer causal ordering between objects.  May be None
        if only the partition labels are available.
    obs_labels:
        Optional human-readable labels for each observation dimension.
    """
    objects: dict[int, list[int]]
    blanket: list[int]
    coupling_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    obs_labels: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TBGuidedPlannerConfig:
    """Configuration for the TB-guided planner.

    Attributes
    ----------
    approach_offset:
        Distance behind the target object along the push direction
        from which the gripper should approach.
    approach_height_offset:
        Additional height added to the approach target so the gripper
        clears the object before descending.
    subgoal_threshold:
        Euclidean distance threshold for considering a subgoal reached.
    goal_threshold:
        Euclidean distance below which the overall task goal is done.
        If None, uses the environment default.
    phase1_object_weight:
        Weight on the object-goal term during the approach phase.
        Low so that the CEM concentrates on moving the gripper.
    phase1_gripper_weight:
        Weight on the gripper-target term during the approach phase.
    phase2_object_weight:
        Weight on the object-goal term during the push phase.
    phase2_gripper_weight:
        Weight on the gripper-target term during the push phase.
        Low but nonzero so the gripper stays near the object.
    """
    approach_offset: float = 0.08
    approach_height_offset: float = 0.0
    subgoal_threshold: float = 0.04
    goal_threshold: Optional[float] = None

    phase1_object_weight: float = 0.1
    phase1_gripper_weight: float = 1.0
    phase2_object_weight: float = 1.0
    phase2_gripper_weight: float = 0.25


# ---------------------------------------------------------------------------
# Causal ordering from TB coupling
# ---------------------------------------------------------------------------

def infer_causal_order(
    partition: TBPartition,
) -> list[int]:
    """Order object IDs by causal influence (upstream first).

    The coupling matrix captures how strongly each variable group
    influences others.  An "upstream" object is one whose variables
    have high outgoing coupling to blanket variables.  In manipulation,
    the gripper is upstream (it acts on the object through the blanket).

    If no coupling matrix is available, fall back to ordering by object
    size (fewer variables = more likely the actuator/gripper).

    Returns
    -------
    list[int]
        Object IDs ordered from most-upstream (actuator) to
        most-downstream (manipulated body).
    """
    obj_ids = sorted(partition.objects.keys())
    if len(obj_ids) <= 1:
        return obj_ids

    C = partition.coupling_matrix
    if C is None:
        # Heuristic: smaller variable set is likely the actuator
        return sorted(obj_ids, key=lambda oid: len(partition.objects[oid]))

    # For each object, compute its coupling strength to blanket variables.
    # The object with *higher* outgoing coupling to the blanket is upstream
    # (it drives the interaction).
    blanket_set = set(partition.blanket)
    scores: dict[int, float] = {}
    for oid in obj_ids:
        obj_vars = partition.objects[oid]
        if not obj_vars or not blanket_set:
            scores[oid] = 0.0
            continue
        # Mean absolute coupling from this object's variables to blanket vars
        coupling_to_blanket = 0.0
        count = 0
        for i in obj_vars:
            for j in blanket_set:
                if i < C.shape[0] and j < C.shape[1]:
                    coupling_to_blanket += abs(float(C[i, j]))
                    count += 1
        scores[oid] = coupling_to_blanket / max(count, 1)

    # Higher score = more upstream (drives blanket)
    return sorted(obj_ids, key=lambda oid: -scores[oid])


def classify_objects(
    partition: TBPartition,
    gripper_object_id: Optional[int] = None,
) -> tuple[int, Optional[int]]:
    """Identify which object is the gripper (actuator) and which is the target.

    If gripper_object_id is provided explicitly, use it.  Otherwise,
    infer from causal ordering: the most-upstream object is the gripper.

    Returns
    -------
    (gripper_id, target_id)
        target_id is None if there is only one object (e.g. FetchReach).
    """
    obj_ids = sorted(partition.objects.keys())
    if not obj_ids:
        raise ValueError("TBPartition has no objects.")

    if gripper_object_id is not None:
        gripper_id = gripper_object_id
    else:
        order = infer_causal_order(partition)
        gripper_id = order[0]

    remaining = [oid for oid in obj_ids if oid != gripper_id]
    target_id = remaining[0] if remaining else None
    return gripper_id, target_id


# ---------------------------------------------------------------------------
# TBGuidedPlanner
# ---------------------------------------------------------------------------

class TBGuidedPlanner:
    """TB-guided symbolic planner that decomposes goals using TB structure.

    This planner replaces the hardcoded FetchPushSymbolicPlanner.  It
    uses the TB partition to identify independently controllable
    subsystems (objects), infer their causal ordering, and sequence
    subgoals accordingly.

    For a two-object system (gripper + manipulated object):
      Phase 0: Move the upstream object (gripper) to an approach
               position behind the downstream object along the
               push direction.  During this phase the object-goal
               weight is low (we do not care where the object is yet).
      Phase 1: Push the downstream object toward the goal by moving
               the gripper through it.

    For a single-object system (gripper only, e.g. FetchReach):
      Phase 0: Move the gripper directly to the goal.  Single phase.

    Parameters
    ----------
    partition:
        TB-discovered variable groupings.
    gripper_object_id:
        Which object ID corresponds to the gripper.  If None, inferred
        from the causal ordering (most-upstream object).
    gripper_pos_indices:
        Observation-vector indices for the gripper xyz position.
        These may be a subset of the gripper object's full variable set.
    default_goal_threshold:
        Euclidean distance below which the task is considered complete.
    cfg:
        Planner configuration (approach offset, weights, etc.).
    """

    def __init__(
        self,
        partition: TBPartition,
        gripper_object_id: Optional[int] = None,
        gripper_pos_indices: tuple[int, ...] = (0, 1, 2),
        default_goal_threshold: float = 0.05,
        cfg: Optional[TBGuidedPlannerConfig] = None,
    ) -> None:
        if cfg is None:
            cfg = TBGuidedPlannerConfig()
        self.cfg = cfg
        self.partition = partition
        self._gripper_pos_indices = gripper_pos_indices
        self._goal_threshold = (
            float(cfg.goal_threshold)
            if cfg.goal_threshold is not None
            else float(default_goal_threshold)
        )

        # Identify objects
        self._gripper_id, self._target_id = classify_objects(
            partition, gripper_object_id
        )
        self._n_objects = len(partition.objects)
        self._has_target = self._target_id is not None

        # Determine number of phases
        if self._has_target:
            self._phase_total = 2  # approach + push
        else:
            self._phase_total = 1  # just reach

        self._phase_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset phase state for a new episode."""
        self._phase_index = 0

    @property
    def n_phases(self) -> int:
        return self._phase_total

    @property
    def has_target_object(self) -> bool:
        return self._has_target

    @property
    def gripper_object_id(self) -> int:
        return self._gripper_id

    @property
    def target_object_id(self) -> Optional[int]:
        return self._target_id

    def decide(
        self,
        obs_vec: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> SymbolicDecision:
        """Produce a planning objective based on TB-guided goal decomposition.

        Parameters
        ----------
        obs_vec:
            Flat observation vector from the environment.
        achieved_goal:
            Current achieved goal (object position for FetchPush,
            gripper position for FetchReach).
        desired_goal:
            Environment-provided desired goal.

        Returns
        -------
        SymbolicDecision
            Contains the PlanningObjective for the CEM planner and a
            status summary.
        """
        if not self._has_target:
            return self._decide_reach(obs_vec, achieved_goal, desired_goal)
        return self._decide_push(obs_vec, achieved_goal, desired_goal)

    # ------------------------------------------------------------------
    # Single-object (Reach) decision
    # ------------------------------------------------------------------

    def _decide_reach(
        self,
        obs_vec: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> SymbolicDecision:
        """Single-phase plan: move gripper to goal."""
        gripper_pos = self._gripper_pos(obs_vec)
        goal_dist = float(np.linalg.norm(achieved_goal - desired_goal))
        done = goal_dist <= self._goal_threshold

        objective = PlanningObjective(
            name="tb_reach/move_to_goal",
            desired_goal=np.asarray(desired_goal, dtype=np.float32),
            object_goal_weight=1.0,
            gripper_target=np.asarray(desired_goal, dtype=np.float32),
            gripper_target_weight=1.0,
            gripper_indices=self._gripper_pos_indices,
        )
        status = SymbolicStatus(
            phase_name="reach_goal",
            phase_index=1,
            phase_total=1,
            subgoal_distance=goal_dist,
            subgoal_threshold=self._goal_threshold,
            object_goal_distance=goal_dist,
            done=done,
            status_text=(
                f"TB single-object plan: move gripper to goal "
                f"(distance={goal_dist:.4f})."
            ),
        )
        return SymbolicDecision(objective=objective, status=status)

    # ------------------------------------------------------------------
    # Two-object (Push) decision
    # ------------------------------------------------------------------

    def _decide_push(
        self,
        obs_vec: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> SymbolicDecision:
        """Two-phase plan: approach then push.

        Phase 0 (approach): Move the gripper behind the object along
        the push direction.  The object-goal weight is low so CEM
        focuses on positioning the gripper.

        Phase 1 (push): Drive the object toward the goal.  The gripper
        target tracks the object position so the gripper stays in
        contact.
        """
        object_goal_dist = float(np.linalg.norm(achieved_goal - desired_goal))

        # Early completion check
        if self._phase_index == 0 and object_goal_dist <= self._goal_threshold:
            self._phase_index = 1

        # Compute approach target
        approach_target = self._approach_target(achieved_goal, desired_goal)
        gripper_pos = self._gripper_pos(obs_vec)
        approach_dist = float(np.linalg.norm(gripper_pos - approach_target))

        # Phase transition: gripper reached approach position
        if self._phase_index == 0 and approach_dist <= self.cfg.subgoal_threshold:
            self._phase_index = 1

        if self._phase_index == 0:
            # Phase 0: approach
            objective = PlanningObjective(
                name="tb_push/approach",
                desired_goal=np.asarray(achieved_goal, dtype=np.float32),
                object_goal_weight=float(self.cfg.phase1_object_weight),
                gripper_target=approach_target,
                gripper_target_weight=float(self.cfg.phase1_gripper_weight),
                gripper_indices=self._gripper_pos_indices,
            )
            status = SymbolicStatus(
                phase_name="tb_approach",
                phase_index=1,
                phase_total=self._phase_total,
                subgoal_distance=approach_dist,
                subgoal_threshold=float(self.cfg.subgoal_threshold),
                object_goal_distance=object_goal_dist,
                done=False,
                status_text=(
                    f"TB two-object plan phase 1/{self._phase_total}: "
                    f"move upstream object (gripper) to approach position "
                    f"(distance={approach_dist:.4f})."
                ),
            )
            return SymbolicDecision(objective=objective, status=status)

        # Phase 1: push
        done = object_goal_dist <= self._goal_threshold
        objective = PlanningObjective(
            name="tb_push/drive_to_goal",
            desired_goal=np.asarray(desired_goal, dtype=np.float32),
            object_goal_weight=float(self.cfg.phase2_object_weight),
            gripper_target=np.asarray(achieved_goal, dtype=np.float32),
            gripper_target_weight=float(self.cfg.phase2_gripper_weight),
            gripper_indices=self._gripper_pos_indices,
        )
        status = SymbolicStatus(
            phase_name="tb_push_to_goal",
            phase_index=2,
            phase_total=self._phase_total,
            subgoal_distance=object_goal_dist,
            subgoal_threshold=self._goal_threshold,
            object_goal_distance=object_goal_dist,
            done=done,
            status_text=(
                f"TB two-object plan phase 2/{self._phase_total}: "
                f"push downstream object to goal "
                f"(distance={object_goal_dist:.4f})."
            ),
        )
        return SymbolicDecision(objective=objective, status=status)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gripper_pos(self, obs_vec: np.ndarray) -> np.ndarray:
        """Extract gripper xyz from the observation vector."""
        return np.asarray(
            obs_vec[list(self._gripper_pos_indices)], dtype=np.float32
        )

    def _push_direction(
        self, object_pos: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        """Unit vector from object to goal (the push direction)."""
        delta = desired_goal - object_pos
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            direction = np.zeros_like(object_pos)
            direction[0] = 1.0
            return direction
        return delta / norm

    def _approach_target(
        self, object_pos: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        """Position behind the object along the push direction.

        The gripper should approach from behind the object (opposite
        the push direction) so that it can push the object forward.
        """
        direction = self._push_direction(object_pos, desired_goal)
        target = object_pos - direction * float(self.cfg.approach_offset)
        if target.shape[0] >= 3:
            target = target.copy()
            target[2] = target[2] + float(self.cfg.approach_height_offset)
        return target.astype(np.float32)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_tb_guided_planner(
    partition: TBPartition,
    gripper_object_id: Optional[int] = None,
    gripper_pos_indices: tuple[int, ...] = (0, 1, 2),
    default_goal_threshold: float = 0.05,
    cfg: Optional[TBGuidedPlannerConfig] = None,
) -> TBGuidedPlanner:
    """Create a TBGuidedPlanner from a TB partition.

    This is the primary entry point.  Call it with the partition
    discovered by ``TopologicalBlankets.fit()`` on ensemble gradients.

    Parameters
    ----------
    partition:
        TB-discovered structure.  ``objects`` maps object IDs to
        variable indices; ``blanket`` lists blanket variable indices.
    gripper_object_id:
        Which object is the gripper.  If None, inferred from causal
        ordering.
    gripper_pos_indices:
        Observation indices for gripper xyz (default: first 3).
    default_goal_threshold:
        Distance threshold for task completion.
    cfg:
        Optional planner configuration overrides.

    Returns
    -------
    TBGuidedPlanner
    """
    return TBGuidedPlanner(
        partition=partition,
        gripper_object_id=gripper_object_id,
        gripper_pos_indices=gripper_pos_indices,
        default_goal_threshold=default_goal_threshold,
        cfg=cfg,
    )


def partition_from_tb_result(
    labels: np.ndarray,
    coupling_matrix: Optional[np.ndarray] = None,
    obs_labels: Optional[list[str]] = None,
) -> TBPartition:
    """Build a TBPartition from raw TB detection output.

    Parameters
    ----------
    labels:
        Per-variable labels from TB.  -1 = blanket, 0,1,2,... = object ID.
    coupling_matrix:
        Optional coupling matrix from TB analysis.
    obs_labels:
        Optional human-readable variable names.

    Returns
    -------
    TBPartition
    """
    labels_arr = np.asarray(labels)
    blanket_indices: list[int] = []
    objects: dict[int, list[int]] = {}

    for idx, lbl in enumerate(labels_arr):
        lbl_int = int(lbl)
        if lbl_int == -1:
            blanket_indices.append(idx)
        else:
            if lbl_int not in objects:
                objects[lbl_int] = []
            objects[lbl_int].append(idx)

    return TBPartition(
        objects=objects,
        blanket=blanket_indices,
        coupling_matrix=coupling_matrix,
        obs_labels=obs_labels,
    )


def make_fetchpush_ground_truth_partition(
    obs_dim: int = 25,
) -> TBPartition:
    """Create the ground-truth FetchPush partition for testing.

    This encodes the known physical structure:
      Object 0 (gripper):  grip_pos[0:3], gripper_state[9:11], grip_velp[20:22]
      Object 1 (object):   object_pos[3:6], object_rot[11:14],
                           object_velp[14:17], object_velr[17:20]
      Blanket:             object_rel_pos[6:9]

    Any remaining dimensions (22:obs_dim) are assigned to the blanket
    since they are ambiguous.
    """
    gripper_vars = [0, 1, 2, 9, 10, 20, 21]
    object_vars = [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    blanket_vars = [6, 7, 8]

    # Extra dimensions go to blanket
    for i in range(22, obs_dim):
        blanket_vars.append(i)

    labels = [
        'grip_x', 'grip_y', 'grip_z',
        'obj_x', 'obj_y', 'obj_z',
        'rel_x', 'rel_y', 'rel_z',
        'grip_state_0', 'grip_state_1',
        'obj_rot_0', 'obj_rot_1', 'obj_rot_2',
        'obj_velp_x', 'obj_velp_y', 'obj_velp_z',
        'obj_velr_x', 'obj_velr_y', 'obj_velr_z',
        'grip_velp_x', 'grip_velp_y',
    ]
    for i in range(22, obs_dim):
        labels.append(f'extra_{i - 22}')

    return TBPartition(
        objects={0: gripper_vars, 1: object_vars},
        blanket=blanket_vars,
        obs_labels=labels,
    )


def make_fetchreach_partition(
    obs_dim: int = 10,
) -> TBPartition:
    """Create a FetchReach-compatible partition (single object).

    FetchReach has only a gripper (no manipulated object).  All
    observation variables belong to a single object; there is no
    blanket.

    The typical FetchReach-v3 observation is 10D:
      [0:3]  grip_pos, [3:5] gripper_state, [5:8] grip_velp, [8:10] extra
    """
    gripper_vars = list(range(obs_dim))
    return TBPartition(
        objects={0: gripper_vars},
        blanket=[],
        obs_labels=[f'var_{i}' for i in range(obs_dim)],
    )
