"""
US-058: Planning Graph from Recursive Blanket Decomposition
=============================================================

Generates a hierarchical planning graph from recursive Topological Blanket
decomposition of LunarLander's 8D state space.

State variables: [x, y, vx, vy, theta, omega, contact_L, contact_R]

Known physical partition:
  Object 0 ("vertical/landing"): {y, vy, contact_L, contact_R} -- indices [1,3,6,7]
  Object 1 ("lateral/orient"):   {x, vx, theta}                -- indices [0,2,4]
  Blanket ("coupling"):          {omega (angular velocity)}     -- index  [5]

The experiment:
  1. Synthesizes LunarLander-like gradient data with the known block structure
  2. Runs recursive TB decomposition (2 levels) via recursive_spectral_detection
  3. Also runs the class-based fit + fit_hierarchical for comparison
  4. Extracts a planning graph: nodes = objects at each hierarchy level,
     edges = blanket connections
  5. Visualizes the graph as a hierarchical layout (networkx + matplotlib)
  6. Defines abstract state space: per-object state projections
  7. Defines abstract actions as blanket-variable transitions
  8. Quantifies dimensionality reduction: full 8D vs factored sub-dims
  9. Describes a concrete planning scenario ("land safely")
  10. Discusses which goal types benefit from the abstraction
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RALPH_DIR = os.path.dirname(SCRIPT_DIR)
TB_PACKAGE_DIR = os.path.dirname(RALPH_DIR)
sys.path.insert(0, TB_PACKAGE_DIR)
sys.path.insert(0, RALPH_DIR)

from topological_blankets import TopologicalBlankets
from topological_blankets.spectral import (
    recursive_spectral_detection,
    build_adjacency_from_hessian,
    build_graph_laplacian,
)
from topological_blankets.features import compute_geometric_features
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_LABELS = ['x', 'y', 'vx', 'vy', 'theta', 'omega', 'contact_L', 'contact_R']
N_VARS = 8

# Ground-truth partition (indices)
GT_OBJ0 = [1, 3, 6, 7]   # vertical/landing: y, vy, contact_L, contact_R
GT_OBJ1 = [0, 2, 4]       # lateral/orient: x, vx, theta
GT_BLANKET = [5]           # omega (angular velocity)


# =========================================================================
# 1. Synthetic LunarLander-like gradient data
# =========================================================================

def generate_lunarlander_gradients(n_samples=2000, coupling_strength=0.3,
                                   noise_scale=0.05, seed=42):
    """
    Generate synthetic gradient data that mimics LunarLander's 8D energy
    landscape with the known block-coupling structure.

    The Hessian is constructed so that:
      - Within-object blocks have strong couplings (0.6-0.9)
      - Cross-object couplings are near zero
      - The blanket variable (omega, index 5) couples to *both* objects
        with moderate strength (coupling_strength)
      - Noise is added to make the problem non-trivial

    Returns:
        gradients: (n_samples, 8) array
        H_true: (8, 8) ground-truth Hessian
    """
    rng = np.random.RandomState(seed)

    # Build ground-truth Hessian with block structure
    H_true = np.zeros((N_VARS, N_VARS))

    # Object 0 internal couplings: {y(1), vy(3), contact_L(6), contact_R(7)}
    # Strong within-object couplings to ensure clear block structure.
    # These represent the tight physical coupling within the vertical/landing
    # subsystem: height drives velocity, velocity drives contact, etc.
    obj0_pairs = [(1, 3, 1.4), (1, 6, 0.9), (1, 7, 0.9),
                  (3, 6, 1.1), (3, 7, 1.1), (6, 7, 1.3)]
    for i, j, w in obj0_pairs:
        H_true[i, j] = w
        H_true[j, i] = w

    # Object 1 internal couplings: {x(0), vx(2), theta(4)}
    # Lateral position, velocity, and angle are tightly coupled.
    obj1_pairs = [(0, 2, 1.4), (0, 4, 0.9), (2, 4, 1.2)]
    for i, j, w in obj1_pairs:
        H_true[i, j] = w
        H_true[j, i] = w

    # Blanket variable omega(5) couples to *both* objects with moderate strength.
    # This is weaker than within-object coupling, making omega a mediator.
    # Couples to theta(4) -- angular velocity is the time derivative of angle
    H_true[5, 4] = coupling_strength
    H_true[4, 5] = coupling_strength
    # Couples to vy(3) -- angular velocity affects vertical stability
    H_true[5, 3] = coupling_strength * 0.7
    H_true[3, 5] = coupling_strength * 0.7
    # Couples to vx(2) -- angular velocity affects horizontal drift
    H_true[5, 2] = coupling_strength * 0.5
    H_true[2, 5] = coupling_strength * 0.5
    # Couples to contact variables weakly
    H_true[5, 6] = coupling_strength * 0.3
    H_true[6, 5] = coupling_strength * 0.3
    H_true[5, 7] = coupling_strength * 0.3
    H_true[7, 5] = coupling_strength * 0.3

    # Diagonal: self-coupling (energy curvature per variable).
    # Contacts have high curvature (binary-like), omega has lowest (freely rotates).
    diag_values = [2.0, 2.2, 1.8, 2.1, 1.6, 1.0, 2.5, 2.5]
    np.fill_diagonal(H_true, diag_values)

    # Add small noise to off-diagonal to prevent perfectly zero cross-object entries
    noise = rng.randn(N_VARS, N_VARS) * noise_scale
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    H_noisy = H_true + noise

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(H_noisy)
    if eigvals.min() < 0.1:
        H_noisy += (0.1 - eigvals.min()) * np.eye(N_VARS)

    # Generate gradient samples: g ~ N(0, H)
    # (In the score matching framework, Cov(grad log p) = H)
    L_chol = np.linalg.cholesky(H_noisy)
    gradients = rng.randn(n_samples, N_VARS) @ L_chol.T

    return gradients, H_true


# =========================================================================
# 2. Recursive TB decomposition
# =========================================================================

def run_recursive_decomposition(gradients, max_levels=3):
    """
    Run recursive spectral detection on the gradient data.

    Returns the hierarchy list and the estimated Hessian.
    """
    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']

    hierarchy = recursive_spectral_detection(
        H_est, max_levels=max_levels, min_vars=3, adjacency_threshold=0.01
    )

    return hierarchy, H_est, features


def run_class_based_decomposition(gradients):
    """
    Run the TopologicalBlankets class-based pipeline for comparison.

    Returns the TB instance after fitting.
    """
    # Try coupling method first (specifically designed for asymmetric partitions),
    # fall back to hybrid if coupling produces degenerate results.
    tb = TopologicalBlankets(method='coupling', n_objects=2, sparsify='threshold')
    tb.fit(gradients)

    blankets = tb.get_blankets()
    objects = tb.get_objects()
    # Sanity check: coupling method should find a small blanket
    if len(blankets) > N_VARS // 2 or len(objects) < 2:
        print("  Coupling method degenerate; falling back to hybrid")
        tb = TopologicalBlankets(method='hybrid', n_objects=2, sparsify='threshold')
        tb.fit(gradients)

    return tb


# =========================================================================
# 3. Planning graph extraction
# =========================================================================

def extract_planning_graph(hierarchy, H_est):
    """
    Build a planning graph from the recursive hierarchy.

    Nodes = objects at each level (internal clusters + blanket)
    Edges = connections through blanket variables

    Returns:
        G: networkx DiGraph
        node_info: dict mapping node_id -> metadata
    """
    G = nx.DiGraph()
    node_info = {}

    for level_data in hierarchy:
        level = level_data['level']
        internals = level_data['internals']
        blanket = level_data['blanket']
        external = level_data['external']

        # Create nodes for this level
        if internals:
            node_id = f"L{level}_internal"
            labels = [STATE_LABELS[i] for i in internals if i < len(STATE_LABELS)]
            G.add_node(node_id, level=level, role='internal',
                       variables=internals, labels=labels)
            node_info[node_id] = {
                'level': level, 'role': 'internal',
                'variables': internals, 'labels': labels,
                'dim': len(internals),
            }

        if blanket:
            node_id = f"L{level}_blanket"
            labels = [STATE_LABELS[i] for i in blanket if i < len(STATE_LABELS)]
            G.add_node(node_id, level=level, role='blanket',
                       variables=blanket, labels=labels)
            node_info[node_id] = {
                'level': level, 'role': 'blanket',
                'variables': blanket, 'labels': labels,
                'dim': len(blanket),
            }

        if external:
            node_id = f"L{level}_external"
            labels = [STATE_LABELS[i] for i in external if i < len(STATE_LABELS)]
            G.add_node(node_id, level=level, role='external',
                       variables=external, labels=labels)
            node_info[node_id] = {
                'level': level, 'role': 'external',
                'variables': external, 'labels': labels,
                'dim': len(external),
            }

        # Edges: blanket mediates between internal and external
        if blanket and internals:
            coupling = np.mean(np.abs(H_est[np.ix_(blanket, internals)]))
            G.add_edge(f"L{level}_internal", f"L{level}_blanket",
                       weight=float(coupling), level=level)
            G.add_edge(f"L{level}_blanket", f"L{level}_internal",
                       weight=float(coupling), level=level)
        if blanket and external:
            coupling = np.mean(np.abs(H_est[np.ix_(blanket, external)]))
            G.add_edge(f"L{level}_external", f"L{level}_blanket",
                       weight=float(coupling), level=level)
            G.add_edge(f"L{level}_blanket", f"L{level}_external",
                       weight=float(coupling), level=level)

        # Cross-level edges: the kept variables at this level become the
        # input to the next level
        if level > 0:
            prev_blanket_id = f"L{level-1}_blanket"
            curr_internal_id = f"L{level}_internal"
            if prev_blanket_id in G and curr_internal_id in G:
                G.add_edge(prev_blanket_id, curr_internal_id,
                           weight=0.5, level=level, cross_level=True)

    return G, node_info


# =========================================================================
# 4. Visualization: hierarchical planning graph
# =========================================================================

def visualize_planning_graph(G, node_info, hierarchy, save_path_func):
    """
    Visualize the planning graph as a hierarchical layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left panel: planning graph ---
    ax = axes[0]
    ax.set_title('Hierarchical Planning Graph\n(Recursive TB Decomposition)', fontsize=13)

    pos = {}
    colors = []
    sizes = []
    labels_dict = {}

    role_colors = {'internal': '#4ECDC4', 'blanket': '#FF6B6B', 'external': '#95E1D3'}
    role_shapes = {'internal': 'o', 'blanket': 's', 'external': 'D'}

    for node_id, info in node_info.items():
        level = info['level']
        role = info['role']

        # Position: x by role, y by level (inverted so level 0 is top)
        role_x = {'external': 0, 'blanket': 1.5, 'internal': 3}
        x = role_x.get(role, 1.5)
        y = -level * 2.0

        pos[node_id] = (x, y)
        colors.append(role_colors.get(role, '#CCCCCC'))
        sizes.append(max(400, info['dim'] * 200))
        var_str = ', '.join(info['labels'][:4])
        if len(info['labels']) > 4:
            var_str += '...'
        labels_dict[node_id] = f"{role.upper()}\n{var_str}\n(dim={info['dim']})"

    # Draw edges
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            is_cross = data.get('cross_level', False)
            style = 'dashed' if is_cross else 'solid'
            alpha = 0.4 if is_cross else 0.7
            w = data.get('weight', 0.5)
            ax.annotate('', xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle='->', lw=1.5 + w * 2,
                                        color='#555555', alpha=alpha,
                                        linestyle=style,
                                        connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    for node_id in G.nodes():
        if node_id in pos:
            x, y = pos[node_id]
            idx = list(node_info.keys()).index(node_id)
            color = colors[idx]
            size = sizes[idx]
            circle = plt.Circle((x, y), radius=0.35, color=color,
                                ec='black', lw=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, labels_dict.get(node_id, node_id),
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    zorder=6)

    ax.set_xlim(-1, 4.5)
    max_level = max(info['level'] for info in node_info.values()) if node_info else 0
    ax.set_ylim(-max_level * 2 - 1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#4ECDC4', label='Internal (object states)'),
        mpatches.Patch(color='#FF6B6B', label='Blanket (coupling vars)'),
        mpatches.Patch(color='#95E1D3', label='External (eliminated)'),
    ]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=9)

    # --- Right panel: abstract state space projection ---
    ax2 = axes[1]
    ax2.set_title('Abstract State Space Projections\n(Dimensionality Reduction)', fontsize=13)

    # Show the factored sub-spaces
    y_positions = [0.85, 0.55, 0.25]
    box_colors = ['#4ECDC4', '#4ECDC4', '#FF6B6B']
    titles = ['Object 0: Vertical/Landing', 'Object 1: Lateral/Orient', 'Blanket: Coupling']
    var_groups = [
        [STATE_LABELS[i] for i in GT_OBJ0],
        [STATE_LABELS[i] for i in GT_OBJ1],
        [STATE_LABELS[i] for i in GT_BLANKET],
    ]
    dims = [len(GT_OBJ0), len(GT_OBJ1), len(GT_BLANKET)]

    for idx, (yp, color, title, vg, dim) in enumerate(
            zip(y_positions, box_colors, titles, var_groups, dims)):
        rect = plt.Rectangle((0.05, yp - 0.08), 0.9, 0.16,
                              facecolor=color, alpha=0.3, edgecolor='black',
                              lw=1.5, transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(0.5, yp + 0.04, title,
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=11, fontweight='bold')
        ax2.text(0.5, yp - 0.04, f"Variables: {', '.join(vg)} | Dim = {dim}",
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=9)

    # Dimensionality summary at the bottom
    total_factored = sum(dims)
    ax2.text(0.5, 0.07,
             f"Full state: 8D  |  Factored: {dims[0]}D + {dims[1]}D + {dims[2]}D = "
             f"{total_factored}D (sum)\n"
             f"Independent planning in each subspace: max single-object dim = {max(dims)}D",
             transform=ax2.transAxes, ha='center', va='center',
             fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')

    plt.tight_layout()
    saved = save_path_func(fig, 'planning_graph_hierarchy', 'planning_graph')
    return saved


# =========================================================================
# 5. Visualization: coupling matrix with partition overlay
# =========================================================================

def visualize_coupling_and_partition(H_est, hierarchy, tb, save_path_func):
    """
    Show the estimated Hessian with the detected partition overlaid,
    and the abstract action space.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: Hessian with partition ---
    ax = axes[0]
    H_abs = np.abs(H_est)
    np.fill_diagonal(H_abs, 0)
    im = ax.imshow(H_abs, cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(N_VARS))
    ax.set_yticks(range(N_VARS))
    ax.set_xticklabels(STATE_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(STATE_LABELS, fontsize=9)
    ax.set_title('Estimated Hessian |H_ij|\n(off-diagonal coupling)', fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Overlay partition boxes
    # Object 0 box
    for (indices, color, label) in [
        (GT_OBJ0, '#4ECDC4', 'Obj 0'),
        (GT_OBJ1, '#FF6B6B', 'Obj 1'),
        (GT_BLANKET, '#FFD93D', 'Blanket'),
    ]:
        i_min, i_max = min(indices) - 0.5, max(indices) + 0.5
        rect = plt.Rectangle((i_min, i_min), i_max - i_min, i_max - i_min,
                              fill=False, edgecolor=color, lw=2.5,
                              linestyle='--', label=label)
        ax.add_patch(rect)
    ax.legend(fontsize=8, loc='upper right')

    # --- Panel 2: Detected assignment ---
    ax2 = axes[1]
    assignment = tb.get_assignment()
    is_blanket = tb._is_blanket

    bar_colors = []
    for i in range(N_VARS):
        if is_blanket[i]:
            bar_colors.append('#FFD93D')
        elif assignment[i] == 0:
            bar_colors.append('#4ECDC4')
        elif assignment[i] == 1:
            bar_colors.append('#FF6B6B')
        else:
            bar_colors.append('#CCCCCC')

    ax2.barh(range(N_VARS), [1]*N_VARS, color=bar_colors, edgecolor='black')
    ax2.set_yticks(range(N_VARS))
    ax2.set_yticklabels(STATE_LABELS, fontsize=10)
    ax2.set_xlim(0, 1.5)
    ax2.set_title('Detected TB Assignment', fontsize=11)

    # Annotate each bar
    for i in range(N_VARS):
        if is_blanket[i]:
            role = 'BLANKET'
        else:
            role = f'Object {assignment[i]}'
        ax2.text(0.5, i, role, ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xticks([])

    # --- Panel 3: Abstract action space ---
    ax3 = axes[2]
    ax3.set_title('Abstract Actions\n(Blanket-Mediated Transitions)', fontsize=11)
    ax3.axis('off')

    actions_text = (
        "Abstract Action Space\n"
        "=====================\n\n"
        "Action 1: ADJUST OMEGA (blanket)\n"
        "  Transitions between lateral/orient\n"
        "  and vertical/landing subsystems.\n"
        "  Physically: fire side thrusters.\n\n"
        "Action 2: VERTICAL CONTROL (within Obj 0)\n"
        "  Change {vy, contact_L, contact_R}\n"
        "  Physically: fire main thruster.\n\n"
        "Action 3: LATERAL CONTROL (within Obj 1)\n"
        "  Change {vx, theta}\n"
        "  Physically: adjust horizontal drift.\n\n"
        "Key insight: Blanket variable omega\n"
        "is the *only* mediator between the\n"
        "two subsystems. Planning can be\n"
        "decomposed into independent sub-goals\n"
        "connected via blanket transitions."
    )
    ax3.text(0.05, 0.95, actions_text, transform=ax3.transAxes,
             fontsize=9.5, va='top', ha='left',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    saved = save_path_func(fig, 'coupling_partition_actions', 'planning_graph')
    return saved


# =========================================================================
# 6. Planning scenario: "land safely"
# =========================================================================

def describe_planning_scenario():
    """
    Describe a concrete planning scenario using the abstract planning graph.

    Returns a structured dict describing the abstract plan for "land safely".
    """
    scenario = {
        'goal': 'Land safely at the pad center',
        'goal_decomposition': {
            'sub_goal_0': {
                'subsystem': 'Object 0 (vertical/landing)',
                'variables': ['y', 'vy', 'contact_L', 'contact_R'],
                'target': 'y -> 0 (ground), vy -> 0 (soft), contacts -> 1 (both legs)',
                'actions': 'Fire main thruster to control descent rate',
            },
            'sub_goal_1': {
                'subsystem': 'Object 1 (lateral/orient)',
                'variables': ['x', 'vx', 'theta'],
                'target': 'x -> 0 (center), vx -> 0 (stopped), theta -> 0 (upright)',
                'actions': 'Fire side thrusters to correct drift and orientation',
            },
            'blanket_coordination': {
                'subsystem': 'Blanket (coupling)',
                'variable': 'omega',
                'target': 'omega -> 0 (no angular velocity)',
                'role': (
                    'omega mediates between lateral control and vertical stability. '
                    'The planner must ensure omega is near zero before final descent, '
                    'because nonzero omega couples angular momentum into vertical dynamics.'
                ),
            },
        },
        'abstract_plan': [
            'Phase 1 (high altitude): Focus on Object 1 sub-goal. '
            'Correct lateral drift (x, vx) and orientation (theta).',
            'Phase 2 (blanket transition): Drive omega toward zero via side thrusters. '
            'This decouples the two subsystems.',
            'Phase 3 (low altitude): Focus on Object 0 sub-goal. '
            'Control descent rate (vy), achieve soft landing (contacts).',
            'Phase 4 (terminal): Verify both contacts. Mission complete.',
        ],
        'dimensionality_benefit': {
            'full_state_dim': 8,
            'obj0_dim': 4,
            'obj1_dim': 3,
            'blanket_dim': 1,
            'max_subproblem_dim': 4,
            'reduction_ratio': 8 / 4,
            'explanation': (
                'Instead of planning in a single 8D continuous space, '
                'the planner operates in at most 4D at a time. '
                'The blanket variable provides a 1D "switch" that controls '
                'which subsystem the planner is actively engaging.'
            ),
        },
    }
    return scenario


def describe_goal_type_analysis():
    """
    Analyze which types of goals benefit most from the TB abstraction.
    """
    analysis = {
        'highly_beneficial': {
            'sequential_goals': (
                'Goals that can be decomposed into sequential phases acting '
                'on different subsystems (e.g., "first align, then descend"). '
                'The blanket variable is the natural handoff point.'
            ),
            'factored_goals': (
                'Goals where success criteria decompose cleanly along the '
                'partition boundary (e.g., "land upright and centered"). '
                'Each criterion maps to exactly one subsystem.'
            ),
            'safety_constrained': (
                'Goals with safety constraints on one subsystem while optimizing '
                'another (e.g., "minimize fuel while staying below max angular velocity"). '
                'The blanket variable is the natural constraint boundary.'
            ),
        },
        'moderately_beneficial': {
            'coupled_dynamics': (
                'Goals requiring simultaneous control of both subsystems '
                '(e.g., "maintain hover at a specific position"). '
                'The factored representation still helps, but the blanket '
                'variable must be actively monitored.'
            ),
        },
        'minimally_beneficial': {
            'single_subsystem': (
                'Goals that involve only one subsystem (e.g., "reach height y=10"). '
                'The abstraction adds no benefit because the problem is already low-dimensional.'
            ),
            'adversarial_coupling': (
                'Goals where the blanket variable is actively driven to extreme values '
                '(e.g., "spin as fast as possible"). These goals break the assumption '
                'that the blanket is a stable mediator.'
            ),
        },
    }
    return analysis


# =========================================================================
# 7. Quantitative metrics
# =========================================================================

def compute_metrics(hierarchy, H_est, tb):
    """
    Compute quantitative metrics for the planning graph analysis.
    """
    # Recursive decomposition metrics
    n_levels = len(hierarchy)
    vars_per_level = [h['n_vars'] for h in hierarchy]
    eigengaps = [h['eigengap'] for h in hierarchy]

    # Class-based decomposition metrics
    assignment = tb.get_assignment()
    is_blanket = tb._is_blanket
    objects = tb.get_objects()

    n_objects = len(objects)
    n_blanket_vars = int(np.sum(is_blanket))
    obj_dims = {f'object_{k}': len(v) for k, v in objects.items()}

    # Dimensionality reduction
    full_dim = N_VARS
    max_sub_dim = max(len(v) for v in objects.values()) if objects else full_dim
    reduction_ratio = full_dim / max_sub_dim

    # Coupling strength analysis
    blanket_indices = np.where(is_blanket)[0]
    internal_indices = np.where(~is_blanket)[0]

    blanket_coupling_mean = 0.0
    blanket_coupling_max = 0.0
    if len(blanket_indices) > 0 and len(internal_indices) > 0:
        cross_block = np.abs(H_est[np.ix_(blanket_indices, internal_indices)])
        blanket_coupling_mean = float(np.mean(cross_block))
        blanket_coupling_max = float(np.max(cross_block))

    # Within-object coupling vs cross-object coupling
    within_couplings = []
    cross_couplings = []
    for k, obj_vars in objects.items():
        for k2, obj_vars2 in objects.items():
            if k == k2:
                block = np.abs(H_est[np.ix_(obj_vars, obj_vars)])
                np.fill_diagonal(block, 0)
                within_couplings.extend(block.flatten().tolist())
            else:
                block = np.abs(H_est[np.ix_(obj_vars, obj_vars2)])
                cross_couplings.extend(block.flatten().tolist())

    within_mean = float(np.mean(within_couplings)) if within_couplings else 0.0
    cross_mean = float(np.mean(cross_couplings)) if cross_couplings else 0.0
    separation_ratio = within_mean / max(cross_mean, 1e-8)

    # Agreement with ground truth
    gt_assignment = np.full(N_VARS, -1)
    for i in GT_OBJ0:
        gt_assignment[i] = 0
    for i in GT_OBJ1:
        gt_assignment[i] = 1
    for i in GT_BLANKET:
        gt_assignment[i] = -1  # blanket

    detected_blanket_vars = set(np.where(is_blanket)[0].tolist())
    gt_blanket_vars = set(GT_BLANKET)
    blanket_recall = len(detected_blanket_vars & gt_blanket_vars) / max(len(gt_blanket_vars), 1)
    blanket_precision = len(detected_blanket_vars & gt_blanket_vars) / max(len(detected_blanket_vars), 1)

    metrics = {
        'recursive_decomposition': {
            'n_levels': n_levels,
            'vars_per_level': vars_per_level,
            'eigengaps': eigengaps,
            'hierarchy_details': [
                {
                    'level': h['level'],
                    'n_vars': h['n_vars'],
                    'internals': h['internals'],
                    'blanket': h['blanket'],
                    'external': h['external'],
                    'eigengap': h['eigengap'],
                }
                for h in hierarchy
            ],
        },
        'flat_decomposition': {
            'n_objects': n_objects,
            'n_blanket_vars': n_blanket_vars,
            'object_dims': obj_dims,
            'blanket_indices': blanket_indices.tolist(),
            'assignment': assignment.tolist(),
        },
        'dimensionality_reduction': {
            'full_dim': full_dim,
            'max_sub_dim': max_sub_dim,
            'reduction_ratio': float(reduction_ratio),
            'factored_dims': [len(v) for v in objects.values()] + [n_blanket_vars],
            'factored_sum': sum(len(v) for v in objects.values()) + n_blanket_vars,
        },
        'coupling_analysis': {
            'blanket_coupling_mean': blanket_coupling_mean,
            'blanket_coupling_max': blanket_coupling_max,
            'within_object_coupling_mean': within_mean,
            'cross_object_coupling_mean': cross_mean,
            'separation_ratio': float(separation_ratio),
        },
        'ground_truth_agreement': {
            'blanket_precision': float(blanket_precision),
            'blanket_recall': float(blanket_recall),
            'detected_blanket_vars': sorted(list(detected_blanket_vars)),
            'gt_blanket_vars': sorted(list(gt_blanket_vars)),
        },
    }

    return metrics


# =========================================================================
# 8. Visualization: planning scenario
# =========================================================================

def visualize_planning_scenario(scenario, save_path_func):
    """
    Visualize the concrete planning scenario.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title('US-058: Abstract Planning Scenario -- "Land Safely"', fontsize=14)
    ax.axis('off')

    # Draw the four phases as a flowchart
    phases = scenario['abstract_plan']
    phase_colors = ['#95E1D3', '#FF6B6B', '#4ECDC4', '#FFD93D']
    phase_labels = ['Phase 1: Align', 'Phase 2: Decouple', 'Phase 3: Descend', 'Phase 4: Land']

    for i, (phase, color, label) in enumerate(zip(phases, phase_colors, phase_labels)):
        y = 0.82 - i * 0.20
        rect = plt.Rectangle((0.05, y - 0.06), 0.55, 0.14,
                              facecolor=color, alpha=0.4, edgecolor='black',
                              lw=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.07, y + 0.03, label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='center')
        # Wrap phase text
        wrapped = phase
        if len(wrapped) > 70:
            mid = wrapped[:70].rfind(' ')
            if mid > 0:
                wrapped = wrapped[:mid] + '\n' + wrapped[mid+1:]
        ax.text(0.07, y - 0.02, wrapped, transform=ax.transAxes,
                fontsize=8.5, va='center', style='italic')

        # Arrow between phases
        if i < len(phases) - 1:
            ax.annotate('', xy=(0.32, y - 0.07), xytext=(0.32, y - 0.06),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Right side: dimensionality benefit
    dim_info = scenario['dimensionality_benefit']
    dim_text = (
        "Dimensionality Benefit\n"
        "======================\n\n"
        f"Full state space:   {dim_info['full_state_dim']}D\n"
        f"Object 0 subspace:  {dim_info['obj0_dim']}D\n"
        f"Object 1 subspace:  {dim_info['obj1_dim']}D\n"
        f"Blanket:            {dim_info['blanket_dim']}D\n"
        f"Max subproblem:     {dim_info['max_subproblem_dim']}D\n"
        f"Reduction ratio:    {dim_info['reduction_ratio']:.1f}x\n\n"
        f"{dim_info['explanation']}"
    )
    ax.text(0.65, 0.75, dim_text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Bottom: goal-type analysis summary
    ax.text(0.65, 0.22,
            "Goal Types Most Benefiting\n"
            "==========================\n"
            "++ Sequential goals (align, then descend)\n"
            "++ Factored goals (upright AND centered)\n"
            "++ Safety-constrained (fuel vs stability)\n"
            "+  Coupled dynamics (hover in place)\n"
            "-  Single-subsystem (reach height)\n"
            "-  Adversarial coupling (spin fast)",
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.tight_layout()
    saved = save_path_func(fig, 'planning_scenario', 'planning_graph')
    return saved


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("US-058: Planning Graph from Recursive TB Decomposition")
    print("=" * 70)

    # 1. Generate synthetic data
    print("\n[1/7] Generating synthetic LunarLander gradient data...")
    gradients, H_true = generate_lunarlander_gradients(
        n_samples=5000, coupling_strength=0.25, noise_scale=0.02
    )
    print(f"  Gradient shape: {gradients.shape}")
    print(f"  Ground truth Hessian condition number: {np.linalg.cond(H_true):.2f}")

    # 2. Recursive decomposition
    print("\n[2/7] Running recursive spectral detection (max 3 levels)...")
    hierarchy, H_est, features = run_recursive_decomposition(gradients, max_levels=3)
    print(f"  Found {len(hierarchy)} hierarchy levels")
    for h in hierarchy:
        int_labels = [STATE_LABELS[i] for i in h['internals'] if i < len(STATE_LABELS)]
        blk_labels = [STATE_LABELS[i] for i in h['blanket'] if i < len(STATE_LABELS)]
        ext_labels = [STATE_LABELS[i] for i in h['external'] if i < len(STATE_LABELS)]
        print(f"  Level {h['level']}: {h['n_vars']} vars, "
              f"eigengap={h['eigengap']:.4f}")
        print(f"    Internal: {int_labels}")
        print(f"    Blanket:  {blk_labels}")
        print(f"    External: {ext_labels}")

    # 3. Class-based decomposition
    print("\n[3/7] Running TopologicalBlankets class-based pipeline...")
    tb = run_class_based_decomposition(gradients)
    objects = tb.get_objects()
    blankets = tb.get_blankets()
    print(f"  Detected {len(objects)} objects, {len(blankets)} blanket variable(s)")
    for k, v in objects.items():
        labels = [STATE_LABELS[i] for i in v if i < len(STATE_LABELS)]
        print(f"  Object {k}: {labels} (dim={len(v)})")
    blanket_labels = [STATE_LABELS[i] for i in blankets if i < len(STATE_LABELS)]
    print(f"  Blanket: {blanket_labels}")

    # 4. Extract planning graph
    print("\n[4/7] Extracting planning graph...")
    G, node_info = extract_planning_graph(hierarchy, H_est)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 5. Compute metrics
    print("\n[5/7] Computing metrics...")
    metrics = compute_metrics(hierarchy, H_est, tb)

    dim_red = metrics['dimensionality_reduction']
    print(f"  Full dim: {dim_red['full_dim']}, Max sub-dim: {dim_red['max_sub_dim']}, "
          f"Reduction: {dim_red['reduction_ratio']:.1f}x")
    print(f"  Factored dims: {dim_red['factored_dims']} (sum={dim_red['factored_sum']})")

    coupling = metrics['coupling_analysis']
    print(f"  Within-object coupling: {coupling['within_object_coupling_mean']:.4f}")
    print(f"  Cross-object coupling:  {coupling['cross_object_coupling_mean']:.4f}")
    print(f"  Separation ratio:       {coupling['separation_ratio']:.2f}")

    gt = metrics['ground_truth_agreement']
    print(f"  Blanket precision: {gt['blanket_precision']:.2f}, "
          f"recall: {gt['blanket_recall']:.2f}")

    # 6. Planning scenario
    print("\n[6/7] Building planning scenario...")
    scenario = describe_planning_scenario()
    goal_analysis = describe_goal_type_analysis()

    print(f"  Goal: {scenario['goal']}")
    print(f"  Abstract plan has {len(scenario['abstract_plan'])} phases")

    # 7. Visualizations and save
    print("\n[7/7] Generating visualizations and saving results...")

    fig1_path = visualize_planning_graph(G, node_info, hierarchy, save_figure)
    fig2_path = visualize_coupling_and_partition(H_est, hierarchy, tb, save_figure)
    fig3_path = visualize_planning_scenario(scenario, save_figure)

    # Combine all results
    all_metrics = {
        **metrics,
        'planning_scenario': scenario,
        'goal_type_analysis': goal_analysis,
        'planning_graph': {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'node_ids': list(node_info.keys()),
            'node_details': {k: {kk: vv for kk, vv in v.items()}
                             for k, v in node_info.items()},
        },
    }

    config = {
        'n_samples': 5000,
        'coupling_strength': 0.25,
        'noise_scale': 0.02,
        'max_levels': 3,
        'method': 'coupling (with hybrid fallback)',
        'n_objects': 2,
        'state_labels': STATE_LABELS,
        'gt_obj0': GT_OBJ0,
        'gt_obj1': GT_OBJ1,
        'gt_blanket': GT_BLANKET,
    }

    results_path = save_results(
        'planning_graph_from_recursive_tb',
        all_metrics,
        config=config,
        notes=(
            'US-058: Planning graph from recursive TB decomposition of '
            'LunarLander 8D state space. Shows hierarchical planning graph, '
            'abstract state projections, dimensionality reduction, and a '
            'concrete "land safely" planning scenario.'
        ),
    )

    print("\n" + "=" * 70)
    print("US-058 COMPLETE")
    print("=" * 70)
    print(f"\nResults JSON: {results_path}")
    print(f"Figure 1 (planning graph):    {fig1_path}")
    print(f"Figure 2 (coupling/actions):  {fig2_path}")
    print(f"Figure 3 (planning scenario): {fig3_path}")

    # Summary
    print("\n--- Summary ---")
    print(f"Recursive decomposition: {len(hierarchy)} levels")
    print(f"Flat decomposition: {len(objects)} objects + {len(blankets)} blanket var(s)")
    print(f"Dimensionality reduction: 8D -> max {dim_red['max_sub_dim']}D "
          f"({dim_red['reduction_ratio']:.1f}x)")
    print(f"Separation ratio (within/cross): {coupling['separation_ratio']:.2f}")
    print(f"Blanket precision: {gt['blanket_precision']:.2f}, "
          f"recall: {gt['blanket_recall']:.2f}")

    return all_metrics


if __name__ == '__main__':
    main()
