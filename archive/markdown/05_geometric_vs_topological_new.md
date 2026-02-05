# Geometric vs Topological Representations of Structure

## The Core Distinction

| | EBM | Bayesian |
|--|-----|----------|
| **Representation** | Geometric | Topological |
| **Primitive** | Energy E(x) | Graph G = (V, E) |
| **Structure lives in** | Curvature, basins, barriers | Edges, paths, cuts |
| **Conditional independence** | High energy barrier | Missing edge (d-separation) |
| **"How independent?"** | Barrier height (continuous) | Independent or not (binary) |

* * *

## 1. Topological Structure (Bayesian)

### The Representation

A directed graph G = (V, E):
- **Nodes V**: Random variables
- **Edges E**: Direct dependencies

### Key Properties

- **Discrete**: Edge exists or doesn't
- **No metric**: No notion of "how far" between variables
- **Connectivity matters**: Paths determine dependence
- **d-separation**: Purely combinatorial criterion

### Structure Learning

Adding/removing edges = discrete topology changes

```
G → G'  (add edge, remove edge, reverse edge)
```

This is fundamentally a *combinatorial search* over graph structures.

* * *

## 2. Geometric Structure (EBM)

### The Representation

An energy function E: X → ℝ defines:
- **Manifold**: The space X with metric induced by Fisher information
- **Scalar field**: E(x) assigns energy to each point
- **Gradient field**: ∇E(x) points toward higher energy
- **Curvature**: Hessian ∇²E(x) describes local shape

### Key Properties

- **Continuous**: Energies vary smoothly
- **Metric structure**: Distances, geodesics, curvature all defined
- **Basins**: Regions flowing to same minimum
- **Barriers**: High-energy ridges separating basins

### Structure Learning

Sculpting the landscape = continuous parameter changes

```
θ → θ + dθ  (gradient flow on parameters)
```

This is fundamentally a *continuous optimization* over landscape shape.

* * *

## 3. How Geometry Induces Topology

The deep connection: *geometry can induce topology*.

### From Metric to Topology (Mathematics)

Every metric space (X, d) induces a topology:
- Open sets = unions of open balls
- Connectivity = existence of continuous paths

But topology can exist without metric (purely combinatorial).

### From Energy to Graph (Our Setting)

The EBM's Hessian induces a graph:
```
Edge (i,j) exists  ⟺  ∂²E/∂xᵢ∂xⱼ ≠ 0
```

**The sparsity pattern of the Hessian IS the graph topology.**

### Basin Boundaries as Topological Features

When energy barrier B(i,j) between regions i and j satisfies:
```
B(i,j) → ∞  ⟹  regions become topologically disconnected
B(i,j) → 0  ⟹  regions merge (topological connection)
```

Continuous geometry changes can induce discrete topology changes.

* * *

## 4. The Hierarchy

```
Geometry (EBM)
    ↓ induces
Topology (Graph)
    ↓ determines
Conditional Independence (Markov Blankets)
    ↓ constrains
Inference & Learning
```

### Going Up (Geometry → Topology)

- Hessian sparsity → graph edges
- Basin structure → connected components
- Barrier heights → edge "strength" (soft topology)

### Going Down (Topology → Geometry)

- Graph structure → constraints on E
- Missing edge (i,j) → require ∂²E/∂xᵢ∂xⱼ = 0
- But many geometries compatible with same topology

* * *

## 5. Implications for Structure Learning

### Bayesian (Topological) Approach

**Pros**:
- Structure is explicit and interpretable
- Conditional independence is exact (not approximate)
- Model comparison is principled (marginal likelihood)

**Cons**:
- Discrete search over exponentially many graphs
- No notion of "almost independent" (edge or no edge)
- Hard to incorporate continuous uncertainty about structure

### EBM (Geometric) Approach

**Pros**:
- Continuous optimization (gradient-based)
- "Soft" structure (barriers can be any height)
- Structure emerges naturally from learning
- Richer representation (curvature, distances)

**Cons**:
- Structure is implicit (must be extracted)
- Conditional independence is approximate (finite barriers)
- No canonical way to compare structures

### The Synthesis

Use geometric optimization with topological extraction:

1. **Optimize** energy landscape (continuous)
2. **Monitor** Hessian sparsity, basin structure (geometric observables)
3. **Extract** topology when needed (threshold barriers → edges)
4. **Compare** structures using free energy (geometric criterion for topological choice)

* * *

## 6. Soft vs Hard Structure

### Hard Structure (Topological)

```
∂²E/∂xᵢ∂xⱼ = 0  exactly
```

Variables i and j are *exactly* conditionally independent.

### Soft Structure (Geometric)

```
∂²E/∂xᵢ∂xⱼ ≈ 0  (small but nonzero)
```

Variables i and j are *approximately* conditionally independent.

### The Spectrum

```
Strong dependence ←————————————→ Independence
     |                                    |
High |∂²E/∂xᵢ∂xⱼ|                  ∂²E/∂xᵢ∂xⱼ = 0
     |                                    |
  (geometry)                         (topology)
```

Geometry gives a continuous spectrum. Topology emerges at the limit.

* * *

## 7. Temperature and the Geometric-Topological Transition

In statistical physics, temperature controls the transition:

### High Temperature (T → ∞)

- All barriers are crossable
- Single connected basin
- Topology: fully connected graph
- Geometry dominates

### Low Temperature (T → 0)

- Only lowest-energy states matter
- Distinct basins become isolated
- Topology: disconnected components
- Topology dominates

### Critical Temperature

- Phase transitions
- Topology changes discontinuously
- Geometric quantities diverge (susceptibility, correlation length)

**For structure learning**: Temperature (or regularization) controls how "hard" the emergent topology is.

* * *

## 8. Mathematical Summary

### Bayesian (Topological)

```
Structure = Graph G = (V, E)
Independence: i ⊥ j | S  ⟺  S d-separates i from j in G
Learning: argmax_G p(G | data)
```

### EBM (Geometric)

```
Structure = Energy landscape E: X → ℝ
Independence: i ⊥ j | rest  ⟺  ∂²E/∂xᵢ∂xⱼ = 0
Learning: argmin_θ F(θ) where E = E(·; θ)
```

### The Bridge

```
Topology(E) = Graph with edges where ∂²E/∂xᵢ∂xⱼ ≠ 0
```

Geometry contains topology as a derived structure.

* * *

## 9. Comparison with Recent Structure Learning Methods

### 9.1 Positioning Matrix (From Grok Analysis)

| Aspect | DMBD | AXIOM | RGM | Topological Blankets |
|--------|------|-------|-----|----------------------------|
| **Inference** | Variational EM | Mixture growing | Renormalization | Sampling + features |
| **Blanket def** | Role assignments ω_i(t) | Slot boundaries | Path-augmented states | High-gradient ridges |
| **Online/Offline** | Dynamic | Online | Hierarchical | Offline |
| **Best for** | Microscopic dynamics | RL/games | Scale-free hierarchy | Trained EBM diagnostics |

### 9.2 When Each Method Excels

**DMBD (Beck & Ramstead 2025)**:
- Time-resolved microscopic trajectories
- Traveling/exchanging-matter objects
- Physical system with continuous dynamics (Newton's cradle, burning fuse)

**AXIOM (Heins et al. 2025)**:
- Online RL with pixel observations
- Sample-efficient games (~10k steps)
- Active exploration with growing structure

**RGM (Friston et al. 2025)**:
- Hierarchical, scale-free structure
- Temporal depth via paths/orbits
- Image/video/music compression

**Topological Blankets (This Project)**:
- Post-hoc analysis of trained EBMs
- Score-based / diffusion model diagnostics
- Equilibrium landscapes with clear basins
- No discrete search or online interaction needed

### 9.3 Complementary Roles

The methods are *complementary*, not competing:
- Use TC as *diagnostic* for what structure AXIOM/RGM learned
- Use TC for *geometric pre-partitioning* to initialize DMBD roles
- TC's *thermodynamic complexity* (Fisher fluctuations) as alternative to AXIOM's BMR

* * *

## 10. Research Program

1. **Geometric → Topological**: When does EBM optimization discover the "correct" Bayesian graph?

2. **Topological → Geometric**: Given a target graph, what's the optimal energy landscape?

3. **Soft Structure**: Can we do inference/learning with continuous "edge strengths" instead of binary edges?

4. **Thermodynamic Topology**: Can temperature/annealing schedules guide topology discovery?

5. **Curvature as Complexity**: Is Fisher information the right geometric measure of structural complexity?

6. **Hybrid Methods**: Can geometric signals guide discrete structure search (AXIOM growing, DMBD assignments)?

7. **Scaling**: Sparse/low-rank Hessian approximations for n > 10⁴ variables?

8. **Dynamics Tracking**: Monitor topology birth/death during training (phase transitions)?
