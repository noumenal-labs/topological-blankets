# Structure Definitions: Bayesian Models vs Energy-Based Models

## Overview

This document formalizes what "structure" means in Bayesian models versus Energy-Based Models (EBMs), identifying the precise gap that structure learning must bridge.

**Core Thesis**: Structure learning IS Markov blanket discovery and typology. Finding the right model structure = finding the right way to partition the world into objects with distinct blanket statistics.

* * *

## 0. What Is Structure? (Mathematical Preliminaries)

### 0.1 The General Pattern

In mathematics, *structure* on a set X is additional data that restricts the "allowed" maps between objects. Structure is characterized by what preserves it.

### Why "Characterized by What Preserves It"?

This is a profound shift in perspective. We don't define structure by saying *what it is* — we define it by saying *what respects it*.

**The naive approach** (what structure "is"):
> "A topology on X is a collection τ of subsets satisfying: ∅, X ∈ τ; closed under arbitrary unions; closed under finite intersections."

**The morphism-centric approach** (what preserves structure):
> "A topology on X is whatever makes continuous maps well-defined. Two spaces have 'the same topology' iff they're homeomorphic."

These are equivalent, but the second view is more powerful because:

1. **Structure becomes operational**: Instead of checking axioms, we check whether maps preserve the structure. The structure exists *precisely to the extent* that there are non-trivial structure-preserving maps.

2. **Comparison becomes natural**: Two objects have "the same structure" iff there's an invertible structure-preserving map (isomorphism) between them. We don't compare axiom-by-axiom; we ask if they're interchangeable via morphisms.

3. **Forgetting structure is a functor**: When we say "topology forgets metric structure," we mean there's a functor Metric → Topological that keeps the object but forgets which maps are allowed. More maps become "legal."

4. **Structure is relational, not intrinsic**: A set doesn't "have" a topology in isolation. It has a topology *relative to* other topological spaces and continuous maps between them. Structure lives in the morphisms.

#### Klein's Erlangen Program (1872)

Felix Klein unified geometry by this principle:

> *"A geometry is the study of invariants under a group of transformations."*

| Geometry | Transformation group | What's preserved |
|----------|---------------------|------------------|
| Euclidean | Isometries (rotate, translate, reflect) | Distances, angles |
| Affine | Affine maps (linear + translation) | Parallelism, ratios |
| Projective | Projective transformations | Cross-ratio, incidence |
| Topology | Homeomorphisms | Connectedness, holes |

Each row is "less structure" — more transformations are allowed, fewer properties are invariant.

**For our project**: EBM geometry is characterized by what reparameterizations θ → θ' preserve. If basins and barriers are preserved, we have "the same geometry." If only connectivity is preserved, we've dropped to topology.

#### The Yoneda Perspective

Category theory takes this further: an object is *completely determined* by its morphisms.

**Yoneda Lemma** (informally):
> An object X is characterized by the collection of all maps into it: Hom(-, X).

Two objects are isomorphic iff they have the same "morphism profile." You never need to look "inside" an object — its relationships to other objects tell you everything.

**For our project**: An EBM E is characterized by:
- Maps *from* data to E (inference: finding low-energy configurations)
- Maps *from* E to other EBMs (reparameterization, coarse-graining)
- Maps *from* E to graphs (the functor F we defined)

The structure of E is not in "what E is" but in "how E relates to everything else."

#### Preservation as Definition

Consider: what IS a group homomorphism φ: G → H?

**Axiomatic**: φ(g · g') = φ(g) · φ(g') for all g, g' ∈ G.

**Preservation view**: φ is a group homomorphism iff it preserves the group structure — meaning the group operation, identity, and inverses are respected.

But here's the key: *we could have defined it the other way around*.

> "The group structure on G is *whatever* is preserved by group homomorphisms."

This circular-seeming definition actually works: the structure and its morphisms are *co-defined*. You can't have one without the other. They're two views of the same thing.

#### Implications for Structure Learning

If structure is characterized by what preserves it, then *structure learning is learning what should be preserved*.

| Framework | Learning question | Morphism question |
|-----------|------------------|-------------------|
| Bayesian | Which graph G fits the data? | What conditional independencies should be preserved? |
| EBM | Which energy E fits the data? | What geometric features should be preserved? |
| Our synthesis | Which (E, G) pair fits? | What should F: EBM → Graph preserve? |

**Key insight**: When we extract topology from geometry, we're asking:
> "What structure in the EBM should be preserved when we forget metric information?"

The answer: *conditional independence* (Markov blankets). The functor F preserves CI structure while forgetting distances.

| Domain | Structure on X | Preserved by |
|--------|----------------|--------------|
| Topology | Open sets τ ⊆ P(X) | Continuous maps |
| Algebra | Operations (·, +, ...) | Homomorphisms |
| Geometry | Metric d: X×X → ℝ | Isometries |
| Differential | Smooth atlas | Diffeomorphisms |
| Order | Relation ≤ | Monotone maps |

#### Worked Examples

**Example 1: Topological Structure**

Consider X = ℝ with two different topologies:
- τ_std = standard topology (open intervals)
- τ_disc = discrete topology (every subset is open)

The map f(x) = x² is:
- Continuous in (ℝ, τ_std) → (ℝ, τ_std) ✓
- Continuous in (ℝ, τ_disc) → (ℝ, τ_std) ✓ (discrete is "finer")
- NOT continuous in (ℝ, τ_std) → (ℝ, τ_disc) ✗

The structure (which sets are "open") determines which maps are allowed. A homeomorphism must preserve this structure in both directions — (ℝ, τ_std) and (ℝ, τ_disc) are NOT homeomorphic even though they have the same underlying set.

**Example 2: Algebraic Structure**

Consider two groups:
- (ℤ, +) = integers under addition
- (ℝ₊, ×) = positive reals under multiplication

The map φ: ℤ → ℝ₊ defined by φ(n) = eⁿ is a homomorphism:
```
φ(n + m) = e^(n+m) = eⁿ · eᵐ = φ(n) · φ(m)  ✓
```

The map ψ(n) = n² is NOT a homomorphism:
```
ψ(2 + 3) = 25 ≠ 4 · 9 = ψ(2) · ψ(3)  ✗
```

The structure (the group operation) restricts which maps "make sense."

**Example 3: Geometric (Metric) Structure**

Consider ℝ² with Euclidean metric d(x,y) = ||x - y||.

Rotation R_θ(x) = [cos θ, -sin θ; sin θ, cos θ] · x is an isometry:
```
d(R_θ(x), R_θ(y)) = ||R_θ(x) - R_θ(y)|| = ||R_θ(x-y)|| = ||x-y|| = d(x,y)  ✓
```

Scaling S_λ(x) = λx (for λ ≠ 1) is NOT an isometry:
```
d(S_λ(x), S_λ(y)) = |λ| · ||x - y|| ≠ ||x - y|| = d(x,y)  ✗
```

Scaling preserves topology (it's a homeomorphism) but destroys metric structure. This shows geometry is "more structure" than topology.

**Example 4: Differential Structure**

Consider ℝ with its standard smooth structure.

The map f(x) = x³ is a diffeomorphism:
- Smooth: f'(x) = 3x² exists and is continuous ✓
- Bijective ✓
- Inverse f⁻¹(x) = x^(1/3) is smooth ✓

The map g(x) = |x| is NOT smooth (not differentiable at 0):
```
g'(0) = lim_{h→0} (|h| - 0)/h  does not exist  ✗
```

Smooth structure is "more structure" than topological structure — fewer maps are allowed.

**Example 5: Order Structure**

Consider (ℝ, ≤) with the usual ordering.

The map f(x) = 2x + 1 is monotone (order-preserving):
```
x ≤ y  ⟹  2x + 1 ≤ 2y + 1  ⟹  f(x) ≤ f(y)  ✓
```

The map g(x) = -x is NOT monotone (it's order-reversing):
```
x ≤ y  ⟹  -x ≥ -y  ⟹  g(x) ≥ g(y)  ✗ (reverses order)
```

Order structure is independent of metric — the same set can have different orders, and order-preserving maps need not preserve distances.

#### The Pattern

In each case:
1. **Structure** = extra data beyond the bare set
2. **Morphisms** = maps that respect this data
3. **Isomorphism** = bijective morphism with morphism inverse
4. **More structure** = fewer allowed morphisms

**The categorical view**: Objects have structure; morphisms preserve it. A category C consists of:
- Objects (sets with structure)
- Morphisms (structure-preserving maps)
- Composition (morphisms compose associatively)
- Identity (every object has identity morphism)

### 0.2 Structure in Probability

For probabilistic models, what is the structure?

**Option 1: The distribution itself**
- Objects: Probability distributions p(x)
- Morphisms: Measure-preserving maps? Sufficient statistics?
- Problem: Too rigid (distributions rarely equal)

**Option 2: Conditional independence relations**
- Objects: Sets of CI statements {X ⊥ Y | Z}
- Morphisms: Maps preserving CI structure
- This is the **graphoid** structure

**Option 3: The generative process**
- Objects: Causal/generative models
- Morphisms: Interventionally-equivalent transformations
- This distinguishes correlation from causation

### 0.3 Two Kinds of Structure in Our Setting

We're dealing with two distinct notions of structure:

**Geometric structure** (EBMs):
```
Objects: Energy functions E: X → ℝ
Morphisms: Reparameterizations θ → θ' preserving level sets, critical points
Structure: Basins, barriers, curvature, geodesics
```

**Combinatorial/Topological structure** (Bayesian):
```
Objects: Graphs G = (V, E)
Morphisms: Graph homomorphisms preserving adjacency
Structure: Connectivity, paths, cuts, d-separation
```

### 0.4 The Functor Perspective

*Key insight*: Geometry → Topology extraction is a *functor*.

```
F: EBM → Graph

F(E) = G_E  (the interaction graph induced by E)
F(θ → θ') = (G_E → G_E')  (how graph changes under reparameterization)
```

Specifically:
```
G_E has edge (i,j) ⟺ ∂²E/∂xᵢ∂xⱼ ≠ 0
```

This functor **forgets** geometric information (distances, curvature) and retains only combinatorial information (connectivity).

**Adjoint?** Is there a functor going the other way?
```
G: Graph → EBM  (construct energy from graph)
```

Yes: Given graph G, define energy:
```
E_G(x) = Σ_{(i,j) ∈ E} φᵢⱼ(xᵢ, xⱼ) + Σᵢ ψᵢ(xᵢ)
```

This is exactly how Markov Random Fields are constructed!

The adjunction F ⊣ G (if it exists precisely) would formalize:
- G embeds graphs into EBMs (adds geometry)
- F extracts graphs from EBMs (forgets geometry)
- The adjunction says these are "optimally" related

### 0.5 What Structure Learning Seeks

Structure learning asks: **Which structure in the target category best explains the data?**

| Framework | Category | Structure | Learning seeks |
|-----------|----------|-----------|----------------|
| Bayesian networks | **Graph** | Edges (d-separation) | Sparsest graph consistent with CI |
| EBMs | **Manifold** | Geometry (curvature) | Landscape with right basins |
| Our synthesis | **Both** | Geometry + Topology | Geometry whose induced topology is optimal |

### 0.6 Levels of Structure

Structure comes in levels of "rigidity":

```
More rigid                                      Less rigid
    ↓                                               ↓
Geometric ────→ Topological ────→ Set-theoretic
(distances)     (connectivity)    (just elements)
```

Each level forgets information:
- Geometry → Topology: Forget distances, keep connectivity
- Topology → Set: Forget connectivity, keep elements

**Our project**: Use the geometric level (EBMs) to discover the topological level (graphs), exploiting that geometry contains more information.

### 0.7 Structure Preservation in Learning

When we learn an EBM (update θ), what structure is preserved?

**Preserved** (ideally):
- Number and type of basins (topology)
- Qualitative barrier structure

**Not preserved**:
- Exact energy values
- Precise curvatures
- Metric distances

Learning that **changes topology** (basin birth/death) is qualitatively different from learning that refines geometry within fixed topology. This is the "phase transition" phenomenon.

* * *

## 1. Structure in Bayesian Models

Bayesian models maintain generative models of the form:

```
P(o, s | m) = P(o | s) P(s)
```

where `o` = observations, `s` = hidden states, `m` = model structure.

### 1.1 Components of Structure

**Graph Topology**
- Which variables exist (state factors, observation modalities)
- Directed edges encoding conditional dependencies
- Example: Does velocity depend on position? Does reward depend on action?

**Temporal Depth**
- How far into past/future the model reasons
- T-step models with state transitions: `P(s_{t+1} | s_t, a_t)`
- Deeper = more planning horizon, but exponential state space

**Factorial Depth**
- Factorization of state space: `s = (s^1, s^2, ..., s^K)`
- Each factor captures independent aspect (location, object identity, context)
- More factors = richer representation, but combinatorial explosion

**Hierarchical Depth**
- Nested models at multiple timescales
- Higher levels: slower dynamics, more abstract states
- Lower levels: faster dynamics, sensorimotor details

### 1.2 The Structure Learning Problem

The space of possible structures is:
- **Discrete**: graphs, not continuous parameters
- **Combinatorial**: O(2^{n^2}) possible DAGs for n variables
- **Non-differentiable**: can't gradient descend over graph topology

Current approaches:
1. **Bayesian model comparison**: Compute P(m|o) for candidate structures
2. **Bayesian model reduction**: Prune unnecessary components
3. **Score-based search**: Hill-climbing over graph space

Key equation for model comparison:
```
ln P(m|o) / P(m'|o) = ΔF + ΔG
```
where ΔF = accuracy-complexity tradeoff, ΔG = expected information gain.

* * *

## 2. Structure in Energy-Based Models

EBMs define a probability distribution via an energy function:

```
E(x, y; θ) : X × Y × Θ → ℝ
p(x, y | θ) = exp(-E(x, y; θ)) / Z(θ)
```

### 2.1 Components of Structure

**Energy Landscape Geometry**
- The shape of E(x; θ) encodes all learned knowledge
- Basins correspond to stable configurations
- Barriers separate distinct modes

**Latent Structure**
- Latent variables z that explain observations y
- E(z, y; θ) defines the joint energy
- Inference = finding low-energy z given y

**Parameter Structure**
- θ parameterizes the energy function
- Different parameterizations induce different landscape geometries
- Learning = optimizing θ to fit data

### 2.2 What EBMs Learn vs What They Assume

**Learned (continuous, differentiable)**
- Parameters θ shaping the landscape
- Implicitly: basin structure, barrier heights, curvature

**Fixed (discrete, assumed)**
- Topology of the generative model (which variables exist)
- Architecture (functional form of E)
- Dimensionality of latent space

### 2.3 The Key Insight

Bayesian models optimize **within** a representational structure (fixed graph, optimize parameters).

EBMs can optimize **the representational structure itself** — the landscape shape IS the representation.

But this is only partially true. EBMs optimize:
- ✓ Landscape shape (via θ)
- ✓ Implicit basin structure
- ✗ Whether to add more latent dimensions
- ✗ Whether to add hierarchical levels
- ✗ Whether the assumed topology is appropriate

* * *

## 3. Comparison Table

| Aspect | Bayesian Model | EBM |
|--------|----------------|-----|
| **What encodes knowledge** | Hidden state beliefs P(s\|o) | Energy landscape E(x;θ) |
| **What learning optimizes** | Model parameters given structure | Landscape parameters given topology |
| **Discrete structure** | Graph topology, factors, hierarchy | Topology often fixed |
| **Continuous structure** | Parameters within each factor | θ parameters |
| **Structure selection** | Bayesian model comparison | Not typically addressed |
| **Inference mechanism** | Belief updates (message passing) | Energy minimization (sampling, optimization) |
| **Free energy** | Over beliefs about states | Over landscape parameters |

* * *

## 4. The Precise Gap

### What EBMs Do Well
EBMs' continuous optimization elegantly handles "within-topology" structure:
- Learning parameters shapes the energy landscape
- Basins emerge naturally from optimization
- The geometry encodes rich relational structure

### What EBMs Cannot Do (Natively)
EBMs cannot answer discrete structural questions:
1. **Dimension selection**: How many latent dimensions?
2. **Topology selection**: What's the right generative structure?
3. **Hierarchical growth**: When should the model add depth?

### The Bridge Needed

A mechanism to make discrete structural decisions using thermodynamic criteria:
- Use free energy (or expected free energy) to compare structures
- Use fluctuation-based estimates for information gain from structure changes
- Integrate with EBM's continuous learning in a three-timescale system:
  - **Fast**: Sampling/optimization on latents (inference)
  - **Slow**: Gradient updates on θ (learning)
  - **Slowest**: Structure selection/growth (architecture search)

* * *

## 5. Formal Statement of the Gap

Let M = {m₁, m₂, ...} be a space of model structures (topologies).

**Bayesian approach** defines:
```
P(m | o) ∝ P(o | m) P(m)
```
and selects m* = argmax_m P(m | o).

**EBM approach** implicitly assumes a fixed m₀ and optimizes:
```
θ* = argmin_θ F(θ; m₀)
```
where F is the variational free energy.

**The gap**: EBMs have no native mechanism for computing or comparing P(m | o) across different topologies m.

**The opportunity**: Thermodynamic quantities (Fisher information, gradient fluctuations) might provide a principled way to estimate marginal likelihoods P(o | m) without explicit integration.

* * *

## 6. Structure as Markov Blanket Discovery

### 6.1 The Blanket Perspective

A Markov blanket B separates internal states Z from external states S:
```
p(s, z | b) = p(s | b) p(z | b)
```

*Key insight*: Model structure = blanket structure. Choosing a model topology is choosing how to partition the world into conditionally independent subsystems.

### 6.2 Blanket Typology

From the "ontological potential function" perspective:
- **Blanket statistics** p(b_τ) define object types
- Two objects are **same type** if they have same blanket statistics
- Structure learning = discovering the right blanket typology

### 6.3 Mapping to EBMs

| Blanket Concept | EBM Equivalent |
|-----------------|----------------|
| Blanket boundary | Basin boundary in energy landscape |
| Internal states Z | Latent variables within a basin |
| External states S | Other basins / observations |
| Blanket statistics | Energy landscape curvature, gradient flow |
| Object type | Distinct basin / mode |

### 6.4 The Unified View

**Structure learning in EBMs** = discovering:
1. **How many basins** (blankets) should exist
2. **Where basin boundaries** (blanket locations) should be
3. **What dynamics** (blanket statistics) each basin has

This reframes the discrete structure problem as a continuous landscape sculpting problem with emergent discretization at basin boundaries.

* * *

## Next Steps

→ The bridge proposal will describe how to:
- Use expected free energy for structure comparison
- Apply thermodynamic criteria for structure growth/pruning
- Implement Markov blanket discovery via basin emergence
