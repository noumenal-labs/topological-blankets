# Mathematical Core: Structure as Partition

## The Fundamental Object

A *structure* is a partition of variables into conditionally independent groups.

Let X = {x₁, x₂, ..., xₙ} be all variables. A structure m defines:
```
m : X → {1, 2, ..., K}
```
assigning each variable to one of K groups, such that groups are conditionally independent given their boundaries.

* * *

## 1. In Bayesian Models

### Graphical Model Representation

Structure m corresponds to a DAG G = (V, E) where:
- V = X (variables are nodes)
- E encodes conditional dependencies

The joint factorizes:
```
p(x | m) = ∏ᵢ p(xᵢ | parents(xᵢ, G))
```

### Markov Blankets from Graphs

For variable xᵢ, its Markov blanket B(xᵢ) consists of:
- Parents of xᵢ
- Children of xᵢ
- Other parents of children of xᵢ (co-parents)

**Property**: xᵢ ⊥ (X \ {xᵢ, B(xᵢ)}) | B(xᵢ)

### Structure Learning

Find m* maximizing:
```
p(m | data) ∝ p(data | m) p(m)
```

where p(data | m) = ∫ p(data | θ, m) p(θ | m) dθ

* * *

## 2. In Energy-Based Models

### Energy Function Representation

Structure is implicit in E : X → ℝ

The probability is:
```
p(x) = exp(-E(x)) / Z
```

### Markov Blankets from Energy

**Definition**: Variables xᵢ and xⱼ are conditionally independent given B if:
```
∂²E/∂xᵢ∂xⱼ = 0  when B is fixed
```

More generally, the **interaction graph** has edge (i,j) iff:
```
∂²E/∂xᵢ∂xⱼ ≠ 0
```

**Basin interpretation**:
- Each basin of E corresponds to a "state" of the system
- Basin boundaries are high-energy barriers
- Variables in different basins are approximately independent (if barrier is high)

### Structure Learning

Minimize free energy:
```
F(θ) = E_q[E(x; θ)] + H[q]
```

Structure emerges from the optimized E(·; θ*).

* * *

## 3. The Bridge: Partition Free Energy

### Unified Objective

For any partition/structure m, define:
```
F(m) = min_θ F(θ | m) + Ω(m)
```

where:
- F(θ | m) = free energy of model with structure m and parameters θ
- Ω(m) = complexity penalty for structure m

### Complexity from Blanket Statistics

**Proposal**: Define complexity in terms of blanket information:
```
Ω(m) = ∑_{blankets B in m} I(B; X_internal(B))
```

where I is mutual information between each blanket and its internal variables.

**Interpretation**:
- More complex structures have more "informative" blankets
- Blankets that strongly constrain their internals = high complexity
- Parsimony favors structures with "simple" blankets

### Thermodynamic Estimation

The key quantities can be estimated from dynamics:

**Fisher Information** (curvature of F):
```
I(θ) = E[(∇ log p)²] ≈ Var[∇E] during sampling
```

**Effective Dimension** (how many parameters matter):
```
d_eff = tr(I(θ)) / λ_max(I(θ))
```

**Blanket Strength** (how separated are regions):
```
B_strength = E[||∇E||²] at basin boundaries
```

* * *

## 4. The Three Representations of Structure

### As a Graph (Bayesian)
```
m = G = (V, E)
```
Nodes = variables, Edges = dependencies

### As a Partition (Abstract)
```
m = {S₁, S₂, ..., Sₖ}  where ∪Sᵢ = X, Sᵢ ∩ Sⱼ = ∅
```
Groups of conditionally independent variables

### As Landscape Geometry (EBM)
```
m ↔ {basins of E, boundaries between basins}
```
Basins = groups, Boundaries = blankets

### Equivalence

All three represent the same information:
- Graph edges ↔ non-zero Hessian entries ↔ within-group connections
- Graph cuts ↔ partition boundaries ↔ basin boundaries
- d-separation ↔ conditional independence ↔ high energy barriers

* * *

## 5. Structure Dynamics

### In Bayesian Models

Structure changes discretely:
```
m → m'  (add/remove edge, merge/split factor)
```

Accepted if:
```
p(m' | data) > p(m | data)
```

### In EBMs

Structure changes continuously (but with discrete effects):
```
θ → θ + dθ
```

As θ changes:
- Basins can merge (barrier drops below threshold)
- Basins can split (new barrier emerges)
- Blankets can sharpen or blur

### Unified Dynamics

Three timescales:
```
Fast:    x(t) → equilibrium in current basin
Slow:    θ(t) → optimal landscape given structure
Slowest: m(t) → optimal structure given θ dynamics
```

The slowest timescale observes the medium timescale:
- If θ dynamics consistently reshape basins → update m
- Use thermodynamic observables as signals for m updates

* * *

## 6. Key Equations

### Free Energy (Bayesian form)
```
F = E_q[log q(z) - log p(z, x)]
  = KL(q || p_posterior) - log p(x)
```

### Free Energy (EBM form)
```
F = E_q[E(z, x; θ)] - H[q]
  = ⟨E⟩ - S  (energy minus entropy)
```

### Structure Comparison
```
log p(m|x) / p(m'|x) = F(m') - F(m) + log p(m)/p(m')
                      = ΔF + Δlog prior
```

### Thermodynamic Structure Criterion
```
F_structure(m) = min_θ ⟨E⟩_m,θ + λ · d_eff(m, θ)
```
where d_eff is estimated from Fisher information eigenspectrum.

### Blanket Emergence Condition
```
New blanket emerges when:
∂²E/∂xᵢ∂xⱼ → 0  for i,j in different proposed groups
```
i.e., when the Hessian becomes block-diagonal.

* * *

## 7. The Core Thesis (Mathematical Form)

*Structure learning = optimizing a partition to minimize free energy.*

In Bayesian models:
```
m* = argmin_m [F(m) + Ω(m)]
    where F(m) = -log p(data | m)  (negative log marginal likelihood)
```

In EBMs:
```
θ* = argmin_θ F(θ)
m*(θ) = partition induced by basins of E(·; θ*)
```

**The bridge**:
```
m* ≈ m*(θ*)  when:
1. EBM is expressive enough to represent optimal Bayesian structure
2. Optimization finds global minimum
3. Thermodynamic criteria correctly identify basin structure
```

* * *

## 8. Open Mathematical Questions

1. **When does EBM basin structure match Bayesian optimal structure?**
   - Sufficient conditions on E parameterization?
   - Role of temperature T in basin formation?

2. **How to formalize "emergent blanket"?**
   - Threshold on Hessian off-diagonal entries?
   - Information-theoretic criterion (mutual information)?

3. **What's the right complexity measure Ω(m)?**
   - Number of blankets?
   - Total blanket "surface area"?
   - Description length of the partition?

4. **Can thermodynamic fluctuations predict Bayesian model evidence?**
   - Fluctuation-dissipation → Fisher information → model complexity?
   - Is there an exact relation or just correlation?
