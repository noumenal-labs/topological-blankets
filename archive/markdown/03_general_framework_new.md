# General Framework: Structure in Energy-Based vs Bayesian Models

* * *

## 1. The Core Distinction

At the most general level, there are two ways to specify a probabilistic model:

### Energy-Based Models (EBMs)

Define a scalar energy function:
```
E(x, y; θ) : X × Y × Θ → ℝ
```

The probability is implicit via the Boltzmann distribution:
```
p(x, y | θ) = exp(-E(x, y; θ)) / Z(θ)
```

**Structure is implicit**: encoded in the geometry of E (basins, ridges, saddles).

### Bayesian Models

Define an explicit joint distribution:
```
p(x, y | θ, m) = p(y | x, θ_likelihood) p(x | θ_prior)
```

where m specifies the factorization structure.

**Structure is explicit**: encoded in the graphical model / factorization.

* * *

## 2. The Equivalence (and Why It's Incomplete)

### Mathematical Equivalence

Every EBM defines a Bayesian model:
```
p(x, y | θ) ∝ exp(-E(x, y; θ))
```

Every Bayesian model defines an EBM:
```
E(x, y; θ, m) = -log p(x, y | θ, m)
```

### The Structural Gap

Despite this equivalence, they differ in how structure enters:

| Aspect | EBM | Bayesian |
|--------|-----|----------|
| Structure location | Implicit in E geometry | Explicit in factorization m |
| Structure learning | Emergent (landscape sculpting) | Discrete search over m |
| Conditional independence | Basin boundaries | Graph d-separation |
| Model comparison | ??? | p(m\|data) ∝ p(data\|m)p(m) |

**The gap**: EBMs have no native mechanism for comparing structures because structure isn't a separate object - it's baked into the energy function.

* * *

## 3. Markov Blankets: The Unifying Concept

### Definition

A Markov blanket B for variables Z separates them from external variables S:
```
p(z, s | b) = p(z | b) p(s | b)
```

Z and S are conditionally independent given B.

### In Bayesian Models

Blankets are determined by graph structure:
- Parents, children, and co-parents of Z
- Read directly from the graphical model
- Structure learning = finding the right graph = finding the right blankets

### In EBMs

Blankets are determined by energy landscape geometry:
- Basin boundaries separate conditionally independent regions
- If moving from region A to region B requires crossing a high-energy barrier, A and B are approximately independent
- Structure learning = shaping the landscape = shaping the blankets

### The Key Insight

*Structure IS Markov blanket structure.*

Both frameworks are ultimately about partitioning variables into conditionally independent groups. They just represent this differently:
- Bayesian: explicit edges in a graph
- EBM: implicit barriers in an energy landscape

* * *

## 4. Structure Learning in Each Framework

### Bayesian Approach

```
m* = argmax_m p(m | data)
   = argmax_m [p(data | m) p(m)]
   = argmax_m [∫ p(data | θ, m) p(θ | m) dθ] p(m)
```

**Challenge**: The integral (marginal likelihood) is intractable for complex models.

**Solutions**:
- Variational bounds (ELBO)
- Bayesian model reduction (local approximation)
- Information criteria (AIC, BIC, MDL)

### EBM Approach

No explicit m to optimize. Instead:
```
θ* = argmin_θ F(θ)
```

where F is free energy, and structure emerges from θ*.

**Challenge**: How to know if the emergent structure is "right"?

**Solutions** (proposed):
- Monitor landscape geometry (basin count, boundary sharpness)
- Use thermodynamic criteria (Fisher information, gradient fluctuations)
- Explicitly parameterize structure and optimize jointly

* * *

## 5. Thermodynamic Criteria for Structure

The key insight from statistical physics: *fluctuations reveal structure*.

### Fisher Information

```
I(θ) = E[(∇_θ log p)²] = -E[∇²_θ log p]
```

Measures sensitivity of the model to parameter changes.

**For structure**: High Fisher information in some directions, low in others, indicates the model has learned distinct "modes" (blankets).

### Fluctuation-Dissipation Relations

From non-equilibrium thermodynamics:
```
Transport coefficient ∝ ∫ ⟨fluctuation(0) · fluctuation(t)⟩ dt
```

**For structure**: Gradient fluctuations during sampling reveal:
- Which dimensions are "active" (high variance = used)
- Where basin boundaries are (high gradient = boundary)
- Effective model complexity (integrated fluctuation magnitude)

### Free Energy Decomposition

```
F = E[E(x,y;θ)] - H[q(x)]
  = Accuracy - Complexity (heuristically)
```

For comparing structures:
```
F(structure_1) vs F(structure_2)
```

Lower F = better structure (if we can define F for each structure).

* * *

## 6. Bridging the Frameworks

### Option A: Explicit Structure in EBMs

Add structure as a parameter:
```
E(x, y; θ, m)
```

Now we can do Bayesian model comparison:
```
p(m | data) ∝ exp(-F(m)) p(m)
```

where F(m) = min_θ F(θ; m).

**Problem**: This loses the "emergent structure" benefit of EBMs.

### Option B: Emergent Structure from EBMs

Let structure emerge, but monitor it:
```
m(θ) = structure_implied_by(E(·; θ))
```

For example:
- Count basins → number of object types
- Measure basin separation → blanket strength
- Analyze Hessian eigenspectrum → effective dimension

**Problem**: Defining m(θ) rigorously is hard.

### Option C: Hybrid (Proposed)

Three-level optimization:
1. **Fast (inference)**: Find low-energy x given current θ, m
2. **Slow (learning)**: Update θ given current m
3. **Slowest (structure)**: Update m based on emergent properties of θ

The key: Use thermodynamic observables to guide discrete m decisions:
- If gradient variance is high → current m is insufficient → grow
- If basins overlap → current m is redundant → merge
- If effective dimension << nominal dimension → current m is wasteful → prune

* * *

## 7. The Unified View

### Structure Learning = Blanket Discovery = Partition Optimization

Regardless of whether we use EBMs or Bayesian models:

1. **The goal** is to find the right conditional independence structure
2. **This structure** corresponds to Markov blankets
3. **Free energy** (in various forms) is the criterion
4. **The challenge** is making discrete structural decisions

### What Each Framework Contributes

**From Bayesian models**:
- Explicit structure representation (graphs, factorizations)
- Principled model comparison (marginal likelihood)
- Compositional structure (hierarchies, factors)

**From EBMs**:
- Continuous optimization (no discrete search)
- Emergent structure (basins form naturally)
- Thermodynamic diagnostics (fluctuations reveal structure)

### The Synthesis

Use EBM optimization with Bayesian structure comparison:

```
1. Parameterize a family of structures M = {m₁, m₂, ...}
2. For each m, define E(x, y; θ, m)
3. Optimize θ via EBM methods (sampling, contrastive, etc.)
4. Compare structures via free energy: F(m) = min_θ F(θ; m) + complexity(m)
5. Use thermodynamic criteria to propose new structures
```

* * *

## 8. Concrete Instantiations

### Latent Variable EBMs

- EBM: E(z, y) = prior_energy(z) + likelihood_energy(z, y)
- Structure: latent dimension, factorization
- Blankets: basins in latent space

### Bayesian Networks

- Bayesian: p(o, s | m) with explicit graph m
- Structure: state factors, temporal depth, hierarchy
- Blankets: explicit in the graphical model

### VAEs

- Hybrid: encoder q(z|x), decoder p(x|z), prior p(z)
- Structure: latent dimension, factorization of z
- Blankets: implicit in the encoder/decoder architecture

### Diffusion Models

- EBM (score-based): learn ∇_x log p(x)
- Structure: noise schedule, architecture
- Blankets: implicit in the score function geometry

* * *

## 9. Research Questions

1. **Can thermodynamic criteria replace discrete structure search?**
   - Use gradient fluctuations instead of model comparison
   - Let blankets emerge from landscape optimization

2. **What's the minimal explicit structure needed?**
   - Maybe just: number of top-level blankets (object count)
   - Let internal structure emerge

3. **How do blankets at different scales interact?**
   - Hierarchical blankets (blankets within blankets)
   - Coarse-graining and renormalization

4. **Is there a universal free energy for structure?**
   - Something like: F_structure = E[energy] + λ·blanket_complexity
   - Where blanket_complexity counts/measures the partition

* * *

## 10. Summary

| Concept | Bayesian | EBM | Unified |
|---------|----------|-----|---------|
| Structure | Graph m | Landscape geometry | Markov blankets |
| Learning | Update θ given m | Update θ (m implicit) | Three timescales |
| Selection | p(m\|data) | Emergent | Free energy + thermodynamic criteria |
| Independence | d-separation | Basin boundaries | Blanket statistics |

**The bridge**: Markov blankets are the common language. Both frameworks are partitioning the world into conditionally independent pieces. The difference is whether this partition is declared upfront (Bayesian) or discovered through optimization (EBM).

**The opportunity**: Use EBM optimization for continuous structure learning, with Bayesian model comparison for discrete decisions, guided by thermodynamic diagnostics.

* * *

## Next Steps

1. Formalize "emergent blanket" mathematically
2. Define thermodynamic structure criteria rigorously
3. Implement hybrid algorithm on toy problems
4. Test whether thermodynamic criteria match Bayesian model comparison
