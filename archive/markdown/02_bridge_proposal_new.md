# Bridge Proposal: Geometric Topology Extraction as Complementary to Discrete Active Inference

## Overview

This document proposes how to bridge EBM continuous landscape optimization with discrete structure selection, positioning Topological Blankets relative to recent active inference works (2024–2025).

**Core Thesis**: Structure learning IS Markov blanket discovery and typology. Topological Blankets provides a continuous, geometric method for this task, complementary to discrete variational approaches.

* * *

## 1. Key Related Works (Accurate Summaries from Papers)

### 1.1 Da Costa (2024) — Toward Universal and Interpretable World Models

**Focus**: Expressive yet tractable Bayesian networks for open-ended learning agents.

**Key contributions**:
- Sparse, compositional class of Bayes nets approximating diverse stochastic processes
- Emphasis on interpretability, scalability, integration with structure learning
- Applications: Modeling video/audio from pixels, pixel-based planning

**Connection to our work**: Defines a target class of "interpretable" sparse Bayes nets. Our method offers a way to discover approximations to this class geometrically from EBMs without explicit discrete search.

### 1.2 Beck & Ramstead (2025) — Dynamic Markov Blanket Detection (DMBD)

**Focus**: Variational Bayesian EM for dynamic blanket detection from microscopic dynamics.

**Key contributions**:
- Uses blanket statistics to define object types and equivalence:
  - **Weak equivalence**: Same steady-state/reward rate
  - **Strong equivalence**: Same boundary paths
- Dynamic assignments ω_i(t) label elements as internal/blanket/external over time
- Applications: Newton's cradle, burning fuse, Lorenz attractor, simulated cell

**Connection to our work**: Closest prior art. DMBD is a discrete variational approach to blanket partitioning. Our method is a continuous geometric alternative—uses gradients/Hessians from Langevin sampling with no explicit role assignments or EM.

### 1.3 Friston et al. (2025) — Scale-Free Active Inference (RGM)

**Focus**: Renormalizing Generative Models for scale-free hierarchical structure.

**Key contributions**:
- Discrete POMDPs augmented with "paths/orbits" as latents for temporal depth
- Scale-invariant via renormalization group
- Discrete analogs of deep conv nets or continuous SSMs
- Applications: Image classification, movie/music compression, Atari planning

**Connection to our work**: Focus on hierarchical, scale-free discrete structures. Our method could provide geometric diagnostics for trained RGMs (if cast as EBMs) or extract equivalent blanket partitions from their energy landscapes.

### 1.4 Heins et al. (2025) — AXIOM

**Focus**: Object-centric architecture with expandable mixture models for RL.

**Key contributions**:
- Four expandable mixtures:
  - **sMM** (Slot): Parses pixels → object slots
  - **iMM** (Identity): Maps features → discrete types
  - **tMM** (Transition): Models dynamics as piecewise linear
  - **rMM** (Recurrent): Models object-object interactions
- Online growing heuristic + periodic Bayesian model reduction (BMR)
- Core priors: Piecewise linear dynamics, sparse interactions
- Results: Masters games in ~10k steps, sample-efficient, interpretable

**Connection to our work**: Discrete expanding structure for object discovery in RL. Similar goal (partition into objects/types), but mixture-based and online discrete. Our approach is offline geometric analysis of a fixed EBM landscape.

* * *

## 2. The Geometric Alternative: Topological Blankets

### 2.1 How It Differs

| Aspect | DMBD / AXIOM / RGM | Topological Blankets |
|--------|--------------------|-----------------------------|
| **Inference type** | Discrete variational (EM, mixtures, renormalization) | Continuous geometric (gradients, Hessians) |
| **Blanket definition** | Explicit role assignments ω_i(t) | High-gradient ridges separating basins |
| **Object discovery** | Mixture components, slot assignments | Low-gradient basins in energy landscape |
| **Dynamics** | Native time-evolving assignments | Static snapshot (extensions possible) |
| **Online/Offline** | Online (AXIOM), dynamic (DMBD) | Offline, post-hoc analysis |

### 2.2 Advantages of Geometric Approach

1. **Works directly on continuous EBMs**: No conversion to discrete structures needed
2. **No bespoke priors**: Post-hoc analysis via sampling, not mixture-specific growing rules
3. **Thermodynamic diagnostics**: Fisher information from fluctuations estimates complexity
4. **Soft structure**: Energy barriers provide continuous "edge strengths" vs binary edges
5. **Scalable**: Sampling-based, no discrete search over exponentially many structures

### 2.3 Potential Synergies

- **Geometric pre-partitioning**: Use basin detection to initialize AXIOM slots or DMBD role assignments
- **Diagnostic tool**: Apply to trained RGM/AXIOM models (convert to energy, extract topology)
- **Hybrid approaches**: Guide discrete structure search with geometric signals
- **Thermodynamic BMR**: Use Fisher fluctuations as alternative to AXIOM's BMR criteria

* * *

## 3. Blanket Statistics from Gradients (DMBD Integration)

DMBD defines object types via blanket statistics (steady-state or path distributions for equivalence). We can proxy these geometrically:

### 3.1 Steady-State Statistics (Weak Equivalence)

```python
def compute_blanket_statistics(gradients, is_blanket):
    """
    Proxy DMBD-style blanket statistics from gradient samples.

    Steady-state statistics characterize blanket "activity level"
    for weak equivalence (same reward rate / steady state).
    """
    blanket_grads = gradients[:, is_blanket]  # (N, n_blanket)

    steady_state = {
        'mean': np.mean(blanket_grads, axis=0),
        'variance': np.var(blanket_grads, axis=0),
        'magnitude': np.mean(np.abs(blanket_grads), axis=0)
    }

    return steady_state
```

### 3.2 Path Statistics (Strong Equivalence)

```python
def compute_path_autocorrelation(gradients, is_blanket, max_lag=50):
    """
    Proxy DMBD-style path statistics via gradient autocorrelation.

    Path statistics capture temporal correlations for strong
    equivalence (same boundary paths).
    """
    blanket_grads = gradients[:, is_blanket]
    n_samples, n_blanket = blanket_grads.shape

    autocorr = np.zeros((n_blanket, max_lag))
    for i in range(n_blanket):
        signal = blanket_grads[:, i]
        for lag in range(max_lag):
            if lag < n_samples - 1:
                autocorr[i, lag] = np.corrcoef(
                    signal[:-lag-1] if lag > 0 else signal[:-1],
                    signal[lag+1:]
                )[0, 1]

    return autocorr
```

### 3.3 Object Typing via Blanket Similarity

```python
def type_objects_by_blanket_statistics(blanket_stats, object_assignment):
    """
    Cluster objects into types based on their blanket statistics.

    Objects with similar blanket profiles (DMBD weak equivalence)
    represent the same "kind of thing".
    """
    from sklearn.cluster import AgglomerativeClustering

    # Build feature vector per object: aggregate blanket stats
    n_objects = object_assignment.max() + 1
    object_features = []

    for obj_id in range(n_objects):
        obj_blanket_mask = ...  # blankets bordering this object
        obj_stats = blanket_stats['variance'][obj_blanket_mask]
        object_features.append([np.mean(obj_stats), np.std(obj_stats)])

    # Cluster to discover types
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
    types = clustering.fit_predict(object_features)

    return types
```

* * *

## 4. Three-Timescale Dynamics (Unified Framework)

Following insights from DMBD (dynamic blankets) and RGM (scale-free hierarchy):

```
Fast (τ_x):     Langevin inference — find low-energy x given θ
                dx = -Γ_x ∇_x E dt + √(2Γ_x T_x) dW

Medium (τ_θ):   Parameter learning — sculpt landscape
                dθ = -Γ_θ ∇_θ F dt (natural gradient via fluctuations)

Slow (τ_m):     Topology extraction — crystallize structure
                Monitor Hessian sparsity, detect blanket emergence
```

### Level 1: Fast - Inference (DMBD: microscopic dynamics)
- Langevin sampling explores energy landscape
- Collects geometric data: trajectories, gradients
- Timescale: ~10-100 steps

### Level 2: Slow - Learning (AXIOM: mixture component updates)
- Parameter updates sculpt landscape
- Natural gradient via Fisher from fluctuations
- Timescale: ~1000-10000 steps

### Level 3: Slowest - Structure Selection (RGM: renormalization)
- Topology extraction via Topological Blankets
- Blanket detection, object clustering, graph construction
- Triggered by convergence or stagnation

* * *

## 5. Comparison: When to Use Which Approach

| Scenario | Best Approach | Reason |
|----------|---------------|--------|
| Online RL with pixel observations | AXIOM | Native mixture growing, active exploration |
| Physical system with microscopic trajectories | DMBD | Dynamic role assignments, path statistics |
| Hierarchical scale-free structure | RGM | Renormalization for temporal/spatial depth |
| Trained EBM diagnostic | Topological Blankets | Post-hoc, no retraining |
| Score-based / diffusion models | Topological Blankets | Native continuous energy |
| Equilibrium vs dynamic | TC (equilibrium) / DMBD (dynamic) | DMBD wins on time-resolved data |

* * *

## 6. Model Comparison Criterion (Bayesian Foundation)

From Bayesian model comparison:
```
ln P(m|o) / P(m'|o) = ΔF + ΔG
```

where:
- ΔF = difference in variational free energy (accuracy - complexity)
- ΔG = difference in expected free energy (epistemic + pragmatic value)

### Translating to EBMs

```
F(m) = E_q[E(x;θ)] + complexity(m)
```

**Thermodynamic complexity estimation** (avoiding explicit marginalization):
```
complexity(m) ≈ (1/2) log det I(θ)  ≈ (1/2) Σᵢ log Var[∂E/∂θᵢ]
```

This uses the Green-Kubo / fluctuation-dissipation relation to estimate Fisher information from gradient variance during sampling.

* * *

## 7. Algorithm: Thermodynamic Blanket Discovery

```python
def thermodynamic_blanket_discovery(E, θ, config):
    """
    Full pipeline combining Topological Blankets with
    DMBD-style blanket statistics for object typing.
    """
    # Phase 1: Geometric data collection (Langevin sampling)
    trajectories, gradients = collect_geometric_data(E, θ, config)

    # Phase 2: Feature computation
    features = compute_features(trajectories, gradients)

    # Phase 3: Blanket detection (high-gradient ridges)
    is_blanket, τ = detect_blankets(features, method=config['threshold'])

    # Phase 3.5: Blanket statistics (DMBD integration)
    blanket_stats = compute_blanket_statistics(gradients, is_blanket)
    path_stats = compute_path_autocorrelation(gradients, is_blanket)

    # Phase 4: Object clustering (basins)
    object_assignment = cluster_objects(features, is_blanket)

    # Phase 5: Object typing via blanket similarity (DMBD weak equivalence)
    object_types = type_objects_by_blanket_statistics(
        blanket_stats, object_assignment
    )

    # Phase 6: Topology extraction
    nodes, edges = extract_topology(object_assignment, blanket_membership)

    return {
        'objects': objects,
        'blankets': blanket_membership,
        'types': object_types,  # NEW: DMBD-style typing
        'graph': (nodes, edges),
        'blanket_stats': blanket_stats,  # NEW: for equivalence testing
        'path_stats': path_stats
    }
```

* * *

## 8. Connection to AXIOM's Expanding Structure

AXIOM grows/prunes mixture components online. We can achieve similar effects geometrically:

### 8.1 Growth (Basin Splitting)

**AXIOM**: Add new mixture component if log evidence favors it.

**Geometric analog**: Detect when a basin should split.
```python
def should_split_basin(features, object_id, threshold=0.5):
    """
    Detect if a basin should split (has internal sub-structure).
    """
    obj_vars = get_object_vars(object_id)
    internal_coupling = features['coupling'][np.ix_(obj_vars, obj_vars)]

    # If internal coupling has block structure, basin should split
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=2, affinity='precomputed')
    labels = sc.fit_predict(internal_coupling)

    # Check if split is meaningful (silhouette score)
    from sklearn.metrics import silhouette_score
    if silhouette_score(internal_coupling, labels) > threshold:
        return True, labels
    return False, None
```

### 8.2 Pruning (Basin Merging)

**AXIOM**: Bayesian Model Reduction merges similar components.

**Geometric analog**: Detect when basins should merge (barrier too low).
```python
def should_merge_basins(features, obj_i, obj_j, energy_threshold=0.1):
    """
    Detect if two basins should merge (barrier too weak).
    """
    # Coupling strength between objects
    obj_i_vars = get_object_vars(obj_i)
    obj_j_vars = get_object_vars(obj_j)
    cross_coupling = features['coupling'][np.ix_(obj_i_vars, obj_j_vars)]

    # If direct coupling is high, blanket isn't separating them
    if np.mean(cross_coupling) > energy_threshold:
        return True
    return False
```

### 8.3 Thermodynamic Alternative to BMR

Instead of comparing component marginal likelihoods, compare free energies:
```python
def thermodynamic_model_reduction(features, object_assignment, λ=0.01):
    """
    Prune objects that don't reduce free energy enough to justify complexity.
    """
    n_objects = object_assignment.max() + 1

    for obj_id in range(n_objects):
        obj_vars = get_object_vars(obj_id)

        # Complexity cost = Fisher information for this object's blanket
        blanket_vars = get_blanket_vars(obj_id)
        complexity = len(blanket_vars) * np.mean(
            features['grad_variance'][blanket_vars]
        )

        # Accuracy gain = coupling strength to other objects
        accuracy = compute_accuracy_gain(features, obj_id)

        # BMR criterion: prune if complexity > accuracy
        if λ * complexity > accuracy:
            merge_object_into_neighbors(obj_id, object_assignment)
```

* * *

## 9. Expected Free Energy for Structure Decisions

From active inference, expected free energy decomposes:
```
G = E_q[ln q(x) - ln p(x,y)]
  = -Information_gain - Expected_value
```

For structure decisions in EBMs:
```
G(m) = -I(m;Y) - E[accuracy(m)]
```

**Thermodynamic estimation**: Fisher information I(θ) under structure m provides a lower bound on I(m;Y) via Cramér-Rao. Higher Fisher information = more informative structure.

* * *

## 10. Summary: Positioning Topological Blankets

### The Core Equivalence

| Traditional Framing | Markov Blanket Framing |
|---------------------|------------------------|
| "How many latent factors?" | "How many objects have distinct blankets?" |
| "What's the right topology?" | "What's the conditional independence structure?" |
| "Should I add a hierarchy level?" | "Are there blankets within blankets?" |
| "What dynamics model?" | "What are the blanket statistics (input-output relations)?" |

### Three Approaches Unified

**Beck & Ramstead (DMBD)**:
- Discovers blankets via dynamic assignment variables ω_i(t)
- Typology from blanket statistics p(b_τ)
- Selection via parsimony (simplest macroscopic description)

**AXIOM (Heins et al.)**:
- Discovers blankets via slot mixture assignments
- Typology from identity mixture clusters
- Selection via Bayesian Model Reduction

**Topological Blankets (This Project)**:
- Blankets = high-gradient ridges in energy landscape
- Typology = similar blanket statistics (weak equivalence proxy)
- Selection via thermodynamic criteria (Fisher fluctuations)

### The EBM Advantage

EBM formulation provides *continuous relaxations* of discrete blanket decisions:
1. **Soft blanket boundaries**: Energy gradients indicate blanket sharpness
2. **Continuous typology**: Mode positions can merge/split smoothly
3. **Thermodynamic selection**: Fluctuation-based complexity without discrete search

### When It Works Best

Topological Blankets is ideal for:
- **Post-hoc analysis** of trained EBMs (score models, diffusion)
- **Diagnostic tool** for understanding what structure models learned
- **Equilibrium regimes** with clear basins and barriers
- **Continuous landscapes** where discrete search is intractable

It complements (not replaces) discrete methods for:
- Online learning with active exploration (use AXIOM)
- Time-resolved dynamics with traveling objects (use DMBD)
- Hierarchical scale-free structure (use RGM)

* * *

## Next Steps

→ Implementation will demonstrate:
1. Blanket statistics integration (DMBD-style typing)
2. Geometric growth/pruning (AXIOM-style structure changes)
3. Comparison experiments on shared toy problems
4. Diagnostic application to trained EBMs
