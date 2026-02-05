# Topological Blankets: Extracting Discrete Structure from Continuous Geometry

**Status**: Working document, iterating
**Core idea**: Use geometric inference (gradients/Hessians from Langevin sampling) to extract Markov blanket topology from EBM energy landscapes.

**Theoretical foundation**: Friston (2025) *A Free Energy Principle: On the Nature of Things* — see `07_friston_foundation_new.md` for exhaustive mathematical grounding.

* * *

## 0. Friston (2025) Grounding (Summary)

Friston derives Markov blankets from sparse coupling in Langevin dynamics:

**Langevin dynamics** (pp. 9-20):
$$\dot{x} = f(x) + \omega, \quad f(x) = -\Gamma \nabla_x E(x)$$

**Conditional independence** (pp. 213-217):
$$\mu \perp \eta \mid b \iff \frac{\partial^2 E}{\partial \mu_i \partial \eta_j} = 0 \quad \forall i,j$$

**This rigorously grounds our hypothesis**: High-gradient ridges = sparse flow separatrices = Markov blankets.

**Spectral detection** (pp. 48-51): Graph Laplacian on Hessian → slow eigenmodes = internals, intermediate = blankets.

**Hierarchical recursion** (pp. 53-64): Adiabatic elimination of fast modes → coarser particles.

Full details: `07_friston_foundation_new.md`

* * *

## 1. Problem Statement

**Given**: An energy-based model E(x; θ) over variables x = (x₁, ..., xₙ)

**Find**: The Markov blanket structure — a partition of variables into objects and their boundaries

* * *

## 2. Formal Setup

### 2.1 The Energy-Based Model

An EBM defines:
```
E : ℝⁿ × Θ → ℝ
p(x | θ) = exp(-E(x; θ)) / Z(θ)
```

### 2.2 Geometric Quantities

From E, we can compute:

**Gradient field**:
```
g(x) = ∇ₓE(x; θ) ∈ ℝⁿ
gᵢ(x) = ∂E/∂xᵢ
```

**Hessian (curvature)**:
```
H(x) = ∇²ₓE(x; θ) ∈ ℝⁿˣⁿ
Hᵢⱼ(x) = ∂²E/∂xᵢ∂xⱼ
```

**Fisher information** (expected Hessian):
```
I(θ) = 𝔼ₚ[H(x)] = 𝔼ₚ[∇²E]
```

### 2.3 Target Topological Structure

A **blanket partition** Π = {O₁, B₁, O₂, B₂, ..., Oₖ, Bₖ, S} where:
- Oₖ = internal variables of object k
- Bₖ = blanket (boundary) variables of object k
- S = external/environment variables

**Constraint**: Oₖ ⊥ S | Bₖ (conditional independence given blanket)

**Induced graph** G = (V, E):
- V = {O₁, O₂, ..., Oₖ} (objects as nodes)
- E = {(Oᵢ, Oⱼ) : Bᵢ ∩ Bⱼ ≠ ∅} (objects connected if blankets overlap)

* * *

## 3. The Core Hypothesis

**Hypothesis**: Markov blankets correspond to high-gradient regions in the energy landscape.

**Intuition**:
- Inside a basin: gradient is small (near minimum)
- On basin boundary: gradient is large (steep slope between basins)
- Blanket = the "ridge" separating conditionally independent regions

**Mathematical statement**:
```
xᵢ ∈ Blanket  ⟺  𝔼[||∂E/∂xᵢ||] > τ
```
for some threshold τ.

**Refinement**: It's not just magnitude, but also **connectivity**:
```
xᵢ ∈ Blanket connecting O_a and O_b  ⟺
    ||∂E/∂xᵢ|| is high  AND
    xᵢ couples to variables in both O_a and O_b
```

* * *

## 4. The Algorithm

### 4.1 Phase 1: Geometric Data Collection

**Input**: Energy function E(x; θ), ability to sample

**Procedure**:
```python
def collect_geometric_data(E, θ, n_samples, n_steps):
    """
    Collect geometric statistics via Langevin sampling.
    """
    trajectories = []
    gradients = []

    for _ in range(n_samples):
        # Initialize
        x = sample_prior()

        for t in range(n_steps):
            # Record
            g = gradient(E, x, θ)
            trajectories.append(x.copy())
            gradients.append(g.copy())

            # Langevin step
            noise = sqrt(2 * lr * T) * randn(n)
            x = x - lr * g + noise

    return trajectories, gradients
```

**Outputs**:
- Trajectory samples {x⁽ᵗ⁾}
- Gradient samples {g⁽ᵗ⁾ = ∇E(x⁽ᵗ⁾)}

### 4.2 Phase 2: Geometric Feature Computation

**Per-variable features**:

```python
def compute_features(trajectories, gradients):
    n_vars = trajectories[0].shape[0]

    # Gradient magnitude per variable
    grad_magnitude = zeros(n_vars)
    for i in range(n_vars):
        grad_magnitude[i] = mean([abs(g[i]) for g in gradients])

    # Gradient variance per variable (stability indicator)
    grad_variance = zeros(n_vars)
    for i in range(n_vars):
        grad_variance[i] = var([g[i] for g in gradients])

    # Hessian estimate via gradient covariance (fluctuation-dissipation)
    G = stack(gradients)  # (n_samples, n_vars)
    H_estimate = cov(G.T)  # (n_vars, n_vars)

    # Normalized coupling matrix
    D = sqrt(diag(H_estimate))
    coupling = abs(H_estimate) / outer(D, D)
    fill_diagonal(coupling, 0)  # Remove self-coupling

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian': H_estimate,
        'coupling': coupling
    }
```

**Outputs**:
- `grad_magnitude[i]`: Average |∂E/∂xᵢ| — blanket indicator
- `grad_variance[i]`: Var[∂E/∂xᵢ] — stability indicator
- `hessian[i,j]`: Estimated ∂²E/∂xᵢ∂xⱼ — interaction strength
- `coupling[i,j]`: Normalized interaction — for clustering

### 4.3 Phase 3: Blanket Detection

**Method A: Gradient Magnitude (Original)**

```python
def compute_blanket_scores(features):
    gm = features['grad_magnitude']
    # Normalize by median (robust to outliers)
    blanket_score = gm / median(gm)
    return blanket_score

def detect_blankets_gradient(blanket_score, method='otsu'):
    if method == 'otsu':
        τ = otsu_threshold(blanket_score)
    elif method == 'percentile':
        τ = percentile(blanket_score, 80)
    is_blanket = blanket_score > τ
    return is_blanket, τ
```

**Method B: Spectral Laplacian (Friston 2025, pp. 48-51)**

Superior for global structure detection; invariant to scaling.

```python
def detect_blankets_spectral(features, n_partitions=3):
    """
    Friston-style spectral blanket detection.

    1. Build adjacency from Hessian sparsity
    2. Compute graph Laplacian L = D - A
    3. Eigen-decompose: slow modes = internal, mid = blanket
    4. Cluster on eigenvector embedding
    """
    H = features['hessian']
    A = (np.abs(H) > threshold).astype(float)
    np.fill_diagonal(A, 0)

    D = np.diag(A.sum(1))
    L = D - A

    eigvals, eigvecs = np.linalg.eigh(L)

    # Cluster on first k non-trivial eigenvectors
    embedding = eigvecs[:, 1:n_partitions+1]
    labels = KMeans(n_clusters=n_partitions).fit_predict(embedding)

    # Blanket = cluster with highest eigenvector variance
    # (connects multiple regions → varied values)
    cluster_var = [np.var(eigvecs[labels == c, 1:4]) for c in range(n_partitions)]
    blanket_cluster = np.argmax(cluster_var)

    is_blanket = labels == blanket_cluster
    return is_blanket, eigvals
```

**Method C: Hybrid (Recommended)**

```python
def detect_blankets_hybrid(features, eigengap_threshold=0.5):
    """
    Use spectral if eigengap strong, else gradient fallback.
    """
    is_blanket_spectral, eigvals = detect_blankets_spectral(features)
    eigengap = np.max(np.diff(eigvals[:6]))

    if eigengap > eigengap_threshold:
        return is_blanket_spectral, 'spectral'
    else:
        blanket_score = compute_blanket_scores(features)
        is_blanket_grad, _ = detect_blankets_gradient(blanket_score)
        return is_blanket_grad, 'gradient'
```

### 4.4 Phase 4: Object Clustering

**Cluster non-blanket variables by coupling**:
```python
def cluster_objects(features, is_blanket):
    coupling = features['coupling']
    internal_vars = ~is_blanket

    # Submatrix of coupling among internal variables
    C_internal = coupling[internal_vars][:, internal_vars]

    # Spectral clustering on coupling matrix
    # Or: connected components with threshold
    n_clusters = estimate_n_clusters(C_internal)  # See Section 5.3
    labels = spectral_clustering(C_internal, n_clusters)

    # Map back to full variable set
    object_assignment = full(n_vars, -1)  # -1 = blanket
    object_assignment[internal_vars] = labels

    return object_assignment
```

### 4.5 Phase 4.5: Blanket Statistics (DMBD Integration)

**Compute DMBD-style blanket statistics for object typing**:
```python
def compute_blanket_statistics(gradients, is_blanket):
    """
    Proxy DMBD-style blanket statistics from gradient samples.

    Steady-state statistics characterize blanket "activity level"
    for weak equivalence (same reward rate / steady state).
    Path autocorrelation captures temporal structure for strong equivalence.
    """
    blanket_grads = gradients[:, is_blanket]  # (n_samples, n_blanket)

    steady_state = {
        'mean': np.mean(blanket_grads, axis=0),
        'variance': np.var(blanket_grads, axis=0),
        'magnitude': np.mean(np.abs(blanket_grads), axis=0)
    }

    # Path autocorrelation (proxy for DMBD strong equivalence)
    max_lag = min(50, len(gradients) // 2)
    autocorr = np.zeros((np.sum(is_blanket), max_lag))
    for i, b_idx in enumerate(np.where(is_blanket)[0]):
        signal = gradients[:, b_idx]
        for lag in range(1, max_lag):
            autocorr[i, lag] = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]

    return {
        'steady_state': steady_state,
        'path_autocorr': autocorr
    }
```

**Use blanket statistics for object typing** (DMBD weak equivalence):
```python
def type_objects_by_blanket_similarity(blanket_stats, object_assignment, blanket_membership):
    """
    Cluster objects into types based on their blanket profiles.
    Objects with similar blanket statistics are the same "kind of thing".
    """
    n_objects = object_assignment.max() + 1
    object_features = []

    for obj_id in range(n_objects):
        # Aggregate blanket stats for this object
        obj_blankets = [b for b, objs in blanket_membership.items() if obj_id in objs]
        if obj_blankets:
            obj_var = np.mean(blanket_stats['steady_state']['variance'][obj_blankets])
            obj_mag = np.mean(blanket_stats['steady_state']['magnitude'][obj_blankets])
        else:
            obj_var, obj_mag = 0, 0
        object_features.append([obj_var, obj_mag])

    # Cluster to discover types
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
    types = clustering.fit_predict(object_features)

    return types
```

### 4.6 Phase 5: Blanket Assignment

**Assign blanket variables to objects they border**:
```python
def assign_blankets(features, object_assignment, is_blanket):
    coupling = features['coupling']
    n_objects = object_assignment.max() + 1
    blanket_vars = where(is_blanket)[0]

    blanket_membership = {}  # blanket_var -> set of objects it borders

    for b in blanket_vars:
        # Which objects does this blanket variable couple to?
        bordering_objects = set()
        for i in range(len(object_assignment)):
            if object_assignment[i] >= 0:  # i is internal to some object
                if coupling[b, i] > coupling_threshold:
                    bordering_objects.add(object_assignment[i])

        blanket_membership[b] = bordering_objects

    return blanket_membership
```

### 4.6 Phase 6: Topology Extraction

**Build the graph**:
```python
def extract_topology(object_assignment, blanket_membership):
    n_objects = object_assignment.max() + 1

    # Nodes = objects
    nodes = list(range(n_objects))

    # Edges = objects that share blanket variables
    edges = set()
    for b, objects in blanket_membership.items():
        for o1 in objects:
            for o2 in objects:
                if o1 < o2:
                    edges.add((o1, o2))

    return nodes, edges
```

### 4.7 Full Algorithm

```python
def topological_blankets(E, θ, config):
    """
    Extract Markov blanket topology from EBM geometry.

    Parameters
    ----------
    E : callable
        Energy function E(x; θ)
    θ : parameters
        Current EBM parameters
    config : dict
        Algorithm configuration

    Returns
    -------
    topology : dict
        - 'objects': list of variable sets (internal to each object)
        - 'blankets': dict mapping blanket vars to bordering objects
        - 'graph': (nodes, edges) tuple
        - 'features': computed geometric features
    """
    # Phase 1: Collect geometric data
    trajectories, gradients = collect_geometric_data(
        E, θ,
        n_samples=config['n_samples'],
        n_steps=config['n_steps']
    )

    # Phase 2: Compute features
    features = compute_features(trajectories, gradients)

    # Phase 3: Detect blankets
    blanket_score = compute_blanket_scores(features)
    is_blanket, τ = detect_blankets(blanket_score, method=config['threshold_method'])

    # Phase 4: Cluster objects
    object_assignment = cluster_objects(features, is_blanket)

    # Phase 5: Assign blankets
    blanket_membership = assign_blankets(features, object_assignment, is_blanket)

    # Phase 6: Extract topology
    nodes, edges = extract_topology(object_assignment, blanket_membership)

    # Package results
    objects = [where(object_assignment == k)[0] for k in range(max(object_assignment) + 1)]

    return {
        'objects': objects,
        'blankets': blanket_membership,
        'graph': (nodes, edges),
        'features': features,
        'blanket_score': blanket_score,
        'threshold': τ
    }
```

* * *

## 5. Open Questions: Threshold Selection

### 5.1 The Problem

How to choose τ such that:
```
xᵢ ∈ Blanket ⟺ blanket_score[i] > τ
```

Too low τ: Everything is a blanket (no internal structure)
Too high τ: Nothing is a blanket (no boundaries detected)

### 5.2 Approach 1: Otsu's Method (Histogram-based)

Treat blanket_score as a histogram and find the threshold that maximizes between-class variance.

```python
def otsu_threshold(scores):
    """
    Find threshold that maximizes separation between two classes.
    """
    # Discretize scores into bins
    hist, bin_edges = histogram(scores, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Total statistics
    total = sum(hist)
    sum_total = sum(hist * bin_centers)

    best_τ, best_variance = 0, 0
    sum_background, weight_background = 0, 0

    for i, (count, center) in enumerate(zip(hist, bin_centers)):
        weight_background += count
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += count * center
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance = weight_background * weight_foreground * (mean_background - mean_foreground)**2

        if variance > best_variance:
            best_variance = variance
            best_τ = center

    return best_τ
```

**Pros**: Automatic, no hyperparameters
**Cons**: Assumes bimodal distribution (blanket vs internal)

### 5.3 Approach 2: Information-Theoretic

Choose τ to maximize information about object membership.

**Intuition**: The blanket should be the minimal set that makes objects conditionally independent.

```python
def information_threshold(blanket_scores, coupling, n_thresholds=50):
    """
    Find threshold that maximizes mutual information between
    inferred objects and the coupling structure.
    """
    thresholds = linspace(min(blanket_scores), max(blanket_scores), n_thresholds)

    best_τ, best_score = 0, -inf

    for τ in thresholds:
        # Tentative blanket assignment
        is_blanket = blanket_scores > τ

        # Cluster non-blanket variables
        internal_vars = ~is_blanket
        if sum(internal_vars) < 2:
            continue

        C_internal = coupling[internal_vars][:, internal_vars]

        # Score: How well-separated are the clusters?
        # Use silhouette score or modularity
        try:
            labels = spectral_clustering(C_internal, n_clusters=2)
            score = silhouette_score(C_internal, labels)
        except:
            score = -inf

        if score > best_score:
            best_score = score
            best_τ = τ

    return best_τ
```

**Pros**: Principled, relates to conditional independence
**Cons**: Requires choosing n_clusters, expensive

### 5.4 Approach 3: Phase Transition Detection

Vary τ and look for discontinuous changes in topology.

**Intuition**: At the "correct" threshold, topology should be stable. At wrong thresholds, small changes in τ cause large changes in structure.

```python
def phase_transition_threshold(blanket_scores, coupling):
    """
    Find threshold at phase transition (stability).
    """
    thresholds = linspace(min(blanket_scores), max(blanket_scores), 100)

    n_objects_list = []
    n_edges_list = []

    for τ in thresholds:
        is_blanket = blanket_scores > τ
        object_assignment = cluster_objects_simple(coupling, is_blanket)
        n_objects = len(unique(object_assignment[object_assignment >= 0]))

        # Count edges (simplified)
        n_edges = count_edges(object_assignment, is_blanket, coupling)

        n_objects_list.append(n_objects)
        n_edges_list.append(n_edges)

    # Find plateau (stable region)
    # Or: find τ where derivative is minimized
    stability = -abs(gradient(n_objects_list))
    best_idx = argmax(stability[10:-10]) + 10  # Avoid edges

    return thresholds[best_idx]
```

**Pros**: Captures notion of "robust" topology
**Cons**: May have multiple stable regions

### 5.5 Approach 4: Bayesian Model Selection

Put a prior on τ and compute posterior.

```python
def bayesian_threshold(blanket_scores, coupling, prior='uniform'):
    """
    Bayesian selection of threshold.
    """
    thresholds = linspace(min(blanket_scores), max(blanket_scores), 50)

    log_posteriors = []

    for τ in thresholds:
        # Prior
        if prior == 'uniform':
            log_prior = 0
        elif prior == 'sparse':
            # Favor fewer blankets
            log_prior = -0.1 * sum(blanket_scores > τ)

        # Likelihood: How well does this explain the coupling structure?
        is_blanket = blanket_scores > τ
        log_likelihood = coupling_likelihood(coupling, is_blanket)

        log_posteriors.append(log_prior + log_likelihood)

    best_idx = argmax(log_posteriors)
    return thresholds[best_idx]

def coupling_likelihood(coupling, is_blanket):
    """
    Log-likelihood that coupling structure is explained by blanket assignment.

    Model: Variables in same object have high coupling.
           Blanket variables have high coupling to multiple objects.
    """
    # Simplified: reward within-object coupling, penalize cross-object
    # (Full version would use proper generative model)
    internal = ~is_blanket
    C_internal = coupling[internal][:, internal]
    C_cross = coupling[internal][:, is_blanket]

    # Internal should be block-diagonal (high within-block)
    # This is a placeholder - real version needs cluster assignments
    log_lik = sum(C_internal) - 0.5 * sum(C_internal**2)

    return log_lik
```

* * *

## 6. Open Questions: Multi-Scale Structure

### 6.1 The Problem

Real systems have *hierarchical* blanket structure:
- Cells have membranes (blankets)
- Organs contain cells (blankets of blankets)
- Organisms contain organs (...)

How to discover blankets at multiple scales?

### 6.2 Approach 1: Temperature Annealing

At different temperatures T, different scales of structure are visible:
- High T: Only coarse structure (large basins merge)
- Low T: Fine structure (small basins distinguishable)

```python
def multiscale_crystallization(E, θ, temperatures):
    """
    Extract topology at multiple scales via temperature.
    """
    topologies = {}

    for T in sorted(temperatures, reverse=True):  # High to low
        # Langevin at this temperature
        trajectories, gradients = collect_geometric_data(E, θ, T=T, ...)

        # Extract topology
        topology = topological_blankets_at_T(trajectories, gradients)
        topologies[T] = topology

    # Build hierarchy: lower-T topologies refine higher-T
    hierarchy = build_hierarchy(topologies)

    return hierarchy
```

### 6.3 Approach 2: Persistent Homology

Track topological features across scales.

**Idea**: As we vary threshold τ (or temperature T):
- Some blankets appear (birth)
- Some blankets disappear (death)
- Long-lived features are "real" structure

```python
def persistent_blankets(blanket_scores, coupling):
    """
    Track blanket structure across thresholds.
    """
    thresholds = sorted(set(blanket_scores))

    persistence_diagram = []

    current_objects = {}  # object_id -> (birth_τ, variables)
    next_id = 0

    for τ in thresholds:
        is_blanket = blanket_scores > τ
        new_objects = cluster_objects_simple(coupling, is_blanket)

        # Track births and deaths
        # (This is simplified - real version needs proper tracking)
        for obj_vars in new_objects:
            if obj_vars not in current_objects.values():
                current_objects[next_id] = (τ, obj_vars)
                next_id += 1

        # Record deaths
        for obj_id, (birth, vars) in list(current_objects.items()):
            if vars not in new_objects:
                persistence_diagram.append((birth, τ, vars))
                del current_objects[obj_id]

    return persistence_diagram
```

### 6.4 Approach 3: Recursive Refinement

Apply crystallization recursively within discovered objects.

```python
def hierarchical_crystallization(E, θ, depth=0, max_depth=3):
    """
    Recursively discover finer structure within objects.
    """
    # Base topology
    topology = topological_blankets(E, θ, ...)

    if depth >= max_depth:
        return topology

    # For each discovered object, look for sub-structure
    for obj_idx, obj_vars in enumerate(topology['objects']):
        if len(obj_vars) < 3:  # Too small to subdivide
            continue

        # Restrict energy to this object's variables
        E_restricted = lambda x: E(embed(x, obj_vars), θ)

        # Recursively crystallize
        sub_topology = hierarchical_crystallization(
            E_restricted, θ, depth=depth+1, max_depth=max_depth
        )

        if len(sub_topology['objects']) > 1:
            topology['sub_structure'][obj_idx] = sub_topology

    return topology
```

* * *

## 7. Open Questions: Dynamics

### 7.1 The Problem

As the EBM learns (θ evolves), the topology may change:
- New objects emerge (basin splits)
- Objects merge (basin merger)
- Blankets sharpen or blur

How to track topology over learning?

### 7.2 Approach 1: Continuous Monitoring

Run crystallization periodically during learning.

```python
def learning_with_topology_tracking(E, data, config):
    """
    Learn EBM while tracking topological changes.
    """
    θ = initialize_parameters()
    topology_history = []

    for epoch in range(config['n_epochs']):
        # Learning step
        θ = learning_step(E, θ, data)

        # Periodic topology extraction
        if epoch % config['topology_interval'] == 0:
            topology = topological_blankets(E, θ, config)
            topology_history.append({
                'epoch': epoch,
                'topology': topology,
                'θ': θ.copy()
            })

            # Detect topology changes
            if len(topology_history) > 1:
                change = compare_topologies(
                    topology_history[-2]['topology'],
                    topology_history[-1]['topology']
                )
                if change['significant']:
                    log(f"Topology change at epoch {epoch}: {change}")

    return θ, topology_history
```

### 7.3 Approach 2: Topology-Aware Learning

Use topology to guide learning.

```python
def topology_aware_learning_step(E, θ, data, current_topology):
    """
    Learning step that respects/encourages current topology.
    """
    # Standard gradient
    grad = compute_gradient(E, θ, data)

    # Topology regularization: encourage sparse Hessian
    # consistent with discovered blanket structure
    H = estimate_hessian(E, θ)

    # Penalize coupling between variables in different objects
    # (not mediated by blanket)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if different_objects(i, j, current_topology):
                if not connected_via_blanket(i, j, current_topology):
                    # Penalize non-zero H[i,j]
                    grad += config['λ_topology'] * sign(H[i,j]) * grad_H_ij(θ)

    θ_new = θ - config['lr'] * grad
    return θ_new
```

### 7.4 Approach 3: Phase Transition Detection

Detect when topology is about to change.

```python
def detect_topology_phase_transition(topology_history):
    """
    Detect impending topology change from trends.
    """
    if len(topology_history) < 3:
        return False, None

    recent = topology_history[-3:]

    # Track blanket scores over time
    blanket_scores = [t['topology']['blanket_score'] for t in recent]

    # Look for variables approaching threshold
    τ = recent[-1]['topology']['threshold']
    current_scores = blanket_scores[-1]

    # Variables close to threshold are "critical"
    critical_vars = abs(current_scores - τ) < 0.1 * τ

    if any(critical_vars):
        # Check trend
        trends = [blanket_scores[-1][i] - blanket_scores[-2][i]
                  for i in where(critical_vars)[0]]

        # If trending toward threshold, phase transition likely
        if any(abs(t) > 0.01 for t in trends):
            return True, where(critical_vars)[0]

    return False, None
```

* * *

## 8. Theoretical Considerations and Robustness

### 8.1 When Does This Work?

**Assumption 1**: Markov blankets correspond to energy barriers.

This holds when:
- Conditional independence ⟺ weak coupling
- Weak coupling ⟺ small ∂²E/∂xᵢ∂xⱼ
- Small Hessian off-diagonal ⟺ low gradient in "between" region

**Assumption 2**: Gradient magnitude indicates blanket membership.

This holds when:
- Basins have clear minima (gradient ≈ 0 inside)
- Boundaries have steep gradients
- Energy landscape is not too flat overall

### 8.2 When It May Fail (Critical Assessment)

From Grok's feedback, the core hypothesis is *physically intuitive but empirically fragile*:

**Failure Mode 1: Rough Multi-Modal Landscapes**
- In high dimensions, gradients are high almost everywhere
- Common in undertrained or poorly regularized EBMs
- Symptom: Everything classified as blanket

**Failure Mode 2: Flat Landscapes**
- Under-regularized or poorly trained models
- Gradients uniformly low
- Symptom: No blankets detected, single merged basin

**Failure Mode 3: Sampling Issues**
- Noise swamps signal at high temperatures
- Insufficient samples for reliable Hessian estimates
- Symptom: Noisy coupling matrix, unstable clusters

**Failure Mode 4: Threshold Sensitivity**
- Heavy-tailed gradient distributions
- Multi-modal histograms defeat Otsu's method
- Symptom: Over/under-splitting depending on τ

### 8.3 Dimensionality and Scaling Concerns

**Coupling Matrix Scaling**:
- Full matrix is O(n²) in memory: n=10⁴ → 800 GB
- Mitigations:
  - Sparse/low-rank Hessian approximations (diagonal + low-rank)
  - Subsample variables for coupling estimation
  - Work in learned representation spaces, not raw pixels

**Sampling Budget**:
- Hessian estimation via covariance needs O(n) samples minimum
- For reliable statistics, typically need 10×-100× more
- Langevin convergence depends on landscape roughness

### 8.2 Relationship to Spectral Graph Theory

The Hessian H relates to the graph Laplacian L:
- For undirected graph: L = D - A (degree minus adjacency)
- Hessian plays similar role: encodes variable interactions

**Connection**:
```
Hᵢⱼ ≠ 0  ⟺  edge (i,j) in interaction graph
```

Spectral clustering on H (or derived coupling matrix) is natural.

### 8.3 Information-Theoretic Interpretation

**Claim**: Blankets are minimal sufficient statistics for conditional independence.

If B is blanket for Z (internal) with respect to S (external):
```
I(Z; S | B) = 0
```

**Gradient connection**:
```
High ||∇_B E|| ⟹ B is "informative" about E
                ⟹ B mediates between Z and S
```

* * *

* * *

## 9. Empirical Validation Strategy (From Grok Feedback)

### 9.1 Progressive Experiment Levels

**Level 1: Quadratic EBMs with Exact Block Structure** (Easiest)
- Analytic gradients/Hessians, exact sampling possible
- Full control over barrier separation strength
- Test: Recovery as function of `blanket_strength` parameter
- Expected: Near-perfect when barriers clear, degrades when weak
- Implementation: `experiments/quadratic_toy_comparison.py`

**Level 2: Mixture-Based EBMs** (Multi-Modal)
- Gaussian Mixture Model as EBM: `E(x) = -log(Σ π_k N(x; μ_k, Σ_k))`
- Ground truth: K components = objects
- Test: Does method recover K without being told?
- Compare to EM clustering baseline

**Level 3: Graphical Models as EBMs** (Direct Topology Ground Truth)
- Ising model: `E(x) = -Σ J_ij x_i x_j - Σ h_i x_i`
- Gaussian graphical model: Precision matrix = Hessian
- Bayesian networks converted to joint energy
- Compare to NOTEARS, DAGMA, PC algorithm

**Level 4: Real Trained EBMs** (Most Realistic)
- Pretrained score-based models on MNIST/CIFAR
- VAE latent spaces as EBMs
- Proxy ground truth from class labels
- Dynamics tracking during training

### 9.2 Metrics (Define Early)

| Category | Metric | Description | Ideal |
|----------|--------|-------------|-------|
| Object Partition | Adjusted Rand Index (ARI) | Clustering accuracy vs ground truth | 1.0 |
| | F1-score (macro) | Per-object precision/recall | 1.0 |
| Blanket Detection | Blanket F1 | Precision/recall of blanket classification | 1.0 |
| | Boundary IoU | Overlap with true blanket sets | 1.0 |
| Induced Graph | Structural Hamming Distance | Edge additions/deletions vs truth | 0 |
| | Graph F1 | Edge-level precision/recall | 1.0 |

### 9.3 Baselines (Always Include)

1. **Standard structure learning**: NOTEARS, DAGMA (continuous), PC/GES (score-based)
2. **Spectral clustering on raw Hessian**: Phase 4 without blanket step
3. **Random partitioning / uniform graph**: Lower bound
4. **DMBD-style clustering**: Role features → KMeans (see comparison code)
5. **AXIOM-style GMM**: Mixture components → boundaries

### 9.4 Ablations

- Gradient magnitude only vs full features (variance + coupling)
- Threshold methods: Otsu vs percentile vs information-theoretic
- With/without persistent homology for multi-scale
- Temperature sensitivity: Low-T sharp vs High-T blurred

* * *

## 10. Summary and Next Steps

### What We Have

1. **Formal algorithm** for extracting topology from EBM geometry
2. **Multiple approaches** for threshold selection
3. **Frameworks** for multi-scale and dynamic topology
4. **DMBD integration** via blanket statistics (Section 3.5)
5. **Robustness analysis** identifying failure modes (Section 8.2)
6. **Validation strategy** with progressive levels (Section 9)

### What's Implemented

1. **Quadratic toy comparison**: `experiments/quadratic_toy_comparison.py`
   - Compares TC vs DMBD-style vs AXIOM-style
   - Sweeps blanket strength, computes ARI/F1
   - Visualizes landscape and results

### What's Needed

1. **Level 2-4 experiments**: GMM, graphical models, real EBMs
2. **Scaling**: Sparse Hessian approximations for large n
3. **Dynamics tracking**: Monitor topology during training
4. **Baselines**: NOTEARS/PC comparison on same problems

### Expected Paper Structure (From Grok)

- Section 5.1: Perfect recovery on quadratics (phase diagrams)
- Section 5.2: Competitive with NOTEARS on graphical models
- Section 5.3: Emergent hierarchy on mixtures
- Section 5.4: Insightful topology on real EBM (e.g., digit basins)

* * *

## 10. Notation Summary

| Symbol | Meaning |
|--------|---------|
| E(x; θ) | Energy function |
| x = (x₁, ..., xₙ) | Variables |
| g = ∇E | Gradient |
| H = ∇²E | Hessian |
| τ | Blanket threshold |
| ωᵢ ∈ {S, B, Z} | Variable role assignment |
| Oₖ | Internal variables of object k |
| Bₖ | Blanket of object k |
| G = (V, E) | Extracted topology graph |
