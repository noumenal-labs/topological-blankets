# Theoretical Foundation: Friston (2025) and Topological Blankets

**Status**: Core theoretical grounding
**Source**: Friston (2025) *A Free Energy Principle: On the Nature of Things* (~255 pages)

This document exhaustively aligns Topological Blankets with the rigorous mathematics from Friston (2025), establishing the project as a computational operationalization of FEP ontology in energy-based models.

* * *

## 1. The Core Convergence

Friston (2025) derives Markov blankets as *ontological primitives* — "things" emerge from sparse coupling in random dynamical systems. The Topological Blankets method operationalizes this in equilibrium EBMs:

| Friston (2025) | Topological Blankets |
|----------------|----------------------------|
| Sparse Langevin flow | High-gradient ridges |
| Zero Jacobian cross-blocks | Hessian sparsity pattern |
| Spectral Laplacian modes | Coupling matrix clustering |
| Surprisal gradients | Energy gradients |
| Recursive blankets | Hierarchical basin refinement |

**Key insight**: Friston provides the *why* (physics of emergence); we provide the *how* (computational extraction from EBMs).

* * *

## 2. Langevin Dynamics as Foundation

### 2.1 Friston's Formulation (pp. 9-20, 41, 87, 90, 105, 119)

Random dynamical systems governed by Langevin equation:

$$\dot{x} = f(x) + \omega$$

where:
- $x \in \mathbb{R}^n$ is the state vector
- $f(x)$ is the **particular flow** (deterministic drift)
- $\omega \sim \mathcal{N}(0, 2\Gamma)$ is Gaussian fluctuations

The flow $f(x)$ can be decomposed (Helmholtz, pp. 112-119):

$$f(x) = (\Gamma + Q) \nabla \ln p(x) = -(\Gamma + Q) \nabla \tilde{\mathcal{S}}(x)$$

where:
- $\Gamma$ is the symmetric diffusion tensor (dissipative)
- $Q$ is the antisymmetric solenoidal flow (conservative)
- $\tilde{\mathcal{S}}(x) = -\ln p(x)$ is the **surprisal** (self-information)

### 2.2 EBM Mapping

In energy-based models, the connection is direct:

$$E(x) \equiv \tilde{\mathcal{S}}(x) = -\ln p(x) + \text{const}$$

Thus:

$$f(x) = -\Gamma \nabla_x E(x)$$

*Our Langevin sampling IS Friston's dynamics* — gradient descent on energy with noise.

### 2.3 Implications for Blanket Detection

High $||\nabla E||$ regions correspond to:
- Steep surprisal gradients
- Regions resisting flow across partitions
- **Separatrices** between conditionally independent basins

This rigorously grounds our hypothesis: *Blankets = high-gradient ridges*.

* * *

## 3. Markov Blanket Partition (The Core Mathematics)

### 3.1 Partition Structure (pp. 25-27, 57, 216-217)

Friston partitions state space into:
- $\eta$ — **External states** (environment)
- $b = (s, a)$ — **Blanket states** (sensory $s$ + active $a$)
- $\mu$ — **Internal states** (the "thing")

The dynamics become:

$$\begin{bmatrix} \dot{\eta} \\ \dot{b} \\ \dot{\mu} \end{bmatrix} = \begin{bmatrix} f_\eta(\eta, b) \\ f_b(\eta, b, \mu) \\ f_\mu(b, \mu) \end{bmatrix} + \omega$$

**Critical structure**:
- External $\eta$ only depends on $(\eta, b)$ — no direct $\mu$ influence
- Internal $\mu$ only depends on $(b, \mu)$ — no direct $\eta$ influence
- Blanket $b$ mediates all cross-partition interactions

### 3.2 Conditional Independence Corollary (pp. 213-217)

**Theorem (Friston)**: Internal states are conditionally independent of external states given blanket:

$$\mu \perp \eta \mid b$$

**iff** the cross-Jacobian blocks are zero:

$$\nabla_{\eta\mu} \tilde{\mathcal{S}} = 0 \quad \text{and} \quad \nabla_{\mu\eta} \tilde{\mathcal{S}} = 0$$

Equivalently, in EBM terms:

$$\frac{\partial^2 E}{\partial \eta_i \partial \mu_j} = 0 \quad \forall i, j$$

**This is exactly our Hessian sparsity criterion**:

$$\text{Edge}(i,j) \iff \frac{\partial^2 E}{\partial x_i \partial x_j} \neq 0$$

### 3.3 Proof Sketch

From the Fokker-Planck equation, the steady-state density factorizes:

$$p(\eta, b, \mu) = p(\eta, b) \cdot p(\mu | b)$$

iff the flow admits no direct $\eta \leftrightarrow \mu$ coupling. The surprisal (energy) then decomposes:

$$\tilde{\mathcal{S}}(\eta, b, \mu) = \tilde{\mathcal{S}}(\eta, b) + \tilde{\mathcal{S}}(\mu | b)$$

with zero cross-derivatives.

* * *

## 4. Spectral Blanket Detection (Friston's Method)

### 4.1 Graph Laplacian Approach (pp. 48-51, 58-61, 67-70)

Friston's spectral method for blanket detection:

**Step 1**: Construct adjacency matrix from Jacobian/coupling:

$$A_{ij} = \begin{cases} 1 & \text{if } |J_{ij}| > \epsilon \\ 0 & \text{otherwise} \end{cases}$$

where $J = \nabla_x f(x)$ is the Jacobian of flow.

In EBMs: $J \approx -\Gamma H$ where $H = \nabla^2 E$ is the Hessian.

**Step 2**: Form graph Laplacian:

$$L = D - A$$

where $D_{ii} = \sum_j A_{ij}$ is the degree matrix.

**Step 3**: Eigen-decomposition:

$$L v_k = \lambda_k v_k$$

with $0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_n$.

**Step 4**: Interpret eigenmodes:
- **Slow modes** (small $\lambda_k$): Internal states (stable, slowly mixing)
- **Intermediate modes**: Blanket states (connecting structure)
- **Fast modes** (large $\lambda_k$): External/noise (rapidly mixing)

### 4.2 Spectral Clustering Interpretation

The **Fiedler vector** $v_1$ (second smallest eigenvalue) partitions the graph:
- Sign of $v_1(i)$ indicates partition membership
- Multiple eigenvectors → multi-way partition

For blanket detection:
- Cluster on $(v_1, v_2, \ldots, v_k)$ using K-means
- Intermediate cluster = blanket variables

### 4.3 Advantages Over Gradient Thresholding

| Gradient Magnitude | Spectral Method |
|--------------------|-----------------|
| Local measure | Global structure |
| Sensitive to scaling | Invariant to rescaling |
| Requires threshold τ | Natural gaps in spectrum |
| Fails on flat landscapes | Detects connectivity |
| Single-scale | Multi-scale via eigengap |

**Recommendation**: Use spectral as primary, gradient as validation/visualization.

* * *

## 5. Hierarchical/Recursive Blankets

### 5.1 Nested Structure (pp. 10-14, 53-64)

Friston emphasizes *recursive* blanket structure:
- Cells have membranes (blankets)
- Organs contain cells (blankets of blankets)
- Organisms contain organs
- Societies contain organisms

Mathematically: Blankets at scale $n$ become **particles** at scale $n+1$.

### 5.2 Adiabatic Elimination (pp. 58-64)

To coarse-grain:
1. Identify fast (external) modes via spectral analysis
2. **Adiabatically eliminate** fast variables (average over their equilibrium)
3. Remaining slow modes = new particle at coarser scale
4. Repeat

**Schur complement** formulation (for quadratic systems):

If $H = \begin{bmatrix} H_{\text{fast}} & H_{\text{cross}} \\ H_{\text{cross}}^T & H_{\text{slow}} \end{bmatrix}$,

the effective Hessian for slow modes is:

$$H_{\text{eff}} = H_{\text{slow}} - H_{\text{cross}}^T H_{\text{fast}}^{-1} H_{\text{cross}}$$

### 5.3 Recursive Algorithm

```python
def recursive_blanket_detection(H, max_levels=3):
    """
    Friston-style recursive blanket detection.

    At each level:
    1. Detect blankets via spectral method
    2. Coarse-grain: Eliminate fast modes (external)
    3. Remaining = new particle → repeat
    """
    hierarchy = []
    current_H = H.copy()
    current_vars = list(range(H.shape[0]))

    for level in range(max_levels):
        # Spectral detection at current scale
        L = np.diag(current_H.sum(1)) - np.abs(current_H)
        eigvals, eigvecs = np.linalg.eigh(L)

        # Identify blanket via Fiedler + clustering
        n_clusters = min(3, len(current_vars))
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters).fit_predict(eigvecs[:, 1:n_clusters])

        # Eigengap analysis: blanket = intermediate eigenvalues
        eigengaps = np.diff(eigvals)
        blanket_start = np.argmax(eigengaps[:len(eigvals)//2]) + 1

        # Record hierarchy
        internals = np.where(labels == 0)[0]
        blanket = np.where(labels == 1)[0]
        external = np.where(labels == 2)[0] if n_clusters > 2 else []

        hierarchy.append({
            'level': level,
            'internals': [current_vars[i] for i in internals],
            'blanket': [current_vars[i] for i in blanket],
            'external': [current_vars[i] for i in external],
            'eigengap': eigengaps[blanket_start-1] if blanket_start > 0 else 0
        })

        # Coarse-grain: Keep blanket + internals as new particle
        keep = list(internals) + list(blanket)
        if len(keep) < 2:
            break

        # Schur complement for effective Hessian
        # (Simplified: just submatrix for now)
        current_H = current_H[np.ix_(keep, keep)]
        current_vars = [current_vars[i] for i in keep]

    return hierarchy
```

* * *

## 6. Gradient Flow on Surprisal

### 6.1 Internal State Dynamics (pp. 112-114, 121-130)

Internal states perform gradient flow on surprisal (with solenoidal component):

$$\dot{\mu} = -\Gamma_\mu \nabla_\mu \tilde{\mathcal{S}}(\mu, b) + Q_\mu \nabla_\mu \tilde{\mathcal{S}}(\mu, b)$$

The dissipative (gradient) part minimizes surprisal; solenoidal part conserves it.

### 6.2 Active States and Agency

Active states $a \subset b$ influence external states:

$$\dot{a} = -\Gamma_a \nabla_a F(\mu, b)$$

where $F$ is the **variational free energy** — internal states' beliefs about external states.

**This is active inference**: Agents act to minimize expected surprisal.

### 6.3 Link to Expected Free Energy

For structure decisions (our structure learning problem):

$$G(\pi) = \mathbb{E}_{q(o|\pi)}[F(\mu, b)] + \text{epistemic value}$$

Friston's expected free energy $G$ guides discrete choices (policies, structures).

**Our thermodynamic criterion** (Fisher complexity) approximates this:

$$\text{complexity}(m) \approx \frac{1}{2} \log \det I(\theta) \approx \frac{1}{2} \sum_i \log \text{Var}[\partial E / \partial \theta_i]$$

* * *

## 7. What Friston Does NOT Cover (Our Novelty)

### 7.1 Absences in Friston (2025)

Exhaustive search confirms NO mentions of:
- "Energy-based model" / "EBM"
- "Basin" / "basin of attraction"
- "Hessian" (as computational object)
- "Gradient magnitude" as blanket indicator
- "Threshold" / "Otsu"
- "Spectral clustering" (uses spectral but not ML clustering)
- "Score-based model" / "diffusion model"

### 7.2 Our Unique Contributions

1. **EBM framing**: Map surprisal to energy function directly
2. **Gradient magnitude hypothesis**: High ||∇E|| = blanket (operational criterion)
3. **Hessian-based coupling**: Fluctuation-dissipation estimation from sampling
4. **Threshold methods**: Otsu, percentile, information-theoretic for discrete partitioning
5. **Quadratic toy validation**: Controlled experiments with known structure
6. **Comparison to ML methods**: NOTEARS, DMBD, AXIOM benchmarks
7. **Application to modern EBMs**: Score-based, diffusion, VAE latent spaces

### 7.3 Synthesis Statement

> Topological Blankets operationalizes Friston (2025)'s physics of emergent things in the computational setting of energy-based models. Where Friston derives blankets from sparse Langevin flow, we detect them via gradient magnitudes and Hessian sparsity in equilibrium samples. This bridges fundamental physics with practical ML structure discovery.

* * *

## 8. Complete Mathematical Framework

### 8.1 Unified Notation

| Symbol | Friston (2025) | Our Method |
|--------|----------------|------------|
| $x$ | Generalized states | EBM variables |
| $\tilde{\mathcal{S}}(x)$ | Surprisal | Energy $E(x)$ |
| $f(x)$ | Particular flow | $-\Gamma \nabla E$ |
| $J$ | Jacobian $\nabla f$ | $-\Gamma H$ (Hessian) |
| $L$ | Graph Laplacian | Coupling Laplacian |
| $\mu, b, \eta$ | Internal, blanket, external | Objects, blankets, environment |

### 8.2 Core Equations Summary

**Langevin dynamics**:
$$dx = -\Gamma \nabla_x E(x) dt + \sqrt{2\Gamma T} dW$$

**Conditional independence** (Friston Corollary):
$$\mu \perp \eta \mid b \iff \frac{\partial^2 E}{\partial \mu_i \partial \eta_j} = 0 \quad \forall i,j$$

**Blanket criterion** (our hypothesis, Friston-grounded):
$$x_i \in \text{Blanket} \iff \mathbb{E}[||\partial E / \partial x_i||] > \tau$$

**Spectral detection** (Friston method):
$$L = D - A, \quad A_{ij} = \mathbf{1}[|H_{ij}| > \epsilon]$$

Blankets in intermediate Laplacian eigenmodes.

**Hierarchical recursion**:
$$H^{(n+1)}_{\text{eff}} = H^{(n)}_{\text{slow}} - H^{(n)}_{\text{cross}}{}^T (H^{(n)}_{\text{fast}})^{-1} H^{(n)}_{\text{cross}}$$

**Fisher complexity**:
$$\text{complexity}(m) \approx \frac{1}{2} \sum_i \log \text{Var}[\nabla_{\theta_i} E]$$

* * *

## 9. Implications for the Project

### 9.1 Theoretical Closure

The project now has *derivational grounding* in FEP physics:
- Blankets aren't ad-hoc — they're ontological necessities from sparse coupling
- Gradient magnitude isn't heuristic — it's the flow separatrix strength
- Spectral detection isn't just clustering — it's Friston's recommended method

### 9.2 Methodological Enhancements

**Phase 3 upgrade**: Hybrid gradient + spectral detection
```
if spectral_eigengap > threshold:
    use spectral partition
else:
    fall back to gradient thresholding
```

**Phase 5 upgrade**: Full recursive detection (Friston hierarchy)

**Validation upgrade**: Test on Friston's "active soup" simulations

### 9.3 Positioning Statements

For papers/presentations:
> "We operationalize Friston (2025)'s physics of emergent things in energy-based models, detecting Markov blankets via gradient magnitudes and spectral methods on Hessian estimates."

> "Topological Blankets provides a non-variational alternative to DMBD/AXIOM, directly extracting blanket topology from equilibrium EBM geometry — grounded in the Free Energy Principle."

* * *

## 10. References

- Friston, K. (2025). *A Free Energy Principle: On the Nature of Things*. (Book manuscript, ~255 pages)
  - Langevin foundations: pp. 9-20, 87, 105, 119
  - Blanket partition: pp. 25-27, 57, 213-217
  - Spectral detection: pp. 48-51, 58-61, 67-70
  - Recursive hierarchy: pp. 10-14, 53-64
  - Gradient flow on surprisal: pp. 112-114, 121-130

- Beck, J. & Ramstead, M.J.D. (2025). Dynamic Markov Blanket Detection. (DMBD)
- Heins, C. et al. (2025). AXIOM. (Object-centric RL)
- Friston, K. et al. (2025). Scale-Free Active Inference. (RGM)
- Da Costa, L. (2024). Toward Universal and Interpretable World Models.
