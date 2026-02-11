# Topological Blankets as a Principled Detector of Factored Representations in Neural Networks

## Technical Report: TB-FWH Bridge (US-107 through US-114)

*Noumenal Labs, February 2026*

---

## 1. Overview

This document describes a novel connection between two independent lines of research:

1. *Topological Blankets (TB)*: A method for extracting discrete Markov blanket structure from continuous energy landscape geometry, operating on the gradient covariance (Hessian) of neural network representations.

2. *The Factored World Hypothesis (FWH)*: The empirical finding that transformers pretrained on next-token prediction learn factored representations, organizing their activations into orthogonal subspaces corresponding to independent generative factors (Shai et al., 2602.02385).

The central claim: TB provides a *principled, information-theoretic detector* of the factored structure that Shai et al. identify via PCA. Where PCA requires "vary-one" analysis (constructing special datasets to isolate factor subspaces), TB discovers the same structure directly from the coupling matrix of arbitrary activations, and additionally identifies *blanket variables* that mediate cross-factor information flow.

## 2. Theoretical Bridge

### 2.1 Shai et al. Framework

Shai et al. formalize a data-generating process as a Generalized Hidden Markov Model (GHMM) with N conditionally independent factors. Each factor n has latent dynamics in a d_n-dimensional space. The key predictions of FWH:

- **Dimension**: Activations concentrate in sum_n(d_n - 1) dimensions (linear in N), not prod_n(d_n) - 1 dimensions (exponential in N).
- **Orthogonality**: Factors occupy orthogonal subspaces.
- **Preference**: Models prefer factored representations even when the joint representation would be more faithful.

### 2.2 TB Framework

TB operates on the gradient covariance matrix C = cov(grad E), where E is the energy landscape defined by the network's loss function. The key structural elements:

- **Coupling matrix**: Off-diagonal blocks of C encode statistical dependencies between dimensions.
- **Graph Laplacian**: Constructed from the adjacency matrix A = |C| (thresholded).
- **Eigengap**: The gap between the N-th and (N+1)-th eigenvalue of the Laplacian detects N connected components.
- **Blanket detection**: Variables with high cross-component coupling are identified as Markov blanket variables.

### 2.3 The Bridge

The connection between FWH and TB is direct:

| FWH Concept | TB Concept | Mathematical Correspondence |
|---|---|---|
| Independent factor | TB object (connected component) | Block-diagonal coupling = zero cross-block gradient covariance |
| Orthogonal subspace | Eigenvector cluster | Spectral gap in graph Laplacian |
| N factors | Eigengap at position N | Lambda_{N+1} - Lambda_N >> Lambda_N - Lambda_{N-1} |
| Factored dimension sum_n(d_n-1) | Active dimensions (non-blanket) | Rank of within-block coupling |
| Cross-factor correlation (noise) | Blanket variables | High cross-block coupling norm |
| Factored attractor | Block-diagonal coupling stability | Eigengap persistence through training |

The key insight: when Shai et al. observe block-diagonal structure in activation covariance, TB's graph Laplacian analysis is the *natural mathematical framework* for detecting and quantifying that structure. TB adds value beyond PCA in three ways:

1. **Automatic factor counting**: The eigengap counts factors without requiring vary-one analysis.
2. **Blanket identification**: TB identifies which dimensions mediate cross-factor information flow, which PCA cannot do.
3. **Robustness**: TB's coupling-based detection is robust to rotation and scaling of factor subspaces.

## 3. Experimental Validation (US-107)

### 3.1 Setup

Following Shai et al. exactly:
- **Data**: 5-factor GHMM (3 Mess3 + 2 Bloch Walk), vocab 433, sequence length 8
- **Model**: GPT-2 decoder-only, 4 layers, d_model=120, d_MLP=480
- **Training**: Adam optimizer, cross-entropy loss, standard next-token prediction

### 3.2 TB Analysis

After training, residual stream activations are extracted at each layer (embedding + 4 transformer blocks). For each layer:

1. Activation differences serve as gradient proxies (the covariance of activation differences estimates the Hessian of the implicit energy function).
2. TB computes the coupling matrix, builds the graph Laplacian, and analyzes the eigengap.
3. Spectral clustering partitions dimensions into factor groups.
4. Blanket detection identifies cross-factor mediating dimensions.

### 3.3 Expected Results

Based on Shai et al.'s findings:
- **Embedding layer**: Dense coupling (no factored structure yet). TB eigengap should be small.
- **Later layers**: Block-diagonal coupling emerges. TB eigengap at position 5 should increase through depth.
- **Dimensionality**: PCA dims-for-95% should converge to ~12 (matching sum_n(d_n-1) = 2+2+2+3+3 = 12).
- **TB advantage**: TB identifies blanket variables that PCA misses.

## 4. Extensions

### 4.1 TB-Structured Bayesian Transformer (US-112)

If TB can detect factored structure, it can also *impose* it. A TB-structured transformer uses the coupling matrix as a structural prior:

- **Sparse attention masks**: Dimensions within the same TB object attend freely; cross-object attention is gated through blanket variables only.
- **Per-factor uncertainty**: Ensemble disagreement computed within each TB-detected subspace, not globally.
- **Eigengap regularization**: A loss term that encourages block-diagonal coupling, accelerating the factoring that Shai et al. show transformers converge toward naturally.

**Hypothesis**: Providing the structural prior upfront should lead to:
- Faster convergence to factored representation
- Better uncertainty quantification (per-factor, not global)
- More robust factoring under noise (the inductive bias compensates for non-factorizable data components)

### 4.2 Surprise-Weighted Learning (US-113)

Per-factor surprise extends scalar surprise (prediction error) into a structured signal:

```
surprise_factor_n = || proj_{subspace_n}(prediction_error) ||^2
```

where `subspace_n` is the TB-detected orthogonal subspace for factor n. This enables:

- **Selective credit assignment**: Only update the factor that's surprising, preserving learned structure in other factors.
- **Factor-specific replay priority**: Replay buffer entries are weighted by per-factor surprise, not total surprise.
- **Connection to Active Inference**: Per-factor surprise maps to per-factor free energy, enabling the *decomposed free energy principle*: each TB object minimizes its own variational free energy.

### 4.3 Surprise-Based Data Annotation for Teleoperation (US-114)

In a teleoperation context (FetchPush manipulation), per-factor surprise drives intelligent handoff:

**Current approach** (scalar surprise): Agent detects "something is wrong" and hands off the entire task.

**Proposed approach** (TB-decomposed surprise):
1. TB partitions the world model into factors: gripper (Object 0), manipulated object (Object 1), relation (blanket).
2. Per-factor surprise computed continuously during autonomous operation.
3. When `surprise_object > threshold_object`, request help *specifically for object manipulation* (e.g., "unexpected object dynamics; please demonstrate push").
4. When `surprise_gripper > threshold_gripper`, request help for gripper control.
5. When `surprise_blanket > threshold_blanket`, request help for the *relationship* between gripper and object (approach strategy).

Benefits:
- **Reduced operator cognitive load**: The operator knows exactly which aspect needs attention.
- **Efficient learning**: Human demonstrations are projected into the relevant factor subspace, accelerating learning for that factor without disturbing others.
- **Fewer unnecessary handoffs**: Scalar surprise often triggers handoffs when only one factor is uncertain; per-factor thresholds are more discriminating.

## 5. Intellectual Property Considerations

### 5.1 Novel Contributions

The following elements appear to be novel:

1. **TB as FWH detector**: Using topological blanket analysis (Markov blanket detection via gradient covariance eigengap) to detect and quantify factored representations in neural network activations. Prior art uses PCA/vary-one analysis.

2. **TB-structured attention**: Attention masks derived from topological blanket coupling matrices, where cross-object attention is gated through blanket variables.

3. **Per-factor surprise via TB decomposition**: Projecting prediction error into TB-detected subspaces to obtain factor-specific surprise signals.

4. **Factor-specific teleoperation handoff**: Using TB-decomposed surprise to generate annotated handoff requests that specify which world-model factor needs human input.

5. **TB-regularized training**: Loss terms that encourage block-diagonal coupling in the gradient covariance, accelerating convergence to factored representations.

### 5.2 Potential Patent Claims

**Claim 1** (Detection): A method for detecting factored representations in neural networks comprising: (a) computing the gradient covariance matrix of network activations, (b) constructing a graph Laplacian from said covariance matrix, (c) analyzing the eigengap of said Laplacian to determine the number of independent factors, and (d) identifying Markov blanket variables as dimensions with high cross-factor coupling.

**Claim 2** (Architecture): A neural network architecture comprising attention mechanisms whose connectivity is structurally constrained by a topological blanket decomposition, wherein attention between dimensions belonging to different detected objects is gated through identified blanket variables.

**Claim 3** (Learning): A method for training neural networks comprising: (a) detecting factored structure via topological blanket analysis, (b) computing per-factor surprise by projecting prediction error into factor subspaces, and (c) weighting gradient updates by factor-specific surprise to enable selective credit assignment.

**Claim 4** (Human-in-the-loop): A method for human-robot collaboration comprising: (a) decomposing a world model into factors via topological blanket analysis, (b) computing per-factor surprise during autonomous operation, (c) generating factor-annotated handoff requests when per-factor surprise exceeds factor-specific thresholds, and (d) projecting human demonstrations into the relevant factor subspace for efficient learning.

## 6. References

- Shai, A. S., Amdahl-Culleton, L., Christensen, C. L., Bigelow, H. R., Rosas, F. E., Boyd, A. B., Alt, E. A., Ray, K. J., and Riechers, P. M. "Transformers learn factored representations." arXiv:2602.02385v1, February 2026.
- Patel, Pattisapu, Ramstead, Dumas. "Towards Psychological World Models." 2025.
- Patel, Pattisapu, Ramstead, Dumas. "Epistemic Foraging and Surprise-Guided Replay (SDWM)." 2025.
- Beck, Ramstead. "Dynamic Markov Blanket Detection (DMBD)." 2025.

---

*This document is a working draft for internal use at Noumenal Labs. Content may be incorporated into publications or patent filings.*
