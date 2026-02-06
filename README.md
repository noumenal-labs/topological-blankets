# Topological Blankets

**Extracting Discrete Markov Blanket Structure from Continuous Energy Landscape Geometry**

*Maxwell J. D. Ramstead, Noumenal Labs*

## Overview

Topological Blankets is a method for extracting discrete Markov blanket topology from continuous energy-based model (EBM) landscapes. It provides a unifying geometric framework for structure learning approaches including RGM, AXIOM, and EBMs.

**Core insight**: Structure learning is discovering topology from geometry. Objects are low-energy basins (or metastable regions); blankets are high-gradient ridges (or mixing bottlenecks).

## Key Equations

```
Blanket criterion:  x_i ∈ Blanket ⟺ 𝔼[‖∂E/∂x_i‖] > τ
Graph functor:      F(E) = G_E where edge(i,j) ⟺ ∂²E/∂x_i∂x_j ≠ 0
Path-based CI:      p(s_τ, z_τ | b_τ) = p(s_τ | b_τ) p(z_τ | b_τ)
Langevin dynamics:  dx = -Γ ∇E dt + √(2ΓT) dW
```

## Repository Structure

```
Noumenal/
├── paper/
│   └── topological_blankets_full.tex    # Main document (~2800 lines)
├── experiments/
│   ├── quadratic_toy_comparison.py      # Level 1 validation
│   └── spectral_friston_detection.py    # Spectral methods
├── reference/
│   └── axiom_text.txt                   # AXIOM reference
└── scripts/
    └── extract_pdf.py
```

## Main Document

The paper (`paper/topological_blankets_full.tex`) contains 11 sections:

1. **Introduction: What is Structure?** Klein's Erlangen program, Yoneda perspective
2. **Structure in Bayesian Models**: Graph topology, conditional independence
3. **Structure in Energy-Based Models**: Energy landscapes, basins, ridges
4. **Structure as Markov Blanket Discovery**: The blanket perspective, typology
5. **Mathematical Core: Structure as Partition**: Partition free energy
6. **Geometric vs Topological Representations**: The bridge between representations
7. **Theoretical Foundation**: Grounded in Friston (2025) FEP, Langevin dynamics
8. **Bridge Proposal: Integration with Active Inference**: DMBD, AXIOM, RGM
9. **The Algorithm: Topological Blankets**: 6-phase pipeline with 17 algorithms
10. **Empirical Validation Strategy**: Levels 1-4
11. **Summary and Conclusions**

### Algorithms (17 total)

**Core detection:**
- Gradient-based blanket detection (Otsu threshold)
- Spectral blanket detection (Fiedler vector, eigengap)
- Hybrid detection (spectral primary, gradient fallback)

**Path-based methods:**
- Path-based blanket detection (committor functions, reactive flux)
- Dynamic Markov Blanket Detection (DMBD)

**Pipeline components:**
- Langevin sampling for geometric data
- Compute geometric features
- Object clustering
- Blanket assignment
- Topology extraction
- Recursive hierarchical extraction

**Structure adaptation:**
- Basin split/merge detection
- Geometric model reduction
- Blanket statistics computation
- Path autocorrelation
- Object typing by blanket statistics

## Theoretical Contributions

1. **Topology Extraction Functor**: F: EBM → Graph where edges indicate non-zero Hessian entries
2. **Gradient-Blanket Correspondence**: High gradient magnitude indicates blanket membership
3. **Path-Based Formulation**: Blanket statistics over trajectories define object types (Beck & Ramstead, 2025)
4. **Maximum Caliber Derivation**: Free energy as ontological potential function
5. **Three-Timescale Dynamics**: Fast inference, medium learning, slow structure extraction

## References

- Friston, K. (2025). A Free Energy Principle: On the Nature of Things.
- Beck, J. & Ramstead, M.J.D. (2025). Dynamic Markov Blanket Detection for Macroscopic Physics Discovery. arXiv:2502.21217
- Heins, C. et al. (2025). AXIOM: Expandable object-centric architecture for RL.
- Schütte, C. & Sarich, M. (2013). Metastability and Markov State Models.
- Jaynes, E.T. (1980). The minimum entropy production principle.
- Da Costa, L. (2024). Toward Universal and Interpretable World Models.

## License

Copyright (c) 2026 Noumenal Labs. All rights reserved.
