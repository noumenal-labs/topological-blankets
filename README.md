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
```

## Repository Structure

```
Noumenal/
├── paper/
│   └── topological_blankets_full.tex    # Main document (~3000 lines)
├── experiments/
│   ├── quadratic_toy_comparison.py      # Level 1 validation
│   └── spectral_friston_detection.py    # Spectral methods
├── archive/
│   └── markdown/                        # Source documents (cleaned)
├── reference/
│   └── axiom_text.txt                   # AXIOM reference
└── scripts/
    └── extract_pdf.py
```

## Main Document

The paper (`paper/topological_blankets_full.tex`) contains:

1. **Introduction**: Structure as preservation; Klein's Erlangen program
2. **Structure in Bayesian Models**: Graph topology, temporal depth
3. **Structure in EBMs**: Energy landscapes, basins, ridges
4. **Markov Blanket Discovery**: Conditional independence from geometry
5. **Mathematical Core**: Topology extraction functor
6. **Geometric vs Topological**: The bridge between representations
7. **Theoretical Foundation**: Grounded in Friston (2025) FEP
8. **Integration with Active Inference**: DMBD, blanket statistics
9. **The Algorithm**: 6-phase pipeline with 16+ algorithms
10. **Empirical Validation Strategy**: Levels 1-4
11. **Summary and Conclusions**

### Key Algorithms

- Gradient-based blanket detection (Otsu threshold)
- Spectral blanket detection (Fiedler vector, eigengap)
- Hybrid detection (spectral primary, gradient fallback)
- Path-based blanket detection (committor functions, reactive flux)
- Dynamic Markov Blanket Detection (DMBD)
- Recursive hierarchical extraction

## Theoretical Contributions

1. **Topology Extraction Functor**: F: EBM → Graph where edges indicate non-zero Hessian entries
2. **Gradient-Blanket Correspondence**: High gradient magnitude indicates blanket membership
3. **Path-Based Formulation**: Blanket statistics over trajectories define object types
4. **Maximum Caliber Derivation**: Free energy as ontological potential function

## References

- Friston, K. (2025). A Free Energy Principle: On the Nature of Things.
- Beck, J. & Ramstead, M.J.D. (2025). Dynamic Markov Blanket Detection for Macroscopic Physics Discovery. arXiv:2502.21217
- Heins, C. et al. (2025). AXIOM: Expandable object-centric architecture for RL.
- Schütte, C. & Sarich, M. (2013). Metastability and Markov State Models.

## License

Copyright (c) 2025 Noumenal Labs. All rights reserved.
