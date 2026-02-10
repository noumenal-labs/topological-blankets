# Topological Blankets Demo

Interactive demonstration of the Topological Blankets method for extracting discrete Markov blanket structure from continuous energy landscape geometry.

## Setup

```bash
pip install -r requirements.txt
pip install jupyter nbconvert
```

## Running the Demo

```bash
cd demo/
jupyter notebook topological_blankets_demo.ipynb
```

Or run non-interactively:

```bash
jupyter nbconvert --execute --to notebook topological_blankets_demo.ipynb
```

## Contents

The notebook walks through:

1. **Theory Overview**: What are Markov blankets and how do energy landscape gradients reveal them?
2. **Synthetic Validation**: Quadratic EBM with known block structure
3. **Bridge Experiments**: 2D score model demonstration
4. **World Model Demo**: Applying TB to a trained Active Inference agent on LunarLander-v3
5. **Multi-Scale Comparison**: State space vs latent space structure
6. **Edge-Compute Implications**: Computational savings from factored inference

The demo loads precomputed results where possible to keep runtime under 10 minutes.
