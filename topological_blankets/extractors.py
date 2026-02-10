"""
Gradient extraction adapters for Topological Blankets.

Provides adapters to extract gradient samples from various model types:
- PyTorch nn.Module models
- JAX pure functions
- Plain Python callables (finite differences)
- Score-based models (where score = -grad E)
"""

import numpy as np
from typing import Callable, Optional


def extract_gradients_callable(energy_fn: Callable,
                                data: np.ndarray,
                                method: str = 'autograd',
                                eps: float = 1e-5) -> np.ndarray:
    """
    Extract gradients from a plain Python callable via finite differences.

    Args:
        energy_fn: Callable that maps x (n_vars,) -> scalar energy.
        data: Array of shape (N, n_vars) of sample points.
        method: Differentiation method ('autograd' for central differences).
        eps: Step size for finite differences.

    Returns:
        Gradients array of shape (N, n_vars).
    """
    N, d = data.shape
    gradients = np.zeros_like(data)

    for i in range(N):
        x = data[i]
        for j in range(d):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            gradients[i, j] = (energy_fn(x_plus) - energy_fn(x_minus)) / (2 * eps)

    return gradients


def extract_gradients_pytorch(model, data: np.ndarray,
                               loss_fn=None,
                               targets: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract gradients from a PyTorch nn.Module.

    Computes grad_x loss(model(x), target) for each sample.
    If no loss_fn/targets provided, computes grad_x sum(model(x)).

    Args:
        model: PyTorch nn.Module.
        data: Array of shape (N, n_vars) of input samples.
        loss_fn: PyTorch loss function (optional).
        targets: Target array of shape (N, ...) for the loss (optional).

    Returns:
        Gradients array of shape (N, n_vars).
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for extract_gradients_pytorch. "
                          "Install via: pip install torch")

    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    if loss_fn is not None and targets is not None:
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        output = model(data_tensor)
        loss = loss_fn(output, targets_tensor)
    else:
        output = model(data_tensor)
        loss = output.sum()

    loss.backward()
    gradients = data_tensor.grad.detach().numpy()

    return gradients


def extract_gradients_jax(energy_fn: Callable,
                           data: np.ndarray) -> np.ndarray:
    """
    Extract gradients from a JAX pure function.

    Uses jax.vmap(jax.grad(energy_fn)) for efficient batched gradient computation.

    Args:
        energy_fn: JAX-compatible callable that maps x (n_vars,) -> scalar.
        data: Array of shape (N, n_vars) of sample points.

    Returns:
        Gradients array of shape (N, n_vars).
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for extract_gradients_jax. "
                          "Install via: pip install jax jaxlib")

    data_jax = jnp.array(data)
    grad_fn = jax.vmap(jax.grad(energy_fn))
    gradients = grad_fn(data_jax)

    return np.array(gradients)


def extract_gradients_from_scores(score_fn: Callable,
                                   data: np.ndarray) -> np.ndarray:
    """
    Extract energy gradients from a score function.

    For score-based models, score(x) = grad_x log p(x) = -grad_x E(x),
    so energy gradients are simply the negated score.

    Args:
        score_fn: Callable that maps x (N, n_vars) or (n_vars,) -> scores.
        data: Array of shape (N, n_vars) of sample points.

    Returns:
        Gradients array of shape (N, n_vars) (negated scores).
    """
    scores = score_fn(data)
    if isinstance(scores, np.ndarray):
        return -scores
    else:
        return -np.array(scores)
