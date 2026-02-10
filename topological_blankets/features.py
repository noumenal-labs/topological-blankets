"""
Geometric feature computation for Topological Blankets.

Extracts gradient-based features from energy landscape samples
for use in blanket detection and object clustering.
"""

import numpy as np
from typing import Dict


def _estimate_covariance_pearson(gradients: np.ndarray) -> np.ndarray:
    """
    Standard Pearson covariance estimate: np.cov(gradients.T).

    This is the default and original Hessian estimation method.
    """
    H_est = np.cov(gradients.T)
    # Handle 1D case where cov returns scalar
    if H_est.ndim == 0:
        H_est = np.array([[float(H_est)]])
    return H_est


def _estimate_covariance_rank(gradients: np.ndarray) -> np.ndarray:
    """
    Rank-based (Spearman) covariance estimate for nonparanormal robustness.

    Implements the nonparanormal extension (Liu, Lafferty, Wasserman 2009):
    1. Compute Spearman rank correlation matrix via scipy.stats.spearmanr.
    2. Scale to covariance: diag(std) @ corr @ diag(std).

    This extends consistency guarantees to any monotone-transformed Gaussian
    (semiparametric model class), making the Hessian estimate robust to
    heavy-tailed and skewed gradient marginals.
    """
    from scipy.stats import spearmanr

    n_vars = gradients.shape[1]

    # Handle 1D case
    if n_vars == 1:
        return np.array([[float(np.var(gradients[:, 0]))]])

    # Spearman rank correlation matrix
    corr_result = spearmanr(gradients)
    rank_corr = corr_result.statistic

    # Ensure it is a matrix (spearmanr returns scalar for 2 variables)
    if rank_corr.ndim == 0:
        rank_corr = np.array([[1.0, float(rank_corr)],
                               [float(rank_corr), 1.0]])

    # Per-variable standard deviations
    stds = np.std(gradients, axis=0)

    # Scale rank correlation to covariance: diag(std) @ corr @ diag(std)
    H_est = np.outer(stds, stds) * rank_corr

    return H_est


def compute_geometric_features(gradients: np.ndarray,
                                covariance_method: str = 'pearson') -> Dict:
    """
    Core feature computation for Topological Blankets.

    Computes gradient magnitude, variance, estimated Hessian (via gradient
    covariance), and normalized coupling matrix from gradient samples.

    Args:
        gradients: Array of shape (N, n_vars) containing gradient samples.
        covariance_method: Method for Hessian estimation.
            - 'pearson' (default): Standard np.cov(gradients.T).
            - 'rank': Spearman rank-based covariance for nonparanormal
              robustness (Liu, Lafferty, Wasserman 2009).

    Returns:
        Dictionary with keys:
            - grad_magnitude: Mean absolute gradient per variable (n_vars,)
            - grad_variance: Variance of gradient per variable (n_vars,)
            - hessian_est: Estimated Hessian via gradient covariance (n_vars, n_vars)
            - coupling: Normalized off-diagonal coupling matrix (n_vars, n_vars)
    """
    grad_magnitude = np.mean(np.abs(gradients), axis=0)
    grad_variance = np.var(gradients, axis=0)

    # Hessian estimate via gradient covariance
    if covariance_method == 'pearson':
        H_est = _estimate_covariance_pearson(gradients)
    elif covariance_method == 'rank':
        H_est = _estimate_covariance_rank(gradients)
    else:
        raise ValueError(
            f"Unknown covariance_method '{covariance_method}'. "
            "Supported: 'pearson', 'rank'."
        )

    # Normalized coupling (off-diagonal)
    D = np.sqrt(np.diag(H_est)) + 1e-8
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': H_est,
        'coupling': coupling
    }
