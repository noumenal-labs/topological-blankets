"""
Spectral methods for Topological Blankets.

Implements Friston's spectral approach to Markov blanket detection
using graph Laplacian eigenmodes.

Based on Friston (2025) "A Free Energy Principle: On the Nature of Things"
pp. 48-51, 58-64, 67-70.

L1 sparsification (US-068) based on:
- Lin, Drton, Shojaie (2016), "Estimation of High-Dimensional Graphical
  Models Using Regularized Score Matching"
- Meinshausen & Buhlmann (2010), "Stability Selection"
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from typing import Dict, List, Optional, Tuple


# =========================================================================
# L1 sparsification: soft thresholding and lambda selection
# =========================================================================

def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft thresholding: S_lambda(x) = sign(x) * max(|x| - lambda, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def l1_sparsify_hessian(H_emp: np.ndarray,
                         lam: float,
                         normalize: bool = True) -> np.ndarray:
    """
    Apply L1 soft-thresholding to off-diagonal Hessian entries.

    Solves the proximal problem:
        min_H  ||H - H_emp||_F^2  +  lambda * ||H_offdiag||_1

    The closed-form solution is element-wise soft thresholding on
    off-diagonal entries, leaving the diagonal untouched.

    Args:
        H_emp: Empirical Hessian matrix (d, d).
        lam: L1 penalty strength (lambda >= 0).
        normalize: If True, normalize by diagonal before thresholding
            (same normalization as the threshold method uses).

    Returns:
        Sparse Hessian matrix (d, d) with soft-thresholded off-diagonals.
    """
    d = H_emp.shape[0]

    if normalize:
        D = np.sqrt(np.abs(np.diag(H_emp)) + 1e-8)
        H_work = H_emp / np.outer(D, D)
    else:
        H_work = H_emp.copy()

    # Apply soft threshold to off-diagonal entries only
    diag_vals = np.diag(H_work).copy()
    H_sparse = _soft_threshold(H_work, lam)
    np.fill_diagonal(H_sparse, diag_vals)

    # Symmetrize (numerical safety after thresholding)
    H_sparse = (H_sparse + H_sparse.T) / 2.0

    return H_sparse


def select_lambda_bic(H_emp: np.ndarray,
                      n_samples: int,
                      lambda_grid: Optional[np.ndarray] = None,
                      normalize: bool = True,
                      n_lambdas: int = 30,
                      ebic_gamma: float = 0.5) -> Tuple[float, dict]:
    """
    Select the L1 penalty lambda via a noise-adaptive universal threshold.

    Computes a data-driven lambda using the universal threshold from
    multiple testing theory (Bickel & Levina 2008, Cai & Liu 2011),
    inspired by regularized score matching (Lin, Drton, Shojaie 2016):

        lambda = (1/sqrt(n)) * sqrt(2 * log(n_offdiag))

    where 1/sqrt(n) is the asymptotic standard deviation of each
    off-diagonal entry in a sample covariance from n observations,
    and n_offdiag = d*(d-1)/2 is the number of simultaneous tests.

    The sqrt(2*log(n_tests)) multiplier controls the family-wise
    error rate, ensuring that pure noise entries are zeroed while
    true edges with magnitude above the noise floor are preserved.

    The method then evaluates an Extended BIC (Foygel & Drton, 2010)
    over a grid centered on this reference lambda to provide fine-
    grained selection:

        EBIC = N * log(RSS / n_offdiag) + k * log(N) + 4 * gamma * k * log(d)

    Args:
        H_emp: Empirical Hessian matrix (d, d).
        n_samples: Number of gradient samples used to estimate H_emp.
        lambda_grid: Explicit grid of lambda values to search. If None,
            a grid centered on the universal threshold is constructed.
        normalize: If True, normalize by diagonal before thresholding.
        n_lambdas: Number of lambda values in the automatic grid.
        ebic_gamma: EBIC gamma parameter (0 = classical BIC, 0.5 =
            recommended default, 1 = strongest high-D penalty).

    Returns:
        Tuple of (best_lambda, info_dict) where info_dict contains the
        selected lambda, reference lambda, EBIC path, and sparsity.
    """
    d = H_emp.shape[0]
    n_offdiag = d * (d - 1) // 2

    if normalize:
        D = np.sqrt(np.abs(np.diag(H_emp)) + 1e-8)
        H_work = H_emp / np.outer(D, D)
    else:
        H_work = H_emp.copy()

    # Off-diagonal absolute values (upper triangle)
    offdiag = np.abs(H_work[np.triu_indices(d, k=1)])

    if lambda_grid is None:
        # Noise-adaptive universal threshold from asymptotic theory.
        #
        # For a covariance matrix estimated from n i.i.d. samples,
        # each off-diagonal entry (after diagonal normalization)
        # has asymptotic std ~ 1/sqrt(n). The universal threshold
        # from multiple testing theory (Bickel & Levina 2008,
        # Cai & Liu 2011) is:
        #
        #   lambda_ref = (1/sqrt(n)) * sqrt(2 * log(n_offdiag))
        #
        # where n_offdiag = d*(d-1)/2 is the number of off-diagonal
        # entries being tested. This controls the family-wise error
        # rate: under the null hypothesis that a given entry is zero,
        # the probability of any false discovery goes to 0 as n grows.
        sigma_est = 1.0 / np.sqrt(max(n_samples, 1))
        lam_ref = sigma_est * np.sqrt(2.0 * np.log(max(n_offdiag, 2)))

        # Ensure lam_ref is in a sensible range
        lam_max = np.max(offdiag) if len(offdiag) > 0 else 1.0
        lam_ref = np.clip(lam_ref, lam_max * 1e-3, lam_max * 0.95)

        # Build grid spanning a wide range around lam_ref for
        # EBIC refinement (factor of 5 each way)
        lam_lo = max(lam_ref / 5.0, 1e-12)
        lam_hi = min(lam_ref * 5.0, lam_max * 1.05)
        lambda_grid = np.logspace(np.log10(lam_lo), np.log10(lam_hi), n_lambdas)
    else:
        lam_ref = None

    bic_values = []
    sparsity_values = []

    log_d = np.log(max(d, 2))
    log_n = np.log(max(n_samples, 2))

    for lam in lambda_grid:
        H_sparse = l1_sparsify_hessian(H_work, lam, normalize=False)

        # Residual: mean squared error over off-diagonal entries
        diff = H_work - H_sparse
        np.fill_diagonal(diff, 0)
        residual = np.sum(diff ** 2) / max(n_offdiag, 1)
        residual = max(residual, 1e-15)  # avoid log(0)

        # Count nonzero off-diagonal upper-triangle entries
        offdiag_sparse = H_sparse[np.triu_indices(d, k=1)]
        k = np.sum(np.abs(offdiag_sparse) > 1e-12)

        # Extended BIC (Foygel & Drton 2010)
        ebic = (n_samples * np.log(residual)
                + k * log_n
                + 4.0 * ebic_gamma * k * log_d)
        bic_values.append(ebic)
        sparsity_values.append(int(k))

    bic_values = np.array(bic_values)
    best_idx = np.argmin(bic_values)
    # If the BIC selected a boundary value (smallest or largest lambda
    # in the grid), it indicates a monotonic BIC surface where the
    # search did not find a true minimum. In this case the reference
    # lambda (universal threshold) is more reliable.
    if lam_ref is not None and (best_idx == 0 or best_idx == len(lambda_grid) - 1):
        best_lambda = float(lam_ref)
    else:
        best_lambda = float(lambda_grid[best_idx])

    # Compute sparsity at the selected lambda
    H_at_best = l1_sparsify_hessian(H_work, best_lambda, normalize=False)
    offdiag_best = H_at_best[np.triu_indices(d, k=1)]
    best_sparsity = int(np.sum(np.abs(offdiag_best) > 1e-12))

    info = {
        'lambda_grid': lambda_grid.tolist(),
        'bic_values': bic_values.tolist(),
        'sparsity_values': sparsity_values,
        'best_idx': int(best_idx),
        'best_lambda': best_lambda,
        'best_bic': float(bic_values[best_idx]),
        'best_sparsity': best_sparsity,
        'total_possible_edges': int(n_offdiag),
        'ebic_gamma': ebic_gamma,
        'lambda_ref': float(lam_ref) if lam_ref is not None else None,
    }

    return best_lambda, info


def select_lambda_cv(H_emp: np.ndarray,
                     gradients: np.ndarray,
                     lambda_grid: Optional[np.ndarray] = None,
                     normalize: bool = True,
                     n_folds: int = 5,
                     n_lambdas: int = 20) -> Tuple[float, dict]:
    """
    Select the L1 penalty lambda via cross-validation on held-out
    gradient log-likelihood.

    Splits gradient samples into folds. For each fold, estimates H on
    the training set, sparsifies at each lambda, then evaluates the
    Frobenius distance to the held-out Hessian estimate.

    Args:
        H_emp: Full-sample empirical Hessian (d, d) (used for grid setup).
        gradients: Raw gradient samples (N, d).
        lambda_grid: Explicit grid of lambda values.
        normalize: If True, normalize by diagonal.
        n_folds: Number of cross-validation folds.
        n_lambdas: Number of lambda values in the automatic grid.

    Returns:
        Tuple of (best_lambda, info_dict).
    """
    N, d = gradients.shape

    if normalize:
        D = np.sqrt(np.abs(np.diag(H_emp)) + 1e-8)
        H_work = H_emp / np.outer(D, D)
    else:
        H_work = H_emp.copy()

    offdiag = np.abs(H_work[np.triu_indices(d, k=1)])
    if lambda_grid is None:
        lam_max = np.max(offdiag) if len(offdiag) > 0 else 1.0
        lam_min = lam_max * 1e-4
        lambda_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambdas)

    # Random fold indices
    rng = np.random.RandomState(42)
    fold_ids = rng.randint(0, n_folds, size=N)

    cv_scores = np.zeros(len(lambda_grid))

    for fold in range(n_folds):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold

        if np.sum(train_mask) < d + 1 or np.sum(test_mask) < d + 1:
            continue

        H_train = np.cov(gradients[train_mask].T)
        H_test = np.cov(gradients[test_mask].T)

        if H_train.ndim == 0:
            H_train = np.array([[float(H_train)]])
        if H_test.ndim == 0:
            H_test = np.array([[float(H_test)]])

        if normalize:
            D_train = np.sqrt(np.abs(np.diag(H_train)) + 1e-8)
            H_train_norm = H_train / np.outer(D_train, D_train)
            D_test = np.sqrt(np.abs(np.diag(H_test)) + 1e-8)
            H_test_norm = H_test / np.outer(D_test, D_test)
        else:
            H_train_norm = H_train
            H_test_norm = H_test

        for li, lam in enumerate(lambda_grid):
            H_sparse = l1_sparsify_hessian(H_train_norm, lam, normalize=False)
            # Held-out Frobenius loss
            loss = np.sum((H_test_norm - H_sparse) ** 2) / (d * d)
            cv_scores[li] += loss

    cv_scores /= n_folds
    best_idx = int(np.argmin(cv_scores))
    best_lambda = float(lambda_grid[best_idx])

    info = {
        'lambda_grid': lambda_grid.tolist(),
        'cv_scores': cv_scores.tolist(),
        'best_idx': best_idx,
        'best_lambda': best_lambda,
        'best_cv_score': float(cv_scores[best_idx]),
        'n_folds': n_folds,
    }

    return best_lambda, info


def stability_selection(H_emp: np.ndarray,
                        gradients: np.ndarray,
                        lam: Optional[float] = None,
                        n_bootstrap: int = 100,
                        threshold_freq: float = 0.6,
                        subsample_ratio: float = 0.5,
                        normalize: bool = True,
                        n_samples_for_bic: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Stability selection for edge identification (Meinshausen & Buhlmann 2010).

    Runs L1 sparsification on n_bootstrap subsamples of the gradient data,
    then keeps edges that appear in more than threshold_freq fraction of
    the bootstrap runs.

    Args:
        H_emp: Full-sample empirical Hessian (d, d).
        gradients: Raw gradient samples (N, d).
        lam: L1 penalty; if None, selected via BIC on the full sample.
        n_bootstrap: Number of bootstrap subsamples.
        threshold_freq: Minimum fraction of bootstrap runs an edge must
            appear in to be kept (default 0.6 = 60%).
        subsample_ratio: Fraction of samples used per bootstrap.
        normalize: If True, normalize by diagonal.
        n_samples_for_bic: Sample count for BIC lambda selection (default: N).

    Returns:
        Tuple of (adjacency_matrix, info_dict) where the adjacency is
        the stability-selected binary edge matrix.
    """
    N, d = gradients.shape

    if lam is None:
        n_for_bic = n_samples_for_bic if n_samples_for_bic else N
        lam, _ = select_lambda_bic(H_emp, n_for_bic, normalize=normalize)

    subsample_size = max(d + 1, int(N * subsample_ratio))
    rng = np.random.RandomState(42)

    edge_counts = np.zeros((d, d))

    for b in range(n_bootstrap):
        idx = rng.choice(N, size=subsample_size, replace=True)
        H_boot = np.cov(gradients[idx].T)
        if H_boot.ndim == 0:
            H_boot = np.array([[float(H_boot)]])

        H_sparse = l1_sparsify_hessian(H_boot, lam, normalize=normalize)

        # Record nonzero off-diagonal entries
        offdiag_mask = (np.abs(H_sparse) > 1e-12)
        np.fill_diagonal(offdiag_mask, False)
        edge_counts += offdiag_mask.astype(float)

    edge_freq = edge_counts / n_bootstrap
    A_stable = (edge_freq >= threshold_freq).astype(float)
    np.fill_diagonal(A_stable, 0)

    info = {
        'n_bootstrap': n_bootstrap,
        'threshold_freq': threshold_freq,
        'subsample_ratio': subsample_ratio,
        'lambda_used': float(lam),
        'n_stable_edges': int(np.sum(A_stable[np.triu_indices(d, k=1)])),
        'edge_frequencies': edge_freq.tolist(),
    }

    return A_stable, info


def build_adjacency_from_hessian(H: np.ndarray,
                                  threshold: float = 0.01,
                                  normalize: bool = True,
                                  sparsify: str = 'threshold',
                                  n_samples: Optional[int] = None,
                                  gradients: Optional[np.ndarray] = None,
                                  l1_lambda: Optional[float] = None,
                                  lambda_method: str = 'bic',
                                  stability_n_bootstrap: int = 100,
                                  stability_threshold: float = 0.6) -> np.ndarray:
    """
    Construct adjacency matrix from Hessian (Friston pp. 48-51).

    Supports three sparsification methods:
      - 'threshold' (default): Hard threshold at a fixed cutoff.
          A_ij = 1 if |H_ij| > threshold.
      - 'l1': L1 soft-thresholding on off-diagonal entries with
          data-adaptive lambda selected via BIC or cross-validation.
      - 'stability': Stability selection, running L1 on bootstrap
          subsamples and keeping edges appearing in >60% of runs.

    In Friston's formulation, H approximates -Gamma^{-1} J where J is
    the Jacobian of flow, so non-zero entries indicate direct influence.

    Args:
        H: Hessian or estimated Hessian matrix (n_vars, n_vars).
        threshold: Coupling threshold for edge creation (used when
            sparsify='threshold').
        normalize: If True, normalize by diagonal (self-coupling strength).
        sparsify: Sparsification method. One of 'threshold', 'l1', 'stability'.
        n_samples: Number of samples used to estimate H (needed for BIC
            lambda selection; required when sparsify='l1' and l1_lambda
            is not provided).
        gradients: Raw gradient samples (N, d). Required for
            sparsify='stability' and for sparsify='l1' with
            lambda_method='cv'.
        l1_lambda: Explicit L1 penalty value. If None, lambda is selected
            automatically via lambda_method.
        lambda_method: How to select lambda when l1_lambda is None.
            One of 'bic' (default) or 'cv' (cross-validation).
        stability_n_bootstrap: Number of bootstrap subsamples for
            stability selection (default 100).
        stability_threshold: Minimum edge frequency for stability
            selection (default 0.6).

    Returns:
        Binary adjacency matrix (n_vars, n_vars).
    """
    if sparsify == 'threshold':
        # Original hard-threshold method
        if normalize:
            D = np.sqrt(np.abs(np.diag(H)) + 1e-8)
            H_norm = np.abs(H) / np.outer(D, D)
        else:
            H_norm = np.abs(H)

        A = (H_norm > threshold).astype(float)
        np.fill_diagonal(A, 0)
        return A

    elif sparsify == 'l1':
        # L1 soft-thresholding with data-adaptive lambda
        if l1_lambda is not None:
            lam = l1_lambda
        elif lambda_method == 'cv' and gradients is not None:
            lam, _ = select_lambda_cv(H, gradients, normalize=normalize)
        else:
            # BIC (default)
            n = n_samples if n_samples is not None else max(100, H.shape[0] * 10)
            lam, _ = select_lambda_bic(H, n, normalize=normalize)

        H_sparse = l1_sparsify_hessian(H, lam, normalize=normalize)

        # Convert to binary adjacency: nonzero off-diagonal -> edge
        A = (np.abs(H_sparse) > 1e-12).astype(float)
        np.fill_diagonal(A, 0)
        return A

    elif sparsify == 'stability':
        if gradients is None:
            raise ValueError(
                "sparsify='stability' requires the 'gradients' argument "
                "to be provided for bootstrap resampling."
            )
        A, _ = stability_selection(
            H, gradients,
            lam=l1_lambda,
            n_bootstrap=stability_n_bootstrap,
            threshold_freq=stability_threshold,
            normalize=normalize,
            n_samples_for_bic=n_samples,
        )
        return A

    else:
        raise ValueError(
            f"Unknown sparsify method '{sparsify}'. "
            "Supported: 'threshold', 'l1', 'stability'."
        )


def build_graph_laplacian(A: np.ndarray,
                          normalized: bool = False) -> np.ndarray:
    """
    Build graph Laplacian from adjacency matrix (Friston pp. 48-51).

    L = D - A  (unnormalized)
    L_norm = I - D^{-1/2} A D^{-1/2}  (normalized)

    Properties:
    - L is positive semi-definite
    - Smallest eigenvalue is 0 (constant eigenvector)
    - Number of zero eigenvalues = number of connected components
    - Eigengap indicates partition strength

    Args:
        A: Adjacency matrix (n_vars, n_vars).
        normalized: If True, use normalized Laplacian.

    Returns:
        Graph Laplacian matrix (n_vars, n_vars).
    """
    D = np.diag(A.sum(axis=1))

    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
        L = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = D - A

    return L


def compute_eigengap(eigvals: np.ndarray) -> Tuple[int, float]:
    """
    Find eigengap indicating natural partition (Friston pp. 58-61).

    The largest gap in the eigenvalue spectrum indicates the number
    of well-separated clusters.

    Args:
        eigvals: Sorted eigenvalues (ascending).

    Returns:
        Tuple of (number of clusters, gap magnitude).
    """
    if len(eigvals) < 2:
        return 1, 0.0

    gaps = np.diff(eigvals)
    best_gap_idx = np.argmax(gaps)
    best_gap = gaps[best_gap_idx]

    n_clusters = best_gap_idx + 1

    return n_clusters, float(best_gap)


def spectral_partition(L: np.ndarray,
                       n_partitions: int = 3,
                       method: str = 'kmeans') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition variables using spectral embedding (Friston pp. 58-61).

    Process:
    1. Compute k smallest eigenvalues/vectors of the Laplacian
    2. Embed points in eigenvector space
    3. Cluster to identify internal, blanket, and external groups

    Args:
        L: Graph Laplacian matrix (n_vars, n_vars).
        n_partitions: Number of partitions to find.
        method: Clustering method ('kmeans' or 'spectral').

    Returns:
        Tuple of (labels, eigenvalues, eigenvectors).
    """
    k = min(n_partitions + 2, len(L) - 1)

    try:
        if len(L) > 100:
            eigvals, eigvecs = eigsh(L, k=k, which='SM')
        else:
            eigvals, eigvecs = eigh(L)
            eigvals = eigvals[:k]
            eigvecs = eigvecs[:, :k]
    except Exception:
        return np.zeros(len(L), dtype=int), np.array([0.0]), np.zeros((len(L), 1))

    # Sort by eigenvalue
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Cluster on eigenvector embedding (skip constant mode)
    end_col = min(n_partitions + 1, eigvecs.shape[1])
    embedding = eigvecs[:, 1:end_col]

    if embedding.shape[1] == 0:
        return np.zeros(len(L), dtype=int), eigvals, eigvecs

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_partitions, random_state=42, n_init=10)
    else:
        clusterer = SpectralClustering(n_clusters=n_partitions,
                                        affinity='nearest_neighbors',
                                        random_state=42)

    try:
        labels = clusterer.fit_predict(embedding)
    except Exception:
        labels = np.zeros(len(L), dtype=int)

    return labels, eigvals, eigvecs


def identify_blanket_from_spectrum(eigvals: np.ndarray,
                                    eigvecs: np.ndarray,
                                    labels: np.ndarray) -> np.ndarray:
    """
    Identify blanket variables from spectral analysis (Friston pp. 67-70).

    Friston's interpretation:
    - Slow modes (small eigenvalues): Stable internal states
    - Fast modes (large eigenvalues): Rapidly mixing external
    - Intermediate: Blanket (mediating structure)

    Heuristic: Blanket = cluster with highest eigenvector variance
    (connects to multiple regions, so varied eigenvector values).

    Args:
        eigvals: Sorted eigenvalues.
        eigvecs: Corresponding eigenvectors.
        labels: Cluster labels from spectral partition.

    Returns:
        Boolean mask identifying blanket variables.
    """
    n_clusters = len(np.unique(labels))

    cluster_variance = []
    for c in range(n_clusters):
        mask = labels == c
        if np.sum(mask) > 0:
            var = np.var(eigvecs[mask, 1:min(4, eigvecs.shape[1])], axis=0).mean()
            cluster_variance.append((c, var))
        else:
            cluster_variance.append((c, 0))

    cluster_variance.sort(key=lambda x: x[1], reverse=True)
    blanket_cluster = cluster_variance[0][0]

    is_blanket = labels == blanket_cluster

    return is_blanket


def schur_complement_reduction(H: np.ndarray,
                                keep_idx: np.ndarray,
                                eliminate_idx: np.ndarray) -> np.ndarray:
    """
    Adiabatic elimination via Schur complement (Friston pp. 58-64).

    When eliminating fast (external) modes, the effective Hessian for
    slow (internal+blanket) modes is:

    H_eff = H_slow - H_cross^T @ H_fast^{-1} @ H_cross

    This "integrates out" the fast variables.

    Args:
        H: Full Hessian matrix.
        keep_idx: Indices of variables to keep (slow + blanket).
        eliminate_idx: Indices of variables to eliminate (fast/external).

    Returns:
        Effective Hessian for kept variables.
    """
    if len(eliminate_idx) == 0:
        return H[np.ix_(keep_idx, keep_idx)]

    H_slow = H[np.ix_(keep_idx, keep_idx)]
    H_fast = H[np.ix_(eliminate_idx, eliminate_idx)]
    H_cross = H[np.ix_(eliminate_idx, keep_idx)]

    try:
        H_fast_inv = np.linalg.inv(H_fast + 1e-6 * np.eye(len(H_fast)))
        H_eff = H_slow - H_cross.T @ H_fast_inv @ H_cross
    except np.linalg.LinAlgError:
        H_eff = H_slow

    return H_eff


def recursive_spectral_detection(H: np.ndarray,
                                  max_levels: int = 3,
                                  min_vars: int = 3,
                                  adjacency_threshold: float = 0.01) -> List[Dict]:
    """
    Friston-style recursive blanket detection (pp. 53-64).

    Algorithm:
    1. Detect blankets via spectral method at current scale
    2. Identify fast (external) modes
    3. Adiabatically eliminate fast modes (Schur complement)
    4. Remaining slow+blanket = new particle
    5. Repeat at coarser scale

    Args:
        H: Hessian matrix (n_vars, n_vars).
        max_levels: Maximum recursion depth.
        min_vars: Minimum variables to continue decomposition.
        adjacency_threshold: Threshold for adjacency construction.

    Returns:
        List of dictionaries, one per level, each containing:
            - level, internals, blanket, external, eigengap, n_vars, eigvals
    """
    hierarchy = []
    current_H = H.copy()
    current_vars = list(range(H.shape[0]))

    for level in range(max_levels):
        if len(current_vars) < min_vars:
            break

        A = build_adjacency_from_hessian(current_H, threshold=adjacency_threshold)
        L = build_graph_laplacian(A)

        eigvals_all, _ = eigh(L)
        n_clusters, eigengap = compute_eigengap(eigvals_all[:min(10, len(eigvals_all))])
        n_clusters = max(2, min(3, n_clusters))

        labels, eigvals, eigvecs = spectral_partition(L, n_partitions=n_clusters)

        is_blanket = identify_blanket_from_spectrum(eigvals, eigvecs, labels)

        # Classify clusters by eigenvalue contribution
        cluster_eigval_score = []
        for c in range(n_clusters):
            mask = labels == c
            if np.sum(mask) > 0:
                slow_proj = np.mean(np.abs(eigvecs[mask, 1:3]))
                cluster_eigval_score.append((c, slow_proj))

        cluster_eigval_score.sort(key=lambda x: x[1])

        if len(cluster_eigval_score) >= 3:
            internal_cluster = cluster_eigval_score[0][0]
            blanket_cluster = cluster_eigval_score[1][0]
            external_cluster = cluster_eigval_score[2][0]
        elif len(cluster_eigval_score) == 2:
            internal_cluster = cluster_eigval_score[0][0]
            blanket_cluster = cluster_eigval_score[1][0]
            external_cluster = None
        else:
            internal_cluster = 0
            blanket_cluster = 0
            external_cluster = None

        internals = [current_vars[i] for i in range(len(labels))
                     if labels[i] == internal_cluster]
        blanket = [current_vars[i] for i in range(len(labels))
                   if labels[i] == blanket_cluster]
        external = [current_vars[i] for i in range(len(labels))
                    if external_cluster is not None and labels[i] == external_cluster]

        hierarchy.append({
            'level': level,
            'internals': internals,
            'blanket': blanket,
            'external': external,
            'eigengap': float(eigengap),
            'n_vars': len(current_vars),
            'eigvals': eigvals[:6].tolist() if len(eigvals) > 6 else eigvals.tolist()
        })

        if external_cluster is not None:
            keep_local = [i for i in range(len(labels))
                          if labels[i] != external_cluster]
            elim_local = [i for i in range(len(labels))
                          if labels[i] == external_cluster]

            if len(keep_local) < min_vars:
                break

            current_H = schur_complement_reduction(current_H,
                                                    np.array(keep_local),
                                                    np.array(elim_local))
            current_vars = [current_vars[i] for i in keep_local]
        else:
            keep_local = [i for i in range(len(labels))
                          if labels[i] == internal_cluster]
            if len(keep_local) < min_vars:
                break
            current_H = current_H[np.ix_(keep_local, keep_local)]
            current_vars = [current_vars[i] for i in keep_local]

    return hierarchy
