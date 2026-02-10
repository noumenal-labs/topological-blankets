"""
PCCA+ fuzzy partition for soft blanket membership.

Implements Perron Cluster Cluster Analysis (Deuflhard & Weber 2005) on the
graph Laplacian eigenvectors to produce fuzzy membership vectors. Blanket
variables naturally emerge as those with high membership in multiple clusters
(i.e., low max-membership), without arbitrary threshold-based detection.

The algorithm:
1. Build a weighted adjacency/transition matrix from the coupling structure.
2. Compute the first k eigenvectors of the transition matrix (or equivalently,
   the normalized graph Laplacian).
3. Find k vertices of the simplex that encloses the eigenvector embedding
   (the "inner simplex problem").
4. Compute fuzzy memberships: chi = V @ inv(A), where A is the k x k matrix
   of simplex vertex coordinates in eigenvector space.
5. Project onto the probability simplex (enforce non-negativity + row-sum = 1).

References:
    - Deuflhard, P. & Weber, M. (2005). Robust Perron cluster analysis in
      conformation dynamics. Linear Algebra and its Applications.
    - Roeblitz, S. & Weber, M. (2013). Fuzzy spectral clustering by PCCA+:
      application to Markov state models and data classification.
      Advances in Data Analysis and Classification.
    - Shi, J. & Malik, J. (2000). Normalized cuts and image segmentation.
      IEEE TPAMI.
    - Ng, A., Jordan, M. & Weiss, Y. (2001). On spectral clustering:
      analysis and an algorithm. NIPS.
"""

import numpy as np
from scipy.linalg import eigh, inv, svd
from typing import Dict, Tuple, Optional


def _find_simplex_vertices(V: np.ndarray, k: int) -> np.ndarray:
    """
    Find k rows of V that form a maximal simplex (inner simplex problem).

    Uses the greedy approach from Deuflhard & Weber: iteratively select the
    row with the largest distance to the current convex hull of selected
    vertices. The first vertex is the row with the largest norm in the
    eigenvector embedding.

    Args:
        V: Eigenvector matrix of shape (n, k), where n is the number of
           variables and k is the number of clusters.
        k: Number of simplex vertices to find.

    Returns:
        Array of k row indices identifying the simplex vertices.
    """
    n = V.shape[0]

    # Start with the row that has the largest norm (most "extreme" point)
    norms = np.linalg.norm(V, axis=1)
    idx = [int(np.argmax(norms))]

    for _ in range(k - 1):
        # Compute distance from each point to the affine hull of selected vertices
        A_sel = V[idx]  # (len(idx), k)

        # Project all points onto the subspace spanned by selected vertices
        # and find the point with maximum residual
        if len(idx) == 1:
            # Distance to single point
            dists = np.linalg.norm(V - A_sel[0], axis=1)
        else:
            # Distance to affine subspace: project V onto span of (A_sel - centroid)
            centroid = A_sel.mean(axis=0)
            directions = A_sel - centroid  # (m, k)
            U, s, Vt = svd(directions, full_matrices=False)
            # Only use significant singular vectors
            rank = np.sum(s > 1e-10)
            if rank == 0:
                # Degenerate: all selected points are the same
                dists = np.linalg.norm(V - centroid, axis=1)
            else:
                basis = Vt[:rank]  # (rank, k), orthonormal rows
                centered = V - centroid  # (n, k)
                projections = centered @ basis.T @ basis  # (n, k)
                residuals = centered - projections
                dists = np.linalg.norm(residuals, axis=1)

        # Exclude already-selected vertices
        for i in idx:
            dists[i] = -1.0

        idx.append(int(np.argmax(dists)))

    return np.array(idx)


def _project_to_simplex(chi: np.ndarray) -> np.ndarray:
    """
    Project each row of chi onto the probability simplex.

    Ensures non-negativity and row-sum = 1. Uses the efficient algorithm
    from Duchi et al. (2008) "Efficient projections onto the l1-ball."

    Args:
        chi: Fuzzy membership matrix of shape (n, k).

    Returns:
        Projected membership matrix with non-negative entries summing to 1
        per row.
    """
    n, k = chi.shape
    result = np.zeros_like(chi)

    for i in range(n):
        v = chi[i].copy()

        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho_candidates = u - cssv / np.arange(1, k + 1)
        rho = np.max(np.where(rho_candidates > 0)[0]) + 1 if np.any(rho_candidates > 0) else k
        theta = cssv[rho - 1] / rho

        result[i] = np.maximum(v - theta, 0.0)

    return result


def pcca_plus(V: np.ndarray, k: int) -> Dict:
    """
    PCCA+ fuzzy clustering on eigenvector embedding.

    Given k eigenvectors of the graph Laplacian (or transition matrix),
    produce a k-dimensional fuzzy membership vector for each variable.
    The membership vector chi[i] = (p_1, ..., p_k) is non-negative and
    sums to 1.

    Variables with max(chi[i]) < threshold are identified as blanket
    variables (high ambiguity, belonging to multiple clusters).

    Args:
        V: Eigenvector matrix of shape (n, k). Typically the first k
           eigenvectors of the normalized graph Laplacian (columns 0 to k-1,
           including the constant eigenvector).
        k: Number of clusters (partitions).

    Returns:
        Dictionary with:
            - memberships: (n, k) fuzzy membership matrix, rows sum to 1
            - hard_labels: (n,) hard partition from argmax of memberships
            - max_membership: (n,) max membership value per variable
            - membership_entropy: (n,) Shannon entropy of membership vector
            - simplex_vertices: (k,) indices of the simplex vertices
    """
    n = V.shape[0]

    if k < 2:
        return {
            'memberships': np.ones((n, 1)),
            'hard_labels': np.zeros(n, dtype=int),
            'max_membership': np.ones(n),
            'membership_entropy': np.zeros(n),
            'simplex_vertices': np.array([0]),
        }

    # Use columns 0 through k-1 of the eigenvectors
    V_k = V[:, :k].copy()

    # Find simplex vertices (inner simplex problem)
    # Note: do NOT row-normalize here; the PCCA+ simplex method operates
    # on the raw eigenvector coordinates. Row-normalization (NJW) is an
    # alternative but collapses the simplex structure that PCCA+ relies on.
    simplex_idx = _find_simplex_vertices(V_k, k)

    # Build transformation matrix A from simplex vertex coordinates
    A = V_k[simplex_idx]  # (k, k)

    # Compute fuzzy memberships: chi = V_k @ inv(A)
    try:
        A_inv = inv(A)
        chi_raw = V_k @ A_inv
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse if A is singular
        A_inv = np.linalg.pinv(A)
        chi_raw = V_k @ A_inv

    # Project onto probability simplex (enforce non-negativity + row-sum = 1)
    chi = _project_to_simplex(chi_raw)

    # Hard labels from argmax
    hard_labels = np.argmax(chi, axis=1)

    # Max membership per variable
    max_membership = np.max(chi, axis=1)

    # Shannon entropy of membership vectors
    # H(p) = -sum(p * log(p)), with 0*log(0) = 0
    chi_safe = np.clip(chi, 1e-15, 1.0)
    membership_entropy = -np.sum(chi * np.log(chi_safe), axis=1)

    return {
        'memberships': chi,
        'hard_labels': hard_labels,
        'max_membership': max_membership,
        'membership_entropy': membership_entropy,
        'simplex_vertices': simplex_idx,
    }


def pcca_blanket_detection(H_est: np.ndarray,
                           n_clusters: int = 3,
                           ambiguity_threshold: float = 0.6,
                           normalized_laplacian: bool = True) -> Dict:
    """
    Detect blanket variables via PCCA+ fuzzy partition.

    Constructs the graph Laplacian from the estimated Hessian, computes
    its eigenvectors, and applies PCCA+ to produce fuzzy membership vectors.
    Blanket variables are identified as those with max membership below the
    ambiguity threshold (they belong significantly to multiple clusters).

    The method uses a *weighted* adjacency matrix (absolute coupling values)
    rather than a binary adjacency, which provides richer spectral structure
    for the PCCA+ simplex method to exploit.

    Args:
        H_est: Estimated Hessian matrix (n_vars, n_vars).
        n_clusters: Number of clusters (objects + blanket partitions).
        ambiguity_threshold: Max-membership threshold for blanket identification.
            Variables with max(chi[i]) < threshold are classified as blanket.
            Default is 0.6, meaning a variable must have at least 60% membership
            in one cluster to be considered "internal" to that cluster.
        normalized_laplacian: If True, use the symmetric normalized Laplacian
            (Shi & Malik 2000, Ng-Jordan-Weiss 2001). Recommended for improved
            spectral clustering quality.

    Returns:
        Dictionary with:
            - is_blanket: (n_vars,) boolean mask of blanket variables
            - memberships: (n_vars, n_clusters) fuzzy membership matrix
            - hard_labels: (n_vars,) hard partition from argmax
            - max_membership: (n_vars,) max membership per variable
            - membership_entropy: (n_vars,) Shannon entropy per variable
            - eigvals: eigenvalues of the Laplacian
            - eigvecs: eigenvectors of the Laplacian
            - simplex_vertices: indices of PCCA+ simplex vertices
            - ambiguity_threshold: the threshold used
    """
    n_vars = H_est.shape[0]

    # Build a weighted adjacency from the absolute Hessian values.
    # Unlike the binary adjacency in spectral.py, the weighted version
    # preserves coupling magnitude, which PCCA+ needs for proper simplex
    # geometry (blanket variables have intermediate coupling to all clusters).
    W = np.abs(H_est).copy()
    np.fill_diagonal(W, 0)

    # Degree matrix
    d = W.sum(axis=1)
    d_safe = np.maximum(d, 1e-10)

    if normalized_laplacian:
        # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
        d_inv_sqrt = 1.0 / np.sqrt(d_safe)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n_vars) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = np.diag(d) - W

    # Compute eigenvectors
    eigvals_full, eigvecs_full = eigh(L)

    # Sort by eigenvalue (ascending); take first n_clusters
    idx = np.argsort(eigvals_full)
    eigvals_full = eigvals_full[idx]
    eigvecs_full = eigvecs_full[:, idx]

    k = min(n_clusters, n_vars - 1)
    eigvals = eigvals_full[:k]
    eigvecs = eigvecs_full[:, :k]

    # PCCA+ on the eigenvector embedding
    pcca_result = pcca_plus(eigvecs, k)

    # Blanket identification uses a two-strategy approach:
    #
    # Strategy 1 (Ambiguity): Variables with max membership < threshold are
    # blanket variables because they belong significantly to multiple clusters.
    #
    # Strategy 2 (Bridging cluster): When PCCA+ forms a distinct "blanket
    # cluster" (variables with high membership in one cluster that bridges
    # between object clusters), identify that cluster using cross-cluster
    # coupling analysis.
    #
    # The method selects whichever strategy identifies a non-trivial blanket.

    is_blanket = pcca_result['max_membership'] < ambiguity_threshold

    # If ambiguity mode found some blanket variables, use them
    n_ambiguous = int(np.sum(is_blanket))

    if n_ambiguous == 0 and n_vars > 3:
        # Strategy 2: Identify the bridging cluster
        # A bridging (blanket) cluster has the property that it couples
        # strongly to all other clusters, while object clusters couple
        # mainly to the blanket cluster.
        hard_labels = pcca_result['hard_labels']
        unique_labels = np.unique(hard_labels)
        n_found = len(unique_labels)

        if n_found >= 2:
            # Compute mean coupling from each cluster to every other cluster
            cross_coupling = np.zeros((n_found, n_found))
            for ci, c1 in enumerate(unique_labels):
                mask_c1 = hard_labels == c1
                for cj, c2 in enumerate(unique_labels):
                    if ci == cj:
                        continue
                    mask_c2 = hard_labels == c2
                    cross_coupling[ci, cj] = W[np.ix_(mask_c1, mask_c2)].mean()

            # Blanket cluster: highest mean cross-coupling to other clusters
            # (blanket couples to all objects; objects couple mainly to blanket)
            mean_cross = np.zeros(n_found)
            for ci in range(n_found):
                others = [cj for cj in range(n_found) if cj != ci]
                if others:
                    mean_cross[ci] = np.mean([cross_coupling[ci, cj]
                                              for cj in others])

            blanket_cluster_idx = int(np.argmax(mean_cross))
            blanket_cluster_label = unique_labels[blanket_cluster_idx]

            blanket_from_cluster = (hard_labels == blanket_cluster_label)

            # Only use if the blanket is a reasonable fraction (< 50%)
            if 0 < np.sum(blanket_from_cluster) < n_vars * 0.5:
                is_blanket = blanket_from_cluster

    # Final fallback: if still no blanket, use entropy-based selection
    if not np.any(is_blanket) and n_vars > 3:
        entropy = pcca_result['membership_entropy']
        n_blanket = max(1, int(np.ceil(n_vars * 0.2)))
        top_entropy_idx = np.argsort(entropy)[-n_blanket:]
        is_blanket = np.zeros(n_vars, dtype=bool)
        is_blanket[top_entropy_idx] = True

    return {
        'is_blanket': is_blanket,
        'memberships': pcca_result['memberships'],
        'hard_labels': pcca_result['hard_labels'],
        'max_membership': pcca_result['max_membership'],
        'membership_entropy': pcca_result['membership_entropy'],
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'simplex_vertices': pcca_result['simplex_vertices'],
        'ambiguity_threshold': ambiguity_threshold,
    }
