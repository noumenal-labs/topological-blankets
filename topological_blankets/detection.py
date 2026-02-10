"""
Blanket detection methods for Topological Blankets.

Provides multiple approaches to identifying Markov blanket variables:
- Gradient-based (Otsu thresholding on gradient magnitude)
- Spectral (Friston eigenvector variance heuristic)
- Hybrid (spectral with gradient fallback)
- Coupling-based (cross-cluster coupling strength)
- Persistence-based (sublevel set filtration on coupling graph, H0 persistence)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .features import compute_geometric_features
from .spectral import (
    build_adjacency_from_hessian,
    build_graph_laplacian,
    spectral_partition,
    compute_eigengap,
    identify_blanket_from_spectrum,
)


def detect_blankets_otsu(features: Dict) -> Tuple[np.ndarray, float]:
    """
    Detect blankets using Otsu's method on gradient magnitude.

    Separates variables into two groups by gradient magnitude, then
    assigns the minority group as blanket (fewer mediating variables
    than internal variables in typical configurations).

    Known limitations:
    - Fails when blanket vars outnumber internal vars (ratio > 1.0)
    - Degrades with asymmetric object sizes (different gradient scales)

    Args:
        features: Dictionary from compute_geometric_features().

    Returns:
        Tuple of (boolean blanket mask, threshold value).
    """
    from skimage.filters import threshold_otsu

    gm = features['grad_magnitude']
    try:
        tau = threshold_otsu(gm)
    except ValueError:
        tau = np.percentile(gm, 80)

    high_group = gm > tau
    low_group = ~high_group

    # Blanket = minority group
    if np.sum(low_group) <= np.sum(high_group):
        is_blanket = low_group
    else:
        is_blanket = high_group

    return is_blanket, float(tau)


def detect_blankets_gradient(gradients: np.ndarray,
                              method: str = 'otsu') -> Tuple[np.ndarray, float]:
    """
    Gradient-magnitude based blanket detection.

    Uses Otsu to separate variables into two groups by gradient magnitude,
    then assigns the minority group as blanket.

    Args:
        gradients: Gradient samples of shape (N, n_vars).
        method: Thresholding method ('otsu', 'percentile', or 'median').

    Returns:
        Tuple of (boolean blanket mask, threshold value).
    """
    grad_magnitude = np.mean(np.abs(gradients), axis=0)

    if method == 'otsu':
        from skimage.filters import threshold_otsu
        try:
            tau = threshold_otsu(grad_magnitude)
        except ValueError:
            tau = np.percentile(grad_magnitude, 80)
    elif method == 'percentile':
        tau = np.percentile(grad_magnitude, 80)
    else:
        tau = np.median(grad_magnitude) * 1.5

    high_group = grad_magnitude > tau
    low_group = ~high_group

    if np.sum(low_group) <= np.sum(high_group):
        is_blanket = low_group
    else:
        is_blanket = high_group

    return is_blanket, float(tau)


def detect_blankets_spectral(H_est: np.ndarray,
                              n_partitions: int = 3) -> Dict:
    """
    Spectral blanket detection via Friston's method.

    Uses graph Laplacian eigenmodes to identify blanket as the cluster
    with highest eigenvector variance.

    Args:
        H_est: Estimated Hessian matrix (n_vars, n_vars).
        n_partitions: Number of spectral partitions.

    Returns:
        Dictionary with is_blanket, eigengap, eigvals, labels.
    """
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    labels, eigvals, eigvecs = spectral_partition(L, n_partitions=n_partitions)

    _, eigengap = compute_eigengap(eigvals)
    is_blanket = identify_blanket_from_spectrum(eigvals, eigvecs, labels)

    return {
        'is_blanket': is_blanket,
        'eigengap': float(eigengap),
        'eigvals': eigvals,
        'spectral_labels': labels
    }


def detect_blankets_hybrid(gradients: np.ndarray,
                            H_est: np.ndarray,
                            eigengap_threshold: float = 0.5) -> Dict:
    """
    Hybrid detection: spectral if eigengap is strong, gradient fallback otherwise.

    Combines Friston's rigorous spectral method with the practical
    gradient-based heuristic.

    Args:
        gradients: Gradient samples of shape (N, n_vars).
        H_est: Estimated Hessian matrix (n_vars, n_vars).
        eigengap_threshold: Minimum eigengap to use spectral method.

    Returns:
        Dictionary with is_blanket, method_used, eigengap, spectral_labels, eigvals.
    """
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    labels, eigvals, eigvecs = spectral_partition(L, n_partitions=3)

    _, eigengap = compute_eigengap(eigvals)

    if eigengap > eigengap_threshold:
        is_blanket = identify_blanket_from_spectrum(eigvals, eigvecs, labels)
        # Sanity check: blanket should be a minority
        if np.sum(is_blanket) > len(is_blanket) / 2:
            is_blanket, _ = detect_blankets_gradient(gradients)
            method_used = 'gradient_fallback'
        else:
            method_used = 'spectral'
    else:
        is_blanket, _ = detect_blankets_gradient(gradients)
        method_used = 'gradient'

    return {
        'is_blanket': is_blanket,
        'method_used': method_used,
        'eigengap': float(eigengap),
        'spectral_labels': labels,
        'eigvals': eigvals
    }


def _cluster_variables_spectral(H_est: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster variables using SpectralClustering on the weighted |H_est|.

    Uses the absolute Hessian values as a precomputed affinity matrix.
    This preserves absolute coupling strength (unlike normalized coupling,
    which dilutes large-cluster couplings).
    """
    from sklearn.cluster import SpectralClustering

    A = np.abs(H_est).copy()
    np.fill_diagonal(A, 0)

    sc = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        assign_labels='discretize', random_state=42
    )
    return sc.fit_predict(A)


def _cluster_variables_agglomerative(H_est: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster variables using AgglomerativeClustering on the weighted |H_est|.

    Uses 1/(|H_est|+eps) as distance, which provides better dynamic range
    than max-A for separating strong from weak couplings.
    """
    from sklearn.cluster import AgglomerativeClustering

    A = np.abs(H_est).copy()
    np.fill_diagonal(A, 0)

    # Inverse-coupling distance: high coupling → low distance
    D = 1.0 / (A + 0.01)
    np.fill_diagonal(D, 0)

    ac = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    )
    return ac.fit_predict(D)


def _score_clusters_entropy(labels: np.ndarray,
                             coupling: np.ndarray,
                             n_clusters: int) -> int:
    """
    Identify blanket cluster by coupling entropy across other clusters.

    Blanket variables couple uniformly to all objects (high entropy).
    Object variables couple mainly to the blanket (low entropy).

    Falls back to within/between ratio when entropy doesn't discriminate
    (e.g., symmetric configurations where all objects look similar).

    Args:
        labels: Cluster labels for all variables.
        coupling: Normalized coupling matrix.
        n_clusters: Number of clusters.

    Returns:
        Index of the identified blanket cluster.
    """
    entropies = np.full(n_clusters, -np.inf)
    ratios = np.full(n_clusters, np.inf)

    for c in range(n_clusters):
        mask_c = labels == c
        n_c = int(np.sum(mask_c))
        if n_c == 0:
            continue

        # Entropy of between-cluster coupling distribution
        profile = []
        for c2 in range(n_clusters):
            if c2 == c:
                continue
            mask_c2 = labels == c2
            if np.sum(mask_c2) == 0:
                profile.append(0)
                continue
            profile.append(coupling[np.ix_(mask_c, mask_c2)].mean())

        profile = np.array(profile)
        total = profile.sum()
        if total > 1e-10:
            p = profile / total
            entropies[c] = -np.sum(p * np.log(p + 1e-10))

        # Within/between ratio
        if n_c <= 1:
            continue
        other = ~mask_c
        within = coupling[np.ix_(mask_c, mask_c)]
        within_mean = (within.sum() - np.trace(within)) / (n_c * (n_c - 1))
        between_mean = coupling[np.ix_(mask_c, other)].mean() if np.sum(other) > 0 else 1e-10
        ratios[c] = within_mean / (between_mean + 1e-10)

    # Use entropy if it discriminates, otherwise fall back to ratio
    valid = entropies > -np.inf
    if np.sum(valid) >= 2:
        valid_e = entropies[valid]
        if np.max(valid_e) - np.min(valid_e) > 0.1:
            return int(np.argmax(entropies))

    return int(np.argmin(ratios))


def detect_blankets_coupling(H_est: np.ndarray,
                              coupling: np.ndarray,
                              n_objects: int) -> np.ndarray:
    """
    Coupling-based blanket detection for asymmetric object sizes.

    Uses a two-pass approach:

    Pass 1 (primary): Cluster all variables into n_objects + 1 groups using
    the weighted |H_est| as affinity. Identify the blanket cluster via
    coupling entropy (blanket couples uniformly across objects) or
    within/between ratio fallback (blanket has lowest ratio).

    Pass 2 (fallback): If Pass 1 produces a blanket > 40% of variables
    (indicating the clustering merged blanket with a large object), cluster
    into n_objects groups and identify blanket variables by their low
    cluster-fit scores using |H_est| values.

    This fixes the known failure mode of Otsu on asymmetric configurations
    (e.g., objects with sizes 2+2+10) by avoiding the minority-group
    assumption entirely.

    Args:
        H_est: Estimated Hessian matrix (n_vars, n_vars).
        coupling: Normalized coupling matrix (n_vars, n_vars).
        n_objects: Expected number of objects (not counting the blanket).

    Returns:
        Boolean mask identifying blanket variables.
    """
    n_vars = H_est.shape[0]
    n_total = n_objects + 1

    if n_vars < n_total + 1:
        return np.zeros(n_vars, dtype=bool)

    # --- Pass 1: SpectralClustering on |H_est|, n_objects+1 clusters ---
    # Identify blanket by coupling entropy (uniform coupling to all objects)
    # with within/between ratio fallback for symmetric cases.
    labels1 = _cluster_variables_spectral(H_est, n_total)
    bc1 = _score_clusters_entropy(labels1, coupling, n_total)
    is_blanket_p1 = (labels1 == bc1)
    n_blanket_p1 = int(np.sum(is_blanket_p1))

    if 0 < n_blanket_p1 < n_vars * 0.4:
        return is_blanket_p1

    # --- Pass 2: AgglomerativeClustering on |H_est|, n_objects+1 clusters ---
    # Agglomerative handles extreme size asymmetry better than SC
    # (e.g., 2+2+10 where SC splits the large object).
    labels2 = _cluster_variables_agglomerative(H_est, n_total)
    bc2 = _score_clusters_entropy(labels2, coupling, n_total)
    is_blanket_p2 = (labels2 == bc2)
    n_blanket_p2 = int(np.sum(is_blanket_p2))

    if 0 < n_blanket_p2 < n_vars * 0.4:
        return is_blanket_p2

    # --- Fallback: use whichever pass produced a smaller blanket ---
    if n_blanket_p1 > 0 and n_blanket_p2 > 0:
        return is_blanket_p1 if n_blanket_p1 <= n_blanket_p2 else is_blanket_p2
    elif n_blanket_p1 > 0:
        return is_blanket_p1
    elif n_blanket_p2 > 0:
        return is_blanket_p2

    return np.zeros(n_vars, dtype=bool)


# =============================================================================
# Persistence-Based Blanket Detection (H0 via Union-Find)
# =============================================================================

class _UnionFind:
    """
    Weighted union-find (disjoint set) data structure for H0 persistence.

    Tracks connected components as edges are added in decreasing weight order.
    Each merge event records the (birth, death, merging_edge) triple that
    constitutes a persistence feature.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        # Track the "birth" threshold for each component. Initially each node
        # is its own component born at +inf (present before any edge is added).
        self.birth = [np.inf] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int, edge_weight: float) -> Optional[Tuple[int, int, float]]:
        """
        Merge the components containing x and y at the given edge weight.

        Returns (root_kept, root_killed, birth_of_killed) if a merge happened,
        or None if x and y were already in the same component. The killed root
        is the "younger" component (the one born later, i.e., with lower birth
        threshold). This follows the elder rule in persistent homology.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None

        # Record birth of killed component *before* modifying anything
        birth_rx = self.birth[rx]
        birth_ry = self.birth[ry]

        # Elder rule: the component born earlier (higher birth value in a
        # descending filtration) survives; the younger one dies.
        # If births are equal, break ties by size (larger survives).
        if birth_rx < birth_ry:
            rx, ry = ry, rx
            birth_rx, birth_ry = birth_ry, birth_rx
        elif birth_rx == birth_ry and self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
            birth_rx, birth_ry = birth_ry, birth_rx

        # rx is the elder (survivor), ry is the younger (killed)
        killed_birth = birth_ry
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        return (rx, ry, killed_birth)


def compute_persistence_diagram(coupling: np.ndarray) -> Dict:
    """
    Compute the H0 persistence diagram from a coupling matrix.

    Implements a sublevel set filtration on the coupling graph in
    *descending* order: strong couplings appear first. At each threshold t
    (decreasing), include edge (i,j) if coupling[i,j] >= t. Initially all
    variables are isolated components. As the threshold decreases, components
    merge. Each merge event is a "death" of one component (the younger one
    by the elder rule).

    The persistence diagram consists of (birth, death) pairs for H0 features.
    Birth = threshold at which the component was created (inf for original
    isolated nodes). Death = threshold at which the component merged into
    an older component. Long-lived features (high persistence = birth - death)
    represent significant structural gaps in the coupling landscape.

    Additionally tracks H1 features (cycles) heuristically: when adding an
    edge that does not merge two components (the endpoints are already
    connected), it closes a cycle. The birth of the cycle is the edge weight.

    Also records snapshots of the component labeling at each merge, enabling
    downstream analysis of which components exist at any threshold.

    Args:
        coupling: Normalized coupling matrix (n_vars, n_vars), symmetric,
                  zero diagonal.

    Returns:
        Dictionary with:
            - h0_diagram: Array of (birth, death) pairs for H0 features
            - h0_edges: List of (i, j) edges that caused each H0 death
            - h0_killed_roots: List of root indices that were killed
            - h0_component_sizes: List of (size_kept, size_killed) at each merge
            - h1_diagram: Array of (birth, death) pairs for H1 (cycle) features
            - h1_edges: List of (i, j) edges that created H1 features
            - edge_weights: Sorted edge weights (descending)
            - n_vars: Number of variables
            - _union_find: The final union-find state (for component queries)
            - _merge_history: List of (threshold, edge, uf_snapshot) at each merge
    """
    n = coupling.shape[0]

    # Extract upper-triangle edges with weights
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = coupling[i, j]
            if w > 0:
                edges.append((w, i, j))

    # Sort descending: strongest couplings first
    edges.sort(key=lambda e: -e[0])

    uf = _UnionFind(n)

    h0_features = []  # (birth, death, edge_i, edge_j, killed_root)
    h0_component_sizes = []
    h1_features = []  # (birth_approx, death_approx, edge_i, edge_j)
    merge_history = []  # (threshold, (i,j), component_labels_before_merge)

    for w, i, j in edges:
        # Snapshot component labels before this edge
        result = uf.union(i, j, w)
        if result is not None:
            # A merge happened: the younger component died at threshold w
            rx_kept, ry_killed, killed_birth = result
            h0_features.append((killed_birth, w, i, j, ry_killed))
            h0_component_sizes.append((uf.size[rx_kept], 0))  # killed is now gone
            merge_history.append((w, (i, j), rx_kept, ry_killed))
        else:
            # No merge: edge closes a cycle (H1 feature)
            h1_features.append((w, 0.0, i, j))

    # Build diagram arrays
    if h0_features:
        h0_diagram = np.array([(b, d) for b, d, _, _, _ in h0_features])
        h0_edges = [(i, j) for _, _, i, j, _ in h0_features]
        h0_killed = [kr for _, _, _, _, kr in h0_features]
    else:
        h0_diagram = np.empty((0, 2))
        h0_edges = []
        h0_killed = []

    if h1_features:
        h1_diagram = np.array([(b, d) for b, d, _, _ in h1_features])
        h1_edges = [(i, j) for _, _, i, j in h1_features]
    else:
        h1_diagram = np.empty((0, 2))
        h1_edges = []

    return {
        'h0_diagram': h0_diagram,
        'h0_edges': h0_edges,
        'h0_killed_roots': h0_killed,
        'h0_component_sizes': h0_component_sizes,
        'h1_diagram': h1_diagram,
        'h1_edges': h1_edges,
        'edge_weights': np.array([w for w, _, _ in edges]),
        'n_vars': n,
        '_union_find': uf,
        '_merge_history': merge_history,
    }


def _identify_blanket_from_persistence(persistence_result: Dict,
                                        n_vars: int,
                                        coupling: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Identify blanket variables from H0 persistence features.

    In a well-structured coupling landscape with objects and blankets,
    the H0 filtration reveals two phases:

    Phase 1 (high coupling): Intra-cluster merges happen at high coupling
    thresholds; variables within tightly coupled blocks merge quickly.

    Phase 2 (low coupling): Inter-cluster bridges form at low coupling
    thresholds, connecting isolated clusters through blanket variables.

    The algorithm uses the persistence structure to find the natural
    number of structural clusters, then identifies blanket variables as
    those with the most balanced (high-entropy) coupling profile across
    the identified clusters.

    Specifically:
    1. Find the optimal number of clusters K from the persistence diagram
       by scanning candidate gap positions and selecting the one that
       yields between 2 and ~sqrt(n) clusters
    2. Replay the filtration to get cluster labels at the gap threshold
    3. Score each variable by the entropy of its coupling distribution
       across clusters: blanket variables couple to multiple clusters
       (high entropy), internal variables couple mainly to one (low entropy)
    4. Separate blanket from internal variables using the largest gap
       in the sorted entropy scores

    Args:
        persistence_result: Output of compute_persistence_diagram().
        n_vars: Total number of variables.
        coupling: Normalized coupling matrix (n_vars, n_vars). If provided,
                  enables coupling-profile-based blanket scoring.

    Returns:
        Boolean mask identifying blanket variables.
    """
    h0_diagram = persistence_result['h0_diagram']
    h0_edges = persistence_result['h0_edges']

    if len(h0_diagram) < 2:
        return np.zeros(n_vars, dtype=bool)

    deaths = h0_diagram[:, 1].copy()

    # Sort death values in descending order
    sorted_death_idx = np.argsort(-deaths)
    sorted_deaths = deaths[sorted_death_idx]
    gaps = np.abs(np.diff(sorted_deaths))

    if len(gaps) == 0:
        return np.zeros(n_vars, dtype=bool)

    # Find the best threshold to cut the dendrogram. Strategy:
    # try multiple candidate cuts (each gap in the death sequence),
    # build clusters at each cut, score the blanket identification
    # quality, and pick the best.
    #
    # For efficiency, limit to the top-k gaps and also include cuts
    # that yield small numbers of clusters (2 to sqrt(n)+2).
    max_target_clusters = max(4, min(int(np.sqrt(n_vars)) + 2, n_vars // 2))

    # Candidate thresholds: every gap position, but prioritize large gaps
    # and positions that yield reasonable cluster counts
    candidates = []
    for gi in range(len(gaps)):
        threshold = (sorted_deaths[gi] + sorted_deaths[gi + 1]) / 2.0
        n_merges = int(np.sum(deaths > threshold))
        n_comp = n_vars - n_merges
        candidates.append((gi, threshold, n_comp, gaps[gi]))

    # Also add thresholds for specific target cluster counts
    for target_k in range(2, max_target_clusters + 1):
        # Need exactly (n_vars - target_k) merges above threshold
        target_merges = n_vars - target_k
        if 0 < target_merges < len(sorted_deaths):
            # Threshold between the target_merges-th and (target_merges+1)-th death
            if target_merges < len(sorted_deaths):
                th = sorted_deaths[target_merges - 1] - 1e-10 if target_merges > 0 else sorted_deaths[0] + 1e-10
                # Actually, set threshold just below the target_merges-th death
                th = sorted_deaths[target_merges - 1] * 0.9999
                n_m = int(np.sum(deaths > th))
                n_c = n_vars - n_m
                if 2 <= n_c <= max_target_clusters:
                    candidates.append((-1, th, n_c, 0.0))

    # Score each candidate by blanket identification quality.
    # A good cut should produce clusters where one cluster has a
    # distinctly different coupling profile (blanket-like).
    best_score = -np.inf
    best_threshold = sorted_deaths[-1] * 0.5  # default
    best_blanket_cluster = 0
    best_labels = None

    if coupling is not None:
        for _, threshold, n_comp, gap_size in candidates:
            if n_comp < 2 or n_comp > max_target_clusters:
                continue

            # Build clusters at this threshold
            uf_test = _UnionFind(n_vars)
            for idx in range(len(h0_diagram)):
                if deaths[idx] > threshold:
                    i, j = h0_edges[idx]
                    uf_test.union(i, j, deaths[idx])

            test_labels = np.array([uf_test.find(v) for v in range(n_vars)])
            test_unique = np.unique(test_labels)
            test_n = len(test_unique)

            if test_n < 2:
                continue

            # Remap to contiguous labels
            lmap = {cl: idx for idx, cl in enumerate(test_unique)}
            mapped = np.array([lmap[cl] for cl in test_labels])

            # Score: use entropy separation between blanket and object
            # clusters. A good cut has one cluster with high entropy
            # (blanket) and the rest with low entropy (objects).
            entropies = []
            sizes = []
            for c in range(test_n):
                mask_c = mapped == c
                n_c = int(np.sum(mask_c))
                sizes.append(n_c)
                if n_c == 0:
                    entropies.append(0)
                    continue
                # Between-cluster coupling profile
                profile = []
                for c2 in range(test_n):
                    if c2 == c:
                        continue
                    mask_c2 = mapped == c2
                    if np.sum(mask_c2) == 0:
                        profile.append(0)
                        continue
                    profile.append(coupling[np.ix_(mask_c, mask_c2)].mean())
                profile = np.array(profile)
                total = profile.sum()
                if total > 1e-10:
                    p = profile / total
                    entropies.append(-np.sum(p * np.log(p + 1e-10)))
                else:
                    entropies.append(0)

            entropies = np.array(entropies)
            sizes = np.array(sizes)

            # The blanket cluster should have the highest entropy
            # and be a minority. Score = max_entropy - second_max_entropy
            # (entropy separation). Penalize if blanket > 40% of vars.
            blanket_c = int(np.argmax(entropies))
            blanket_size = sizes[blanket_c]

            if blanket_size >= n_vars * 0.5:
                continue  # skip: blanket too large

            sorted_e = np.sort(entropies)[::-1]
            if len(sorted_e) >= 2:
                entropy_sep = sorted_e[0] - sorted_e[1]
            else:
                entropy_sep = sorted_e[0]

            # Score: entropy separation + gap size bonus + size penalty
            score = entropy_sep + 0.1 * gap_size
            # Prefer cuts where blanket is < 40% of vars
            if blanket_size > n_vars * 0.4:
                score -= 0.5
            # Prefer having >= 2 non-blanket clusters (at least 2 objects)
            n_object_clusters = test_n - 1
            if n_object_clusters >= 2:
                score += 0.2

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_blanket_cluster = blanket_c
                best_labels = mapped.copy()

    gap_threshold = best_threshold

    # If the scoring loop already found a good cut with labeled clusters,
    # use it directly.
    if best_labels is not None and coupling is not None:
        is_blanket = (best_labels == best_blanket_cluster)
        return is_blanket

    # Otherwise, replay filtration up to the gap and identify blanket
    uf_partial = _UnionFind(n_vars)
    for idx in range(len(h0_diagram)):
        d = deaths[idx]
        if d > gap_threshold:
            i, j = h0_edges[idx]
            uf_partial.union(i, j, d)

    cluster_labels = np.array([uf_partial.find(v) for v in range(n_vars)])
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return np.zeros(n_vars, dtype=bool)

    if coupling is not None:
        label_map = {cl: idx for idx, cl in enumerate(unique_clusters)}
        mapped_labels = np.array([label_map[cl] for cl in cluster_labels])

        blanket_cluster_idx = _score_clusters_entropy(
            mapped_labels, coupling, n_clusters)

        is_blanket = (mapped_labels == blanket_cluster_idx)

        if np.sum(is_blanket) > n_vars * 0.5:
            sizes = np.array([np.sum(mapped_labels == c) for c in range(n_clusters)])
            blanket_cluster_idx = int(np.argmin(sizes))
            is_blanket = (mapped_labels == blanket_cluster_idx)

        return is_blanket

    # Fallback: without coupling matrix, use bridge-edge participation
    blanket_count = np.zeros(n_vars, dtype=int)
    for idx, (i, j) in enumerate(h0_edges):
        if deaths[idx] < gap_threshold:
            blanket_count[i] += 1
            blanket_count[j] += 1

    is_blanket = blanket_count > 0

    if np.sum(is_blanket) > n_vars * 0.5:
        is_blanket = blanket_count >= 2
        if np.sum(is_blanket) == 0 or np.sum(is_blanket) > n_vars * 0.5:
            thresh = np.percentile(blanket_count[blanket_count > 0], 75)
            is_blanket = blanket_count >= thresh

    return is_blanket


def compute_persistence_bootstrap(gradients: np.ndarray,
                                   n_bootstrap: int = 200,
                                   covariance_method: str = 'pearson',
                                   random_state: int = 42) -> Dict:
    """
    Bootstrap confidence intervals on persistence features (Fasy et al. 2014).

    Resamples the gradient matrix (with replacement on rows) n_bootstrap
    times, computes the persistence diagram for each resample, and reports
    confidence bands on each feature's (birth, death) coordinates.

    Args:
        gradients: Gradient samples of shape (N, n_vars).
        n_bootstrap: Number of bootstrap resamples.
        covariance_method: Method for Hessian estimation ('pearson' or 'rank').
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - base_diagram: The persistence diagram from the full dataset
            - bootstrap_diagrams: List of n_bootstrap persistence diagrams
            - birth_ci: (n_features, 2) array of [lo, hi] confidence intervals
                        on birth values (2.5th and 97.5th percentiles)
            - death_ci: (n_features, 2) array of [lo, hi] confidence intervals
                        on death values
            - persistence_ci: (n_features, 2) array of [lo, hi] confidence
                              intervals on persistence values
            - significant_mask: Boolean array indicating which features have
                                persistence CI lower bound > 0
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_vars = gradients.shape

    # Base diagram from full dataset
    base_features = compute_geometric_features(gradients, covariance_method=covariance_method)
    base_pd = compute_persistence_diagram(base_features['coupling'])

    # Bootstrap resamples
    bootstrap_diagrams = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_gradients = gradients[idx]
        boot_features = compute_geometric_features(boot_gradients,
                                                    covariance_method=covariance_method)
        boot_pd = compute_persistence_diagram(boot_features['coupling'])
        bootstrap_diagrams.append(boot_pd)

    # Compute confidence intervals on the base diagram features
    n_features = len(base_pd['h0_diagram'])
    if n_features == 0:
        return {
            'base_diagram': base_pd,
            'bootstrap_diagrams': bootstrap_diagrams,
            'birth_ci': np.empty((0, 2)),
            'death_ci': np.empty((0, 2)),
            'persistence_ci': np.empty((0, 2)),
            'significant_mask': np.array([], dtype=bool),
        }

    # For each base feature, collect matched bootstrap features.
    # Matching by rank: sort both base and bootstrap diagrams by persistence
    # (descending) and align by ordinal rank.
    base_births = base_pd['h0_diagram'][:, 0].copy()
    base_deaths = base_pd['h0_diagram'][:, 1].copy()
    finite_b = np.isfinite(base_births)
    max_fb = np.max(base_births[finite_b]) if np.any(finite_b) else 1.0
    base_births_f = np.where(np.isfinite(base_births), base_births, max_fb * 2)
    base_pers = base_births_f - base_deaths

    # Sort by persistence descending
    rank_order = np.argsort(-base_pers)

    # Collect bootstrap values at each rank
    boot_births_arr = np.full((n_bootstrap, n_features), np.nan)
    boot_deaths_arr = np.full((n_bootstrap, n_features), np.nan)
    boot_pers_arr = np.full((n_bootstrap, n_features), np.nan)

    for b_idx, bpd in enumerate(bootstrap_diagrams):
        if len(bpd['h0_diagram']) == 0:
            continue
        b_births = bpd['h0_diagram'][:, 0].copy()
        b_deaths = bpd['h0_diagram'][:, 1].copy()
        fb = np.isfinite(b_births)
        mfb = np.max(b_births[fb]) if np.any(fb) else 1.0
        b_births_f = np.where(np.isfinite(b_births), b_births, mfb * 2)
        b_pers = b_births_f - b_deaths
        b_rank = np.argsort(-b_pers)

        n_match = min(n_features, len(b_rank))
        for r in range(n_match):
            base_idx = rank_order[r]
            boot_idx = b_rank[r]
            boot_births_arr[b_idx, base_idx] = b_births_f[boot_idx]
            boot_deaths_arr[b_idx, base_idx] = b_deaths[boot_idx]
            boot_pers_arr[b_idx, base_idx] = b_pers[boot_idx]

    # Compute percentile CIs (ignoring NaN)
    birth_ci = np.zeros((n_features, 2))
    death_ci = np.zeros((n_features, 2))
    persistence_ci = np.zeros((n_features, 2))

    for f in range(n_features):
        valid_b = boot_births_arr[:, f][~np.isnan(boot_births_arr[:, f])]
        valid_d = boot_deaths_arr[:, f][~np.isnan(boot_deaths_arr[:, f])]
        valid_p = boot_pers_arr[:, f][~np.isnan(boot_pers_arr[:, f])]

        if len(valid_b) > 0:
            birth_ci[f] = np.percentile(valid_b, [2.5, 97.5])
        if len(valid_d) > 0:
            death_ci[f] = np.percentile(valid_d, [2.5, 97.5])
        if len(valid_p) > 0:
            persistence_ci[f] = np.percentile(valid_p, [2.5, 97.5])

    # A feature is "significant" if the lower bound of its persistence CI > 0
    significant_mask = persistence_ci[:, 0] > 0

    return {
        'base_diagram': base_pd,
        'bootstrap_diagrams': bootstrap_diagrams,
        'birth_ci': birth_ci,
        'death_ci': death_ci,
        'persistence_ci': persistence_ci,
        'significant_mask': significant_mask,
    }


def detect_blankets_persistence(features: Dict,
                                 gradients: Optional[np.ndarray] = None,
                                 n_bootstrap: int = 200,
                                 covariance_method: str = 'pearson',
                                 random_state: int = 42) -> Dict:
    """
    Persistence-based blanket detection via H0 sublevel set filtration.

    Replaces Otsu thresholding with a topologically principled, multi-scale
    approach. Instead of picking a single threshold, the method tracks how
    the topology of the coupling graph changes across all thresholds.
    Features (connected components) that persist across a wide range of
    thresholds represent real structural gaps; blanket variables are those
    involved in bridging these gaps.

    This addresses two known failure modes of Otsu:
    1. Asymmetric objects: Otsu's bimodal assumption fails when object sizes
       differ significantly (e.g., 2+2+10 configurations).
    2. Non-bimodal coupling distributions: when coupling values are not
       bimodally distributed, Otsu picks an arbitrary threshold.

    The persistence approach is threshold-free: significant features are
    identified by their persistence (lifetime in the filtration), which is
    intrinsically robust to the distribution shape.

    Optionally computes bootstrap confidence intervals on persistence
    features (Fasy et al. 2014) to quantify statistical reliability.

    Args:
        features: Dictionary from compute_geometric_features() (must contain
                  'coupling').
        gradients: Original gradient samples, needed for bootstrap CI.
                   If None, bootstrap is skipped.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        covariance_method: Method for Hessian estimation in bootstrap.
        random_state: Random seed.

    Returns:
        Dictionary with:
            - is_blanket: Boolean blanket mask
            - persistence_diagram: Full H0+H1 persistence result
            - significant_features: Indices of significant H0 features
            - persistence_values: Array of persistence values per H0 feature
            - bootstrap: Bootstrap CI result (if gradients provided),
                         or None
    """
    coupling = features['coupling']
    n_vars = coupling.shape[0]

    # Compute persistence diagram
    pd_result = compute_persistence_diagram(coupling)

    # Identify blanket variables from significant H0 features
    is_blanket = _identify_blanket_from_persistence(pd_result, n_vars, coupling=coupling)

    # Compute persistence values for reporting
    h0_diagram = pd_result['h0_diagram']
    if len(h0_diagram) > 0:
        births = h0_diagram[:, 0]
        deaths = h0_diagram[:, 1]
        finite_mask = np.isfinite(births)
        if np.any(finite_mask):
            max_fb = np.max(births[finite_mask])
        else:
            max_fb = 1.0
        births_finite = np.where(np.isfinite(births), births, max_fb * 2)
        persistence_values = births_finite - deaths
        median_p = np.median(persistence_values)
        significant_indices = np.where(persistence_values > median_p)[0]
    else:
        persistence_values = np.array([])
        significant_indices = np.array([], dtype=int)

    # Bootstrap confidence intervals if gradients are available
    bootstrap_result = None
    if gradients is not None and n_bootstrap > 0:
        bootstrap_result = compute_persistence_bootstrap(
            gradients, n_bootstrap=n_bootstrap,
            covariance_method=covariance_method,
            random_state=random_state
        )

    return {
        'is_blanket': is_blanket,
        'persistence_diagram': pd_result,
        'significant_features': significant_indices,
        'persistence_values': persistence_values,
        'bootstrap': bootstrap_result,
    }


def detect_blankets_pcca(H_est: np.ndarray,
                         n_clusters: int = 3,
                         ambiguity_threshold: float = 0.6,
                         normalized_laplacian: bool = True) -> Dict:
    """
    PCCA+ fuzzy partition blanket detection.

    Uses Perron Cluster Cluster Analysis (Deuflhard & Weber 2005) on the
    graph Laplacian eigenvectors to produce fuzzy membership vectors. Blanket
    variables are identified as those with max membership below the ambiguity
    threshold, meaning they belong significantly to multiple clusters.

    This replaces the eigenvector variance heuristic with a principled fuzzy
    partition where boundary/blanket variables naturally have distributed
    membership.

    Args:
        H_est: Estimated Hessian matrix (n_vars, n_vars).
        n_clusters: Number of clusters for PCCA+ partitioning.
        ambiguity_threshold: Max-membership threshold for blanket detection.
            Variables with max(chi[i]) < threshold are classified as blanket.
        normalized_laplacian: If True, use the symmetric normalized Laplacian
            (recommended per Shi & Malik 2000, Ng-Jordan-Weiss 2001).

    Returns:
        Dictionary with:
            - is_blanket: Boolean blanket mask
            - memberships: (n_vars, n_clusters) fuzzy membership matrix
            - hard_labels: Hard partition from argmax of memberships
            - max_membership: Max membership per variable
            - membership_entropy: Shannon entropy per variable
            - eigvals: Laplacian eigenvalues
            - eigvecs: Laplacian eigenvectors
            - simplex_vertices: PCCA+ simplex vertex indices
            - ambiguity_threshold: The threshold used
    """
    from .pcca import pcca_blanket_detection

    return pcca_blanket_detection(
        H_est,
        n_clusters=n_clusters,
        ambiguity_threshold=ambiguity_threshold,
        normalized_laplacian=normalized_laplacian,
    )
