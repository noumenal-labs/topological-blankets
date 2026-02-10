"""
US-046: Scale TB to 500D and 1000D with Sparse Hessian Approximations
======================================================================

Benchmarks 4 sparse Hessian approximation methods against the full Hessian
at dimensions 50, 100, 200, 500, 1000.

Methods:
  1. Diagonal: diag(H) only; coupling estimated from gradient correlation
  2. Diagonal + rank-k: diag(H) + top-k SVD correction (adaptive k)
  3. Random projection: Project gradients to M << d dims, compute M x M
     Hessian, reconstruct d x d coupling via variable-projection affinity
  4. Block-diagonal: Bootstrap structure from a low-D subsample, then
     compute Hessian within each discovered block

Key insights:
  - Langevin sampling does not converge at high-D because the precision
    matrix condition number grows with d (eigmax ~ d * intra_strength),
    forcing tiny step sizes. Since the landscape is a quadratic EBM with
    known stationary distribution N(0, inv(Theta)/temp), we sample
    directly from the Gaussian and compute gradients analytically.
  - At high-D, gradient-magnitude Otsu fails because blanket variables
    become a small minority and gradient magnitude distributions overlap.
    The entropy-based spectral detection (clustering into n_objects+1
    groups, selecting the cluster with highest cross-cluster coupling
    entropy as blanket) is required for d > 50.

Acceptance criteria:
  - ARI > 0.7 at 500D for at least one sparse method
  - 1000D completes without OOM (ARI documented even if degraded)
  - Wall-clock and memory comparison at 50, 100, 200, 500, 1000
  - Pareto frontier plot: ARI vs compute time
  - Best method identified per dimension regime
  - Results JSON + PNGs saved to results/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import tracemalloc
import sys
import os
import warnings
import gc
warnings.filterwarnings('ignore')

# Project root for imports
ralph_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
repo_root = os.path.join(ralph_root, '..')
sys.path.insert(0, ralph_root)
sys.path.insert(0, repo_root)

from topological_blankets.features import compute_geometric_features
from topological_blankets.detection import (
    detect_blankets_otsu, detect_blankets_coupling, detect_blankets_spectral
)
from topological_blankets.clustering import cluster_internals
from topological_blankets.spectral import (
    build_adjacency_from_hessian, build_graph_laplacian, spectral_partition,
    identify_blanket_from_spectrum, compute_eigengap
)
from experiments.quadratic_toy_comparison import (
    QuadraticEBMConfig, build_precision_matrix, get_ground_truth,
    compute_metrics
)
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# High-D blanket detection: direct entropy-based spectral method
# =========================================================================

def _spectral_cluster_variables(H_est, n_clusters):
    """
    Cluster variables using SpectralClustering on |H_est| as affinity.

    Uses the absolute Hessian values as a precomputed affinity matrix,
    which preserves the coupling-strength signal for spectral partitioning.
    """
    from sklearn.cluster import SpectralClustering

    A = np.abs(H_est).copy()
    np.fill_diagonal(A, 0)

    sc = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        assign_labels='discretize', random_state=42
    )
    return sc.fit_predict(A)


def detect_blankets_high_d(features, n_objects):
    """
    Blanket detection suitable for high-dimensional settings.

    At high-D, gradient-magnitude Otsu fails because blanket vars are a
    small minority (~15-20%) and gradient magnitude distributions overlap.

    This method clusters variables into n_objects+1 groups using spectral
    clustering on |H_est|, then identifies the blanket cluster as the one
    with the highest cross-cluster coupling entropy: the blanket couples
    uniformly to all objects (high entropy), while each object couples
    mainly to the blanket (low entropy).

    The entropy-based identification is applied directly without the
    library's 0.1-threshold fallback, which can misfire on symmetric
    configurations.

    Falls back to gradient-magnitude percentile if spectral methods fail.
    """
    H_est = features['hessian_est']
    coupling = features['coupling']
    n_vars = H_est.shape[0]

    n_total = n_objects + 1

    if n_vars < n_total + 1:
        return np.zeros(n_vars, dtype=bool)

    # Spectral clustering into n_objects+1 groups
    try:
        labels = _spectral_cluster_variables(H_est, n_total)
    except Exception:
        # Fallback: gradient magnitude percentile
        gm = features['grad_magnitude']
        threshold = np.percentile(gm, 20)
        return gm <= threshold

    # Identify blanket cluster by coupling entropy.
    # The blanket cluster should have the highest entropy of its
    # cross-cluster coupling profile (uniform coupling to all objects).
    entropies = np.full(n_total, -np.inf)
    sizes = np.zeros(n_total, dtype=int)

    for c in range(n_total):
        mask_c = labels == c
        n_c = int(np.sum(mask_c))
        sizes[c] = n_c
        if n_c == 0:
            continue

        profile = []
        for c2 in range(n_total):
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

    # Select the cluster with the highest entropy, preferring smaller
    # clusters (blanket should be a minority).
    # Score = entropy - penalty if cluster is too large (> 40% of vars)
    scores = entropies.copy()
    for c in range(n_total):
        if sizes[c] > 0.4 * n_vars:
            scores[c] -= 1.0  # penalize large clusters
        if sizes[c] == 0:
            scores[c] = -np.inf

    blanket_cluster = int(np.argmax(scores))
    is_blanket = (labels == blanket_cluster)
    n_blanket = int(np.sum(is_blanket))

    # Validate: blanket should be between 5% and 50% of variables
    if 0.05 * n_vars <= n_blanket <= 0.5 * n_vars:
        return is_blanket

    # Fallback: try the library's coupling-based detection
    try:
        is_blanket_lib = detect_blankets_coupling(H_est, coupling, n_objects)
        n_blanket_lib = int(np.sum(is_blanket_lib))
        if 0.05 * n_vars <= n_blanket_lib <= 0.5 * n_vars:
            return is_blanket_lib
    except Exception:
        pass

    # Last resort: gradient magnitude percentile
    gm = features['grad_magnitude']
    threshold = np.percentile(gm, 20)
    is_blanket_pct = gm <= threshold
    if np.sum(is_blanket_pct) > 0:
        return is_blanket_pct

    return is_blanket


# =========================================================================
# Sparse Hessian method 1: Diagonal with gradient-correlation coupling
# =========================================================================

def hessian_diagonal(gradients):
    """
    Diagonal Hessian approximation with gradient-correlation coupling.

    The Hessian is restricted to its diagonal. Coupling is estimated from
    the correlation of gradient time-series across variables. This preserves
    pairwise structure needed for spectral clustering without computing the
    full covariance.
    """
    n_samples, n_vars = gradients.shape
    grad_magnitude = np.mean(np.abs(gradients), axis=0)
    grad_variance = np.var(gradients, axis=0)
    H_est = np.diag(grad_variance)

    # Correlation matrix as coupling proxy
    g_centered = gradients - gradients.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(g_centered ** 2, axis=0)) + 1e-8
    g_normed = g_centered / norms

    # For very high-D, compute correlation in chunks to control memory
    if n_vars > 500:
        coupling = _chunked_correlation(g_normed, n_samples, chunk_size=200)
    else:
        coupling = np.abs(g_normed.T @ g_normed) / n_samples
    np.fill_diagonal(coupling, 0)

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': H_est,
        'coupling': coupling,
    }


def _chunked_correlation(g_normed, n_samples, chunk_size=200):
    """Compute correlation matrix in chunks to reduce peak memory."""
    n_vars = g_normed.shape[1]
    coupling = np.zeros((n_vars, n_vars))
    for i in range(0, n_vars, chunk_size):
        ie = min(i + chunk_size, n_vars)
        for j in range(i, n_vars, chunk_size):
            je = min(j + chunk_size, n_vars)
            block = np.abs(g_normed[:, i:ie].T @ g_normed[:, j:je]) / n_samples
            coupling[i:ie, j:je] = block
            if i != j:
                coupling[j:je, i:ie] = block.T
    return coupling


# =========================================================================
# Sparse Hessian method 2: Diagonal + rank-k (adaptive rank)
# =========================================================================

def hessian_diagonal_rank_k(gradients, rank=None):
    """
    Diagonal + rank-k Hessian approximation with dimension-adaptive rank.

    H_approx = diag(var) + V_k S_k^2 V_k^T

    The rank scales with dimension: k = max(10, d//10), capped at 100.
    Uses scipy.sparse.linalg.svds for O(Ndk) efficiency.
    """
    n_samples, n_vars = gradients.shape
    grad_variance = np.var(gradients, axis=0)
    grad_magnitude = np.mean(np.abs(gradients), axis=0)

    if rank is None:
        rank = min(max(10, n_vars // 10), 100)

    k = min(rank, min(n_samples, n_vars) - 2)
    if k < 1:
        H_est = np.diag(grad_variance)
    else:
        grad_centered = gradients - gradients.mean(axis=0)
        scaled = grad_centered / np.sqrt(max(n_samples - 1, 1))

        try:
            from scipy.sparse.linalg import svds
            U, S, Vt = svds(scaled, k=k)
            low_rank = Vt.T @ np.diag(S ** 2) @ Vt
            H_est = np.diag(grad_variance) + low_rank
        except Exception:
            H_est = np.diag(grad_variance)

    D = np.sqrt(np.abs(np.diag(H_est))) + 1e-8
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': H_est,
        'coupling': coupling,
    }


# =========================================================================
# Sparse Hessian method 3: Random projection (Johnson-Lindenstrauss)
# =========================================================================

def hessian_random_projection(gradients, target_dim=None, n_projections=5):
    """
    Random projection Hessian approximation.

    Generates multiple random projection matrices, computes approximate
    coupling from each, and averages to reduce JL distortion.
    """
    n_samples, n_vars = gradients.shape

    if target_dim is None:
        target_dim = min(max(int(8 * np.log(n_vars + 1)), 30), n_vars)

    if target_dim >= n_vars:
        return compute_geometric_features(gradients)

    grad_magnitude = np.mean(np.abs(gradients), axis=0)
    grad_variance = np.var(gradients, axis=0)

    # Average coupling over multiple random projections
    coupling_accum = np.zeros((n_vars, n_vars))
    H_accum = np.zeros((n_vars, n_vars))

    for proj_idx in range(n_projections):
        rng = np.random.RandomState(42 + proj_idx)
        R = rng.randn(n_vars, target_dim) / np.sqrt(target_dim)

        G_proj = gradients @ R
        H_proj = np.cov(G_proj.T)
        if H_proj.ndim == 0:
            H_proj = np.array([[float(H_proj)]])

        H_approx = R @ H_proj @ R.T
        H_accum += H_approx

        D_proj = np.sqrt(np.abs(np.diag(H_approx))) + 1e-8
        C_proj = np.abs(H_approx) / np.outer(D_proj, D_proj)
        np.fill_diagonal(C_proj, 0)
        coupling_accum += C_proj

    coupling = coupling_accum / n_projections
    np.fill_diagonal(coupling, 0)
    H_est = H_accum / n_projections

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': H_est,
        'coupling': coupling,
    }


# =========================================================================
# Sparse Hessian method 4: Block-diagonal (bootstrap)
# =========================================================================

def hessian_block_diagonal(gradients, n_objects=2, subsample_frac=0.3,
                           min_subsample=30):
    """
    Block-diagonal Hessian approximation via structure bootstrap.

    1. Subsample a fraction of dimensions.
    2. Run full TB on the subsampled dimensions to discover block structure.
    3. Assign remaining dimensions to blocks via gradient correlation.
    4. Compute the full Hessian within each block only (block-diagonal).
    """
    n_samples, n_vars = gradients.shape
    grad_magnitude = np.mean(np.abs(gradients), axis=0)
    grad_variance = np.var(gradients, axis=0)

    n_sub = max(min_subsample, int(n_vars * subsample_frac))
    n_sub = min(n_sub, n_vars)

    rng = np.random.RandomState(42)
    sub_dims = rng.choice(n_vars, size=n_sub, replace=False)
    sub_dims.sort()

    # Run full TB on subsampled dimensions with coupling-based detection
    G_sub = gradients[:, sub_dims]
    sub_features = compute_geometric_features(G_sub)
    sub_is_blanket = detect_blankets_high_d(sub_features, n_objects)
    sub_assignment = cluster_internals(
        sub_features, sub_is_blanket, n_clusters=n_objects
    )

    # Assign all dimensions to blocks via gradient correlation
    block_ids = np.unique(sub_assignment)
    block_centroids = {}
    for bid in block_ids:
        mask = sub_assignment == bid
        dims_in_block = sub_dims[mask]
        if len(dims_in_block) > 0:
            block_centroids[bid] = np.mean(gradients[:, dims_in_block], axis=1)

    full_assignment = np.full(n_vars, -1, dtype=int)

    if len(block_centroids) > 0:
        centroids_matrix = np.column_stack(
            [block_centroids[bid] for bid in sorted(block_centroids.keys())]
        )
        bid_list = sorted(block_centroids.keys())

        g_centered = gradients - gradients.mean(axis=0, keepdims=True)
        c_centered = centroids_matrix - centroids_matrix.mean(axis=0, keepdims=True)
        g_norms = np.sqrt(np.sum(g_centered ** 2, axis=0)) + 1e-8
        c_norms = np.sqrt(np.sum(c_centered ** 2, axis=0)) + 1e-8

        corr_matrix = (g_centered / g_norms).T @ (c_centered / c_norms)
        corr_matrix /= n_samples

        best_block_idx = np.argmax(np.abs(corr_matrix), axis=1)
        for dim in range(n_vars):
            full_assignment[dim] = bid_list[best_block_idx[dim]]

    # Compute block-diagonal Hessian
    H_est = np.zeros((n_vars, n_vars))
    for bid in block_ids:
        block_dims = np.where(full_assignment == bid)[0]
        if len(block_dims) < 2:
            for d_idx in block_dims:
                H_est[d_idx, d_idx] = grad_variance[d_idx]
            continue
        G_block = gradients[:, block_dims]
        H_block = np.cov(G_block.T)
        if H_block.ndim == 0:
            H_block = np.array([[float(H_block)]])
        ix = np.ix_(block_dims, block_dims)
        H_est[ix] = H_block

    D = np.sqrt(np.abs(np.diag(H_est))) + 1e-8
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    return {
        'grad_magnitude': grad_magnitude,
        'grad_variance': grad_variance,
        'hessian_est': H_est,
        'coupling': coupling,
    }


# =========================================================================
# TB pipeline runner with pluggable Hessian
# =========================================================================

def run_tb_with_features(features, n_objects):
    """
    Run TB detection + clustering given pre-computed features.

    Uses the high-D coupling-aware detection method.
    """
    is_blanket = detect_blankets_high_d(features, n_objects)
    assignment = cluster_internals(features, is_blanket, n_clusters=n_objects)

    return {
        'assignment': assignment,
        'is_blanket': is_blanket,
    }


# =========================================================================
# Synthetic landscape generation (direct Gaussian sampling)
# =========================================================================

def generate_landscape(total_dim, n_objects, blanket_fraction=0.2,
                       intra_strength=8.0, blanket_strength=1.0,
                       n_samples_multiplier=80, seed=42, temp=0.1):
    """
    Generate a synthetic quadratic EBM landscape at a given dimension.

    Uses *direct Gaussian sampling* instead of Langevin dynamics. For
    a quadratic EBM E(x) = (1/2) x^T Theta x, the stationary distribution
    is N(0, inv(Theta)/temp). We sample x directly from this distribution
    and compute gradients analytically as grad_E = Theta @ x.

    This avoids the Langevin convergence problem at high-D, where the
    condition number kappa(Theta) = eigmax/eigmin grows with dimension,
    forcing step sizes so small that the chain never mixes.

    Parameters:
    - blanket_fraction=0.2 (20%) keeps blanket detectable at 1000D
    - intra_strength=8.0 for strong within-object coupling
    - blanket_strength=1.0 for clear cross-object mediation
    - n_samples_multiplier=80 for adequate gradient coverage at high-D
      (the covariance matrix has d*(d+1)/2 parameters; at d=500 that is
      125,250, so we need n >> d for good estimation. 80*d gives ~4x)
    - temp=0.1 controls the sampling temperature
    """
    vars_per_blanket = max(3, int(total_dim * blanket_fraction))
    remaining = total_dim - vars_per_blanket
    vars_per_object = remaining // n_objects
    vars_per_blanket = total_dim - n_objects * vars_per_object

    cfg = QuadraticEBMConfig(
        n_objects=n_objects,
        vars_per_object=vars_per_object,
        vars_per_blanket=vars_per_blanket,
        intra_strength=intra_strength,
        blanket_strength=blanket_strength,
    )

    actual_dim = n_objects * vars_per_object + vars_per_blanket
    Theta = build_precision_matrix(cfg)
    truth = get_ground_truth(cfg)

    # Direct sampling from N(0, inv(Theta)/temp)
    Sigma = np.linalg.inv(Theta) / temp
    # Ensure numerical symmetry
    Sigma = (Sigma + Sigma.T) / 2.0
    L = np.linalg.cholesky(Sigma)

    rng = np.random.RandomState(seed)
    n_samples = max(3000, actual_dim * n_samples_multiplier)
    z = rng.randn(n_samples, actual_dim)
    samples = z @ L.T  # x ~ N(0, Sigma)

    # Gradient of quadratic EBM: grad_E(x) = Theta @ x
    gradients = samples @ Theta.T

    return gradients, truth, cfg


# =========================================================================
# Benchmark runner
# =========================================================================

def benchmark_single(gradients, truth, n_objects, method_name, hessian_fn,
                     hessian_kwargs=None):
    """
    Benchmark a single sparse Hessian method on a single landscape.
    Returns dict with ARI, F1, wall-clock time, peak memory.
    """
    if hessian_kwargs is None:
        hessian_kwargs = {}

    gc.collect()
    tracemalloc.start()
    t0 = time.time()

    features = hessian_fn(gradients, **hessian_kwargs)
    result = run_tb_with_features(features, n_objects)

    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = compute_metrics(result, truth)

    return {
        'method': method_name,
        'object_ari': float(metrics['object_ari']),
        'blanket_f1': float(metrics['blanket_f1']),
        'full_ari': float(metrics['full_ari']),
        'time_s': float(elapsed),
        'memory_mb': float(peak / (1024 * 1024)),
    }


# =========================================================================
# Main experiment
# =========================================================================

def run_scaling_high_d():
    """
    Run the full US-046 scaling experiment.

    Tests 5 methods (full + 4 sparse) at 5 dimensions (50..1000),
    with 3 trials per configuration.
    """
    print("=" * 70)
    print("US-046: Scale TB to 500D and 1000D with Sparse Hessian Approximations")
    print("=" * 70)

    dimensions = [50, 100, 200, 500, 1000]
    n_objects = 2
    n_trials = 3

    methods = [
        ('full', compute_geometric_features, {}),
        ('diagonal', hessian_diagonal, {}),
        ('diag_rank_k', hessian_diagonal_rank_k, {'rank': None}),
        ('random_proj', hessian_random_projection, {'n_projections': 5}),
        ('block_diag', hessian_block_diagonal, {'n_objects': n_objects}),
    ]

    all_results = {}

    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"  Dimension: {dim}")
        print(f"{'='*60}")

        dim_results = {}

        # Skip full Hessian at 1000D to avoid d^2 covariance memory
        methods_for_dim = methods if dim <= 500 else [
            m for m in methods if m[0] != 'full'
        ]

        for mname, mfn, mkwargs in methods_for_dim:
            trial_results = []

            for trial in range(n_trials):
                print(f"  {mname:15s} trial {trial+1}/{n_trials} ... ",
                      end='', flush=True)

                try:
                    gradients, truth, cfg = generate_landscape(
                        dim, n_objects, seed=42 + trial
                    )

                    res = benchmark_single(
                        gradients, truth, n_objects,
                        mname, mfn, mkwargs
                    )
                    trial_results.append(res)
                    print(f"ARI={res['object_ari']:.3f}  "
                          f"F1={res['blanket_f1']:.3f}  "
                          f"time={res['time_s']:.2f}s  "
                          f"mem={res['memory_mb']:.1f}MB")

                except MemoryError:
                    print("OOM!")
                    trial_results.append({
                        'method': mname, 'object_ari': 0.0,
                        'blanket_f1': 0.0, 'full_ari': 0.0,
                        'time_s': float('inf'),
                        'memory_mb': float('inf'),
                        'oom': True,
                    })
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    trial_results.append({
                        'method': mname, 'object_ari': 0.0,
                        'blanket_f1': 0.0, 'full_ari': 0.0,
                        'time_s': float('inf'),
                        'memory_mb': float('inf'),
                        'error': str(e),
                    })
                finally:
                    gc.collect()

            # Aggregate across trials
            valid = [r for r in trial_results
                     if r.get('time_s', float('inf')) < float('inf')]
            aris = [r['object_ari'] for r in valid]
            f1s = [r['blanket_f1'] for r in valid]
            times = [r['time_s'] for r in valid]
            mems = [r['memory_mb'] for r in valid]

            dim_results[mname] = {
                'mean_ari': float(np.mean(aris)) if aris else 0.0,
                'std_ari': float(np.std(aris)) if aris else 0.0,
                'mean_f1': float(np.mean(f1s)) if f1s else 0.0,
                'std_f1': float(np.std(f1s)) if f1s else 0.0,
                'mean_time_s': float(np.mean(times)) if times else float('inf'),
                'mean_memory_mb': float(np.mean(mems)) if mems else float('inf'),
                'per_trial': trial_results,
            }

            s = dim_results[mname]
            print(f"  >> {mname:15s} AVG: "
                  f"ARI={s['mean_ari']:.3f}+/-{s['std_ari']:.3f}  "
                  f"F1={s['mean_f1']:.3f}  "
                  f"time={s['mean_time_s']:.2f}s  "
                  f"mem={s['mean_memory_mb']:.1f}MB")

        all_results[str(dim)] = dim_results

    # =====================================================================
    # Identify best method per dimension regime
    # =====================================================================
    print("\n" + "=" * 70)
    print("  Best Method per Dimension Regime")
    print("=" * 70)

    best_methods = {}
    for dim in dimensions:
        dim_key = str(dim)
        if dim_key not in all_results:
            continue
        dr = all_results[dim_key]
        best_name = max(dr.keys(), key=lambda m: dr[m]['mean_ari'])
        best_methods[dim_key] = {
            'method': best_name,
            'ari': dr[best_name]['mean_ari'],
            'time_s': dr[best_name]['mean_time_s'],
            'memory_mb': dr[best_name]['mean_memory_mb'],
        }
        print(f"  dim={dim:5d}: {best_name:15s} "
              f"(ARI={dr[best_name]['mean_ari']:.3f}, "
              f"time={dr[best_name]['mean_time_s']:.2f}s, "
              f"mem={dr[best_name]['mean_memory_mb']:.1f}MB)")

    # =====================================================================
    # Check acceptance criteria
    # =====================================================================
    print("\n" + "=" * 70)
    print("  Acceptance Criteria Check")
    print("=" * 70)

    sparse_methods = ['diagonal', 'diag_rank_k', 'random_proj', 'block_diag']

    # Criterion 1: ARI > 0.7 at 500D for at least one sparse method
    ari_500 = {m: all_results.get('500', {}).get(m, {}).get('mean_ari', 0)
               for m in sparse_methods}
    best_500 = max(ari_500.items(), key=lambda x: x[1])
    criterion_1 = best_500[1] > 0.7
    print(f"  [{'PASS' if criterion_1 else 'FAIL'}] "
          f"ARI > 0.7 at 500D: best = {best_500[0]} "
          f"with ARI={best_500[1]:.3f}")

    # Criterion 2: 1000D completes without OOM
    ran_1000 = '1000' in all_results and len(all_results['1000']) > 0
    any_oom = False
    if ran_1000:
        for m, v in all_results['1000'].items():
            for tr in v.get('per_trial', []):
                if tr.get('oom', False):
                    any_oom = True
    criterion_2 = ran_1000 and not any_oom
    if ran_1000:
        ari_1000 = {m: v['mean_ari'] for m, v in all_results['1000'].items()}
        print(f"  [{'PASS' if criterion_2 else 'FAIL'}] "
              f"1000D without OOM: {ari_1000}")
    else:
        print(f"  [FAIL] 1000D did not run")

    # Criterion 3: 4 sparse methods benchmarked
    criterion_3 = all(m in all_results.get('50', {}) for m in sparse_methods)
    print(f"  [{'PASS' if criterion_3 else 'FAIL'}] "
          f"4 sparse methods benchmarked")

    all_pass = criterion_1 and criterion_2 and criterion_3

    # =====================================================================
    # Save results
    # =====================================================================
    config = {
        'dimensions': dimensions,
        'n_objects': n_objects,
        'n_trials': n_trials,
        'methods': ['full'] + sparse_methods,
        'blanket_fraction': 0.2,
        'intra_strength': 8.0,
        'blanket_strength': 1.0,
        'n_samples_multiplier': 80,
        'sampling_method': 'direct_gaussian',
        'detection_method': 'entropy-based spectral (high-D adapted)',
    }

    save_payload = {
        'scaling_results': all_results,
        'best_methods': best_methods,
        'acceptance_criteria': {
            'ari_gt_0.7_at_500d': criterion_1,
            '1000d_no_oom': criterion_2,
            '4_sparse_methods': criterion_3,
            'all_pass': all_pass,
        },
    }

    save_results('scaling_high_d', save_payload, config,
                 notes='US-046: Scale TB to 500D/1000D with 4 sparse Hessian methods. '
                       'Uses direct Gaussian sampling (not Langevin) for convergent '
                       'gradient statistics and entropy-based spectral detection for '
                       'high-D robustness.')

    # =====================================================================
    # Plots
    # =====================================================================
    _plot_scaling_curves(all_results, dimensions)
    _plot_pareto_frontier(all_results, dimensions)
    _plot_memory_comparison(all_results, dimensions)

    print(f"\nUS-046 {'PASSED' if all_pass else 'completed (check criteria above)'}.")
    return all_results, all_pass


# =========================================================================
# Plotting
# =========================================================================

_ALL_METHODS = ['full', 'diagonal', 'diag_rank_k', 'random_proj', 'block_diag']
_COLORS = {
    'full': '#2ecc71', 'diagonal': '#e74c3c',
    'diag_rank_k': '#3498db', 'random_proj': '#9b59b6',
    'block_diag': '#f39c12',
}
_LABELS = {
    'full': 'Full Hessian', 'diagonal': 'Diagonal',
    'diag_rank_k': 'Diag + rank-k', 'random_proj': 'Random Projection',
    'block_diag': 'Block-Diagonal',
}
_MARKERS = {
    'full': 'o', 'diagonal': 's', 'diag_rank_k': '^',
    'random_proj': 'D', 'block_diag': 'v',
}


def _plot_scaling_curves(all_results, dimensions):
    """Three-panel figure: ARI, Time, Memory vs dimension."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for m in _ALL_METHODS:
        dims, aris, stds = [], [], []
        for d in dimensions:
            dk = str(d)
            if dk in all_results and m in all_results[dk]:
                dims.append(d)
                aris.append(all_results[dk][m]['mean_ari'])
                stds.append(all_results[dk][m]['std_ari'])
        if dims:
            axes[0].errorbar(dims, aris, yerr=stds, label=_LABELS[m],
                             color=_COLORS[m], marker=_MARKERS[m],
                             capsize=3, linewidth=2, markersize=6)

    axes[0].axhline(y=0.7, color='gray', linestyle='--', alpha=0.5,
                    label='ARI=0.7 target')
    axes[0].set_xlabel('Dimension', fontsize=11)
    axes[0].set_ylabel('Object ARI', fontsize=11)
    axes[0].set_title('Object Recovery vs Dimension', fontsize=12)
    axes[0].legend(fontsize=8, loc='lower left')
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].set_xscale('log')
    axes[0].set_xticks(dimensions)
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].grid(True, alpha=0.3)

    for m in _ALL_METHODS:
        dims, ts = [], []
        for d in dimensions:
            dk = str(d)
            if dk in all_results and m in all_results[dk]:
                t = all_results[dk][m]['mean_time_s']
                if t < float('inf'):
                    dims.append(d)
                    ts.append(t)
        if dims:
            axes[1].plot(dims, ts, label=_LABELS[m], color=_COLORS[m],
                         marker=_MARKERS[m], linewidth=2, markersize=6)

    axes[1].set_xlabel('Dimension', fontsize=11)
    axes[1].set_ylabel('Wall-clock Time (s)', fontsize=11)
    axes[1].set_title('Runtime vs Dimension', fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xticks(dimensions)
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].grid(True, alpha=0.3)

    for m in _ALL_METHODS:
        dims, mems = [], []
        for d in dimensions:
            dk = str(d)
            if dk in all_results and m in all_results[dk]:
                mem = all_results[dk][m]['mean_memory_mb']
                if mem < float('inf'):
                    dims.append(d)
                    mems.append(mem)
        if dims:
            axes[2].plot(dims, mems, label=_LABELS[m], color=_COLORS[m],
                         marker=_MARKERS[m], linewidth=2, markersize=6)

    axes[2].set_xlabel('Dimension', fontsize=11)
    axes[2].set_ylabel('Peak Memory (MB)', fontsize=11)
    axes[2].set_title('Memory Usage vs Dimension', fontsize=12)
    axes[2].legend(fontsize=8)
    axes[2].set_yscale('log')
    axes[2].set_xscale('log')
    axes[2].set_xticks(dimensions)
    axes[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'scaling_curves', 'scaling_high_d')


def _plot_pareto_frontier(all_results, dimensions):
    """Pareto frontier: ARI vs compute time for all methods and dimensions."""
    fig, ax = plt.subplots(figsize=(10, 7))

    points = []
    for d in dimensions:
        dk = str(d)
        if dk not in all_results:
            continue
        for m in _ALL_METHODS:
            if m not in all_results[dk]:
                continue
            t = all_results[dk][m]['mean_time_s']
            ari = all_results[dk][m]['mean_ari']
            if t < float('inf'):
                points.append((t, ari, m, d))

    for m in _ALL_METHODS:
        method_pts = [(t, a, d) for t, a, mm, d in points if mm == m]
        if not method_pts:
            continue
        ts = [p[0] for p in method_pts]
        aris = [p[1] for p in method_pts]
        dims = [p[2] for p in method_pts]
        sizes = [30 + 20 * np.log2(d / 50 + 1) for d in dims]
        ax.scatter(ts, aris, c=_COLORS[m], s=sizes, label=_LABELS[m],
                   alpha=0.8, edgecolors='black', linewidths=0.5, zorder=3)
        for t, a, d in method_pts:
            ax.annotate(f'{d}D', (t, a), fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

    sorted_pts = sorted(points, key=lambda p: p[0])
    pareto = []
    best_ari = -1
    for t, a, m, d in sorted_pts:
        if a > best_ari:
            best_ari = a
            pareto.append((t, a))
    if len(pareto) > 1:
        ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
                'k--', linewidth=1.5, alpha=0.5, label='Pareto frontier',
                zorder=2)

    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5,
               label='ARI=0.7 target')
    ax.set_xlabel('Wall-clock Time (s)', fontsize=11)
    ax.set_ylabel('Object ARI', fontsize=11)
    ax.set_title('Accuracy-Cost Pareto Frontier\n'
                 '(all methods, all dimensions)', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'pareto_frontier', 'scaling_high_d')


def _plot_memory_comparison(all_results, dimensions):
    """Grouped bar chart of memory usage at each dimension."""
    fig, ax = plt.subplots(figsize=(12, 6))

    dims_available = [d for d in dimensions if str(d) in all_results]
    n_dims = len(dims_available)
    n_methods = len(_ALL_METHODS)
    bar_width = 0.15
    x = np.arange(n_dims)

    short_labels = {
        'full': 'Full', 'diagonal': 'Diagonal',
        'diag_rank_k': 'Diag+rank-k', 'random_proj': 'RandProj',
        'block_diag': 'BlockDiag',
    }
    for i, m in enumerate(_ALL_METHODS):
        mems = []
        for d in dims_available:
            dk = str(d)
            if m in all_results[dk]:
                mem = all_results[dk][m]['mean_memory_mb']
                mems.append(mem if mem < float('inf') else 0)
            else:
                mems.append(0)
        ax.bar(x + i * bar_width, mems, bar_width,
               label=short_labels[m], color=_COLORS[m],
               edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Peak Memory (MB)', fontsize=11)
    ax.set_title('Memory Usage Comparison by Dimension', fontsize=12)
    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([str(d) for d in dims_available])
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, 'memory_comparison', 'scaling_high_d')


# =========================================================================
# Entry point
# =========================================================================

if __name__ == '__main__':
    all_results, all_pass = run_scaling_high_d()
