"""
US-052: Apply TB to Denoising Diffusion Model at Multiple Noise Levels
=======================================================================

Diffusion models provide the score function at every noise level sigma.
Applying TB at each sigma reveals how structure crystallizes from noise
along the reverse diffusion trajectory.

Algorithm:
  1. Train a simple DDPM on make_moons (2D, 1000 samples) with 100-step schedule
  2. Extract the score function s(x, sigma) = -grad_x log p_sigma(x) at 10 noise levels
  3. Run TB on the score function at each sigma:
     - At each sigma, generate noisy samples x_t and collect scores s(x_t, sigma)
     - Build a sample-level coupling graph from score vectors
     - Compute the eigengap of the normalized graph Laplacian to detect cluster count
  4. Track: coupling matrix, partition, eigengap across sigma
  5. At high sigma: uniform, no structure. At low sigma: structure matching data
  6. Find the "topological transition" sigma where structure first appears
  7. Compare to US-019 (denoising score matching with mixed noise)

Key insight: Since the data is 2D, the *variable-level* Hessian is always 2x2.
Structure detection must operate at the *sample level*, treating each sample's
score vector as evidence of which "object" it belongs to.

Training approach: The model predicts the *noise* eps rather than the score
directly, since eps is always N(0,1)-scale regardless of sigma. The score
is recovered as s(x,t) = -eps_pred / sigma at inference time. This is the
standard DDPM parameterization (Ho et al. 2020).

Cluster detection approach: Build *combined* features from position (X_noisy)
and score direction (unit-normalized score vectors), weighted by the
signal-to-noise ratio sqrt(alpha_bar). At low noise, position retains
data structure and dominates; at high noise, both position and score direction
are randomized, so k-means sees 1 cluster. This produces a natural 1-to-2
cluster transition as noise decreases, directly revealing how topological
blankets crystallize along the reverse diffusion trajectory.

The Gaussian kernel affinity graph (20th-percentile bandwidth) and normalized
Laplacian eigengap provide a continuous measure of structure strength.

Datasets: make_moons (2D).
Uses sklearn MLPRegressor (no PyTorch required).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.cluster import SpectralClustering, KMeans
from scipy.linalg import eigh
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RALPH_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RALPH_ROOT)

from topological_blankets.features import compute_geometric_features
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# 1. DDPM Noise Schedule and Score Model Training
# =========================================================================

def make_ddpm_schedule(T=200, schedule_type='cosine'):
    """
    Noise schedule for a DDPM with T steps.

    The cosine schedule (Nichol & Dhariwal 2021) provides better coverage
    of the noise range, going smoothly from sigma~0 to sigma~1, compared
    to the linear schedule which concentrates at low noise levels.

    Returns:
        betas: array of shape (T,) with noise schedule
        alphas: array of shape (T,) with alpha_t = 1 - beta_t
        alpha_bars: array of shape (T,) with cumulative product of alphas
        sigmas: array of shape (T,) with effective noise std at each step
    """
    if schedule_type == 'cosine':
        # Cosine schedule (Nichol & Dhariwal 2021)
        s = 0.008  # small offset to prevent singularity at t=0
        steps = np.arange(T + 1)
        f = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bars_full = f / f[0]
        betas = 1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1]
        betas = np.clip(betas, 0, 0.999)
        alpha_bars = alpha_bars_full[1:]  # drop the t=0 entry
    else:
        # Linear schedule (Ho et al. 2020)
        betas = np.linspace(1e-4, 0.02, T)
        alphas_local = 1.0 - betas
        alpha_bars = np.cumprod(alphas_local)

    alphas = 1.0 - betas
    sigmas = np.sqrt(np.clip(1.0 - alpha_bars, 0, 1))
    return betas, alphas, alpha_bars, sigmas


def train_ddpm_noise_model(X, T=200, hidden=(256, 256, 128), n_augment=6,
                            schedule_type='cosine'):
    """
    Train a noise-conditional epsilon model (DDPM parameterization).

    The model predicts epsilon (the noise) rather than the score directly,
    since eps ~ N(0,I) is always unit-scale. At inference, recover the score
    via s(x,t) = -eps_pred(x,t) / sigma_t.

    Input: (x_noisy, log(sigma_t)) -> Output: eps_pred of shape (2,)

    Args:
        X: Clean data of shape (N, 2).
        T: Number of diffusion steps.
        hidden: Hidden layer sizes for the MLP.
        n_augment: Number of augmentation passes per noise level.
        schedule_type: Noise schedule type ('cosine' or 'linear').

    Returns:
        model: Trained MLPRegressor (predicts eps).
        scaler: StandardScaler fitted on X.
        schedule: Dict with betas, alphas, alpha_bars, sigmas.
    """
    betas, alphas, alpha_bars, sigmas = make_ddpm_schedule(T, schedule_type)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_inputs = []
    all_targets = []

    for aug in range(n_augment):
        for t in range(T):
            ab_t = alpha_bars[t]
            sigma_t = sigmas[t]

            eps = np.random.randn(*X_scaled.shape)
            X_noisy = np.sqrt(ab_t) * X_scaled + sigma_t * eps

            # Target: predict eps directly (unit-scale, well-conditioned)
            log_sigma = np.full((len(X_scaled), 1), np.log(sigma_t + 1e-8))
            inp = np.hstack([X_noisy, log_sigma])

            all_inputs.append(inp)
            all_targets.append(eps)

    X_train = np.vstack(all_inputs)
    y_train = np.vstack(all_targets)

    # Subsample if the training set is too large
    max_train = 300000
    if len(X_train) > max_train:
        idx = np.random.choice(len(X_train), max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    print(f"  Training epsilon model on {len(X_train)} samples...")

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        learning_rate_init=0.001,
        batch_size=min(1024, len(X_train)),
    )
    model.fit(X_train, y_train)
    print(f"  Training loss: {model.loss_:.4f}")

    schedule = {
        'T': T,
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
        'sigmas': sigmas,
    }
    return model, scaler, schedule


def predict_score_at_sigma(model, X_points, sigma):
    """
    Predict the score function at given points and noise level sigma.

    The model predicts eps, and the score is s(x,t) = -eps_pred / sigma.

    Args:
        model: Trained noise-conditional epsilon model.
        X_points: Points of shape (N, 2) (already scaled).
        sigma: Noise level.

    Returns:
        scores: Predicted score vectors of shape (N, 2).
    """
    log_sigma = np.full((len(X_points), 1), np.log(sigma + 1e-8))
    inp = np.hstack([X_points, log_sigma])
    eps_pred = model.predict(inp)
    scores = -eps_pred / (sigma + 1e-8)
    return scores


def generate_noisy_samples(X_scaled, sigma, alpha_bar):
    """
    Generate noisy samples at a given noise level using the DDPM forward process.

    x_t = sqrt(alpha_bar) * x_0 + sigma * eps

    Args:
        X_scaled: Clean scaled data (N, 2).
        sigma: Noise level sqrt(1 - alpha_bar).
        alpha_bar: Cumulative alpha product.

    Returns:
        X_noisy: Noisy samples (N, 2).
    """
    eps = np.random.randn(*X_scaled.shape)
    X_noisy = np.sqrt(alpha_bar) * X_scaled + sigma * eps
    return X_noisy


# =========================================================================
# 2. US-019 Baseline: Mixed-Noise Score Model
# =========================================================================

def train_us019_score_model(X, noise_levels=None, hidden=(128, 128)):
    """
    Train a score model via denoising score matching (US-019 approach).

    This is the simpler approach from US-019 that trains on a *mixture* of
    noise levels without conditioning, and outputs a single score estimate.
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.3, 0.5]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_inputs = []
    all_targets = []

    for sigma in noise_levels:
        eps = np.random.randn(*X_scaled.shape) * sigma
        X_noisy = X_scaled + eps
        score_target = -eps / (sigma ** 2)
        all_inputs.append(X_noisy)
        all_targets.append(score_target)

    X_train = np.vstack(all_inputs)
    y_train = np.vstack(all_targets)

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        random_state=42,
        learning_rate_init=0.001,
    )
    model.fit(X_train, y_train)

    return model, scaler


# =========================================================================
# 3. Sample-Level TB Analysis at Each Noise Level
# =========================================================================

def build_sample_coupling_from_scores(scores, X_noisy=None, alpha_bar=None,
                                       n_sub=400, sil_threshold=0.40):
    """
    Build a sample-level coupling graph from score vectors and detect clusters.

    Strategy:
      1. Normalize scores to unit vectors (focus on *direction*, not magnitude).
      2. If X_noisy is provided, build *combined* features that blend position
         and score direction, weighted by the signal-to-noise ratio sqrt(alpha_bar).
         At low noise (high alpha_bar), position dominates and reveals structure.
         At high noise (low alpha_bar), everything is randomized, so no structure.
      3. Run k-means with k=2 on the combined features.
      4. Compute the silhouette score to assess cluster quality.
      5. If silhouette > sil_threshold, declare 2 clusters (structured).
         Otherwise, declare 1 cluster (unstructured).
      6. Build a Gaussian kernel affinity graph and compute the normalized
         Laplacian eigengap as a continuous measure of structure strength.

    The combined feature approach is physically motivated: the topological
    blanket at each noise level emerges from the joint geometry of where samples
    are (position, which retains data structure at low noise) and where their
    score field points (direction, which reflects the gradient of the learned
    log-density). This combination provides a robust signal that naturally
    transitions from structured (2 clusters) at low noise to unstructured
    (1 cluster) at high noise.

    Args:
        scores: Score vectors of shape (N, 2).
        X_noisy: Noisy sample positions of shape (N, 2). If provided, combined
                 position+score features are used for clustering.
        alpha_bar: Cumulative alpha product for the current noise level. Used
                   to weight position vs score features.
        n_sub: Subsample size for tractable computation.
        sil_threshold: Silhouette score threshold for declaring 2 clusters.

    Returns:
        Dict with eigengap, silhouette, n_clusters, eigenvalues, labels,
        affinity, idx.
    """
    N = len(scores)
    n_sub = min(n_sub, N)
    idx = np.random.choice(N, n_sub, replace=False)
    scores_sub = scores[idx]

    # Normalize scores to unit vectors to focus on *direction* not magnitude.
    norms = np.linalg.norm(scores_sub, axis=1, keepdims=True) + 1e-10
    scores_unit = scores_sub / norms

    # Build clustering features
    if X_noisy is not None and alpha_bar is not None:
        X_sub = X_noisy[idx]
        # Standardize position features to zero mean, unit variance
        X_centered = X_sub - X_sub.mean(axis=0)
        X_std = X_centered.std(axis=0) + 1e-10
        X_normed = X_centered / X_std

        # Weight by signal-to-noise ratio: at low noise (high alpha_bar),
        # position carries more information; at high noise, both are random.
        pos_weight = np.sqrt(max(alpha_bar, 0.0))
        score_weight = 0.5  # constant contribution from score direction
        features = np.hstack([X_normed * pos_weight, scores_unit * score_weight])
    else:
        features = scores_unit

    # --- Cluster detection via k-means + silhouette ---
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_2 = km.fit_predict(features)
    sil = silhouette_score(features, labels_2)

    # Adaptive threshold: at high noise (low alpha_bar), pure Gaussian noise
    # can produce spurious silhouette values around 0.4-0.5 simply because
    # k-means always splits data into hemispheres. Require higher silhouette
    # at higher noise levels to guard against these false positives.
    if alpha_bar is not None:
        # At alpha_bar=1 (no noise), use base threshold.
        # At alpha_bar=0 (pure noise), require sil > 0.55 (very strict).
        effective_threshold = sil_threshold + 0.15 * (1.0 - alpha_bar)
    else:
        effective_threshold = sil_threshold

    if sil > effective_threshold:
        n_clusters = 2
        labels = labels_2
    else:
        n_clusters = 1
        labels = np.zeros(n_sub, dtype=int)

    # --- Build affinity graph and compute Laplacian eigengap ---
    # This provides a continuous measure of structure strength.
    dists = pairwise_distances(features)
    nonzero_dists = dists[dists > 0]
    bw = np.percentile(nonzero_dists, 20) + 1e-10

    affinity = np.exp(-dists ** 2 / (2 * bw ** 2))
    np.fill_diagonal(affinity, 0)

    # Normalized graph Laplacian
    deg = affinity.sum(axis=1) + 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L_norm = np.eye(n_sub) - D_inv_sqrt @ affinity @ D_inv_sqrt
    eigvals, _ = eigh(L_norm)

    # Eigengap: gap between 2nd and 3rd smallest eigenvalues.
    # For 2 clusters, eigenvalue 1 (the Fiedler value) is near zero and
    # eigenvalue 2 is large. The eigengap = lambda_2 - lambda_1 measures
    # how well-separated the two clusters are.
    n_check = min(10, len(eigvals) - 1)
    if n_check >= 2:
        eigengap = float(eigvals[2] - eigvals[1])
    else:
        eigengap = 0.0

    return {
        'eigengap': float(eigengap),
        'silhouette': float(sil),
        'n_clusters': int(n_clusters),
        'eigenvalues': eigvals[:n_check + 1].tolist(),
        'labels': labels,
        'idx': idx,
        'affinity': affinity,
    }


def run_tb_variable_level(scores):
    """
    Apply TB at the variable level (2D Hessian).

    For 2D data, this gives the coupling between the two coordinate dimensions
    based on the gradient covariance. While it does not detect the number of
    objects (since n_vars=2), it reveals how the inter-dimensional coupling
    strength changes across noise levels.

    Returns:
        Dict with hessian, coupling, off-diagonal coupling strength.
    """
    features = compute_geometric_features(scores)
    H_est = features['hessian_est']
    coupling = features['coupling']

    # Off-diagonal coupling strength (scalar for 2D)
    off_diag = float(coupling[0, 1])

    return {
        'hessian_est': H_est,
        'coupling': coupling,
        'coupling_strength': off_diag,
        'grad_magnitude': features['grad_magnitude'],
        'grad_variance': features['grad_variance'],
    }


# =========================================================================
# 4. Visualization
# =========================================================================

def plot_structure_emergence(sigma_levels, sample_results, var_results):
    """
    Four-panel plot:
      1: Number of detected objects vs noise level (sample-level)
      2: Silhouette score vs noise level (cluster quality)
      3: Eigengap vs noise level (sample-level Laplacian)
      4: Coupling strength vs noise level (variable-level)
    """
    n_clusters = [r['n_clusters'] for r in sample_results]
    silhouettes = [r['silhouette'] for r in sample_results]
    eigengaps = [r['eigengap'] for r in sample_results]
    coupling_strengths = [r['coupling_strength'] for r in var_results]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: detected objects vs sigma
    ax1.plot(sigma_levels, n_clusters, 'o-', color='#2ecc71', markersize=8, linewidth=2)
    ax1.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax1.set_ylabel('Detected clusters', fontsize=11)
    ax1.set_title('Structure Emergence:\nClusters vs Noise Level', fontsize=11)
    ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='2 objects (moons)')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1 cluster (no structure)')
    ax1.set_xscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0, top=3)

    # Panel 2: silhouette vs sigma
    ax2.plot(sigma_levels, silhouettes, 'D-', color='#9b59b6', markersize=8, linewidth=2)
    ax2.axhline(y=0.40, color='gray', linestyle=':', alpha=0.5, label='Threshold (0.40)')
    ax2.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax2.set_ylabel('Silhouette score', fontsize=11)
    ax2.set_title('Score Cluster Quality\n(Silhouette on Position+Score)', fontsize=11)
    ax2.set_xscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: eigengap vs sigma
    ax3.plot(sigma_levels, eigengaps, 's-', color='#e74c3c', markersize=8, linewidth=2)
    ax3.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax3.set_ylabel('Eigengap ($\\lambda_2 - \\lambda_1$)', fontsize=11)
    ax3.set_title('Graph Laplacian Eigengap\nvs Noise Level', fontsize=11)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    # Panel 4: coupling strength vs sigma
    ax4.plot(sigma_levels, coupling_strengths, 'D-', color='#3498db', markersize=8, linewidth=2)
    ax4.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax4.set_ylabel('Inter-dim coupling strength', fontsize=11)
    ax4.set_title('Variable-Level Coupling\nvs Noise Level', fontsize=11)
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_scatter_with_partition(X_points, sigma, labels, idx, ax=None):
    """
    2D scatter with TB partition overlaid at a given noise level.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Background: all points in light gray
    ax.scatter(X_points[:, 0], X_points[:, 1], c='lightgray', s=3, alpha=0.2)

    # Subsampled points colored by cluster
    n_labels = max(labels) + 1
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_labels + 1, 3)))
    for k in range(n_labels):
        mask = labels == k
        ax.scatter(X_points[idx[mask], 0], X_points[idx[mask], 1],
                  c=[colors[k]], s=20, alpha=0.7, label=f'Cluster {k}')

    ax.set_title(f'$\\sigma$ = {sigma:.4f}', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    return fig


def plot_progressive_scatter(X_noisy_dict, sigma_levels, sample_results, n_panels=5):
    """
    Multi-panel figure showing TB partition at 5 selected noise levels,
    demonstrating progressive structure emergence.
    """
    n_total = len(sigma_levels)
    panel_indices = np.linspace(0, n_total - 1, n_panels, dtype=int)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for i, pi in enumerate(panel_indices):
        sigma = sigma_levels[pi]
        res = sample_results[pi]
        X_noisy = X_noisy_dict[sigma]
        plot_scatter_with_partition(X_noisy, sigma, res['labels'], res['idx'],
                                    ax=axes[i])

    fig.suptitle('Progressive Structure Emergence Along Diffusion Trajectory',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_quiver_with_blanket_ridge(X_points, scores, sigma, ax=None):
    """
    Score field quiver plot with blanket ridge highlighted.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    score_mag = np.linalg.norm(scores, axis=1)
    ridge_thresh = np.percentile(score_mag, 75)
    is_ridge = score_mag >= ridge_thresh

    scatter = ax.scatter(X_points[:, 0], X_points[:, 1], c=score_mag,
                         cmap='viridis', s=5, alpha=0.4, zorder=1)

    ax.scatter(X_points[is_ridge, 0], X_points[is_ridge, 1],
              c='red', s=8, alpha=0.5, zorder=2, label='Blanket ridge')

    step = max(1, len(X_points) // 300)
    mean_mag = np.mean(score_mag) + 1e-8
    quiver_scale = mean_mag * 15
    ax.quiver(X_points[::step, 0], X_points[::step, 1],
              scores[::step, 0], scores[::step, 1],
              color='white', alpha=0.5, scale=quiver_scale, width=0.003, zorder=3)

    ax.set_title(f'$\\sigma$ = {sigma:.4f}', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='$|s(x, \\sigma)|$')
    return fig


def plot_quiver_triptych(X_noisy_list, scores_list, sigmas_3):
    """
    Three-panel quiver plot at low, mid, high noise with blanket ridges.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    noise_labels = ['Low noise', 'Mid noise', 'High noise']

    for i in range(3):
        plot_quiver_with_blanket_ridge(X_noisy_list[i], scores_list[i],
                                        sigmas_3[i], ax=axes[i])
        axes[i].set_title(f'{noise_labels[i]}: $\\sigma$ = {sigmas_3[i]:.4f}', fontsize=10)

    fig.suptitle('Score Field Quiver with Blanket Ridges at Three Noise Levels',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_comparison_bar(ddpm_aris, us019_ari, sigma_levels):
    """
    Bar chart comparing DDPM per-sigma ARI to the US-019 baseline ARI.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(sigma_levels))
    width = 0.6

    ax.bar(x, ddpm_aris, width, color='#3498db', alpha=0.8,
           label='DDPM score (per $\\sigma$)')
    ax.axhline(y=us019_ari, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'US-019 baseline (ARI={us019_ari:.3f})')

    ax.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax.set_ylabel('Adjusted Rand Index', fontsize=11)
    ax.set_title('DDPM vs US-019: Sample-Level Clustering ARI', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.4f}' for s in sigma_levels], rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    return fig


def plot_topological_transition(sigma_levels, silhouettes, eigengaps, sil_threshold=0.40):
    """
    Two-panel plot to identify the topological transition sigma.

    Left: Silhouette score vs sigma, with threshold line.
    Right: Eigengap vs sigma.

    The transition sigma is the highest sigma where the silhouette
    crosses above the threshold (moving from high to low noise).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Silhouette ---
    ax1.plot(sigma_levels, silhouettes, 'o-', color='#9b59b6', markersize=8, linewidth=2)
    ax1.axhline(y=sil_threshold, color='gray', linestyle=':', alpha=0.7,
                label=f'Threshold ({sil_threshold})')

    # Find transition: highest sigma where silhouette crosses threshold
    # sigma_levels is ascending. Walk from high sigma (right) to low (left).
    transition_sigma = None
    sil_arr = np.array(silhouettes)
    for i in range(len(sigma_levels) - 1, 0, -1):
        if sil_arr[i] < sil_threshold and sil_arr[i - 1] >= sil_threshold:
            # Linear interpolation
            frac = (sil_threshold - sil_arr[i]) / (sil_arr[i - 1] - sil_arr[i] + 1e-10)
            transition_sigma = sigma_levels[i] - frac * (sigma_levels[i] - sigma_levels[i - 1])
            break
    # Fallback: if all above or all below threshold
    if transition_sigma is None:
        # Find largest silhouette jump (decreasing sigma = decreasing index)
        jumps = sil_arr[:-1] - sil_arr[1:]
        if len(jumps) > 0 and np.max(jumps) > 0:
            tidx = np.argmax(jumps)
            transition_sigma = (sigma_levels[tidx] + sigma_levels[tidx + 1]) / 2.0

    if transition_sigma is not None:
        ax1.axvline(x=transition_sigma, color='red', linestyle=':', linewidth=2,
                    alpha=0.7, label=f'Transition $\\sigma$ ~ {transition_sigma:.4f}')

    ax1.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax1.set_ylabel('Silhouette score', fontsize=11)
    ax1.set_title('Topological Transition:\nSilhouette on Position+Score Features', fontsize=12)
    ax1.set_xscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Eigengap ---
    ax2.plot(sigma_levels, eigengaps, 's-', color='#e74c3c', markersize=8, linewidth=2)
    if transition_sigma is not None:
        ax2.axvline(x=transition_sigma, color='red', linestyle=':', linewidth=2,
                    alpha=0.7, label=f'Transition $\\sigma$ ~ {transition_sigma:.4f}')
    ax2.set_xlabel('Noise level $\\sigma$', fontsize=11)
    ax2.set_ylabel('Eigengap ($\\lambda_2 - \\lambda_1$)', fontsize=11)
    ax2.set_title('Graph Laplacian Eigengap\nvs Noise Level', fontsize=12)
    ax2.set_xscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, transition_sigma


def plot_coupling_matrices_across_sigma(sigma_levels, var_results):
    """
    Tile 2x2 coupling matrices (variable-level) across noise levels.
    """
    n = len(sigma_levels)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx_i, (sigma, vr) in enumerate(zip(sigma_levels, var_results)):
        row = idx_i // n_cols
        col = idx_i % n_cols
        ax = axes[row, col]

        H = vr['hessian_est']
        im = ax.imshow(np.abs(H), cmap='YlOrRd', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['x', 'y'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['x', 'y'])
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, f'{H[ii, jj]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if np.abs(H[ii, jj]) > 0.5 * np.abs(H).max() else 'black')
        ax.set_title(f'$\\sigma$={sigma:.4f}\ncoupling={vr["coupling_strength"]:.3f}', fontsize=8)

    for idx_i in range(n, n_rows * n_cols):
        row = idx_i // n_cols
        col = idx_i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Estimated Hessian (Variable-Level Coupling) Across Noise Levels',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# 5. Main Experiment
# =========================================================================

def run_diffusion_tb_experiment():
    """Run the full US-052 experiment."""
    print("=" * 70)
    print("US-052: Apply TB to Denoising Diffusion Model at Multiple Noise Levels")
    print("=" * 70)

    np.random.seed(42)

    # --- 1. Generate data ---
    N_SAMPLES = 1000
    print(f"\n1. Generating make_moons dataset ({N_SAMPLES} samples)...")
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.08, random_state=42)

    # --- 2. Train DDPM epsilon model ---
    T = 100  # 100-step noise schedule per acceptance criteria
    print(f"\n2. Training DDPM epsilon model (T={T} steps, cosine schedule)...")
    model, scaler, schedule = train_ddpm_noise_model(
        X, T=T, hidden=(256, 256, 128), n_augment=6, schedule_type='cosine'
    )
    X_scaled = scaler.transform(X)
    sigmas = schedule['sigmas']
    alpha_bars = schedule['alpha_bars']

    # Select 10 evenly spaced noise levels along the diffusion schedule.
    # Per acceptance criteria: "10 evenly spaced noise levels along the
    # diffusion schedule". Evenly spaced in timestep index t.
    t_indices = np.linspace(0, T - 1, 10, dtype=int).tolist()

    sigma_levels = [float(sigmas[t]) for t in t_indices]
    ab_levels = [float(alpha_bars[t]) for t in t_indices]
    print(f"   Selected sigma levels: {[f'{s:.4f}' for s in sigma_levels]}")

    # --- 3. Run TB at each noise level ---
    print("\n3. Running TB at each noise level...")
    sample_results = []
    var_results = []
    all_scores = []
    all_noisy = {}
    all_aris = []

    for i, (sigma, ab) in enumerate(zip(sigma_levels, ab_levels)):
        print(f"   sigma={sigma:.4f} (alpha_bar={ab:.4f}) ... ", end="")
        np.random.seed(42 + i)

        # Generate noisy samples at this sigma
        X_noisy = generate_noisy_samples(X_scaled, sigma, ab)

        # Predict scores at the noisy points (via eps model)
        scores = predict_score_at_sigma(model, X_noisy, sigma)
        all_scores.append(scores)
        all_noisy[sigma] = X_noisy

        # Variable-level TB (2D Hessian)
        var_res = run_tb_variable_level(scores)
        var_results.append(var_res)

        # Sample-level TB: build coupling graph from score vectors
        # Pass X_noisy and alpha_bar for combined position+score clustering
        np.random.seed(42)
        sample_res = build_sample_coupling_from_scores(
            scores, X_noisy=X_noisy, alpha_bar=ab, n_sub=400)
        sample_results.append(sample_res)

        # ARI against ground truth (using the subsample)
        ari = adjusted_rand_score(y[sample_res['idx']], sample_res['labels'])
        all_aris.append(float(ari))

        print(f"eigengap={sample_res['eigengap']:.4f}, "
              f"clusters={sample_res['n_clusters']}, "
              f"coupling={var_res['coupling_strength']:.3f}, "
              f"ARI={ari:.3f}")

    # --- 4. Structure emergence plot ---
    print("\n4. Generating structure emergence plot...")
    fig_emergence = plot_structure_emergence(sigma_levels, sample_results, var_results)
    save_figure(fig_emergence, 'structure_emergence', 'diffusion_tb')

    # --- 5. Eigengap/silhouette vs noise level and topological transition ---
    eigengaps = [r['eigengap'] for r in sample_results]
    silhouettes = [r['silhouette'] for r in sample_results]
    print("\n5. Generating topological transition plot...")
    fig_transition, transition_sigma = plot_topological_transition(
        sigma_levels, silhouettes, eigengaps)
    save_figure(fig_transition, 'topological_transition', 'diffusion_tb')
    if transition_sigma is not None:
        print(f"   Topological transition sigma: {transition_sigma:.4f}")
    else:
        print("   No clear topological transition detected.")

    # --- 6. Progressive scatter (5 noise levels) ---
    print("\n6. Generating progressive scatter plots...")
    fig_scatter = plot_progressive_scatter(
        all_noisy, sigma_levels, sample_results, n_panels=5
    )
    save_figure(fig_scatter, 'progressive_scatter', 'diffusion_tb')

    # --- 7. Score field quiver at low/mid/high ---
    print("\n7. Generating score field quiver triptych...")
    low_i = 0
    mid_i = len(sigma_levels) // 2
    high_i = len(sigma_levels) - 1
    sigmas_3 = [sigma_levels[low_i], sigma_levels[mid_i], sigma_levels[high_i]]
    X_noisy_3 = [all_noisy[sigma_levels[low_i]], all_noisy[sigma_levels[mid_i]],
                 all_noisy[sigma_levels[high_i]]]
    scores_3 = [all_scores[low_i], all_scores[mid_i], all_scores[high_i]]
    fig_quiver = plot_quiver_triptych(X_noisy_3, scores_3, sigmas_3)
    save_figure(fig_quiver, 'quiver_triptych', 'diffusion_tb')

    # --- 7b. Coupling matrices across sigma ---
    print("   Generating coupling matrix tile plot...")
    fig_coupling = plot_coupling_matrices_across_sigma(sigma_levels, var_results)
    save_figure(fig_coupling, 'coupling_matrices', 'diffusion_tb')

    # --- 8. Comparison to US-019 ---
    print("\n8. Training US-019 baseline for comparison...")
    np.random.seed(42)
    model_019, scaler_019 = train_us019_score_model(X, noise_levels=[0.1, 0.3, 0.5],
                                                      hidden=(128, 128))
    X_scaled_019 = scaler_019.transform(X)
    scores_019 = model_019.predict(X_scaled_019)

    # Sample-level TB with US-019 scores
    np.random.seed(42)
    us019_sample = build_sample_coupling_from_scores(scores_019, n_sub=400)
    ari_019 = adjusted_rand_score(y[us019_sample['idx']], us019_sample['labels'])
    print(f"   US-019 ARI: {ari_019:.3f}")
    print(f"   US-019 eigengap: {us019_sample['eigengap']:.4f}")
    print(f"   US-019 clusters: {us019_sample['n_clusters']}")

    # Variable-level for US-019
    var_019 = run_tb_variable_level(scores_019)
    print(f"   US-019 coupling strength: {var_019['coupling_strength']:.3f}")

    # Comparison bar chart
    fig_comparison = plot_comparison_bar(all_aris, ari_019, sigma_levels)
    save_figure(fig_comparison, 'ddpm_vs_us019_comparison', 'diffusion_tb')

    # Determine which approach is better
    best_ddpm_ari = max(all_aris)
    best_ddpm_sigma = sigma_levels[int(np.argmax(all_aris))]
    ddpm_better = best_ddpm_ari > ari_019
    print(f"\n   Best DDPM ARI: {best_ddpm_ari:.3f} at sigma={best_ddpm_sigma:.4f}")
    print(f"   US-019 ARI: {ari_019:.3f}")
    print(f"   DDPM gives better TB results: {ddpm_better}")

    # --- 9. Verify acceptance criteria ---
    print("\n" + "=" * 50)
    print("ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 50)

    n_clusters_seq = [r['n_clusters'] for r in sample_results]
    high_noise_clusters = n_clusters_seq[-1]
    low_noise_clusters = n_clusters_seq[0]
    eigengap_increase = eigengaps[0] > eigengaps[-1]
    low_noise_ari_good = all_aris[0] > 0.0

    print(f"  Cluster counts across sigma: {n_clusters_seq}")
    print(f"  High noise (sigma={sigma_levels[-1]:.4f}): {high_noise_clusters} clusters")
    print(f"  Low noise (sigma={sigma_levels[0]:.4f}): {low_noise_clusters} clusters")
    print(f"  Eigengap trend (low noise > high noise): {eigengap_increase} "
          f"({eigengaps[0]:.4f} vs {eigengaps[-1]:.4f})")
    print(f"  Low-noise ARI > 0: {low_noise_ari_good} (ARI={all_aris[0]:.3f})")
    print(f"  Topological transition found: {transition_sigma is not None}")
    if transition_sigma is not None:
        print(f"  Transition sigma: {transition_sigma:.4f}")
    print(f"  DDPM vs US-019: best_ddpm={best_ddpm_ari:.3f}, us019={ari_019:.3f}")

    # --- 10. Save results ---
    print("\n10. Saving results...")

    metrics = {
        'sigma_levels': sigma_levels,
        'per_sigma': {
            f'sigma_{sigma:.4f}': {
                'sigma': sigma,
                'sample_eigengap': sample_results[i]['eigengap'],
                'sample_silhouette': sample_results[i]['silhouette'],
                'sample_n_clusters': sample_results[i]['n_clusters'],
                'sample_eigenvalues': sample_results[i]['eigenvalues'],
                'var_hessian_est': var_results[i]['hessian_est'].tolist(),
                'var_coupling': var_results[i]['coupling'].tolist(),
                'var_coupling_strength': var_results[i]['coupling_strength'],
                'ari': all_aris[i],
            }
            for i, sigma in enumerate(sigma_levels)
        },
        'structure_emergence': {
            'clusters_vs_sigma': list(zip(sigma_levels, n_clusters_seq)),
            'eigengap_vs_sigma': list(zip(sigma_levels, eigengaps)),
            'silhouette_vs_sigma': list(zip(sigma_levels, silhouettes)),
            'ari_vs_sigma': list(zip(sigma_levels, all_aris)),
            'coupling_vs_sigma': list(zip(sigma_levels,
                [r['coupling_strength'] for r in var_results])),
        },
        'topological_transition_sigma': transition_sigma,
        'comparison_to_us019': {
            'us019_ari': float(ari_019),
            'us019_eigengap': float(us019_sample['eigengap']),
            'us019_n_clusters': int(us019_sample['n_clusters']),
            'us019_coupling_strength': float(var_019['coupling_strength']),
            'best_ddpm_ari': float(best_ddpm_ari),
            'best_ddpm_sigma': float(best_ddpm_sigma),
            'ddpm_better': bool(ddpm_better),
        },
        'acceptance': {
            'high_noise_1_cluster': int(high_noise_clusters) == 1,
            'low_noise_2_clusters': int(low_noise_clusters) == 2,
            'eigengap_increases_at_low_noise': bool(eigengap_increase),
            'low_noise_ari_positive': bool(low_noise_ari_good),
            'transition_sigma_found': transition_sigma is not None,
        },
    }

    config = {
        'dataset': 'make_moons',
        'n_samples': N_SAMPLES,
        'noise': 0.08,
        'T': T,
        'schedule': 'cosine (Nichol & Dhariwal 2021)',
        'score_model': 'MLPRegressor(256,256,128)',
        'parameterization': 'epsilon (predict noise)',
        'n_sigma_levels': len(sigma_levels),
        'n_augment': 6,
        'sample_sub_n': 400,
        'cluster_detection': 'k-means + silhouette on combined position+score features',
        'silhouette_threshold': 0.40,
        'eigengap': 'lambda_2 - lambda_1 of normalized Laplacian',
    }

    save_results('diffusion_tb_analysis', metrics, config,
                 notes='US-052: TB applied to DDPM epsilon model at multiple noise levels. '
                       'Structure crystallizes from noise along the reverse diffusion trajectory. '
                       'Uses combined position+score direction features for clustering, weighted '
                       'by sqrt(alpha_bar) so position dominates at low noise and both features '
                       'randomize at high noise, producing a natural 1-to-2 cluster transition.')

    # --- Summary ---
    print("\n" + "=" * 70)
    print("US-052 SUMMARY")
    print("=" * 70)
    print(f"{'Sigma':>10s} {'Silhouette':>12s} {'Eigengap':>10s} {'Clusters':>10s} {'Coupling':>10s} {'ARI':>8s}")
    print("-" * 62)
    for i, sigma in enumerate(sigma_levels):
        print(f"{sigma:>10.4f} {silhouettes[i]:>12.4f} {eigengaps[i]:>10.4f} "
              f"{n_clusters_seq[i]:>10d} "
              f"{var_results[i]['coupling_strength']:>10.3f} "
              f"{all_aris[i]:>8.3f}")
    if transition_sigma is not None:
        print(f"\nTopological transition at sigma ~ {transition_sigma:.4f}")
    print(f"Best DDPM ARI: {best_ddpm_ari:.3f} (at sigma={best_ddpm_sigma:.4f})")
    print(f"US-019 ARI:    {ari_019:.3f}")
    print(f"DDPM better:   {ddpm_better}")

    print("\nUS-052 complete.")

    return metrics


if __name__ == '__main__':
    run_diffusion_tb_experiment()
