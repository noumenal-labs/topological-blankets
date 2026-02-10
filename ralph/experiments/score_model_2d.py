"""
US-019: 2D Score Model Demonstration
=====================================

Train a small MLP to estimate the score function on 2D synthetic datasets.
Apply TB to score samples. Show blanket detection overlaid on 2D scatter.

Datasets: moons, circles, blobs, aniso.
Uses sklearn MLPRegressor (no PyTorch required).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topological_blankets.core import topological_blankets as tb_pipeline
from topological_blankets.features import compute_geometric_features
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure


# =========================================================================
# Dataset generators
# =========================================================================

def make_dataset(name, n_samples=2000, noise=0.08):
    """Generate 2D dataset by name."""
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif name == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.6, random_state=42)
    elif name == 'aniso':
        np.random.seed(42)
        X1 = np.random.randn(n_samples // 2, 2) @ np.array([[0.5, -0.8], [0.8, 0.5]]) + [2, 0]
        X2 = np.random.randn(n_samples // 2, 2) @ np.array([[0.5, 0.8], [-0.8, 0.5]]) + [-2, 0]
        X = np.vstack([X1, X2])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


# =========================================================================
# Score estimation via denoising score matching
# =========================================================================

def train_score_model(X, noise_levels=None, hidden=(128, 128)):
    """
    Train a score model via denoising score matching.

    For each sample x, add noise eps ~ N(0, sigma^2 I), then train network
    to predict -eps/sigma^2 (which approximates the score at x + eps).
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.3, 0.5]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate training data: noisy inputs -> score targets
    all_inputs = []
    all_targets = []

    for sigma in noise_levels:
        eps = np.random.randn(*X_scaled.shape) * sigma
        X_noisy = X_scaled + eps
        # Score target: -eps / sigma^2 (denoising score matching)
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


def collect_score_samples(model, scaler, X, n_score_samples=3000):
    """Collect score function evaluations on data points."""
    X_scaled = scaler.transform(X[:n_score_samples])
    scores = model.predict(X_scaled)
    return scores, X_scaled


# =========================================================================
# TB on score samples
# =========================================================================

def apply_tb_to_scores(scores, n_objects):
    """
    Apply TB pipeline to score samples.

    Score = -grad E, so scores are negated energy gradients.
    TB works on gradients, and the sign doesn't affect the Hessian estimate
    (covariance is sign-invariant), so pass scores directly.
    """
    result = tb_pipeline(scores, n_objects=n_objects, method='gradient')
    return result


# =========================================================================
# Visualization
# =========================================================================

def plot_score_field_and_blankets(X, scores, tb_result, dataset_name, y_true=None):
    """2D scatter with TB assignment and score field quiver overlay."""
    assignment = tb_result['assignment']
    is_blanket = tb_result['is_blanket']

    # Since TB operates on variables (dimensions), not samples,
    # for 2D data we get partition of the 2 dimensions.
    # This tells us which dimensions are "blanket" vs "object" dimensions.
    # For sample-level visualization, color samples by their dominant
    # gradient dimension assignment.

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original data with true labels
    ax = axes[0]
    if y_true is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='Set1',
                            s=5, alpha=0.5)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='gray', s=5, alpha=0.5)
    ax.set_title(f'{dataset_name}: Ground Truth')
    ax.set_aspect('equal')

    # Panel 2: Score field as quiver
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=2, alpha=0.3)
    # Subsample for quiver
    step = max(1, len(X) // 400)
    ax.quiver(X[::step, 0], X[::step, 1],
              scores[::step, 0], scores[::step, 1],
              color='#2ecc71', alpha=0.6, scale=50)
    ax.set_title(f'{dataset_name}: Score Field')
    ax.set_aspect('equal')

    # Panel 3: TB coupling analysis
    ax = axes[2]
    features = tb_result['features']
    coupling = features['coupling']
    H_est = features['hessian_est']

    # For 2D, the Hessian is 2x2. Show it as annotated matrix.
    im = ax.imshow(np.abs(H_est), cmap='YlOrRd', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['x', 'y'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['x', 'y'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{H_est[i,j]:.2f}', ha='center', va='center', fontsize=12)
    ax.set_title(f'Estimated Hessian (blanket dims: {np.where(is_blanket)[0]})')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def plot_sample_level_tb(X, scores, n_objects, dataset_name, y_true=None):
    """
    Apply TB at the sample level by treating each sample as a variable.

    For 2D data with many samples, subsample to keep dimension manageable,
    then show TB partition overlaid on scatter.
    """
    # Subsample to ~200 points for tractable Hessian
    n_sub = min(200, len(X))
    idx = np.random.choice(len(X), n_sub, replace=False)
    X_sub = X[idx]
    scores_sub = scores[idx]

    # Transpose: treat samples as "variables" and dimensions as "observations"
    # This gives a (2, n_sub) gradient matrix; TB needs (N, n_vars)
    # Instead, use the score vectors directly: (n_sub, 2)
    # The Hessian estimate will be 2x2, which is too small.

    # Better approach: use pairwise score differences to build sample-level coupling.
    # Compute coupling between samples based on score similarity.
    from sklearn.metrics import pairwise_distances
    score_dists = pairwise_distances(scores_sub)
    affinity = np.exp(-score_dists / (2 * np.median(score_dists) ** 2 + 1e-10))

    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=n_objects, affinity='precomputed',
                            random_state=42)
    sample_labels = sc.fit_predict(affinity + 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_objects + 1))

    # Full data in gray
    ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=3, alpha=0.2)

    # Subsampled points colored by TB cluster
    for k in range(n_objects):
        mask = sample_labels == k
        ax.scatter(X_sub[mask, 0], X_sub[mask, 1], c=[colors[k]],
                  s=30, alpha=0.8, label=f'Object {k}')

    ax.set_title(f'{dataset_name}: Sample-level Spectral Clustering on Scores')
    ax.legend()
    ax.set_aspect('equal')

    return fig, sample_labels, X_sub, y_true[idx] if y_true is not None else None


# =========================================================================
# Metrics
# =========================================================================

def compute_sample_ari(pred_labels, true_labels):
    """Compute ARI between predicted and true sample labels."""
    if true_labels is None:
        return float('nan')
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(true_labels, pred_labels))


# =========================================================================
# Main experiment
# =========================================================================

def run_score_model_experiment():
    """Run the full 2D score model demonstration."""
    print("=" * 70)
    print("US-019: 2D Score Model Demonstration")
    print("=" * 70)

    datasets = {
        'moons': {'n_objects': 2, 'noise': 0.08},
        'circles': {'n_objects': 2, 'noise': 0.06},
        'blobs': {'n_objects': 3, 'noise': 0.08},
        'aniso': {'n_objects': 2, 'noise': 0.08},
    }

    all_results = {}

    for name, cfg in datasets.items():
        print(f"\n--- Dataset: {name} ---")
        n_objects = cfg['n_objects']

        X, y = make_dataset(name, n_samples=3000, noise=cfg['noise'])

        # Train score model
        model, scaler = train_score_model(X)
        scores, X_scaled = collect_score_samples(model, scaler, X)

        # Variable-level TB (2D)
        var_result = apply_tb_to_scores(scores, n_objects=n_objects)
        print(f"  Variable-level TB: assignment={var_result['assignment']}, "
              f"blanket={np.where(var_result['is_blanket'])[0]}")

        # Score field + Hessian visualization
        fig_var = plot_score_field_and_blankets(X_scaled, scores, var_result, name, y)
        save_figure(fig_var, f'score_field_{name}', 'score_model_2d')

        # Sample-level clustering on scores
        np.random.seed(42)
        fig_sample, pred_labels, X_sub, y_sub = plot_sample_level_tb(
            X_scaled, scores, n_objects, name, y)
        save_figure(fig_sample, f'sample_clusters_{name}', 'score_model_2d')

        # ARI
        ari = compute_sample_ari(pred_labels, y_sub)
        print(f"  Sample-level ARI: {ari:.3f}")

        all_results[name] = {
            'n_objects': n_objects,
            'variable_assignment': var_result['assignment'].tolist(),
            'variable_blanket': var_result['is_blanket'].tolist(),
            'hessian_est': var_result['features']['hessian_est'].tolist(),
            'sample_ari': ari,
        }

    config = {
        'datasets': list(datasets.keys()),
        'n_samples': 3000,
        'score_model': 'MLPRegressor(128,128)',
        'noise_levels': [0.1, 0.3, 0.5],
    }

    save_results('score_model_2d', all_results, config,
                 notes='US-019: 2D score model demo. TB applied to learned score fields.')

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"{'Dataset':<12s} {'ARI':>6s}")
    print("-" * 20)
    for name, r in all_results.items():
        print(f"{name:<12s} {r['sample_ari']:>6.3f}")

    print("\nUS-019 complete.")
    return all_results


if __name__ == '__main__':
    run_score_model_experiment()
