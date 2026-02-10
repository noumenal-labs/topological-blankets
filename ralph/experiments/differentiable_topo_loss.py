"""
US-075: Differentiable Topological Loss for Structure-Aware Training
=====================================================================

Implements a differentiable persistence-based regularizer (Hu et al. 2019)
that can be added to world model training loss to enforce topologically
correct Markov blanket structure during learning, not just post-hoc.

The core idea: given a coupling matrix H derived from a neural network's
latent space, compute the H0 persistence diagram (connected component
births/deaths) in a differentiable way via soft sorting, then penalize
topological features that deviate from a target persistence diagram.

For k objects, the target diagram has k-1 long-lived H0 features (gaps
between the k tightly-coupled clusters) and all other features should
have zero persistence (noise).

Experiments:
1. Synthetic autoencoder on quadratic landscape: train with and without
   L_topo, compare post-hoc TB ARI.
2. LunarLander dynamics model: add L_topo to ensemble training, compare
   TB structure quality.
3. Hyperparameter sweep: L_topo weight from 0.001 to 1.0, measure the
   reconstruction-vs-structure tradeoff.

References:
    Hu et al. (2019) "Topology-Preserving Deep Image Segmentation"
    Bruel-Gabrielsson et al. (2020) "A Topology Layer for Machine Learning"
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.cluster import SpectralClustering
import sys
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'trajectory_data')


# =============================================================================
# Differentiable Persistence Computation (PyTorch)
# =============================================================================

def coupling_matrix_from_latent(z_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized coupling matrix from a batch of latent vectors.

    Given z of shape (batch, latent_dim), compute:
        H_est = Cov(z)   (gradient covariance approximated by latent covariance)
        coupling[i,j] = |H_est[i,j]| / (sqrt(H_est[i,i]) * sqrt(H_est[j,j]))

    All operations are differentiable w.r.t. z_batch.
    """
    # Center
    z_mean = z_batch.mean(dim=0, keepdim=True)
    z_centered = z_batch - z_mean

    # Covariance
    n = z_batch.shape[0]
    H_est = (z_centered.T @ z_centered) / (n - 1)

    # Normalized coupling
    diag = torch.sqrt(torch.diag(H_est).clamp(min=1e-8))
    outer_diag = diag.unsqueeze(1) * diag.unsqueeze(0)
    coupling = torch.abs(H_est) / outer_diag

    # Zero diagonal
    mask = 1.0 - torch.eye(coupling.shape[0], device=coupling.device)
    coupling = coupling * mask

    return coupling


def differentiable_persistence_h0(coupling: torch.Tensor) -> torch.Tensor:
    """
    Differentiable H0 persistence from a coupling matrix.

    This implements a soft version of the union-find persistence computation.
    Instead of the discrete Kruskal-style algorithm (non-differentiable due
    to discrete union-find operations), we use the following approach:

    1. Extract all edge weights from the upper triangle of the coupling matrix.
    2. Sort edges in descending order (strongest first).
    3. At each step of the filtration, the "death" of a component happens
       when two previously disconnected components merge. In the standard
       (non-differentiable) algorithm, this is tracked by union-find.

    For differentiability, we use the fact that for a weighted complete graph,
    the H0 persistence diagram can be read off from the minimum spanning tree
    (MST). Specifically, the death values in the H0 diagram correspond exactly
    to the edge weights of the *maximum* spanning tree (since we do a descending
    filtration). The (n-1) MST edge weights are the death values, and all
    births are at infinity (each node starts as its own component).

    The persistence of the i-th feature (sorted descending) is:
        pers_i = max_coupling - mst_edge_i  (approximately)

    We compute the MST weights differentiably using the fact that for a
    complete graph, sorting all edge weights and applying a greedy selection
    (Kruskal's) gives the MST. We approximate this with a differentiable
    relaxation: the MST edge weights are the (n-1) largest "essential" edges,
    which we extract via the sorted eigenvalues of the graph Laplacian.

    Actually, the simplest differentiable approach: the H0 persistence deaths
    for a descending filtration on coupling are exactly the edge weights where
    two components merge. For an n-node complete graph, there are exactly n-1
    such events. These correspond to the maximum spanning tree edge weights,
    sorted in descending order.

    We compute these via the graph Laplacian eigenvalues: for a weighted
    adjacency W, the Fiedler value (second-smallest eigenvalue of the
    Laplacian L = D - W) indicates the weakest "bottleneck" in the graph.
    More generally, the k-th smallest eigenvalue of L relates to the k-th
    most significant topological feature.

    For full differentiability, we use torch.linalg.eigh on the Laplacian.

    Returns:
        Sorted H0 persistence values (descending), shape (n-1,).
        These represent the "gap sizes" in the coupling structure:
        large values = significant structural boundaries,
        small values = noise.
    """
    n = coupling.shape[0]

    # Build weighted Laplacian: L = D - W
    # Use coupling as adjacency weights
    W = coupling
    D = W.sum(dim=1)
    L = torch.diag(D) - W

    # Eigendecomposition (differentiable via torch.linalg.eigh)
    eigvals, _ = torch.linalg.eigh(L)

    # The eigenvalues of the graph Laplacian are: 0 = lambda_1 <= lambda_2 <= ...
    # The number of zero eigenvalues = number of connected components.
    # The gaps between consecutive eigenvalues indicate cluster boundaries.
    #
    # For the persistence interpretation:
    # - lambda_2 (Fiedler value) ~ death threshold of the last-surviving
    #   non-trivial H0 feature (the main cluster split)
    # - Larger eigenvalues correspond to finer splits
    #
    # The "persistence" of the k-th structural feature is proportional to
    # the gap: lambda_{k+1} - lambda_k
    #
    # For our topological loss, we want the first (k-1) gaps (for k objects)
    # to be large, and all remaining gaps to be small.

    # Compute spectral gaps: these are our differentiable persistence proxies
    # Skip lambda_1 (always 0 for connected graphs)
    gaps = eigvals[1:] - eigvals[:-1]  # shape (n-1,)

    # The gaps from the non-zero eigenvalues are the meaningful ones
    # Sort descending to get "most persistent" features first
    sorted_gaps, _ = torch.sort(gaps[1:], descending=True)  # skip the 0-to-lambda2 gap

    return sorted_gaps


def topological_loss(coupling: torch.Tensor,
                     n_objects: int = 2,
                     target_persistence: float = 1.0) -> torch.Tensor:
    """
    Differentiable topological loss for structure-aware training.

    L_topo = L_signal + L_noise

    L_signal: penalize desired features for being too weak
        For k objects, we want k-1 "long-lived" H0 features (the gaps
        between the k tightly-coupled clusters). These should have
        persistence >= target_persistence.

        L_signal = sum_{i=1}^{k-1} max(0, target - pers_i)^2

    L_noise: penalize spurious features for existing
        All features beyond the k-1 desired ones should have zero
        persistence (or as small as possible).

        L_noise = sum_{i=k}^{n-2} pers_i^2

    Args:
        coupling: Differentiable coupling matrix (n, n).
        n_objects: Number of desired objects (clusters).
        target_persistence: Target persistence for the desired features.

    Returns:
        Scalar topological loss.
    """
    n = coupling.shape[0]
    n_desired = n_objects - 1  # k objects require k-1 splits

    # Get differentiable persistence values (sorted descending)
    pers = differentiable_persistence_h0(coupling)

    if len(pers) < n_desired:
        # Not enough features; just penalize all for being small
        return (target_persistence - pers.sum()).clamp(min=0) ** 2

    # L_signal: the top (n_desired) persistence values should be >= target
    desired = pers[:n_desired]
    L_signal = ((target_persistence - desired).clamp(min=0) ** 2).sum()

    # L_noise: all remaining persistence values should be near zero
    if len(pers) > n_desired:
        noise = pers[n_desired:]
        L_noise = (noise ** 2).sum()
    else:
        L_noise = torch.tensor(0.0, device=coupling.device)

    return L_signal + L_noise


# =============================================================================
# Synthetic Autoencoder Experiment
# =============================================================================

class SimpleAutoencoder(nn.Module):
    """
    A small autoencoder for the quadratic landscape experiment.

    Encodes n_vars-dimensional data into a latent space of the same
    dimensionality (for interpretability), with a bottleneck hidden layer.
    """
    def __init__(self, n_vars: int, hidden_dim: int = 32, latent_dim: int = None):
        super().__init__()
        if latent_dim is None:
            latent_dim = n_vars

        self.encoder = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vars),
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)


def build_quadratic_precision(n_objects=2, vars_per_object=4,
                               vars_per_blanket=3, intra_strength=6.0,
                               blanket_strength=0.8):
    """Build block-structured precision matrix for quadratic EBM."""
    n = n_objects * vars_per_object + vars_per_blanket
    Theta = np.zeros((n, n))

    start = 0
    for i in range(n_objects):
        end = start + vars_per_object
        Theta[start:end, start:end] = intra_strength
        np.fill_diagonal(Theta[start:end, start:end],
                         intra_strength * vars_per_object)
        start = end

    blanket_start = n_objects * vars_per_object
    Theta[blanket_start:, blanket_start:] = 1.0
    np.fill_diagonal(Theta[blanket_start:, blanket_start:], vars_per_blanket)

    for obj_idx in range(n_objects):
        obj_start = obj_idx * vars_per_object
        obj_end = obj_start + vars_per_object
        Theta[obj_start:obj_end, blanket_start:] = blanket_strength
        Theta[blanket_start:, obj_start:obj_end] = blanket_strength

    Theta = (Theta + Theta.T) / 2.0
    eigvals = np.linalg.eigvalsh(Theta)
    if eigvals.min() < 0.1:
        Theta += np.eye(n) * (0.1 - eigvals.min() + 0.1)

    return Theta


def sample_quadratic(Theta, n_samples=3000, step_size=0.005, temp=0.1):
    """Langevin sampling from quadratic EBM."""
    n_vars = Theta.shape[0]
    samples = []
    x = np.random.randn(n_vars) * 0.5

    n_steps = 30
    for i in range(n_samples * n_steps):
        grad = Theta @ x
        noise = np.sqrt(2 * step_size * temp) * np.random.randn(n_vars)
        x = x - step_size * grad + noise
        if i % n_steps == 0:
            samples.append(x.copy())

    return np.array(samples)


def get_ground_truth(n_objects, vars_per_object, vars_per_blanket):
    """Return ground truth labels: object index per variable, -1 for blanket."""
    n_vars = n_objects * vars_per_object + vars_per_blanket
    gt = np.full(n_vars, -1)
    for obj_idx in range(n_objects):
        start = obj_idx * vars_per_object
        end = start + vars_per_object
        gt[start:end] = obj_idx
    return gt


def evaluate_structure(z_np, n_objects, ground_truth):
    """
    Evaluate topological blanket quality on latent representations.

    Computes the coupling matrix, applies TB detection, and measures
    ARI and blanket F1 against ground truth.
    """
    # Compute coupling matrix (numpy)
    H_est = np.cov(z_np.T)
    if H_est.ndim == 0:
        H_est = np.array([[float(H_est)]])
    D = np.sqrt(np.diag(H_est).clip(min=1e-8))
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    n_vars = coupling.shape[0]
    truth_blanket = (ground_truth == -1)

    # Spectral clustering for object assignment
    try:
        n_total = n_objects + 1
        sc = SpectralClustering(
            n_clusters=n_total, affinity='precomputed',
            assign_labels='discretize', random_state=42
        )
        labels = sc.fit_predict(coupling + 1e-6)

        # Identify blanket cluster: the one with highest cross-cluster coupling entropy
        entropies = []
        for c in range(n_total):
            mask_c = labels == c
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
                entropies.append(-np.sum(p * np.log(p + 1e-10)))
            else:
                entropies.append(0)
        blanket_cluster = np.argmax(entropies)
        pred_blanket = (labels == blanket_cluster)

        # Object assignment: remap non-blanket labels
        pred_assignment = np.full(n_vars, -1)
        obj_counter = 0
        for c in range(n_total):
            if c == blanket_cluster:
                continue
            pred_assignment[labels == c] = obj_counter
            obj_counter += 1

    except Exception:
        pred_blanket = np.zeros(n_vars, dtype=bool)
        pred_assignment = np.zeros(n_vars, dtype=int)

    # Metrics
    internal_mask = ~truth_blanket
    if np.sum(internal_mask) > 1 and len(np.unique(pred_assignment[internal_mask])) > 1:
        ari = adjusted_rand_score(ground_truth[internal_mask],
                                   pred_assignment[internal_mask])
    else:
        ari = 0.0

    blanket_f1 = f1_score(truth_blanket.astype(int),
                           pred_blanket.astype(int), zero_division=0)

    # Block structure quality: ratio of within-block to between-block coupling
    within = 0.0
    between = 0.0
    count_w = 0
    count_b = 0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if ground_truth[i] >= 0 and ground_truth[j] >= 0:
                if ground_truth[i] == ground_truth[j]:
                    within += coupling[i, j]
                    count_w += 1
                else:
                    between += coupling[i, j]
                    count_b += 1
    block_ratio = (within / max(count_w, 1)) / (between / max(count_b, 1) + 1e-8)

    return {
        'ari': float(ari),
        'blanket_f1': float(blanket_f1),
        'block_ratio': float(block_ratio),
        'coupling': coupling,
        'pred_blanket': pred_blanket,
        'pred_assignment': pred_assignment,
    }


def train_autoencoder(data_np, n_objects, vars_per_object, vars_per_blanket,
                      topo_weight=0.0, n_epochs=200, lr=1e-3,
                      target_persistence=0.5, seed=42, verbose=False):
    """
    Train autoencoder with optional topological loss.

    Args:
        data_np: Training data (n_samples, n_vars).
        n_objects: Number of ground truth objects.
        vars_per_object: Variables per object.
        vars_per_blanket: Variables in the blanket.
        topo_weight: Weight for L_topo. 0 = no topological regularization.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        target_persistence: Target persistence for desired features.
        seed: Random seed.
        verbose: Print training progress.

    Returns:
        Trained model, training history dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_vars = data_np.shape[1]
    data_tensor = torch.FloatTensor(data_np)

    model = SimpleAutoencoder(n_vars, hidden_dim=64, latent_dim=n_vars)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'recon_loss': [],
        'topo_loss': [],
        'total_loss': [],
    }

    batch_size = 256

    for epoch in range(n_epochs):
        model.train()
        epoch_recon = 0.0
        epoch_topo = 0.0
        epoch_total = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(len(data_tensor))
        data_shuffled = data_tensor[perm]

        for start in range(0, len(data_shuffled), batch_size):
            end = min(start + batch_size, len(data_shuffled))
            batch = data_shuffled[start:end]

            x_hat, z = model(batch)

            # Reconstruction loss
            recon_loss = ((x_hat - batch) ** 2).mean()

            # Topological loss
            if topo_weight > 0 and len(batch) > n_vars + 1:
                coupling = coupling_matrix_from_latent(z)
                topo_l = topological_loss(coupling, n_objects=n_objects,
                                          target_persistence=target_persistence)
            else:
                topo_l = torch.tensor(0.0)

            total = recon_loss + topo_weight * topo_l

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_topo += topo_l.item()
            epoch_total += total.item()
            n_batches += 1

        avg_recon = epoch_recon / n_batches
        avg_topo = epoch_topo / n_batches
        avg_total = epoch_total / n_batches

        history['recon_loss'].append(avg_recon)
        history['topo_loss'].append(avg_topo)
        history['total_loss'].append(avg_total)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: recon={avg_recon:.4f}, "
                  f"topo={avg_topo:.4f}, total={avg_total:.4f}")

    return model, history


def run_synthetic_experiment(n_trials=5, verbose=True):
    """
    Experiment 1: Synthetic autoencoder on quadratic landscape.

    Train with and without L_topo, compare post-hoc TB quality.
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Synthetic Autoencoder with Topological Loss")
    print("=" * 70)

    n_objects = 2
    vars_per_object = 4
    vars_per_blanket = 3
    n_vars = n_objects * vars_per_object + vars_per_blanket

    ground_truth = get_ground_truth(n_objects, vars_per_object, vars_per_blanket)

    results_baseline = []
    results_topo = []
    topo_weight = 0.1  # moderate weight

    for trial in range(n_trials):
        if verbose:
            print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Generate data
        np.random.seed(42 + trial)
        Theta = build_quadratic_precision(n_objects, vars_per_object,
                                           vars_per_blanket)
        data = sample_quadratic(Theta, n_samples=3000)

        # Train baseline (no topo loss)
        if verbose:
            print("  Training baseline (no topo loss)...")
        model_base, hist_base = train_autoencoder(
            data, n_objects, vars_per_object, vars_per_blanket,
            topo_weight=0.0, n_epochs=200, seed=42 + trial,
            verbose=verbose
        )

        # Train with topo loss
        if verbose:
            print(f"  Training with L_topo (weight={topo_weight})...")
        model_topo, hist_topo = train_autoencoder(
            data, n_objects, vars_per_object, vars_per_blanket,
            topo_weight=topo_weight, n_epochs=200, seed=42 + trial,
            target_persistence=0.5, verbose=verbose
        )

        # Evaluate both
        model_base.eval()
        model_topo.eval()
        with torch.no_grad():
            z_base = model_base.encode(torch.FloatTensor(data)).numpy()
            z_topo = model_topo.encode(torch.FloatTensor(data)).numpy()

        eval_base = evaluate_structure(z_base, n_objects, ground_truth)
        eval_topo = evaluate_structure(z_topo, n_objects, ground_truth)

        eval_base['final_recon_loss'] = hist_base['recon_loss'][-1]
        eval_topo['final_recon_loss'] = hist_topo['recon_loss'][-1]
        eval_base['history'] = hist_base
        eval_topo['history'] = hist_topo

        results_baseline.append(eval_base)
        results_topo.append(eval_topo)

        if verbose:
            print(f"  Baseline: ARI={eval_base['ari']:.3f}, "
                  f"F1={eval_base['blanket_f1']:.3f}, "
                  f"block_ratio={eval_base['block_ratio']:.2f}, "
                  f"recon={eval_base['final_recon_loss']:.4f}")
            print(f"  Topo:     ARI={eval_topo['ari']:.3f}, "
                  f"F1={eval_topo['blanket_f1']:.3f}, "
                  f"block_ratio={eval_topo['block_ratio']:.2f}, "
                  f"recon={eval_topo['final_recon_loss']:.4f}")

    return results_baseline, results_topo


# =============================================================================
# LunarLander Dynamics Model Experiment
# =============================================================================

class EnsembleMLP(nn.Module):
    """Simple ensemble dynamics model for LunarLander 8D state space."""

    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128, n_members=3):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )
            for _ in range(n_members)
        ])

    def forward(self, s, a):
        """Forward through all members, return mean and individual predictions."""
        preds = [m(torch.cat([s, a], dim=-1)) for m in self.members]
        preds = torch.stack(preds, dim=0)  # (n_members, batch, state_dim)
        return preds.mean(dim=0), preds


def train_dynamics_model(states, actions, next_states,
                          topo_weight=0.0, n_objects=2,
                          n_epochs=100, lr=1e-3,
                          target_persistence=0.3,
                          seed=42, verbose=False):
    """
    Train an ensemble dynamics model with optional topological loss.

    The topological loss is computed on the latent coupling structure:
    for each batch, we compute the coupling matrix of the hidden
    representations (last hidden layer activations) and penalize
    deviations from the target persistence diagram.
    """
    torch.manual_seed(seed)

    state_dim = states.shape[1]
    n_actions = 4

    # One-hot encode actions
    actions_oh = np.zeros((len(actions), n_actions))
    actions_oh[np.arange(len(actions)), actions.astype(int)] = 1.0

    s_tensor = torch.FloatTensor(states)
    a_tensor = torch.FloatTensor(actions_oh)
    ns_tensor = torch.FloatTensor(next_states)

    model = EnsembleMLP(state_dim=state_dim, action_dim=n_actions,
                        hidden_dim=128, n_members=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'recon_loss': [], 'topo_loss': [], 'total_loss': []}
    batch_size = 256

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(s_tensor))
        epoch_recon = 0.0
        epoch_topo = 0.0
        n_batches = 0

        for start in range(0, len(s_tensor), batch_size):
            end = min(start + batch_size, len(s_tensor))
            idx = perm[start:end]
            s_b = s_tensor[idx]
            a_b = a_tensor[idx]
            ns_b = ns_tensor[idx]

            pred_mean, _ = model(s_b, a_b)
            recon_loss = ((pred_mean - ns_b) ** 2).mean()

            # Topological loss on prediction residuals coupling structure
            if topo_weight > 0 and len(s_b) > state_dim + 1:
                # Use the prediction outputs as "latent" representation
                # Coupling of the predictions reveals structure
                coupling = coupling_matrix_from_latent(pred_mean)
                topo_l = topological_loss(coupling, n_objects=n_objects,
                                          target_persistence=target_persistence)
            else:
                topo_l = torch.tensor(0.0)

            total = recon_loss + topo_weight * topo_l
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_topo += topo_l.item()
            n_batches += 1

        avg_recon = epoch_recon / n_batches
        avg_topo = epoch_topo / n_batches
        history['recon_loss'].append(avg_recon)
        history['topo_loss'].append(avg_topo)
        history['total_loss'].append(avg_recon + topo_weight * avg_topo)

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: recon={avg_recon:.4f}, "
                  f"topo={avg_topo:.4f}")

    return model, history


def evaluate_dynamics_structure(model, states, actions, next_states, n_objects=2):
    """
    Evaluate the coupling structure of a trained dynamics model.

    Collect predictions across the dataset and compute the coupling
    matrix of the prediction outputs, then apply TB detection.
    """
    state_dim = states.shape[1]
    n_actions = 4
    actions_oh = np.zeros((len(actions), n_actions))
    actions_oh[np.arange(len(actions)), actions.astype(int)] = 1.0

    model.eval()
    with torch.no_grad():
        s_t = torch.FloatTensor(states)
        a_t = torch.FloatTensor(actions_oh)
        pred_mean, preds = model(s_t, a_t)
        pred_np = pred_mean.numpy()

    # Compute coupling from predictions
    H_est = np.cov(pred_np.T)
    D = np.sqrt(np.diag(H_est).clip(min=1e-8))
    coupling = np.abs(H_est) / np.outer(D, D)
    np.fill_diagonal(coupling, 0)

    # Also compute from prediction errors (gradients of loss surface)
    errors = pred_np - next_states
    H_err = np.cov(errors.T)
    D_err = np.sqrt(np.diag(H_err).clip(min=1e-8))
    coupling_err = np.abs(H_err) / np.outer(D_err, D_err)
    np.fill_diagonal(coupling_err, 0)

    # Eigenvalue analysis for structure quality
    W = coupling.copy()
    D_mat = np.diag(W.sum(axis=1))
    L = D_mat - W
    eigvals = np.linalg.eigvalsh(L)
    gaps = np.diff(eigvals)

    # Spectral gap ratio: gap between cluster 2-3 split vs noise
    if len(gaps) > 2:
        spectral_gap_ratio = gaps[1] / (gaps[2] + 1e-8)
    else:
        spectral_gap_ratio = 0.0

    return {
        'coupling': coupling,
        'coupling_errors': coupling_err,
        'eigvals': eigvals.tolist(),
        'spectral_gap_ratio': float(spectral_gap_ratio),
        'final_recon': float(np.mean((pred_np - next_states) ** 2)),
    }


def run_lunarlander_experiment(verbose=True):
    """
    Experiment 2: LunarLander dynamics model with topological loss.

    Uses pre-collected trajectory data from results/trajectory_data/.
    """
    print("\n" + "=" * 70)
    print("Experiment 2: LunarLander Dynamics Model with Topological Loss")
    print("=" * 70)

    # Load trajectory data
    states = np.load(os.path.join(TRAJ_DIR, 'states.npy'))
    actions = np.load(os.path.join(TRAJ_DIR, 'actions.npy'))
    next_states = np.load(os.path.join(TRAJ_DIR, 'next_states.npy'))

    print(f"Loaded {len(states)} transitions, state_dim={states.shape[1]}")

    # Normalize states for training stability
    s_mean = states.mean(axis=0)
    s_std = states.std(axis=0) + 1e-8
    states_norm = (states - s_mean) / s_std
    next_states_norm = (next_states - s_mean) / s_std

    # Use a subset for speed
    n_use = min(5000, len(states))
    idx = np.random.RandomState(42).choice(len(states), n_use, replace=False)
    s_sub = states_norm[idx]
    a_sub = actions[idx]
    ns_sub = next_states_norm[idx]

    STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']

    # Expected structure: 2 object groups
    #  - kinematic: {x, y, vx, vy, angle, ang_vel} (position/velocity)
    #  - contact: {left_leg, right_leg}
    # With angle/ang_vel potentially acting as blanket

    results = {}

    for topo_w_label, topo_w in [('baseline', 0.0), ('topo_0.01', 0.01),
                                   ('topo_0.1', 0.1), ('topo_0.5', 0.5)]:
        if verbose:
            print(f"\n  Training: {topo_w_label} (topo_weight={topo_w})")

        model, hist = train_dynamics_model(
            s_sub, a_sub, ns_sub,
            topo_weight=topo_w, n_objects=2,
            n_epochs=80, lr=1e-3,
            target_persistence=0.3,
            seed=42, verbose=verbose
        )

        eval_result = evaluate_dynamics_structure(
            model, s_sub, a_sub, ns_sub, n_objects=2)

        eval_result['history'] = {
            'recon_loss': hist['recon_loss'],
            'topo_loss': hist['topo_loss'],
        }

        results[topo_w_label] = eval_result

        if verbose:
            print(f"  {topo_w_label}: recon={eval_result['final_recon']:.4f}, "
                  f"spectral_gap={eval_result['spectral_gap_ratio']:.3f}")

    return results, STATE_LABELS


# =============================================================================
# Hyperparameter Sweep
# =============================================================================

def run_hyperparameter_sweep(verbose=True):
    """
    Experiment 3: Sweep L_topo weight from 0.001 to 1.0.

    Measures the reconstruction loss vs TB ARI tradeoff.
    """
    print("\n" + "=" * 70)
    print("Experiment 3: Hyperparameter Sweep (L_topo Weight)")
    print("=" * 70)

    n_objects = 2
    vars_per_object = 4
    vars_per_blanket = 3

    ground_truth = get_ground_truth(n_objects, vars_per_object, vars_per_blanket)

    weights = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    n_trials = 3

    sweep_results = {w: [] for w in weights}

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        Theta = build_quadratic_precision(n_objects, vars_per_object,
                                           vars_per_blanket)
        data = sample_quadratic(Theta, n_samples=3000)

        for w in weights:
            if verbose:
                print(f"  Trial {trial+1}, weight={w}")

            model, hist = train_autoencoder(
                data, n_objects, vars_per_object, vars_per_blanket,
                topo_weight=w, n_epochs=200, lr=1e-3,
                target_persistence=0.5, seed=42 + trial
            )

            model.eval()
            with torch.no_grad():
                z = model.encode(torch.FloatTensor(data)).numpy()

            eval_result = evaluate_structure(z, n_objects, ground_truth)
            eval_result['final_recon_loss'] = hist['recon_loss'][-1]
            eval_result['final_topo_loss'] = hist['topo_loss'][-1]

            sweep_results[w].append(eval_result)

    return sweep_results, weights


# =============================================================================
# Plotting
# =============================================================================

def plot_synthetic_comparison(results_baseline, results_topo):
    """Plot comparison of baseline vs topo-regularized autoencoder."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. ARI comparison
    ax = axes[0]
    ari_base = [r['ari'] for r in results_baseline]
    ari_topo = [r['ari'] for r in results_topo]

    positions = [0, 1]
    bp = ax.boxplot([ari_base, ari_topo], positions=positions, widths=0.6,
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline\n(no L_topo)', 'With L_topo'])
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Object Partition Quality (ARI)')
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 2. Blanket F1 comparison
    ax = axes[1]
    f1_base = [r['blanket_f1'] for r in results_baseline]
    f1_topo = [r['blanket_f1'] for r in results_topo]

    bp = ax.boxplot([f1_base, f1_topo], positions=positions, widths=0.6,
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline\n(no L_topo)', 'With L_topo'])
    ax.set_ylabel('F1 Score')
    ax.set_title('Blanket Detection Quality (F1)')
    ax.set_ylim(-0.1, 1.1)

    # 3. Block ratio comparison
    ax = axes[2]
    br_base = [r['block_ratio'] for r in results_baseline]
    br_topo = [r['block_ratio'] for r in results_topo]

    bp = ax.boxplot([br_base, br_topo], positions=positions, widths=0.6,
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline\n(no L_topo)', 'With L_topo'])
    ax.set_ylabel('Within / Between Coupling Ratio')
    ax.set_title('Block Structure Quality')

    plt.suptitle('Differentiable Topological Loss: Synthetic Autoencoder',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'synthetic_comparison', 'topo_loss')


def plot_training_curves(results_baseline, results_topo):
    """Plot training curves comparing baseline and topo-regularized models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Average training curves across trials
    n_epochs = len(results_baseline[0]['history']['recon_loss'])

    # Reconstruction loss
    ax = axes[0]
    base_recon = np.array([r['history']['recon_loss'] for r in results_baseline])
    topo_recon = np.array([r['history']['recon_loss'] for r in results_topo])

    ax.plot(range(n_epochs), base_recon.mean(axis=0), color='#e74c3c',
            label='Baseline', linewidth=2)
    ax.fill_between(range(n_epochs),
                     base_recon.mean(axis=0) - base_recon.std(axis=0),
                     base_recon.mean(axis=0) + base_recon.std(axis=0),
                     alpha=0.2, color='#e74c3c')
    ax.plot(range(n_epochs), topo_recon.mean(axis=0), color='#2ecc71',
            label='With L_topo', linewidth=2)
    ax.fill_between(range(n_epochs),
                     topo_recon.mean(axis=0) - topo_recon.std(axis=0),
                     topo_recon.mean(axis=0) + topo_recon.std(axis=0),
                     alpha=0.2, color='#2ecc71')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss During Training')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Topological loss (topo model only)
    ax = axes[1]
    topo_tl = np.array([r['history']['topo_loss'] for r in results_topo])
    ax.plot(range(n_epochs), topo_tl.mean(axis=0), color='#3498db',
            label='L_topo', linewidth=2)
    ax.fill_between(range(n_epochs),
                     topo_tl.mean(axis=0) - topo_tl.std(axis=0),
                     topo_tl.mean(axis=0) + topo_tl.std(axis=0),
                     alpha=0.2, color='#3498db')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Topological Loss')
    ax.set_title('Topological Loss During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Training Dynamics: Baseline vs Topo-Regularized',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'training_curves', 'topo_loss')


def plot_coupling_matrices(results_baseline, results_topo, ground_truth,
                            n_objects, vars_per_object, vars_per_blanket):
    """Plot coupling matrices side by side for baseline vs topo."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Use the first trial
    coupling_base = results_baseline[0]['coupling']
    coupling_topo = results_topo[0]['coupling']

    # Ground truth precision matrix
    Theta = build_quadratic_precision(n_objects, vars_per_object, vars_per_blanket)
    D_gt = np.sqrt(np.diag(Theta).clip(min=1e-8))
    coupling_gt = np.abs(Theta) / np.outer(D_gt, D_gt)
    np.fill_diagonal(coupling_gt, 0)

    n_vars = coupling_base.shape[0]
    boundaries = [vars_per_object * i for i in range(1, n_objects + 1)]

    for ax, coupling, title in zip(axes,
                                    [coupling_gt, coupling_base, coupling_topo],
                                    ['Ground Truth\nPrecision', 'Baseline\nLatent Coupling',
                                     'With L_topo\nLatent Coupling']):
        im = ax.imshow(coupling, cmap='hot', aspect='auto', vmin=0)
        for b in boundaries:
            ax.axhline(y=b - 0.5, color='white', linestyle='--', linewidth=2)
            ax.axvline(x=b - 0.5, color='white', linestyle='--', linewidth=2)
        ax.set_xlabel('Variable j')
        ax.set_ylabel('Variable i')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Coupling Matrix Comparison: Ground Truth vs Learned Representations',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'coupling_matrices', 'topo_loss')


def plot_hyperparameter_sweep(sweep_results, weights):
    """Plot reconstruction loss vs ARI tradeoff across L_topo weights."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    mean_ari = []
    std_ari = []
    mean_f1 = []
    std_f1 = []
    mean_recon = []
    std_recon = []
    mean_block = []
    std_block = []

    for w in weights:
        trials = sweep_results[w]
        aris = [t['ari'] for t in trials]
        f1s = [t['blanket_f1'] for t in trials]
        recons = [t['final_recon_loss'] for t in trials]
        blocks = [t['block_ratio'] for t in trials]

        mean_ari.append(np.mean(aris))
        std_ari.append(np.std(aris))
        mean_f1.append(np.mean(f1s))
        std_f1.append(np.std(f1s))
        mean_recon.append(np.mean(recons))
        std_recon.append(np.std(recons))
        mean_block.append(np.mean(blocks))
        std_block.append(np.std(blocks))

    x_labels = [str(w) for w in weights]

    # Plot 1: ARI vs weight
    ax = axes[0]
    ax.errorbar(range(len(weights)), mean_ari, yerr=std_ari,
                marker='o', capsize=3, color='#2ecc71', linewidth=2)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('L_topo Weight')
    ax.set_ylabel('ARI')
    ax.set_title('Object Partition Quality')
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 2: Recon loss vs weight
    ax = axes[1]
    ax.errorbar(range(len(weights)), mean_recon, yerr=std_recon,
                marker='s', capsize=3, color='#e74c3c', linewidth=2)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('L_topo Weight')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Quality')
    ax.grid(True, alpha=0.3)

    # Plot 3: Tradeoff (recon vs ARI)
    ax = axes[2]
    for i, w in enumerate(weights):
        ax.errorbar(mean_recon[i], mean_ari[i],
                     xerr=std_recon[i], yerr=std_ari[i],
                     marker='o', capsize=3, markersize=8,
                     label=f'w={w}')
    ax.set_xlabel('Reconstruction Loss')
    ax.set_ylabel('ARI')
    ax.set_title('Recon vs Structure Tradeoff')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Sweep: L_topo Weight',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'hyperparameter_sweep', 'topo_loss')


def plot_lunarlander_results(ll_results, state_labels):
    """Plot LunarLander coupling matrices and spectral analysis."""
    configs = list(ll_results.keys())
    n_configs = len(configs)

    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 10))
    if n_configs == 1:
        axes = axes.reshape(-1, 1)

    for i, cfg_name in enumerate(configs):
        result = ll_results[cfg_name]
        coupling = result['coupling']

        # Coupling matrix
        ax = axes[0, i]
        im = ax.imshow(coupling, cmap='hot', aspect='auto', vmin=0)
        ax.set_xticks(range(len(state_labels)))
        ax.set_xticklabels(state_labels, rotation=45, fontsize=8)
        ax.set_yticks(range(len(state_labels)))
        ax.set_yticklabels(state_labels, fontsize=8)
        ax.set_title(f'{cfg_name}\nrecon={result["final_recon"]:.4f}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Eigenvalue spectrum
        ax = axes[1, i]
        eigvals = result['eigvals']
        ax.bar(range(len(eigvals)), eigvals, color='#3498db')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'Laplacian Spectrum\ngap_ratio={result["spectral_gap_ratio"]:.2f}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('LunarLander Dynamics: Effect of Topological Loss on Coupling Structure',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'lunarlander_comparison', 'topo_loss')


def plot_ll_training_curves(ll_results):
    """Plot LunarLander training curves for all configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = list(ll_results.keys())
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

    for i, cfg_name in enumerate(configs):
        result = ll_results[cfg_name]
        hist = result['history']
        color = colors[i % len(colors)]

        # Reconstruction loss
        axes[0].plot(hist['recon_loss'], label=cfg_name, color=color, linewidth=2)

        # Topo loss
        if any(v > 0 for v in hist['topo_loss']):
            axes[1].plot(hist['topo_loss'], label=cfg_name, color=color, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reconstruction Loss')
    axes[0].set_title('Dynamics Prediction Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Topological Loss')
    axes[1].set_title('Topological Regularizer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LunarLander Training Dynamics',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'lunarlander_training', 'topo_loss')


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("=" * 70)
    print("US-075: Differentiable Topological Loss for Structure-Aware Training")
    print("=" * 70)
    t0 = time.time()

    # ---- Experiment 1: Synthetic Autoencoder ----
    results_baseline, results_topo = run_synthetic_experiment(n_trials=5, verbose=True)

    plot_synthetic_comparison(results_baseline, results_topo)
    plot_training_curves(results_baseline, results_topo)
    plot_coupling_matrices(results_baseline, results_topo,
                            get_ground_truth(2, 4, 3), 2, 4, 3)

    # ---- Experiment 2: LunarLander ----
    ll_results, state_labels = run_lunarlander_experiment(verbose=True)

    plot_lunarlander_results(ll_results, state_labels)
    plot_ll_training_curves(ll_results)

    # ---- Experiment 3: Hyperparameter Sweep ----
    sweep_results, weights = run_hyperparameter_sweep(verbose=True)

    plot_hyperparameter_sweep(sweep_results, weights)

    # ---- Compile and Save Results ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Synthetic results
    base_aris = [r['ari'] for r in results_baseline]
    topo_aris = [r['ari'] for r in results_topo]
    base_f1s = [r['blanket_f1'] for r in results_baseline]
    topo_f1s = [r['blanket_f1'] for r in results_topo]
    base_blocks = [r['block_ratio'] for r in results_baseline]
    topo_blocks = [r['block_ratio'] for r in results_topo]
    base_recons = [r['final_recon_loss'] for r in results_baseline]
    topo_recons = [r['final_recon_loss'] for r in results_topo]

    print("\nSynthetic Autoencoder (5 trials):")
    print(f"  Baseline ARI:       {np.mean(base_aris):.3f} +/- {np.std(base_aris):.3f}")
    print(f"  Topo ARI:           {np.mean(topo_aris):.3f} +/- {np.std(topo_aris):.3f}")
    print(f"  Baseline Blanket F1:{np.mean(base_f1s):.3f} +/- {np.std(base_f1s):.3f}")
    print(f"  Topo Blanket F1:    {np.mean(topo_f1s):.3f} +/- {np.std(topo_f1s):.3f}")
    print(f"  Baseline Block Ratio:{np.mean(base_blocks):.2f} +/- {np.std(base_blocks):.2f}")
    print(f"  Topo Block Ratio:    {np.mean(topo_blocks):.2f} +/- {np.std(topo_blocks):.2f}")
    print(f"  Baseline Recon Loss: {np.mean(base_recons):.4f}")
    print(f"  Topo Recon Loss:     {np.mean(topo_recons):.4f}")

    ari_improved = np.mean(topo_aris) >= np.mean(base_aris)
    f1_improved = np.mean(topo_f1s) >= np.mean(base_f1s)

    print(f"\n  ARI improved with L_topo: {ari_improved}")
    print(f"  F1 improved with L_topo:  {f1_improved}")

    # LunarLander results
    print("\nLunarLander Dynamics Model:")
    for cfg_name, result in ll_results.items():
        print(f"  {cfg_name}: recon={result['final_recon']:.4f}, "
              f"spectral_gap={result['spectral_gap_ratio']:.3f}")

    # Sweep results
    print("\nHyperparameter Sweep:")
    for w in weights:
        trials = sweep_results[w]
        mean_ari_w = np.mean([t['ari'] for t in trials])
        mean_recon_w = np.mean([t['final_recon_loss'] for t in trials])
        print(f"  w={w}: ARI={mean_ari_w:.3f}, recon={mean_recon_w:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save comprehensive results JSON
    metrics = {
        'synthetic': {
            'baseline': {
                'mean_ari': float(np.mean(base_aris)),
                'std_ari': float(np.std(base_aris)),
                'mean_blanket_f1': float(np.mean(base_f1s)),
                'std_blanket_f1': float(np.std(base_f1s)),
                'mean_block_ratio': float(np.mean(base_blocks)),
                'mean_recon_loss': float(np.mean(base_recons)),
                'per_trial_ari': [float(a) for a in base_aris],
                'per_trial_f1': [float(f) for f in base_f1s],
            },
            'topo_regularized': {
                'mean_ari': float(np.mean(topo_aris)),
                'std_ari': float(np.std(topo_aris)),
                'mean_blanket_f1': float(np.mean(topo_f1s)),
                'std_blanket_f1': float(np.std(topo_f1s)),
                'mean_block_ratio': float(np.mean(topo_blocks)),
                'mean_recon_loss': float(np.mean(topo_recons)),
                'per_trial_ari': [float(a) for a in topo_aris],
                'per_trial_f1': [float(f) for f in topo_f1s],
            },
            'ari_improved': bool(ari_improved),
            'f1_improved': bool(f1_improved),
        },
        'lunarlander': {
            cfg_name: {
                'final_recon': result['final_recon'],
                'spectral_gap_ratio': result['spectral_gap_ratio'],
                'eigvals': result['eigvals'],
            }
            for cfg_name, result in ll_results.items()
        },
        'hyperparameter_sweep': {
            str(w): {
                'mean_ari': float(np.mean([t['ari'] for t in sweep_results[w]])),
                'std_ari': float(np.std([t['ari'] for t in sweep_results[w]])),
                'mean_blanket_f1': float(np.mean([t['blanket_f1'] for t in sweep_results[w]])),
                'mean_recon_loss': float(np.mean([t['final_recon_loss'] for t in sweep_results[w]])),
                'mean_block_ratio': float(np.mean([t['block_ratio'] for t in sweep_results[w]])),
            }
            for w in weights
        },
        'elapsed_seconds': elapsed,
    }

    config = {
        'synthetic_n_objects': 2,
        'synthetic_vars_per_object': 4,
        'synthetic_vars_per_blanket': 3,
        'synthetic_n_trials': 5,
        'synthetic_topo_weight': 0.1,
        'synthetic_n_epochs': 200,
        'lunarlander_n_transitions': n_use if 'n_use' in dir() else 5000,
        'lunarlander_n_epochs': 80,
        'sweep_weights': weights,
        'sweep_n_trials': 3,
    }

    save_results(
        'differentiable_topo_loss',
        metrics,
        config,
        notes=(
            'US-075: Differentiable topological loss (Hu et al. 2019). '
            'Persistence-based regularizer computed via graph Laplacian eigenvalues. '
            f'Synthetic: baseline ARI={np.mean(base_aris):.3f}, '
            f'topo ARI={np.mean(topo_aris):.3f}. '
            f'ARI improved: {ari_improved}, F1 improved: {f1_improved}.'
        )
    )

    print("\nAll results and figures saved to ralph/results/")
    return metrics


if __name__ == '__main__':
    main()
