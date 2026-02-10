"""
US-028/029: Dreamer Autoencoder Analysis
=========================================

Train a Dreamer-style autoencoder (Encoder 8D->64D, Decoder 64D->8D) from scratch
on Active Inference trajectory data, then apply Topological Blankets to the 64D
latent space.

NO pretrained Dreamer checkpoint exists. All telecorder .pt files are PyTorch
ensemble models. The Encoder+Decoder from thrml_wm_mini/models.py are trained
here on reconstruction loss using the trajectory data already collected by
world_model_analysis.py (4508 transitions).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import types
import warnings
warnings.filterwarnings('ignore')

# Project paths
NOUMENAL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TELECORDER_DIR = os.environ.get('TELECORDER_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'telecorder', 'services', 'connectors', 'lunarlander', 'src'))

sys.path.insert(0, NOUMENAL_DIR)

from experiments.utils.results import save_results
from experiments.utils.plotting import save_figure

STATE_LABELS = ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg']


def setup_telecorder_imports():
    """Bypass telecorder_lunarlander.__init__.py which imports zenoh (unavailable on Windows)."""
    pkg_path = os.path.join(TELECORDER_DIR, 'telecorder_lunarlander')
    pkg = types.ModuleType('telecorder_lunarlander')
    pkg.__path__ = [pkg_path]
    pkg.__package__ = 'telecorder_lunarlander'
    sys.modules['telecorder_lunarlander'] = pkg

    sys.path.insert(0, TELECORDER_DIR)


def load_trajectory_data():
    """Load trajectory data collected by world_model_analysis.py."""
    data_dir = os.path.join(NOUMENAL_DIR, 'results', 'trajectory_data')

    states = np.load(os.path.join(data_dir, 'states.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))
    next_states = np.load(os.path.join(data_dir, 'next_states.npy'))

    print(f"Loaded trajectory data: {states.shape[0]} transitions, {states.shape[1]}D state")
    return states, actions, next_states


# =========================================================================
# US-028: Train Dreamer autoencoder
# =========================================================================

def train_autoencoder(states, n_epochs=500, lr=1e-3, batch_size=256, seed=42):
    """
    Train Encoder (8D->64D) + Decoder (64D->8D) on reconstruction loss.

    Uses the Dreamer model architecture from telecorder thrml_wm_mini/models.py
    (3-layer MLPs with ReLU, hidden_dim=64, latent_dim=64).
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    setup_telecorder_imports()
    from telecorder_lunarlander.thrml_wm_mini.models import Encoder, Decoder
    from telecorder_lunarlander.thrml_wm_mini.constants import LATENT_DIM

    print(f"\n{'='*70}")
    print(f"US-028: Train Dreamer Autoencoder (8D -> {LATENT_DIM}D -> 8D)")
    print(f"{'='*70}")

    # Normalize data
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)
    state_std = np.where(state_std < 1e-6, 1.0, state_std)  # avoid div by zero for boolean dims
    states_norm = (states - state_mean) / state_std

    states_jax = jnp.array(states_norm, dtype=jnp.float32)
    n_samples = len(states_jax)
    print(f"Training data: {n_samples} samples, normalized")
    print(f"State mean: {state_mean.round(3)}")
    print(f"State std:  {state_std.round(3)}")

    # Initialize models
    key = jax.random.PRNGKey(seed)
    k_enc, k_dec = jax.random.split(key)
    encoder = Encoder(k_enc, state_dim=8)
    decoder = Decoder(k_dec, state_dim=8)

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter((encoder, decoder), eqx.is_array))

    # Loss function
    def compute_loss(enc_dec, batch):
        encoder_l, decoder_l = enc_dec
        z = encoder_l(batch)
        recon = decoder_l(z)
        mse = jnp.mean((recon - batch) ** 2)
        return mse

    @eqx.filter_jit
    def train_step(enc_dec, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(enc_dec, batch)

        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(enc_dec, eqx.is_array)
        )
        enc_dec_new = eqx.apply_updates(enc_dec, updates)
        return enc_dec_new, opt_state_new, loss

    # Training loop
    losses = []
    key_shuffle = jax.random.PRNGKey(seed + 1)
    n_batches = max(1, n_samples // batch_size)

    print(f"\nTraining for {n_epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Batches per epoch: {n_batches}")

    for epoch in range(n_epochs):
        key_shuffle, subkey = jax.random.split(key_shuffle)
        perm = jax.random.permutation(subkey, n_samples)
        epoch_loss = 0.0

        enc_dec = (encoder, decoder)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch = states_jax[idx]
            enc_dec, opt_state, loss = train_step(enc_dec, opt_state, batch)
            epoch_loss += float(loss)
        encoder, decoder = enc_dec

        epoch_loss /= n_batches
        losses.append(epoch_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: MSE = {epoch_loss:.6f}")

    final_loss = losses[-1]
    print(f"\nFinal reconstruction MSE: {final_loss:.6f}")

    # Encode all data to latent space
    z_all = np.array(encoder(states_jax))
    print(f"Latent representations shape: {z_all.shape}")

    # Latent dimension usage analysis
    z_var = np.var(z_all, axis=0)
    z_var_sorted = np.sort(z_var)[::-1]
    active_dims = np.sum(z_var > 0.01)
    print(f"Active latent dimensions (var > 0.01): {active_dims}/{LATENT_DIM}")
    print(f"Top-8 latent variances: {z_var_sorted[:8].round(4)}")
    print(f"Bottom-8 latent variances: {z_var_sorted[-8:].round(6)}")

    # Reconstruction quality check
    recon_all = np.array(decoder(jnp.array(z_all)))
    per_dim_mse = np.mean((recon_all - states_norm) ** 2, axis=0)
    print(f"\nPer-dimension reconstruction MSE:")
    for i, (label, mse) in enumerate(zip(STATE_LABELS, per_dim_mse)):
        print(f"  {label:12s}: {mse:.6f}")

    return encoder, decoder, z_all, {
        'losses': [float(l) for l in losses],
        'final_mse': float(final_loss),
        'n_epochs': n_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'latent_dim': int(LATENT_DIM),
        'n_transitions': n_samples,
        'state_mean': state_mean.tolist(),
        'state_std': state_std.tolist(),
        'latent_variance': z_var.tolist(),
        'active_dims': int(active_dims),
        'per_dim_mse': per_dim_mse.tolist(),
        'per_dim_labels': STATE_LABELS,
    }


def compute_latent_gradients(encoder, decoder, states_norm):
    """
    Compute reconstruction loss gradients in latent space.

    grad_z ||decode(z) - obs||^2

    These gradients reveal the energy landscape geometry in the learned 64D space.
    """
    import jax
    import jax.numpy as jnp

    states_jax = jnp.array(states_norm, dtype=jnp.float32)
    z_all = encoder(states_jax)

    # Per-sample reconstruction loss gradient w.r.t. z
    def single_recon_loss(z, obs_target):
        recon = decoder.forward_single(z)
        return jnp.mean((recon - obs_target) ** 2)

    grad_fn = jax.vmap(jax.grad(single_recon_loss), in_axes=(0, 0))
    gradients = grad_fn(z_all, states_jax)
    gradients = np.array(gradients)

    print(f"Computed latent gradients: shape {gradients.shape}")
    print(f"Gradient magnitude stats: mean={np.mean(np.abs(gradients)):.6f}, "
          f"max={np.max(np.abs(gradients)):.6f}")

    return gradients, np.array(z_all)


# =========================================================================
# US-029: Apply TB to 64D latent space
# =========================================================================

def analyze_latent_space(gradients, z_all, encoder, decoder, states_norm):
    """Apply TB to 64D latent-space gradients."""
    import jax
    import jax.numpy as jnp

    from topological_blankets.core import topological_blankets as tb_pipeline
    from topological_blankets.features import compute_geometric_features
    from topological_blankets.spectral import (
        build_adjacency_from_hessian, build_graph_laplacian,
        compute_eigengap, recursive_spectral_detection
    )
    from scipy.linalg import eigh

    print(f"\n{'='*70}")
    print(f"US-029: TB Analysis of Dreamer 64D Latent Space")
    print(f"{'='*70}")

    # Compute features
    features = compute_geometric_features(gradients)
    H_est = features['hessian_est']
    coupling = features['coupling']

    print(f"Hessian shape: {H_est.shape}")
    print(f"Coupling matrix shape: {coupling.shape}")

    # Spectral analysis
    A = build_adjacency_from_hessian(H_est)
    L = build_graph_laplacian(A)
    eigvals, eigvecs = eigh(L)

    # Find eigengap for reasonable number of clusters (cap at 10)
    n_check = min(20, len(eigvals))
    n_clusters, eigengap = compute_eigengap(eigvals[:n_check])
    n_clusters = min(n_clusters, 8)  # reasonable upper bound
    print(f"Spectral analysis: eigengap={eigengap:.4f}, suggested clusters={n_clusters}")

    # Run TB with different n_objects
    results_by_method = {}
    for method in ['gradient', 'coupling', 'hybrid']:
        try:
            result = tb_pipeline(gradients, n_objects=n_clusters, method=method)
            assign = result['assignment']
            blanket = result['is_blanket']
            n_blanket = int(np.sum(blanket))
            objects = {}
            for obj_id in sorted(set(assign)):
                if obj_id >= 0:
                    objects[int(obj_id)] = int(np.sum(assign == obj_id))
            print(f"  {method}: {len(objects)} objects, {n_blanket} blanket dims")
            for obj_id, size in objects.items():
                print(f"    Object {obj_id}: {size} dims")
            results_by_method[method] = {
                'assignment': assign.tolist(),
                'is_blanket': blanket.tolist(),
                'n_blanket': n_blanket,
                'object_sizes': objects,
            }
        except Exception as e:
            print(f"  {method}: failed ({e})")
            results_by_method[method] = {'error': str(e)}

    # Hierarchical detection
    try:
        hierarchy = recursive_spectral_detection(H_est, max_levels=3)
        hierarchy_data = [{k: v.tolist() if hasattr(v, 'tolist') else v
                          for k, v in level.items()} for level in hierarchy]
        print(f"Hierarchical detection: {len(hierarchy)} levels")
    except Exception as e:
        hierarchy_data = [{'error': str(e)}]
        print(f"Hierarchical detection failed: {e}")

    # Latent-to-physical mapping via decoder Jacobian
    print("\nComputing latent-to-physical mapping (decoder Jacobian)...")
    states_jax = jnp.array(states_norm, dtype=jnp.float32)

    # Compute Jacobian at representative z points (subsample for speed)
    n_jac_samples = min(500, len(z_all))
    idx_sample = np.random.RandomState(42).choice(len(z_all), n_jac_samples, replace=False)
    z_sample = jnp.array(z_all[idx_sample])

    def decoder_single(z):
        return decoder.forward_single(z)

    jac_fn = jax.vmap(jax.jacobian(decoder_single))
    jacobians = np.array(jac_fn(z_sample))  # (n_samples, 8, 64)
    mean_jac = np.mean(np.abs(jacobians), axis=0)  # (8, 64): mean abs Jacobian

    print(f"Mean Jacobian shape: {mean_jac.shape} (physical_dims x latent_dims)")

    # Also compute correlation between latent dims and physical state
    z_centered = z_all - z_all.mean(axis=0)
    s_centered = states_norm - states_norm.mean(axis=0)
    z_std = np.std(z_all, axis=0, keepdims=True)
    z_std = np.where(z_std < 1e-8, 1.0, z_std)
    s_std = np.std(states_norm, axis=0, keepdims=True)
    s_std = np.where(s_std < 1e-8, 1.0, s_std)
    correlation = (z_centered / z_std).T @ (s_centered / s_std) / len(z_all)  # (64, 8)

    # For each latent dim, find strongest physical correlate
    strongest_correlate = np.argmax(np.abs(correlation), axis=1)
    correlate_strength = np.max(np.abs(correlation), axis=1)

    print(f"\nLatent-to-physical correlation summary:")
    for phys_idx, label in enumerate(STATE_LABELS):
        mapped = np.where(strongest_correlate == phys_idx)[0]
        if len(mapped) > 0:
            strengths = correlate_strength[mapped]
            print(f"  {label:12s}: {len(mapped)} latent dims, "
                  f"max correlation={np.max(strengths):.3f}")

    return {
        'n_clusters_spectral': int(n_clusters),
        'eigengap': float(eigengap),
        'eigenvalues': eigvals[:20].tolist(),
        'methods': results_by_method,
        'hierarchy': hierarchy_data,
        'coupling_matrix': coupling.tolist(),
        'mean_jacobian': mean_jac.tolist(),
        'latent_physical_correlation': correlation.tolist(),
        'strongest_correlate_per_latent': strongest_correlate.tolist(),
        'correlate_strength_per_latent': correlate_strength.tolist(),
        'state_labels': STATE_LABELS,
    }


# =========================================================================
# Visualization
# =========================================================================

def plot_training_curve(losses):
    """Plot autoencoder training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, color='#3498db', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Dreamer Autoencoder Training')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_latent_variance(z_var):
    """Plot variance of each latent dimension (sorted)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_var = np.sort(z_var)[::-1]
    ax.bar(range(len(sorted_var)), sorted_var, color='#2ecc71', alpha=0.8)
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Active threshold (0.01)')
    ax.set_xlabel('Latent Dimension (sorted by variance)')
    ax.set_ylabel('Variance')
    ax.set_title(f'Latent Dimension Usage ({np.sum(z_var > 0.01)}/64 active)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_coupling_matrix_64d(coupling, assignment=None):
    """Plot 64x64 coupling matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 9))
    coupling_abs = np.abs(np.array(coupling))
    im = ax.imshow(coupling_abs, cmap='YlOrRd', aspect='auto')

    if assignment is not None:
        # Draw partition boundaries
        assign = np.array(assignment)
        sorted_idx = np.argsort(assign)
        coupling_sorted = coupling_abs[sorted_idx][:, sorted_idx]
        im.set_data(coupling_sorted)

        # Mark boundaries between objects
        unique_labels = sorted(set(assign))
        boundaries = []
        for label in unique_labels:
            count = int(np.sum(assign == label))
            if boundaries:
                boundaries.append(boundaries[-1] + count)
            else:
                boundaries.append(count)
        for b in boundaries[:-1]:
            ax.axhline(y=b - 0.5, color='cyan', linewidth=1, alpha=0.7)
            ax.axvline(x=b - 0.5, color='cyan', linewidth=1, alpha=0.7)

    ax.set_title('Dreamer 64D Latent: Coupling Matrix')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_eigenvalue_spectrum(eigvals, title="Eigenvalue Spectrum"):
    """Plot eigenvalue spectrum."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(eigvals)), eigvals, 'o-', color='#2ecc71', markersize=4)
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_latent_physical_correlation(correlation, state_labels):
    """Plot correlation heatmap between latent dims and physical state."""
    fig, ax = plt.subplots(figsize=(12, 5))
    corr = np.array(correlation)  # (64, 8)
    im = ax.imshow(np.abs(corr).T, cmap='viridis', aspect='auto')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Physical State Variable')
    ax.set_yticks(range(len(state_labels)))
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_title('Latent-to-Physical Correlation (|r|)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_jacobian_heatmap(mean_jac, state_labels):
    """Plot mean absolute decoder Jacobian."""
    fig, ax = plt.subplots(figsize=(12, 5))
    jac = np.array(mean_jac)  # (8, 64)
    im = ax.imshow(jac, cmap='magma', aspect='auto')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Physical State Variable')
    ax.set_yticks(range(len(state_labels)))
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_title('Decoder Jacobian (mean |dPhysical/dLatent|)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


# =========================================================================
# Main experiment
# =========================================================================

def run_us028():
    """US-028: Train Dreamer autoencoder and extract 64D latent representations."""
    # Load trajectory data
    states, actions, next_states = load_trajectory_data()

    # Train autoencoder
    encoder, decoder, z_all, training_meta = train_autoencoder(states, n_epochs=500, lr=1e-3)

    # Compute reconstruction loss gradients in latent space
    state_mean = np.array(training_meta['state_mean'])
    state_std = np.array(training_meta['state_std'])
    states_norm = (states - state_mean) / state_std

    print("\n--- Computing latent-space gradients ---")
    latent_gradients, z_all_checked = compute_latent_gradients(encoder, decoder, states_norm)

    # Save data for US-029
    data_dir = os.path.join(NOUMENAL_DIR, 'results', 'trajectory_data')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'dreamer_latents.npy'), z_all)
    np.save(os.path.join(data_dir, 'dreamer_latent_gradients.npy'), latent_gradients)
    np.save(os.path.join(data_dir, 'dreamer_states_norm.npy'), states_norm)

    # Plots
    fig_train = plot_training_curve(training_meta['losses'])
    save_figure(fig_train, 'dreamer_training_curve', 'dreamer_analysis')

    fig_var = plot_latent_variance(np.array(training_meta['latent_variance']))
    save_figure(fig_var, 'dreamer_latent_variance', 'dreamer_analysis')

    # Save results
    save_results('dreamer_autoencoder_training', training_meta, {
        'architecture': 'Encoder(8->64->64->64) + Decoder(64->64->64->8)',
        'source': 'telecorder thrml_wm_mini/models.py',
        'data_source': 'Active Inference trajectories (world_model_analysis.py)',
    }, notes='US-028: Dreamer autoencoder trained from scratch. No pretrained checkpoint. '
             '8D->64D->8D reconstruction with MSE loss.')

    print("\nUS-028 complete.")
    return encoder, decoder, z_all, latent_gradients, states_norm, training_meta


def run_us029(encoder, decoder, z_all, latent_gradients, states_norm, training_meta):
    """US-029: Apply TB to Dreamer 64D latent space."""
    latent_results = analyze_latent_space(
        latent_gradients, z_all, encoder, decoder, states_norm)

    # Plots
    # Coupling matrix
    assignment = None
    for method in ['hybrid', 'coupling', 'gradient']:
        if method in latent_results['methods'] and 'assignment' in latent_results['methods'][method]:
            assignment = latent_results['methods'][method]['assignment']
            break

    fig_coupling = plot_coupling_matrix_64d(latent_results['coupling_matrix'], assignment)
    save_figure(fig_coupling, 'dreamer_coupling_64d', 'dreamer_analysis')

    # Eigenvalue spectrum
    fig_eig = plot_eigenvalue_spectrum(
        latent_results['eigenvalues'],
        f"Dreamer 64D Latent: Eigenvalue Spectrum (gap={latent_results['eigengap']:.4f})")
    save_figure(fig_eig, 'dreamer_eigenvalue_spectrum', 'dreamer_analysis')

    # Latent-physical correlation
    fig_corr = plot_latent_physical_correlation(
        latent_results['latent_physical_correlation'], STATE_LABELS)
    save_figure(fig_corr, 'dreamer_latent_physical_correlation', 'dreamer_analysis')

    # Decoder Jacobian
    fig_jac = plot_jacobian_heatmap(latent_results['mean_jacobian'], STATE_LABELS)
    save_figure(fig_jac, 'dreamer_decoder_jacobian', 'dreamer_analysis')

    # Save results
    save_results('dreamer_tb_analysis', latent_results, {
        'latent_dim': 64,
        'state_dim': 8,
        'training_mse': training_meta['final_mse'],
    }, notes='US-029: TB analysis of Dreamer 64D latent space. Coupling matrix, spectral analysis, '
             'hierarchical detection, latent-to-physical mapping via decoder Jacobian.')

    print("\nUS-029 complete.")
    return latent_results


if __name__ == '__main__':
    encoder, decoder, z_all, latent_grads, states_norm, train_meta = run_us028()
    run_us029(encoder, decoder, z_all, latent_grads, states_norm, train_meta)
