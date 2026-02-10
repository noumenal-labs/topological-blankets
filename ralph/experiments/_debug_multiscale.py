"""Debug script for understanding the spectral behavior of the hierarchical landscape."""
import numpy as np
from scipy.linalg import eigh
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from topological_blankets.spectral import build_adjacency_from_hessian, build_graph_laplacian, compute_eigengap

# Build hierarchical precision matrix inline
n_macro, sub_per_macro, vps = 2, 2, 5
n_sub = n_macro * sub_per_macro
n_vars = n_sub * vps
Theta = np.zeros((n_vars, n_vars))

sub_idx = 0
for m in range(n_macro):
    for s in range(sub_per_macro):
        st, en = sub_idx*vps, (sub_idx+1)*vps
        Theta[st:en, st:en] = 8.0
        np.fill_diagonal(Theta[st:en, st:en], 40.0)
        sub_idx += 1

for m in range(n_macro):
    for s1 in range(sub_per_macro):
        for s2 in range(s1+1, sub_per_macro):
            i1, i2 = m*sub_per_macro+s1, m*sub_per_macro+s2
            st1,en1 = i1*vps,(i1+1)*vps
            st2,en2 = i2*vps,(i2+1)*vps
            Theta[st1:en1, st2:en2] = 2.0
            Theta[st2:en2, st1:en1] = 2.0

for m1 in range(n_macro):
    for m2 in range(m1+1, n_macro):
        for s1 in range(sub_per_macro):
            for s2 in range(sub_per_macro):
                i1, i2 = m1*sub_per_macro+s1, m2*sub_per_macro+s2
                st1,en1 = i1*vps,(i1+1)*vps
                st2,en2 = i2*vps,(i2+1)*vps
                Theta[st1:en1, st2:en2] = 0.1
                Theta[st2:en2, st1:en1] = 0.1

Theta = (Theta + Theta.T)/2.0
eigvals_theta = np.linalg.eigvalsh(Theta)
if eigvals_theta.min() < 0.1:
    Theta += np.eye(n_vars)*(0.1 - eigvals_theta.min() + 0.1)

# Sample
np.random.seed(42)
x = np.random.randn(n_vars)
grads = []
for i in range(5000*50):
    g = Theta @ x
    x = x - 0.005*g + np.sqrt(2*0.005*0.1)*np.random.randn(n_vars)
    if i % 50 == 0:
        grads.append((Theta@x).copy())
gradients = np.array(grads)

# Hessian estimate and coupling
H = np.cov(gradients.T)
D = np.sqrt(np.diag(H)) + 1e-8
coupling = np.abs(H) / np.outer(D, D)
np.fill_diagonal(coupling, 0)

print("Coupling block structure (representative entries):")
for i in range(4):
    row = []
    for j in range(4):
        val = np.mean(coupling[i*5:(i+1)*5, j*5:(j+1)*5])
        row.append(f"{val:.4f}")
    print(f"  sub{i}: {row}")

# Binary adjacency (default threshold 0.01)
A = build_adjacency_from_hessian(H, threshold=0.01)
L = build_graph_laplacian(A)
eigs, vecs = eigh(L)
print(f"\nBinary adjacency Laplacian eigenvalues (first 10): {eigs[:10].round(4)}")
gaps = np.diff(eigs[:10])
print(f"Gaps: {gaps.round(4)}")
n_c, eg = compute_eigengap(eigs[:10])
print(f"Detected n_clusters={n_c}, eigengap={eg:.4f}")

# Try weighted Laplacian
print("\n--- Weighted approach ---")
L_w = np.diag(coupling.sum(axis=1)) - coupling
eigs_w, vecs_w = eigh(L_w)
print(f"Weighted Laplacian eigenvalues (first 10): {eigs_w[:10].round(4)}")
gaps_w = np.diff(eigs_w[:10])
print(f"Gaps: {gaps_w.round(4)}")

# Try higher threshold
for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    A_t = build_adjacency_from_hessian(H, threshold=thresh)
    L_t = build_graph_laplacian(A_t)
    eigs_t, _ = eigh(L_t)
    n_c_t, eg_t = compute_eigengap(eigs_t[:10])
    near_zero = np.sum(eigs_t < 0.1)
    print(f"  threshold={thresh}: near_zero_eigs={near_zero}, eigengap_n={n_c_t}, eigengap={eg_t:.4f}")

# What does coupling look like with noise?
for sigma in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    np.random.seed(42)
    noised = gradients + np.random.randn(*gradients.shape) * sigma
    H_n = np.cov(noised.T)
    D_n = np.sqrt(np.diag(H_n)) + 1e-8
    coup_n = np.abs(H_n) / np.outer(D_n, D_n)
    np.fill_diagonal(coup_n, 0)

    # Block averages
    intra_sub = []
    intra_macro_cross = []
    inter_macro = []
    for i in range(4):
        for j in range(4):
            block_avg = np.mean(coup_n[i*5:(i+1)*5, j*5:(j+1)*5])
            if i == j:
                continue
            gt_m_i = 0 if i < 2 else 1
            gt_m_j = 0 if j < 2 else 1
            if gt_m_i == gt_m_j:
                intra_macro_cross.append(block_avg)
            else:
                inter_macro.append(block_avg)
        intra_sub.append(np.mean(coup_n[i*5:(i+1)*5, i*5:(i+1)*5]))

    L_w_n = np.diag(coup_n.sum(axis=1)) - coup_n
    eigs_n, _ = eigh(L_w_n)
    near_zero = np.sum(eigs_n < 0.5)

    print(f"\nsigma={sigma}: intra_sub_avg={np.mean(intra_sub):.4f}, "
          f"intra_macro_cross={np.mean(intra_macro_cross):.4f}, "
          f"inter_macro={np.mean(inter_macro):.4f}")
    print(f"  Weighted eigs[:8]: {eigs_n[:8].round(4)}")
    print(f"  Near-zero (<0.5): {near_zero}")
