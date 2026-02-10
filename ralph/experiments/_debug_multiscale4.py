"""Debug: understand the noise floor and thresholding approach."""
import numpy as np
from scipy.linalg import eigh
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from topological_blankets.features import compute_geometric_features

# Build landscape
n_vars = 20; vps = 5
Theta = np.zeros((n_vars, n_vars))
sub_idx = 0
for m in range(2):
    for s in range(2):
        st, en = sub_idx*vps, (sub_idx+1)*vps
        Theta[st:en, st:en] = 10.0
        np.fill_diagonal(Theta[st:en, st:en], 50.0)
        sub_idx += 1
for m in range(2):
    i1, i2 = m*2, m*2+1
    Theta[i1*vps:(i1+1)*vps, i2*vps:(i2+1)*vps] = 1.0
    Theta[i2*vps:(i2+1)*vps, i1*vps:(i1+1)*vps] = 1.0
for s1 in range(2):
    for s2 in range(2):
        i1, i2 = s1, 2+s2
        Theta[i1*vps:(i1+1)*vps, i2*vps:(i2+1)*vps] = 0.02
        Theta[i2*vps:(i2+1)*vps, i1*vps:(i1+1)*vps] = 0.02
Theta = (Theta+Theta.T)/2.0
e = np.linalg.eigvalsh(Theta)
if e.min() < 0.1: Theta += np.eye(n_vars)*(0.1-e.min()+0.1)

np.random.seed(42)
x = np.random.randn(n_vars)
grads = []
for i in range(5000*50):
    g = Theta@x; x = x-0.005*g+np.sqrt(2*0.005*0.1)*np.random.randn(n_vars)
    if i%50==0: grads.append((Theta@x).copy())
gradients = np.array(grads)
N = len(gradients)

# True coupling levels:
H_clean = np.cov(gradients.T)
D_clean = np.sqrt(np.diag(H_clean)) + 1e-8
coup_clean = np.abs(H_clean) / np.outer(D_clean, D_clean)
np.fill_diagonal(coup_clean, 0)

print("Clean coupling block structure:")
for i in range(4):
    for j in range(4):
        if i == j:
            print(f"  sub{i}-sub{j} (intra): {np.mean(coup_clean[i*5:(i+1)*5, i*5:(i+1)*5]):.4f}")
        elif (i < 2 and j < 2) or (i >= 2 and j >= 2):
            print(f"  sub{i}-sub{j} (macro): {np.mean(coup_clean[i*5:(i+1)*5, j*5:(j+1)*5]):.4f}")
        else:
            print(f"  sub{i}-sub{j} (inter): {np.mean(coup_clean[i*5:(i+1)*5, j*5:(j+1)*5]):.4f}")

# Theoretical noise floor for coupling:
# If g_noised = g + N(0,sigma^2 I), then Cov(g_noised) = Cov(g) + sigma^2 I
# Normalized coupling: |H_ij + delta_ij*sigma^2| / sqrt((H_ii+sigma^2)(H_jj+sigma^2))
# For i != j: |H_ij| / sqrt((H_ii+sigma^2)(H_jj+sigma^2))
# This shrinks off-diagonal coupling. For truly independent vars, H_ij ~ O(1/sqrt(N))
# due to finite-sample noise. The noise floor is ~ 1/sqrt(N) * H_diag / (H_diag + sigma^2)
# A simpler estimate: for N samples of d-dim random noise, the expected spurious coupling is
# ~ 1/sqrt(N). For the noised gradient, the spurious coupling is reduced by the ratio
# sigma^2 / (var_g + sigma^2) which -> 1 as sigma -> inf.

grad_var = np.mean(np.var(gradients, axis=0))
print(f"\nGradient variance per dim: {grad_var:.3f}")
print(f"N = {N}")
print(f"Expected noise floor (1/sqrt(N)): {1.0/np.sqrt(N):.4f}")

for sigma in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
    np.random.seed(42)
    noised = gradients + np.random.randn(*gradients.shape)*sigma
    H_n = np.cov(noised.T)
    D_n = np.sqrt(np.diag(H_n)) + 1e-8
    coup_n = np.abs(H_n) / np.outer(D_n, D_n)
    np.fill_diagonal(coup_n, 0)

    # Block means
    intra_sub = np.mean([np.mean(coup_n[i*5:(i+1)*5, i*5:(i+1)*5]) for i in range(4)])
    intra_macro = np.mean([np.mean(coup_n[0:5,5:10]), np.mean(coup_n[5:10,0:5]),
                           np.mean(coup_n[10:15,15:20]), np.mean(coup_n[15:20,10:15])])
    inter_macro = np.mean([np.mean(coup_n[i*5:(i+1)*5, j*5:(j+1)*5])
                          for i in range(2) for j in range(2,4)])

    # Noise floor: expected coupling between independent dims is ~1/sqrt(N)
    noise_floor = 1.0 / np.sqrt(N)

    # Threshold at 2x noise floor (signal must be at least 2x above noise)
    threshold = noise_floor * 2

    # Count connected components at this threshold
    adj = (coup_n > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    visited = np.zeros(n_vars, dtype=bool)
    n_comp = 0
    labels = np.full(n_vars, -1, dtype=int)
    for start in range(n_vars):
        if visited[start]: continue
        queue = [start]; visited[start] = True; labels[start] = n_comp
        while queue:
            node = queue.pop(0)
            for nb in range(n_vars):
                if not visited[nb] and adj[node,nb] > 0:
                    visited[nb] = True; labels[nb] = n_comp; queue.append(nb)
        n_comp += 1

    # Also try dynamic threshold based on coupling distribution
    off_diag = coup_n[np.triu_indices(n_vars, k=1)]
    dynamic_thresh = np.mean(off_diag) + np.std(off_diag)

    adj2 = (coup_n > dynamic_thresh).astype(float)
    np.fill_diagonal(adj2, 0)
    visited2 = np.zeros(n_vars, dtype=bool)
    n_comp2 = 0
    for start in range(n_vars):
        if visited2[start]: continue
        queue = [start]; visited2[start] = True
        while queue:
            node = queue.pop(0)
            for nb in range(n_vars):
                if not visited2[nb] and adj2[node,nb] > 0:
                    visited2[nb] = True; queue.append(nb)
        n_comp2 += 1

    print(f"sigma={sigma:5.1f}: intra={intra_sub:.4f} macro={intra_macro:.4f} inter={inter_macro:.4f} "
          f"| noise_floor={noise_floor:.4f} thresh={threshold:.4f} n_comp={n_comp} "
          f"| dyn_thresh={dynamic_thresh:.4f} n_comp2={n_comp2}")
