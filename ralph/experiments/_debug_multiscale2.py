"""Debug: understand coupling matrix behavior across noise levels."""
import numpy as np
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from topological_blankets.features import compute_geometric_features

# Build same landscape
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
            Theta[i1*vps:(i1+1)*vps, i2*vps:(i2+1)*vps] = 2.0
            Theta[i2*vps:(i2+1)*vps, i1*vps:(i1+1)*vps] = 2.0
for m1 in range(n_macro):
    for m2 in range(m1+1, n_macro):
        for s1 in range(sub_per_macro):
            for s2 in range(sub_per_macro):
                i1 = m1*sub_per_macro+s1; i2 = m2*sub_per_macro+s2
                Theta[i1*vps:(i1+1)*vps, i2*vps:(i2+1)*vps] = 0.1
                Theta[i2*vps:(i2+1)*vps, i1*vps:(i1+1)*vps] = 0.1
Theta = (Theta+Theta.T)/2.0
e = np.linalg.eigvalsh(Theta)
if e.min() < 0.1: Theta += np.eye(n_vars)*(0.1-e.min()+0.1)

# Sample
np.random.seed(42)
x = np.random.randn(n_vars)
grads = []
for i in range(5000*50):
    g = Theta@x; x = x - 0.005*g + np.sqrt(2*0.005*0.1)*np.random.randn(n_vars)
    if i%50==0: grads.append((Theta@x).copy())
gradients = np.array(grads)

print("Gradient std per dim:", np.std(gradients, axis=0)[:5].round(3))
print("Gradient signal strength:", np.mean(np.std(gradients, axis=0)).round(3))

for sigma in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    np.random.seed(42)
    noised = gradients + np.random.randn(*gradients.shape)*sigma
    features = compute_geometric_features(noised)
    coupling = features['coupling']

    # Block means
    blocks = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if i == j:
                blocks[i,j] = np.mean(coupling[i*5:(i+1)*5, i*5:(i+1)*5])
            else:
                blocks[i,j] = np.mean(coupling[i*5:(i+1)*5, j*5:(j+1)*5])

    intra_sub = np.mean([blocks[i,i] for i in range(4)])
    intra_macro = np.mean([blocks[0,1], blocks[1,0], blocks[2,3], blocks[3,2]])
    inter_macro = np.mean([blocks[i,j] for i in range(2) for j in range(2,4)] +
                          [blocks[i,j] for i in range(2,4) for j in range(2)])

    # Distance = 1 - coupling, hierarchical clustering
    dist = 1.0 - coupling
    np.fill_diagonal(dist, 0)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method='average')
    merge_dists = Z[:,2]
    gaps = np.diff(merge_dists)

    # Count clusters at different thresholds
    for t in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        labels = fcluster(Z, t=t, criterion='distance')
        nc = len(np.unique(labels))
        if nc in [1,2,4]:
            pass  # interesting

    # Best gap
    best_gap_idx = np.argmax(gaps)
    n_clust = n_vars - (best_gap_idx + 1)

    # Also compute contrast ratio
    signal_range = intra_sub - inter_macro

    print(f"sigma={sigma:6.2f}: intra_sub={intra_sub:.4f} intra_macro={intra_macro:.4f} "
          f"inter_macro={inter_macro:.4f} contrast={signal_range:.4f} "
          f"dendro_n={n_clust} merge_gaps_top3={sorted(gaps)[-3:]}")
