"""Debug: silhouette scores at different k for different noise levels."""
import numpy as np
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from topological_blankets.features import compute_geometric_features

# Build with wider separation
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

print(f"Gradient std: {np.mean(np.std(gradients, axis=0)):.3f}")

for sigma in [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    np.random.seed(42)
    noised = gradients + np.random.randn(*gradients.shape)*sigma
    features = compute_geometric_features(noised)
    coupling = features['coupling']

    dist = 1.0 - coupling
    np.fill_diagonal(dist, 0)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method='average')

    sils = {}
    for k in [2, 3, 4, 5, 6]:
        labels = fcluster(Z, t=k, criterion='maxclust') - 1
        n_uniq = len(np.unique(labels))
        if n_uniq >= 2:
            s = silhouette_score(dist, labels, metric='precomputed')
            sils[k] = f"{s:.3f}"
        else:
            sils[k] = "N/A"

    # Block coupling means
    blocks = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            blocks[i,j] = np.mean(coupling[i*5:(i+1)*5, j*5:(j+1)*5])
    intra_sub = np.mean([blocks[i,i] for i in range(4)])
    intra_macro = np.mean([blocks[0,1], blocks[1,0], blocks[2,3], blocks[3,2]])
    inter_macro = np.mean([blocks[i,j] for i in range(2) for j in range(2,4)] +
                          [blocks[i,j] for i in range(2,4) for j in range(2)])

    print(f"sigma={sigma:5.1f}: sils={sils}  coupling: sub={intra_sub:.4f} macro={intra_macro:.4f} inter={inter_macro:.4f}")
