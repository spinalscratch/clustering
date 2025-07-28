import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS

np.random.seed(60)

# Cluster 1
cluster1_center = [-3, 3]
cluster1_data = cluster1_center + 1.2 * np.random.randn(120, 2)

# Cluster 2
cluster2_center = [3, 3]
cluster2_data = cluster2_center + 1.0 * np.random.randn(100, 2)

# Cluster 3
cluster3_center = [0, -3]
cluster3_data = cluster3_center + 1.5 * np.random.randn(70, 2)

# rumore
noise_data = 8 * (np.random.rand(50, 2) - 0.5)

# combina dati
X = np.vstack((cluster1_data, cluster2_data, cluster3_data, noise_data))

optics_model = OPTICS(min_samples=15, xi=0.03, min_cluster_size=0.04) # Questi sono gli iperparametri originali
optics_model.fit(X) # Applica OPTICS sui dati

reachability = optics_model.reachability_[optics_model.ordering_]
core_distances = optics_model.core_distances_[optics_model.ordering_]
ordered_indices = optics_model.ordering_

fig = plt.figure(figsize=(10, 6))

finite_reachability = reachability[np.isfinite(reachability)]
max_finite_reachability = finite_reachability.max() if finite_reachability.size > 0 else 0
reachability[np.isinf(reachability)] = max_finite_reachability * 1.2 # o un altro valore appropriato

plt.plot(reachability, color='blue', alpha=0.7)

plt.title("Reachability Plot (OPTICS)")
plt.xlabel("Indice dei punti")
plt.ylabel("Reachability distance")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
