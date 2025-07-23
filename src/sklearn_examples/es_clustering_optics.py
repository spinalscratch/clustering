import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
import os

# genera dati artificiali
np.random.seed(60)

# cluster 1
cluster1_center = [-3, 3]
cluster1_data = cluster1_center + 1.2 * np.random.randn(120, 2)

# cluster 2
cluster2_center = [3, 3]
cluster2_data = cluster2_center + 1.0 * np.random.randn(100, 2)

# cluster 3
cluster3_center = [0, -3]
cluster3_data = cluster3_center + 1.5 * np.random.randn(70, 2)

# per aggiungere rumore
noise_data = 8 * (np.random.rand(50, 2) - 0.5)

# combina tutti i dati
X = np.vstack((cluster1_data, cluster2_data, cluster3_data, noise_data))

# plot prima del clustering (punti dello stesso colore)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=15, c='darkblue', alpha=0.7)
plt.title("Original data")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)

# applica OPTICS
optics_model = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
optics_model.fit(X)

labels = optics_model.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)

# plot con i cluster separati per colore (outlier in nero)
plt.subplot(1, 2, 2)
colors_map = plt.colormaps.get_cmap('tab10')
cluster_colors = [colors_map(i / (n_clusters_ - 1)) if n_clusters_ > 1 else colors_map(0)
                  for i in range(n_clusters_)]
if n_clusters_ == 0:
    cluster_colors = []

# dizionario per assegnamento colori/etichette (si poteva fare diversamente)
color_dict = {}
cluster_idx = 0
for k in sorted(list(unique_labels)):
    if k == -1:
        color_dict[k] = 'black'
    else:
        color_dict[k] = cluster_colors[cluster_idx]
        cluster_idx += 1


for k in unique_labels:
    class_member_mask = (labels == k)
    xy = X[class_member_mask]

    point_color = color_dict[k]
    point_size = 15
    marker_style = 'o'
    edge_color = 'none'
    linewidths = 0
    alpha = 0.8

    if k == -1:
        zorder_val = 10
        label_text = 'Outlier'
    else:
        zorder_val = 1
        label_text = f'Cluster {k}'

    plt.scatter(xy[:, 0], xy[:, 1], s=point_size, color=point_color, marker=marker_style,
                edgecolors=edge_color, linewidths=linewidths,
                label=label_text, alpha=alpha, zorder=zorder_val)

plt.title(f"Clustering OPTICS (Num. cluster: {n_clusters_})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='best', fontsize='small')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# salva il plot
output_folder = "../../results/plots/other_examples"
output_filename = "optics_clustering_plot.png"
output_path = os.path.join(output_folder, output_filename)

# crea la cartella se non esiste
os.makedirs(output_folder, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close() # Chiudi la figura per liberare memoria, dato che non la mostriamo
print(f"Plot salvato in: {output_path}")