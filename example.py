import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS

# --- 1. Genera Dati Artificiali più dispersi ---
np.random.seed(60)  # Manteniamo lo stesso seed per dati simili

# Cluster 1
cluster1_center = [-3, 3]
cluster1_data = cluster1_center + 1.2 * np.random.randn(120, 2)

# Cluster 2
cluster2_center = [3, 3]
cluster2_data = cluster2_center + 1.0 * np.random.randn(100, 2)

# Cluster 3 (più piccolo e disperso)
cluster3_center = [0, -3]
cluster3_data = cluster3_center + 1.5 * np.random.randn(70, 2)

# Aggiungiamo del rumore sparso nell'intero spazio
noise_data = 8 * (np.random.rand(50, 2) - 0.5)

# Combina tutti i dati
X = np.vstack((cluster1_data, cluster2_data, cluster3_data, noise_data))

# --- 2. Plot 1: Tutti i punti dello stesso colore (prima del clustering) ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=15, c='darkblue', alpha=0.7)
plt.title("Original data")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)

# --- 3. Applica OPTICS Clustering ---
optics_model = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
optics_model.fit(X)

labels = optics_model.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)

# --- 4. Plot 2: Tutti i punti clusterizzati con colori diversi e outlier in NERO ---
plt.subplot(1, 2, 2)

# Scegli una colormap per i cluster (escludendo il nero per il rumore)
# Utilizziamo 'tab10' che ha 10 colori distinti, sufficienti per i nostri cluster
# Inizializziamo un contatore per assegnare i colori ai cluster, saltando il nero
color_idx = 0
colors_map = plt.cm.get_cmap('tab10', n_clusters_)

for k in unique_labels:
    if k == -1:
        # Colore nero puro per il rumore/outlier
        point_color = 'black'
        marker_style = 'o'  # Circolo
        edge_color = 'none'  # Nessun bordo
        point_size = 15  # Dimensione standard, come gli altri punti
        zorder_val = 10  # Disegna sopra
        label_text = 'Outlier'
    else:
        # Colori diversi per i cluster
        point_color = colors_map(color_idx)
        color_idx += 1 # Incrementa l'indice solo per i cluster validi
        marker_style = 'o'
        edge_color = 'none'  # Nessun bordo
        point_size = 15  # Dimensione standard
        zorder_val = 1  # Disegna sotto
        label_text = f'Cluster {k}'

    class_member_mask = (labels == k)
    xy = X[class_member_mask]

    plt.scatter(xy[:, 0], xy[:, 1], s=point_size, c=point_color, marker=marker_style,
                edgecolors=edge_color, linewidths=0, # Rimosso il bordo
                label=label_text, alpha=0.8, zorder=zorder_val)

plt.title(f"Clustering OPTICS (Num. cluster: {n_clusters_})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='best', fontsize='small')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()