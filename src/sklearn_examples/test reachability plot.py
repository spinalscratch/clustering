import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler # Utile per scalare i dati se necessario

# --- 1. Genera Dati Artificiali più dispersi (stessi dati di prima) ---
np.random.seed(60)

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

# --- Opzionale: Scalare i dati (spesso consigliato per algoritmi basati sulla distanza) ---
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# Usiamo X non scalato per mantenere la coerenza con l'esempio precedente
# Altrimenti, useremmo X_scaled nel modello OPTICS

# --- 2. Applica OPTICS Clustering ---
# min_samples: il numero minimo di punti in un quartiere per considerare un punto un core point
# xi: un valore soglia per l'estrazione del cluster (sensibilità alle "valli" nel reachability plot)
# min_cluster_size: la dimensione minima di un cluster
optics_model = OPTICS(min_samples=15, xi=0.03, min_cluster_size=0.04) # Questi sono gli iperparametri originali
optics_model.fit(X) # Applica OPTICS sui dati

# --- 3. Estrai i dati per il Reachability Plot ---
# Reachability distances: Le distanze di reachability calcolate per ogni punto
reachability = optics_model.reachability_[optics_model.ordering_]
# Core distances: Le distanze del core per ogni punto, nel loro ordine di elaborazione
core_distances = optics_model.core_distances_[optics_model.ordering_]
# Indici dei punti nell'ordine di elaborazione di OPTICS
ordered_indices = optics_model.ordering_

# --- 4. Crea il Reachability Plot ---
plt.figure(figsize=(10, 6))

# Trama le reachability distances.
# Le 'valli' (minimi) indicano i cluster, le 'montagne' (massimi) indicano i punti outlier o la separazione tra cluster.
# Impostiamo il valore di reachability infinita a un valore grande per visualizzarlo correttamente
# (tipicamente i punti non raggiungibili che non appartengono a nessun cluster denso).
# Useremo np.inf come un valore grande che sarà visibile.
# Per chiarezza, sostituiremo np.inf con il massimo valore finito nel plot + un offset, se presente.

# Sostituisci np.inf con un valore elevato per la visualizzazione, se presente
finite_reachability = reachability[np.isfinite(reachability)]
max_finite_reachability = finite_reachability.max() if finite_reachability.size > 0 else 0
reachability[np.isinf(reachability)] = max_finite_reachability * 1.2 # o un altro valore appropriato

plt.plot(reachability, color='blue', alpha=0.7)

plt.title("Reachability Plot (OPTICS)")
plt.xlabel("Indice dei punti")
plt.ylabel("Reachability distance")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Spiegazione del Reachability Plot ---
print("\n--- Spiegazione del Reachability Plot ---")
print("Il Reachability Plot è uno strumento diagnostico per gli algoritmi di clustering basati sulla densità come OPTICS.")
print("- L'asse orizzontale mostra i punti dati nell'ordine in cui OPTICS li ha elaborati.")
print("- L'asse verticale mostra la 'distanza di reachability' per ciascun punto.")
print("  - Una bassa distanza di reachability indica che il punto è densamente connesso ai suoi vicini.")
print("  - Un'alta distanza di reachability (picchi) indica punti che sono 'lontani' dai cluster densi.")
print("I cluster appaiono come 'valli' (regioni di bassa reachability) nel plot.")
print("Gli outlier e le aree a bassa densità appaiono come 'montagne' (picchi di alta reachability).")
print("La 'profondità' delle valli e l'altezza dei picchi possono aiutare a determinare i cluster e gli outlier.")
print("Modificando i parametri `xi` e `min_cluster_size` di OPTICS, puoi controllare come queste valli vengono interpretate in cluster effettivi.")