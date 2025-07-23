import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os

n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)  # Anisotropic blobs
X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)  # Unequal variance
X_filtered = np.vstack(
    (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))  # Unevenly sized blobs
y_filtered = [0] * 500 + [1] * 100 + [2] * 10

output_folder = "../../results/plots/other_examples"

# crea la cartella di output se non esiste
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Cartella '{output_folder}' creata.")

# Grafico dei dati originali
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=8)
plt.title("Dati Originali")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(output_folder, "dati_originali.png"))
plt.close()

# Grafico dei dati anisotropici
plt.figure(figsize=(8, 6))
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y, s=8)
plt.title("Dati Anisotropici")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(output_folder, "dati_anisotropici.png"))
plt.close()

# Grafico dei dati con varianza disomogenea
plt.figure(figsize=(8, 6))
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied, s=8)
plt.title("Dati con Varianza Disomogenea")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(output_folder, "dati_varianza_disomogenea.png"))
plt.close()

# Grafico dei dati con cluster di dimensioni diverse
plt.figure(figsize=(8, 6))
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered, s=8)
plt.title("Dati con Cluster di Dimensioni Diverse")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(output_folder, "dati_cluster_diversi.png"))
plt.close()

print(f"I plot sono stati salvati nella cartella: {output_folder}")
