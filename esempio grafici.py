import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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

# Grafico dei dati originali con puntini più piccoli
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=8)
plt.title("Dati Originali")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Grafico dei dati anisotropici con puntini più piccoli
plt.figure(figsize=(8, 6))
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y, s=8)
plt.title("Dati Anisotropici")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Grafico dei dati con varianza disomogenea con puntini più piccoli
plt.figure(figsize=(8, 6))
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied, s=8)
plt.title("Dati con Varianza Disomogenea")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Grafico dei dati con cluster di dimensioni diverse con puntini più piccoli
plt.figure(figsize=(8, 6))
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered, s=8)
plt.title("Dati con Cluster di Dimensioni Diverse")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
