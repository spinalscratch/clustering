import sklearn.datasets
from sklearn.cluster import OPTICS
import numpy as np
import dbcv
import matplotlib.pyplot as plt


def main():
    #genera dati
    X, y_true = sklearn.datasets.make_moons(n_samples=500, noise=0.05, random_state=1782)

    #aggiungi rumore
    rng = np.random.RandomState(1082)
    quantiles = np.quantile(X, (0, 1), axis=0)
    X_noise = rng.uniform(low=quantiles[0], high=quantiles[1], size=(100, 2))

    #combina i dati
    X = np.vstack((X, X_noise))

    #applica OPTICS
    clustering = OPTICS(min_samples=10, xi=0.07, min_cluster_size=0.15).fit(X)
    y_pred = clustering.labels_

    #calcola DBCV
    score = dbcv.dbcv(X, y_pred)
    print("DBCV score:", score)

    #visualizzazione
    plt.figure(figsize=(12, 6))

    unique_labels = np.unique(y_pred)

    custom_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan']

    for k in unique_labels:
        if k == -1:
            col = 'gray'
            label = 'Noise'
        else:
            col = custom_colors[
                k % len(custom_colors)]
            label = f'Cluster {k}'

        plt.scatter(X[y_pred == k, 0], X[y_pred == k, 1],
                    c=[col], label=label, alpha=0.6)

    plt.title(f'Clustering con OPTICS - DBCV score: {score:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    #reachability plot
    plt.figure(figsize=(12, 4))
    plt.plot(clustering.reachability_[clustering.ordering_], 'b-', alpha=0.8)
    plt.title('Reachability plot')
    plt.xlabel('Punti ordinati')
    plt.ylabel('Distanza')
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == '__main__':
    main()