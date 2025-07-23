import warnings
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import dbcv

def calculate_dbcv_score(X, labels):
    try:
        score_result = dbcv.dbcv(X, labels)
        if isinstance(score_result, tuple):
            return score_result[0]
        else:
            return score_result
    except Exception as e:
        print(f"Errore nel calcolo del punteggio DBCV: {e}")
        return np.nan

def count_clusters(labels):
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        return len(unique_labels) - 1
    return len(unique_labels)

def main():
    n_samples = 500
    seed = 30

    # generazione dei dataset di esempio
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
    )
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)

    random_state_aniso = 170
    X_aniso_base, y_aniso_base = datasets.make_blobs(n_samples=n_samples, random_state=random_state_aniso)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso_base, transformation)
    aniso = (X_aniso, y_aniso_base)

    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state_aniso
    )

    default_optics_params = {
        "min_samples": 7,
        "xi": 0.05,
        "min_cluster_size": 0.1,
    }

    # elenco dei dataset e dei parametri specifici per ciascuno
    datasets_to_process = [
        (noisy_circles, {"min_samples": 7, "xi": 0.08, "n_clusters_expected": 2}),
        (noisy_moons, {"min_samples": 7, "xi": 0.1, "n_clusters_expected": 2}),
        (varied, {"min_samples": 7, "xi": 0.01, "min_cluster_size": 0.2, "n_clusters_expected": 3}),
        (aniso, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2, "n_clusters_expected": 3}),
        (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2, "n_clusters_expected": 3}),
    ]


    for i_dataset, (dataset, algo_specific_params) in enumerate(datasets_to_process):
        X, _ = dataset
        X = StandardScaler().fit_transform(X)

        params = default_optics_params.copy()
        params.update(algo_specific_params)

        print(f"\nProcessing Dataset {i_dataset + 1}: {algo_specific_params.get('name', 'Unnamed Dataset')}")
        print(f"  Parametri OPTICS: min_samples={params['min_samples']}, xi={params['xi']}, min_cluster_size={params['min_cluster_size']}")

        optics = OPTICS(
            min_samples=params["min_samples"],
            xi=params["xi"],
            min_cluster_size=params["min_cluster_size"],
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            optics.fit(X)
            y_pred = optics.labels_.astype(int)

        n_clusters = count_clusters(y_pred)
        dbcv_score = calculate_dbcv_score(X, y_pred)

        print(f"  Numero di cluster trovati (escluso rumore): {n_clusters}")
        print(f"  DBCV score: {dbcv_score:.4f}")

        unique_labels, counts = np.unique(y_pred, return_counts=True)
        print("  Distribuzione dei cluster:")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"    Noise (outliers): {count} elementi")
            else:
                print(f"    Cluster {label}: {count} elementi")

        print("-" * 60)

if __name__ == "__main__":
    main()