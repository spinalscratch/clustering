import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from dbcv import dbcv
from sklearn import cluster, datasets
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Genera dataset
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

#Set up parametri
plt.figure(figsize=(10, 15))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.05, hspace=0.25
)

plot_num = 1

default_base = {
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "random_state": 42,
}

datasets = [
    (
        noisy_circles,
        {
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {}),
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    #Crea oggetti cluster
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
        n_init=10,  # Impostazione esplicita di n_init per evitare FutureWarning
    )
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )

    clustering_algorithms = [
        ("K-Means", two_means),
        ("OPTICS", optics),
    ]

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)

        #Aggiunge il titolo solo per la prima riga
        if i_dataset == 0:
            plt.title(name, size=16, pad=20)

        colors = np.array(
            list(
                islice(
                    cycle([
                        "#377eb8", "#ff7f00", "#4daf4a",
                        "#f781bf", "#a65628", "#984ea3",
                        "#999999", "#e41a1c", "#dede00"
                    ]),
                    int(max(y_pred) + 1),
                )
            )
        )
        #Aggiunge il nero per i punti di rumore (outliers)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=15, color=colors[y_pred], alpha=0.8)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks([])
        plt.yticks([])

        #Calcola e visualizza i punteggi
        n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        silhouette = "N/A"
        dbcv = "N/A"

        if n_clusters_ > 1:
            try:
                silhouette = f"{silhouette_score(X, y_pred):.2f}"

                #kDBCV restituisce una coppia (punteggio, punteggi_individuali)
                #estraggo solo il primo elemento (il punteggio generale)
                score_result = dbcv.dbcv(X, y_pred)

                #Controlliamo se il risultato è una coppia e prendiamo il primo valore
                if isinstance(score_result, tuple):
                    dbcv_val = score_result[0]
                else:
                    dbcv_val = score_result  #se dovesse restituire un singolo valore

                dbcv = f"{dbcv_val:.2f}"

            except Exception as e:
                # In caso di errore durante il calcolo del punteggio
                print(f"Could not calculate scores for {name} on dataset {i_dataset}: {e}")
                pass

        #Posizione per il testo dei punteggi
        plt.text(
            -2.4,
            -2.4,
            f"Silhouette: {silhouette}\nDBCV: {dbcv}",
            size=10,
            verticalalignment="bottom",
            horizontalalignment="left",
        )

        plt.text(
            0.99,
            0.01,
            f"{(t1 - t0):.2f}s",
            transform=plt.gca().transAxes,
            size=12,
            horizontalalignment="right",
        )
        plot_num += 1

plt.show()
