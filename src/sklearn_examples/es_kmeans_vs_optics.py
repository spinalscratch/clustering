import time
import warnings
from itertools import cycle, islice
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

output_folder = "../../results/plots/sklearn_plots"
# crea la cartella se non esiste
os.makedirs(output_folder, exist_ok=True)

# genera datasets
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

plt.figure(figsize=(10, 15))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.15
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

    # crea cluster
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
        n_init='auto',
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

        if i_dataset == 0:
            plt.title(name, size=16, pad=10)

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
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=15, color=colors[y_pred], alpha=0.8)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks([])
        plt.yticks([])
        plt.text(
            0.99,
            0.01,
            f"{(t1 - t0):.2f}s",
            transform=plt.gca().transAxes,
            size=12,
            horizontalalignment="right",
        )
        plot_num += 1

plt.savefig(os.path.join(output_folder, "sklearn_optics_kmeans_comparison.png"))
plt.close()