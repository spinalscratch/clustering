import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, MeanShift, Birch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import dbcv
import os
import time
import matplotlib.pyplot as plt

# Import dei parametri da file esterno
from bin.params_algo_csv6 import optics_params, meanshift_params, birch_params

PLOTS_OUTPUT_DIR = "../../results/csv6/plots"

def get_plot_filename(algo_name, params_str, dbcv_score, plot_type):
    params_str = params_str.replace(':', '_').replace('{', '').replace('}', '').replace('\'', '').replace(', ', '_').replace('.', 'dot')
    score_str = f"DBCV_{dbcv_score:.4f}".replace('.', 'dot')
    filename = f"{algo_name}_{plot_type}_{params_str}_{score_str}.png"
    return os.path.join(PLOTS_OUTPUT_DIR, filename)

def create_and_save_scatterplot(X_reduced, labels, algo_name, params, dbcv_score, exec_time, reduction_method):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1

    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap("Dark2")
    colors = [cmap(i % 20) for i in range(len(unique_labels))]

    for idx, label in enumerate(unique_labels):
        cluster_points = X_reduced[labels == label]
        cluster_size = len(cluster_points)

        if label == -1:
            cluster_color = "k"
            point_label = f"Noise ({cluster_size})"
        else:
            cluster_color = colors[idx % 20]
            point_label = f"Cluster {label} ({cluster_size})"

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[cluster_color],
            label=point_label,
            s=20,
            alpha=0.7
        )

    plt.title(
        f"{algo_name} con Riduzione {reduction_method}\n"
        f"Parametri: {params}\n"
        f"Cluster Trovati: {n_clusters}, Punteggio DBCV: {dbcv_score:.4f}, Tempo: {exec_time:.2f}s",
        fontsize=12
    )
    plt.xlabel(f"{reduction_method} Component 1")
    plt.ylabel(f"{reduction_method} Component 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # salva il grafico
    filename = get_plot_filename(algo_name.replace(" #", "_"), str(params), dbcv_score, reduction_method)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"--> Grafico salvato in: {filename}")



def main():

    #caricamento e pulizia del dataset
    df = pd.read_csv('../../data/raw/journal.pone.0216416_Takashi2019_diabetes_type1_dataset_preprocessed.csv')
    df = df.dropna(how='all').dropna(axis=1, how='all').dropna()

    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aggiunta di piccolo rumore per evitare duplicati (importante per alcuni algoritmi)
    rng = np.random.default_rng(seed=42)
    X_scaled += rng.normal(0, 1e-8, X_scaled.shape)

    # riduzione della dimensionalità
    print("Esecuzione riduzione dimensionalità con PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print("Esecuzione riduzione dimensionalità con t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X_scaled)
    print("Riduzione dimensionalità completata.\n")


    def count_clusters(labels):
        unique = np.unique(labels)
        return len(unique) if -1 not in unique else len(unique) - 1

    # ---- OPTICS ----
    print("\n### Esecuzione OPTICS ###")
    for i, params in enumerate(optics_params):
        algo_id = f"OPTICS #{i}"
        print(f"\nAvvio {algo_id} con parametri: {params}")

        start_time = time.time()
        optics = OPTICS(**params)
        labels = optics.fit_predict(X_scaled)
        exec_time = time.time() - start_time

        clusters = count_clusters(labels)
        if clusters > 1:
            score = dbcv.dbcv(X_scaled, labels)
            print(f"{algo_id}: {clusters} cluster, DBCV: {score:.4f}, Tempo: {exec_time:.2f}s")

            if 0.99 > score > 0.01:
                # Scatterplot PCA
                create_and_save_scatterplot(X_pca, labels, algo_id, params, score, exec_time, "PCA")
                # Scatterplot t-SNE
                create_and_save_scatterplot(X_tsne, labels, algo_id, params, score, exec_time, "t-SNE")

                # Reachability Plot per OPTICS
                print(f"--> Generazione Reachability Plot per {algo_id}...")
                reachability = optics.reachability_[optics.ordering_]
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(reachability)), reachability)
                plt.title(f"Reachability Plot per {algo_id}\nParametri: {params}\nDBCV: {score:.4f}")
                plt.xlabel("Ordinamento dei punti")
                plt.ylabel("Distanza di raggiungibilità")
                plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                reach_filename = get_plot_filename(algo_id.replace(' #', '_'), str(params), score, "Reachability")
                plt.savefig(reach_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"--> Grafico salvato in: {reach_filename}")
        else:
            print(f"{algo_id}: Trovati {clusters} cluster. Impossibile calcolare DBCV o generare grafici.")


    # ---- MeanShift ----
    print("\n### Esecuzione MeanShift ###")
    for i, params in enumerate(meanshift_params):
        algo_id = f"MeanShift #{i}"
        print(f"\nAvvio {algo_id} con parametri: {params}")

        start_time = time.time()
        ms = MeanShift(**params)
        labels = ms.fit_predict(X_scaled)
        exec_time = time.time() - start_time

        clusters = count_clusters(labels)
        if clusters > 1:
            score = dbcv.dbcv(X_scaled, labels)
            print(f"{algo_id}: {clusters} cluster, DBCV: {score:.4f}, Tempo: {exec_time:.2f}s")
            if 0.99 > score > 0.01:
                create_and_save_scatterplot(X_pca, labels, algo_id, params, score, exec_time, "PCA")
                create_and_save_scatterplot(X_tsne, labels, algo_id, params, score, exec_time, "t-SNE")
        else:
            print(f"{algo_id}: Trovati {clusters} cluster. Impossibile calcolare DBCV o generare grafici.")

    # ---- BIRCH ----
    print("\n### Esecuzione BIRCH ###")
    for i, params in enumerate(birch_params):
        algo_id = f"BIRCH #{i}"
        # BIRCH con n_clusters=None spesso non trova cluster, si può forzare un valore se necessario
        if 'n_clusters' not in params or params['n_clusters'] is None:
            print(f"\n{algo_id}: Parametro 'n_clusters' non specificato, potrebbe non trovare cluster validi.")

        print(f"Avvio {algo_id} con parametri: {params}")

        start_time = time.time()
        birch = Birch(**params)
        labels = birch.fit_predict(X_scaled)
        exec_time = time.time() - start_time

        clusters = count_clusters(labels)
        if clusters > 1:
            score = dbcv.dbcv(X_scaled, labels)
            print(f"{algo_id}: {clusters} cluster, DBCV: {score:.4f}, Tempo: {exec_time:.2f}s")
            if 0.99 > score > 0.01:
                create_and_save_scatterplot(X_pca, labels, algo_id, params, score, exec_time, "PCA")
                create_and_save_scatterplot(X_tsne, labels, algo_id, params, score, exec_time, "t-SNE")
        else:
            print(f"{algo_id}: Trovati {clusters} cluster. Impossibile calcolare DBCV o generare grafici.")

if __name__ == "__main__":
    main()