import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, MeanShift, Birch
import dbcv


def main():
    # Caricamento e pulizia del dataset
    df = pd.read_csv('../../data/raw/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.dropna()

    # Estrazione delle features
    X = df.values

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aggiunta di piccolo rumore per evitare duplicati
    rng = np.random.default_rng(seed=42)
    X_scaled += rng.normal(0, 1e-8, X_scaled.shape)

    # Funzione per calcolare il numero di cluster (escluso il rumore)
    def count_clusters(labels):
        unique = np.unique(labels)
        # Esclude il noise (label -1) se presente
        num_clusters = len(unique) if -1 not in unique else len(unique) - 1
        return num_clusters

    # Funzione per stampare la distribuzione dei cluster
    def print_cluster_distribution(labels, algorithm_name):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nDistribuzione cluster per {algorithm_name}:")
        for label, count in zip(unique, counts):
            if label == -1:
                print(f"  Noise (outliers): {count} elementi")
            else:
                print(f"  Cluster {label}: {count} elementi")
        print(f"  Totale elementi: {len(labels)}\n")

    # Applicazione OPTICS
    optics = OPTICS(min_samples=10, xi=0.01, min_cluster_size=0.15)
    optics_labels = optics.fit_predict(X_scaled)
    optics_clusters = count_clusters(optics_labels)
    optics_score = dbcv.dbcv(X_scaled, optics_labels)
    print_cluster_distribution(optics_labels, "OPTICS")
    print(f"OPTICS: {optics_clusters} cluster, DBCV score: {optics_score:.4f}")

    # Applicazione MeanShift
    ms = MeanShift(bandwidth=4)
    ms_labels = ms.fit_predict(X_scaled)
    ms_clusters = count_clusters(ms_labels)
    ms_score = dbcv.dbcv(X_scaled, ms_labels)
    print_cluster_distribution(ms_labels, "MeanShift")
    print(f"MeanShift: {ms_clusters} cluster, DBCV score: {ms_score:.4f}")

    # Applicazione BIRCH
    birch = Birch(n_clusters=None, threshold=3.5, branching_factor=50)
    birch_labels = birch.fit_predict(X_scaled)
    birch_clusters = count_clusters(birch_labels)
    birch_score = dbcv.dbcv(X_scaled, birch_labels)
    print_cluster_distribution(birch_labels, "BIRCH")
    print(f"BIRCH: {birch_clusters} cluster, DBCV score: {birch_score:.4f}")


if __name__ == "__main__":
    main()