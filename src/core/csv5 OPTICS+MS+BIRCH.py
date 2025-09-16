import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, MeanShift, Birch
import dbcv
import os
from datetime import datetime
from contextlib import redirect_stdout

# Import dei parametri da file esterno
from bin.params_algo_csv5 import optics_params, meanshift_params, birch_params

def main():
    output_dir = "../../results/csv5"
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"risultati_cluster_csv5_{today_str}.txt")

    with open(output_path, "w", encoding="utf-8") as f, redirect_stdout(f):
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

        def count_clusters(labels):
            unique = np.unique(labels)
            return len(unique) if -1 not in unique else len(unique) - 1

        def print_cluster_distribution(labels, algorithm_name):
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nDistribuzione cluster per {algorithm_name}:")
            for label, count in zip(unique, counts):
                if label == -1:
                    print(f"  Noise (outliers): {count} elementi")
                else:
                    print(f"  Cluster {label}: {count} elementi")
            print(f"  Totale elementi: {len(labels)}\n")

        # ---- OPTICS ----
        print("\n### Risultati per OPTICS ###")
        for i, params in enumerate(optics_params):
            optics = OPTICS(**params)
            labels = optics.fit_predict(X_scaled)
            clusters = count_clusters(labels)
            score = dbcv.dbcv(X_scaled, labels)

            if 0.99 > score > 0.01:
                print(f"\n[OPTICS #{i}] Parametri: {params}")
                print_cluster_distribution(labels, f"OPTICS #{i}")
                print(f"OPTICS #{i}: {clusters} cluster, DBCV score: {score:.4f}")

        # ---- MeanShift ----
        print("\n### Risultati per MeanShift ###")
        for i, params in enumerate(meanshift_params):
            ms = MeanShift(**params)
            labels = ms.fit_predict(X_scaled)
            clusters = count_clusters(labels)
            score = dbcv.dbcv(X_scaled, labels)

            if 0.99 > score > 0.01:
                print(f"\n[MeanShift #{i}] Parametri: {params}")
                print_cluster_distribution(labels, f"MeanShift #{i}")
                print(f"MeanShift #{i}: {clusters} cluster, DBCV score: {score:.4f}")

        # ---- BIRCH ----
        print("\n### Risultati per BIRCH ###")
        for i, params in enumerate(birch_params):
            birch = Birch(**params)
            labels = birch.fit_predict(X_scaled)
            clusters = count_clusters(labels)
            score = dbcv.dbcv(X_scaled, labels)

            if 0.99 > score > 0.01:
                print(f"\n[BIRCH #{i}] Parametri: {params}")
                print_cluster_distribution(labels, f"BIRCH #{i}")
                print(f"BIRCH #{i}: {clusters} cluster, DBCV score: {score:.4f}")

        print(f"\n--- Output salvato in: {output_path} ---")

if __name__ == "__main__":
    main()
