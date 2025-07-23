import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#lettura file
df = pd.read_csv("10_7717_peerj_5665_dataYM2018_neuroblastoma.csv", header=None)

#imposta i nomi delle colonne
df.columns = [
    "age", "sex", "site", "stage", "risk", "time_months",
    "autologous_stem_cell_transplantation", "radiation",
    "degree_of_differentiation", "UH_or_FH", "MYCN_status",
    "surgical_methods", "outcome"
]

#rimuove eventuali spazi dalle celle (importante per colonne tipo "MYCN_status ")
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

#converte tutto in numerico, le stringhe non convertibili diventano NaN
df = df.apply(pd.to_numeric, errors='coerce')

#rimuove righe vuote o con valori NaN
df = df.dropna()

#separa i dati da clusterizzare (tutte le colonne tranne outcome)
X = df.drop("outcome", axis=1)

#standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#OPTICS
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
optics.fit(X_scaled)

#aggiungi label del cluster al dataframe
df['cluster'] = optics.labels_

#pca
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=df['cluster'], palette="Set2", legend="full", s=50
)
plt.title("Clustering OPTICS (ridotto con PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

