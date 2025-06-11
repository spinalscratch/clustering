import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Caricamento dei dati
df = pd.read_csv("journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv")

# Selezionare le features chiave dal paper: CRP, LymC, PLTC
X = df[['CRP', 'LymC', 'PLTC']]

# Scalatura dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicazione di BIRCH
birch_model = Birch(threshold=0.5, n_clusters=20)  # threshold e n_clusters si possono calibrare
birch_model.fit(X_scaled)
labels = birch_model.labels_

# Analisi dei risultati
n_clusters = len(np.unique(labels))
print(f"Numero di cluster stimati: {n_clusters}")

# Mappatura dei cluster ai gruppi clinici
results = []
for i in range(n_clusters):
    cluster_data = df[labels == i]
    sepsis_ratio = cluster_data['Group'].mean()
    rule_compliance = np.mean(
        (cluster_data['CRP'] >= 4.0) &
        (cluster_data['LymC'] < 0.45) &
        (cluster_data['PLTC'] < 150)
    )
    results.append({
        'Cluster': i,
        'Size': len(cluster_data),
        'Sepsis_Ratio': sepsis_ratio,
        'Rule_Compliance': rule_compliance
    })

results_df = pd.DataFrame(results)
print("\nAnalisi dei cluster:")
print(results_df.sort_values('Sepsis_Ratio', ascending=False))

# Visualizzazione 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X['CRP'],
    X['LymC'],
    X['PLTC'],
    c=labels,
    cmap='viridis',
    s=50,
    alpha=0.6
)

ax.set_xlabel('CRP (mg/dL)', fontsize=12)
ax.set_ylabel('LymC (x10³/μL)', fontsize=12)
ax.set_zlabel('PLTC (x10³/μL)', fontsize=12)
ax.set_title('BIRCH Clustering', fontsize=14)


plt.legend(*scatter.legend_elements(), title='Cluster')
plt.tight_layout()
plt.show()

# Identificazione del cluster "sepsi"
sepsis_cluster = results_df.loc[results_df['Sepsis_Ratio'].idxmax()]
print(f"\nCluster sepsi identificato: {sepsis_cluster['Cluster']}")
print(f"Ratio sepsi: {sepsis_cluster['Sepsis_Ratio']:.2%}")
print(f"Conformità alla regola CRP≥4.0, LymC<0.45, PLT<150: {sepsis_cluster['Rule_Compliance']:.2%}")
