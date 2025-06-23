import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import Birch # Import Birch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Caricamento del Dataset
file_path = 'journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Dataset '{file_path}' caricato con successo. Dimensioni: {df.shape}")
    print("\nPrime 5 righe del dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato. Assicurati che si trovi nella stessa directory dello script o specifica il percorso completo.")
    exit()

#Salva le colonne originali per l'interpretazione dei cluster
df_original_copy = df.copy()

#Pre-elaborazione dei Dati
numerical_features = [
    'Age', 'APACHE II', 'SOFA', 'CRP', 'WBCC', 'NeuC', 'LymC',
    'EOC', 'NLCR', 'PLTC', 'MPV', 'LOS-ICU'
]
categorical_features = ['sex_woman', 'diagnosis_0EC_1M_2_AC']
outcome_features = ['Group', 'Mortality']

print("\nVerifica dei valori mancanti:")
print(df[numerical_features + categorical_features].isnull().sum())

for col in numerical_features:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputato i valori mancanti di '{col}' con la mediana: {median_val}")

for col in categorical_features:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Imputato i valori mancanti di '{col}' con la moda: {mode_val}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_scaled = preprocessor.fit_transform(df)

feature_names_out = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
X_scaled_df_info = pd.DataFrame(X_scaled, columns=feature_names_out)
print("\nDati pre-elaborati (prime 5 righe, scalati e codificati):")
print(X_scaled_df_info.head())
print(f"Shape dei dati pre-elaborati per il clustering: {X_scaled.shape}")

#Applicazione del Clustering BIRCH
print("\n--- Applicazione dell'algoritmo BIRCH ---")


birch_model = Birch(n_clusters=16, threshold=0.5)

birch_model.fit(X_scaled)
df_original_copy['Cluster'] = birch_model.labels_

n_clusters_birch = len(np.unique(df_original_copy['Cluster']))
print(f"Numero di cluster identificati da BIRCH: {n_clusters_birch}")

#Interpretazione dei Cluster
print("\n--- Interpretazione dei Cluster ---")

print("\nConteggio dei punti per cluster:")
print(df_original_copy['Cluster'].value_counts().sort_index())

print("\nCaratteristiche medie per Cluster (sulle scale originali):")
for i in range(n_clusters_birch):
    cluster_data = df_original_copy[df_original_copy['Cluster'] == i]
    if not cluster_data.empty:
        print(f"\n--- Cluster {i} (N={len(cluster_data)} pazienti) ---")
        print("  Medie Numeriche:")
        print(cluster_data[numerical_features].mean().round(2))
        print("\n  Distribuzione di 'sex_woman':")
        print(cluster_data['sex_woman'].value_counts(normalize=True).round(2))
        print("\n  Distribuzione delle diagnosi ('diagnosis_0EC_1M_2_AC'):")
        print(cluster_data['diagnosis_0EC_1M_2_AC'].value_counts(normalize=True).round(2))
        print("\n  Tasso di Mortalità nel cluster:")
        if 'Mortality' in df_original_copy.columns:
            mortality_rate = cluster_data['Mortality'].mean()
            print(f"    Mortality = 1: {mortality_rate:.2f} | Mortality = 0: {1-mortality_rate:.2f}")
        else:
            print("    Colonna 'Mortality' non trovata.")
        print("\n  LOS-ICU Medio nel cluster:")
        if 'LOS-ICU' in df_original_copy.columns:
            print(f"    {cluster_data['LOS-ICU'].mean():.2f} giorni")
        else:
            print("    Colonna 'LOS-ICU' non trovata.")
        print("\n  Distribuzione dei 'Group' nel cluster:")
        if 'Group' in df_original_copy.columns:
            print(cluster_data['Group'].value_counts(normalize=True).round(2))
        else:
            print("    Colonna 'Group' non trovata.")

#Visualizzazione con PCA e Interpretazione dei Carichi
print("\n--- Visualizzazione dei Cluster con PCA ---")

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df_original_copy['Cluster'] # Assicurati di usare i cluster BIRCH

#Aumenta il numero di colori se n_clusters_birch è grande
colors = sns.color_palette("viridis", n_clusters_birch) # O un'altra palette come "tab20" se ci sono molti cluster
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster',
    palette=colors,
    data=pca_df,
    legend='full',
    alpha=0.7,
    s=70
)
plt.title(f'Visualizzazione dei Cluster BIRCH (PCA, {n_clusters_birch} cluster)', fontsize=16)
plt.xlabel(f'Componente Principale 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'Componente Principale 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True)
plt.show()

print(f"\nVarianza spiegata dalla PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Varianza spiegata dalla PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Varianza totale spiegata dalle prime 2 PC: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.2f}%")

#Interpretazione dei Carichi (Loadings)
print("\n--- Interpretazione dei Carichi delle Componenti Principali ---")

#Crea un DataFrame per visualizzare i carichi in modo più leggibile
loadings_df = pd.DataFrame(pca.components_.T, columns=['PC1_Loadings', 'PC2_Loadings'], index=feature_names_out)

#Ordina per valore assoluto per vedere le variabili più influenti
print("\nVariabili più influenti sulla PC1 (ordinate per valore assoluto):")
print(loadings_df['PC1_Loadings'].abs().sort_values(ascending=False).head(10))

print("\nVariabili più influenti sulla PC2 (ordinate per valore assoluto):")
print(loadings_df['PC2_Loadings'].abs().sort_values(ascending=False).head(10))