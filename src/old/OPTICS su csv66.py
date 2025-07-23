import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Caricamento del Dataset ---
file_path = 'journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Dataset '{file_path}' caricato con successo. Dimensioni: {df.shape}")
    print("\nPrime 5 righe del dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato. Assicurati che si trovi nella stessa directory dello script o specifica il percorso completo.")
    exit()

# Salva le colonne originali per l'interpretazione dei cluster
df_original_copy = df.copy()

# --- 2. Pre-elaborazione dei Dati ---
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

# --- 3. Applicazione del Clustering OPTICS ---
print("\n--- Applicazione dell'algoritmo OPTICS ---")

# Parametri di OPTICS:
# min_samples: Il numero minimo di campioni in un vicinato per considerarlo un punto "core".
#              Valori più alti tendono a trovare cluster più densi e a etichettare più punti come rumore.
# xi: La soglia di "ripidità" per estrarre i cluster. Un valore più piccolo porta a cluster più larghi,
#     un valore più grande a cluster più specifici e più isolati.
# cluster_method: 'xi' o 'dbscan'. Se 'xi', usa il parametro xi per estrarre cluster.
#                 Se 'dbscan', usa un valore eps (che non è un parametro diretto di OPTICS
#                 ma si riferisce alla distanza massima tra due campioni per essere considerati
#                 nello stesso vicinato. Non lo useremo direttamente qui, ma sappi che optics
#                 può emulare dbscan con eps).

# Proviamo con min_samples=5 e cluster_method='xi' con un valore di xi
optics_model = OPTICS(min_samples=5, xi=0.01, cluster_method='xi') # Prova a sperimentare con questi valori
optics_model.fit(X_scaled)

# Assegna le etichette dei cluster al dataframe originale
df_original_copy['Cluster'] = optics_model.labels_

# OPTICS può assegnare -1 ai punti considerati rumore
n_clusters_optics = len(np.unique(df_original_copy['Cluster'])) - (1 if -1 in df_original_copy['Cluster'] else 0)
print(f"Numero di cluster identificati da OPTICS (escluso il rumore): {n_clusters_optics}")
if -1 in df_original_copy['Cluster']:
    noise_points = sum(df_original_copy['Cluster'] == -1)
    print(f"Numero di punti etichettati come rumore (-1): {noise_points}")
    print(f"Percentuale di punti etichettati come rumore: {(noise_points / len(df_original_copy) * 100):.2f}%")

# --- 4. Interpretazione dei Cluster ---
print("\n--- Interpretazione dei Cluster ---")

print("\nConteggio dei punti per cluster:")
print(df_original_copy['Cluster'].value_counts().sort_index())

# Loop per stampare le medie per ogni cluster (escludendo il rumore se presente)
print("\nCaratteristiche medie per Cluster (sulle scale originali):")
for i in sorted(df_original_copy['Cluster'].unique()):
    if i == -1:
        print(f"\n--- Cluster -1 (Rumore, N={sum(df_original_copy['Cluster'] == -1)} pazienti) ---")
        # Puoi decidere se mostrare le medie per il rumore o meno. Spesso non sono molto interpretabili.
        # print("  Medie Numeriche (Rumore):")
        # print(df_original_copy[df_original_copy['Cluster'] == -1][numerical_features].mean().round(2))
        continue # Salta l'interpretazione dettagliata per il rumore

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

# --- 5. Visualizzazione con PCA e Interpretazione dei Carichi ---
print("\n--- Visualizzazione dei Cluster con PCA ---")

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df_original_copy['Cluster'] # Assicurati di usare i cluster OPTICS

# Numero di cluster effettivi (escludendo il rumore per la palette)
# Se ci sono cluster validi, allora n_clusters_for_palette = n_clusters_optics + 1 (per il rumore -1 se lo si vuole colorare)
# Se vuoi dare un colore specifico al rumore, devi creare una colormap custom.
# Per ora, usiamo una palette per il numero di cluster validi.
# Il rumore sarà probabilmente in un colore default o non verrà mostrato bene se non specificato.
# Per visualizzare bene il rumore, assegno un colore specifico.
unique_clusters = sorted(pca_df['Cluster'].unique())
if -1 in unique_clusters:
    # Se il rumore è presente, gli assegno un colore grigio
    colors = sns.color_palette("viridis", n_clusters_optics)
    # Aggiungo il grigio per il cluster -1
    cmap_list = list(colors)
    cmap_list.insert(0, (0.5, 0.5, 0.5)) # Grigio per il rumore (cluster -1)
    custom_palette = sns.color_palette(cmap_list)
    # Mapping da cluster_label a indice colore
    cluster_to_color_idx = {cluster: i for i, cluster in enumerate(unique_clusters)}
    pca_df['Cluster_Color_Idx'] = pca_df['Cluster'].map(cluster_to_color_idx)
    hue_col = 'Cluster_Color_Idx'
    palette_to_use = custom_palette
else:
    colors = sns.color_palette("viridis", n_clusters_optics)
    hue_col = 'Cluster'
    palette_to_use = colors

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PC1', y='PC2',
    hue=hue_col,
    palette=palette_to_use,
    data=pca_df,
    legend='full',
    alpha=0.7,
    s=70
)

# Aggiungi etichette personalizzate alla legenda per il rumore
handles, labels = plt.gca().get_legend_handles_labels()
if -1 in unique_clusters:
    labels = ['Rumore (-1)'] + [str(c) for c in unique_clusters if c != -1]
plt.legend(handles=handles, labels=labels, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.title(f'Visualizzazione dei Cluster OPTICS (PCA, {n_clusters_optics} cluster validi)', fontsize=16)
plt.xlabel(f'Componente Principale 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'Componente Principale 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True)
plt.show()

print(f"\nVarianza spiegata dalla PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Varianza spiegata dalla PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Varianza totale spiegata dalle prime 2 PC: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.2f}%")

# --- Interpretazione dei Carichi (Loadings) ---
print("\n--- Interpretazione dei Carichi delle Componenti Principali ---")

# Crea un DataFrame per visualizzare i carichi in modo più leggibile
loadings_df = pd.DataFrame(pca.components_.T, columns=['PC1_Loadings', 'PC2_Loadings'], index=feature_names_out)

# Ordina per valore assoluto per vedere le variabili più influenti
print("\nVariabili più influenti sulla PC1 (ordinate per valore assoluto):")
print(loadings_df['PC1_Loadings'].abs().sort_values(ascending=False).head(10))

print("\nVariabili più influenti sulla PC2 (ordinate per valore assoluto):")
print(loadings_df['PC2_Loadings'].abs().sort_values(ascending=False).head(10))