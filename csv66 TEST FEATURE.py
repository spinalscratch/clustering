import pandas as pd
import dbcv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    df = pd.read_csv('journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv')
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)

    # Salva feature e target
    feature_cols = df.columns
    X = df[feature_cols]
    categorical_features = ['sex_woman', 'diagnosis_0EC_1M_2_AC', 'Group', 'Mortality']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Standardizza
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Clustering con BIRCH
    birch = Birch(n_clusters=2, threshold=2)
    labels = birch.fit_predict(X_scaled)
    df['cluster'] = labels

    # Stampa numero elementi per cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nDistribuzione elementi nei cluster:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} elementi")
    print(f"Totale elementi: {len(df)}\n")

    # Calcolo DBCV
    dbcv_score = dbcv.dbcv(X_scaled.values, labels)
    print(f"DBCV Score: {dbcv_score:.4f}")

    # Test statistici: ANOVA/t-test per numeriche, χ² per categoriche
    results = []

    for col in X.columns:
        if col in numerical_features:
            # Split per cluster
            group0 = df[df['cluster'] == 0][col]
            group1 = df[df['cluster'] == 1][col]
            stat, p = ttest_ind(group0, group1, equal_var=False)
            results.append({'feature': col, 'type': 'numerical', 'test': 't-test', 'p_value': p})
        else:
            # Categoriale: χ²
            contingency = pd.crosstab(df[col], df['cluster'])
            try:
                stat, p, _, _ = chi2_contingency(contingency)
            except:
                p = 1.0
            results.append({'feature': col, 'type': 'categorical', 'test': 'chi2', 'p_value': p})

    # Ordina per p-value
    results_df = pd.DataFrame(results).sort_values(by='p_value')

    print("\nFeature con p-value basso (più discriminanti tra cluster)")
    print(results_df)

    # Silhouette score
    print(f"\nSilhouette Score: {silhouette_score(X_scaled, labels):.3f}")