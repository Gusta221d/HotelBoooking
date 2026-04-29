import pandas as pd
import numpy as np
import time
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

print("--------- ESTABILIDADE E WARD ---------")

try:
    df = pd.read_csv('../hotel_bookings_clean.csv')
except FileNotFoundError:
    df = pd.read_csv('hotel_bookings_clean.csv')

categorical_features = ['hotel', 'arrival_date_month', 'meal', 'market_segment', 
                        'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']
numeric_features = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 
                    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 
                    'babies', 'is_repeated_guest', 'previous_cancellations', 
                    'previous_bookings_not_canceled', 'required_car_parking_spaces', 
                    'total_of_special_requests']

#pipeline de preprocessor entre numericas e categoricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features), #aplica StandardScaler nas numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) #aplica OneHotEncoder nas categóricas
    ])

#transforma dados brutos na matriz de características ->x_euclid
X_euclid = preprocessor.fit_transform(df) 
results = []
k_winner = 8 #define o número de clusters k-8 como o winner
representation_id = 'EUCLID-SemADR'
labels_by_seed = {}

#teste com5 seeds
print("\n[Sensibilidade] A iniciar teste de estabilidade com 5 seeds para K=8...")
seeds_to_test = [42, 67, 123, 2026, 71] #lista de seeds aleatórias

#inicio do loop para cada seed
for s in seeds_to_test:
    print(f"-> A correr seed {s}...", end=" ", flush=True)
    start_time = time.time()
    kmeans = KMeans(n_clusters=k_winner, random_state=s, n_init=10) #inicia o kmeans com a seed atual
    labels = kmeans.fit_predict(X_euclid)
    labels_by_seed[s] = labels
    runtime = round(time.time() - start_time, 2)#obtem o tempo de execucao

    #calcula as metricas de validade
    silhouette = silhouette_score(X_euclid, labels, sample_size=30000, random_state=s) 
    ch_score = calinski_harabasz_score(X_euclid, labels)
    db_score = davies_bouldin_score(X_euclid, labels)

    #guarda os resultados
    results.append({
        'representation_id': representation_id,
        'algorithm': 'K-Means Stability',
        'k': k_winner,
        'seed': s,
        'runtime_sec': runtime,
        'silhouette': round(silhouette, 4),
        'calinski_harabasz': round(ch_score, 2),
        'davies_bouldin': round(db_score, 4)
    })
    print(f"Sil: {silhouette:.4f}")

#calcula estabilidade por ARI entre pares de seeds
ari_values = []
for s1, s2 in itertools.combinations(seeds_to_test, 2):
    ari = adjusted_rand_score(labels_by_seed[s1], labels_by_seed[s2])
    ari_values.append(ari)

ari_mean = float(np.mean(ari_values))
ari_std = float(np.std(ari_values))
print(f"\n[Estabilidade] ARI médio entre seeds: {ari_mean:.4f} ± {ari_std:.4f}")

print("\nA iniciar Clustering Hierárquico Ward")
print("-> A aplicar Regra de Subamostragem: N=20000")

np.random.seed(42) #seed fixa para a amostragem
sample_indices = np.random.choice(X_euclid.shape[0], 20000, replace=False) #seleciona 20000 índices random
X_sample = X_euclid[sample_indices] #cria uma amostra para o calculo hierarquico

start_time = time.time()
ward_model = AgglomerativeClustering(n_clusters=k_winner, linkage='ward') #inicia o modelo hierarquico
labels_ward = ward_model.fit_predict(X_sample)
ward_runtime = round(time.time() - start_time, 2)

#calcula metricas para o modelo hierarquico
ward_sil = silhouette_score(X_sample, labels_ward, random_state=42)
ward_ch = calinski_harabasz_score(X_sample, labels_ward)
ward_db = davies_bouldin_score(X_sample, labels_ward)

#guarda os resultados do ward
results.append({
    'representation_id': representation_id + '-Sample20k',
    'algorithm': 'Hierarchical Ward',
    'k': k_winner,
    'seed': 42,
    'runtime_sec': ward_runtime,
    'silhouette': round(ward_sil, 4),
    'calinski_harabasz': round(ward_ch, 2),
    'davies_bouldin': round(ward_db, 4)
})
print(f"Concluído em {ward_runtime}s. Silhouette: {ward_sil:.4f}")

#diagnóstico da família hierárquica: gerar dendrograma numa amostra menor
print("\n[Diagnóstico] A gerar dendrograma Ward (amostra N=1500)...")
np.random.seed(42)
dendro_indices = np.random.choice(X_euclid.shape[0], 1500, replace=False)
X_dendro = X_euclid[dendro_indices]
Z = linkage(X_dendro, method='ward')

#altura de corte aproximada para k_winner=8 clusters
n_d = X_dendro.shape[0]
cut_idx = n_d - k_winner
if cut_idx > 0:
    lower_h = Z[cut_idx - 1, 2]
    upper_h = Z[cut_idx, 2]
    cut_height = (lower_h + upper_h) / 2.0
else:
    cut_height = Z[0, 2]

fig_dir = 'graficos_relatorio'
os.makedirs(fig_dir, exist_ok=True)
plt.figure(figsize=(14, 6))
dendrogram(Z, no_labels=True, color_threshold=None)
#linha vertical para o corte aproximado
plt.axhline(
    y=cut_height,
    color='black',
    linestyle='--',
    linewidth=1.2,
    label=f'Corte aproximado (k={k_winner})'
)
plt.legend(loc='upper right')
plt.title('Dendrograma (Ward, N=1500)')
plt.xlabel('Observações')
plt.ylabel('Distância Ward')
plt.tight_layout()
dendro_path = os.path.join(fig_dir, 'dendrograma_ward_n1500.png')
plt.savefig(dendro_path, dpi=200)
plt.close()
print(f"[Diagnóstico] Dendrograma guardado em: {dendro_path}")

log_path = 'experiments.csv' if os.path.exists('experiments.csv') else '../experiments.csv'
df_results = pd.DataFrame(results)
df_results.to_csv(log_path, mode='a', header=False, index=False) 
print(f"\n[Sucesso] Resultados de Estabilidade e Ward adicionados a {log_path}!")

#guarda relatório de estabilidade separado para auditoria
stability_path = 'stability_report.csv' if os.path.exists('experiments.csv') else '../stability_report.csv'
df_ari = pd.DataFrame({
    'representation_id': [representation_id],
    'k': [k_winner],
    'seeds_tested': [','.join(map(str, seeds_to_test))],
    'ari_mean': [round(ari_mean, 4)],
    'ari_std': [round(ari_std, 4)],
    'n_pairs': [len(ari_values)]
})
df_ari.to_csv(stability_path, index=False)
print(f"[Sucesso] Relatório ARI guardado em: {stability_path}")