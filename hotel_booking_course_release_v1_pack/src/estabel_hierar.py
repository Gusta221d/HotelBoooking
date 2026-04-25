import pandas as pd
import numpy as np
import time
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

#teste com5 seeds
print("\n[Sensibilidade] A iniciar teste de estabilidade com 5 seeds para K=8...")
seeds_to_test = [42, 67, 123, 2026, 71] #lista de seeds aleatórias

#inicio do loop para cada seed
for s in seeds_to_test:
    print(f"-> A correr seed {s}...", end=" ", flush=True)
    start_time = time.time()
    kmeans = KMeans(n_clusters=k_winner, random_state=s, n_init=10) #inicia o kmeans com a seed atual
    labels = kmeans.fit_predict(X_euclid)
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

log_path = 'experiments.csv' if os.path.exists('experiments.csv') else '../experiments.csv'
df_results = pd.DataFrame(results)
df_results.to_csv(log_path, mode='a', header=False, index=False) 
print(f"\n[Sucesso] Resultados de Estabilidade e Ward adicionados a {log_path}!")