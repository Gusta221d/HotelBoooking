import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append('src')
try:
    from ikmeans import iKMeans
except ImportError:
    print("ERRO: Não existe ficheiro ikmeans.py na pasta src")
    sys.exit(1)

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

print("--------- MODELAÇÃO, IK-MEANS E LOGGING ---------")

file_path = '../hotel_bookings_clean.csv'
try:
    df = pd.read_csv(file_path)
    print(f"[Info] Dados carregados com sucesso! ({df.shape[0]} reservas)")
except FileNotFoundError:
    print(f"ERRO: Não encontrei o {file_path}")
    sys.exit(1)

categorical_features = ['hotel', 'arrival_date_month', 'meal', 'market_segment', 
                        'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']
numeric_features_base = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 
                         'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 
                         'babies', 'is_repeated_guest', 'previous_cancellations', 
                         'previous_bookings_not_canceled', 'required_car_parking_spaces', 
                         'total_of_special_requests']

#variantes para testar impacto do ADR e sensibilidade de scaling
variants = [
    {
        'representation_id': 'EUCLID-ComADR-Standard',
        'numeric_features': numeric_features_base + ['adr'],
        'scaler': StandardScaler()
    },
    {
        'representation_id': 'EUCLID-SemADR-Standard',
        'numeric_features': numeric_features_base,
        'scaler': StandardScaler()
    },
    {
        'representation_id': 'EUCLID-SemADR-Robust',
        'numeric_features': numeric_features_base,
        'scaler': RobustScaler()
    }
]

k_grid = [3, 4, 5, 6, 7, 8] #valores de K para o protocolo experimental
seed = 42 #seed fixa
results = [] #lista para guardar as metricas

print("\nA começar Protocolo Experimental\n")

for variant in variants:
    representation_id = variant['representation_id']
    numeric_features = variant['numeric_features']
    scaler = variant['scaler']
    print(f"--> A construir matriz geométrica: {representation_id}")
    
    #comeco do pipeline de preprocessor para euclidian
    preprocessor = ColumnTransformer(
        transformers=[
            #aplica escala normalizada aos valores numericos & codifica as categorias
            ('num', scaler, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    #dataframe na matriz numerica para EUCLID
    X_euclid = preprocessor.fit_transform(df)

    #ikmeans para encontrar centroides
    print("A executar iK-Means para encontrar seeds")
    ikm = iKMeans(min_cluster_size=11, random_state=seed)
    ikm_centroids = ikm.find_anomalous_patterns(X_euclid) #procura patterns anómalos nos dados
    k_ikm = len(ikm_centroids) #guarda o nº de clusters
    print(f"[iK-Means] Encontrou {k_ikm} clusters válidos!")

    #loop para modelos kmeans com K valores distintos
    for k in k_grid:
        print(f"    A treinar K-Means (K={k})...", end=" ", flush=True)
        start_time = time.time()
        
        #utilizacao de centroides do ikmeans se K coincidir
        if k == k_ikm:
            kmeans = KMeans(n_clusters=k, init=ikm_centroids, n_init=1, random_state=seed)
            algorithm_name = 'K-Means (init=iK-Means)'
        # senão inicialização aleatória
        else:
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            algorithm_name = 'K-Means Baseline'

        #atribui clusters aos dados  
        labels = kmeans.fit_predict(X_euclid)
        runtime = round(time.time() - start_time, 2)
        
        #calculo dos indices
        silhouette = silhouette_score(X_euclid, labels, sample_size=30000, random_state=seed) 
        ch_score = calinski_harabasz_score(X_euclid, labels)
        db_score = davies_bouldin_score(X_euclid, labels)
        
        print(f"Concluído em {runtime}s")
        
        #guarda os resultados
        results.append({
            'representation_id': representation_id,
            'algorithm': algorithm_name,
            'k': k,
            'seed': seed,
            'runtime_sec': runtime,
            'silhouette': round(silhouette, 4),
            'calinski_harabasz': round(ch_score, 2),
            'davies_bouldin': round(db_score, 4)
        })
    print("-" * 30)

#exporta os resultados para ficheiros csv
print("\n A GRAVAR LOG ")
df_results = pd.DataFrame(results)#dados para dataframe 
output_csv = 'experiments.csv'
df_results.to_csv(output_csv, index=False)

print(f"[Sucesso] Experiências gravadas no ficheiro: {output_csv}")
print("\nTabela Resumo:")
print(df_results.sort_values(by='silhouette', ascending=False).head(5)[['representation_id', 'algorithm', 'k', 'silhouette', 'davies_bouldin']].to_string(index=False))