import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

#carrega os dados
df = pd.read_csv('../hotel_bookings_course_release_v1.csv')

#filtra anomalias e remove leakage
df_clean = df[(df['adr'] > 0) & (df['adr'] < 5000)].copy()
#lista de variaveis a remover 
cols_drop = [
    'is_canceled', 'reservation_status', 'reservation_status_date',
    'agent', 'company', 'country', 'arrival_date_year'
]
df_model = df_clean.drop(columns=cols_drop, errors='ignore')

#separa features e criar o pipeline
num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
#define o pipeline para as variaveis num
pipe_num = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])
#define o pipeline para var texto
pipe_cat = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
#junta ambos os pipelines
preproc = ColumnTransformer(
    transformers=[
        ('num', pipe_num, num_cols),
        ('cat', pipe_cat, cat_cols)
    ])
print("A processar os dados")

X_final = preproc.fit_transform(df_model)
print(f"Dimensão dos dados processados: {X_final.shape}")

#treina os kmeans
k_values = range(3, 7)
resultados = []

#inicia o ciclo para o processo
print("\n--- K-Means a serem Treinados ---")
for k in k_values:
    print(f"A treinar modelo com k={k}...")
    
    #algoritmo de kmeans, garantindo a estabilidade e reprodução
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_final) #treina e atribui o num do cluster a cada line
    
    #calcula metricas
    sil = silhouette_score(X_final, labels, sample_size=30000, random_state=42)
    ch = calinski_harabasz_score(X_final, labels)
    
    resultados.append({
        'K': k,
        'Inércia': kmeans.inertia_,
        'Silhouette': sil,
        'Calinski-Harabasz': ch
    })

#mostra a tabela final de resultados
df_res = pd.DataFrame(resultados)
print("\n--- Resultados da Avaliação ---")
print(df_res.to_string(index=False))