import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

#carrega e limpa dados
df = pd.read_csv('../hotel_bookings_course_release_v1.csv')
df_clean = df[(df['adr'] > 0) & (df['adr'] < 5000)].copy()
#lista de variaveis a remover 
cols_drop = [
    'is_canceled', 'reservation_status', 'reservation_status_date',
    'agent', 'company', 'country', 'arrival_date_year'
]
df_model = df_clean.drop(columns=cols_drop, errors='ignore')

#pipeline de valores num e de texto
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
    transformers=[('num', pipe_num, num_cols), ('cat', pipe_cat, cat_cols)]
)

X_final = preproc.fit_transform(df_model)

#treina o modelo final -k3
print("A treinar K-Means em K=3")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_final)

#cria perfil do cluster
print("\n--- Perfil do Cluster ---")
#define o conjunto de variaveis, agrupa os dados pelo cluster e calcula a media
vars_chave = ['lead_time', 'adr', 'stays_in_week_nights', 'stays_in_weekend_nights', 'adults', 'children']
perfil = df_clean.groupby('Cluster')[vars_chave].mean().round(2)

#obtem se o numero de reservas 
perfil['Total_Reservas'] = df_clean.groupby('Cluster').size()
print(perfil.to_string())

#distribuicao de variaveis texto 
print("\n--- Distribuição por Hoteis ---")
print(pd.crosstab(df_clean['Cluster'], df_clean['hotel'], normalize='index').round(2) * 100)