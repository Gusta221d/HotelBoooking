import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#carrega os dados
df = pd.read_csv('hotel_booking_course_release_v1_pack\hotel_bookings_course_release_v1.csv')

#filtra anomalias 
df_clean = df[(df['adr'] > 0) & (df['adr'] < 5000)].copy()

#remove variaveis
cols_drop = [
    'is_canceled', 'reservation_status', 'reservation_status_date', 
    'agent', 'company',
    'country', 'arrival_date_year' 
]
df_model = df_clean.drop(columns=cols_drop, errors='ignore')

#separa por features, extraindo lista consoante o tipo de dados
num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()

#criar o pipeline
#dados num
pipe_num = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])
#dados em texto
pipe_cat = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preproc = ColumnTransformer(
    transformers=[
        ('num', pipe_num, num_cols),
        ('cat', pipe_cat, cat_cols)
    ])

X_final = preproc.fit_transform(df_model)
print("Pipeline concluída")
print(f"Dimensão final para o modelo K-Means: {X_final.shape}")