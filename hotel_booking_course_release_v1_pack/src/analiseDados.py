import pandas as pd

#carrega o dataset da pasta
df = pd.read_csv('hotel_booking_course_release_v1_pack\hotel_bookings_course_release_v1.csv')

print("--- Dimensões do dataset ---")
print(f"Total de reservas: {df.shape[0]}")
print(f"Total de atributos: {df.shape[1]}\n")

#separa as variáveis entre numerica e de texto 
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

#conta os nulos
print("--- Valores Nulos Numéricos ---")
nulos_num = df[num_cols].isnull().sum()
print(nulos_num[nulos_num > 0] if nulos_num.sum() > 0 else "Sem nulo")

print("\n--- Valores Nulos de Categoria ---")
nulos_cat = df[cat_cols].isnull().sum()
print(nulos_cat[nulos_cat > 0] if nulos_cat.sum() > 0 else "Sem nulo")

#estatísticas para detetar anomalias
print("\n--- Estatisticas de Deteção de Anomalias ---")

#mostra o min e max para ver se existe valores impossíveis
print(df[num_cols].describe().T[['min', 'max', 'mean', '50%']])