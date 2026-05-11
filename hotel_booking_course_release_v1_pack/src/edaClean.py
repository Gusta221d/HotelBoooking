import pandas as pd 
import numpy as np
import hashlib
import os 

print("---------  Clean dados --------- ")
file_path = '../hotel_bookings_course_release_v1.csv'

#calculo para o hash SHA-256
with open(file_path, "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
print(f"\n[Integrity] Dataset SHA-256 Hash: {file_hash}")

df = pd.read_csv(file_path)
n_original = len(df)


#lista de colunas a removes
cols_to_drop = [
    'agent', 'company', 'country',
    'is_canceled', 'reservation_status', 'reservation_status_date',
    'assigned_room_type', 'booking_changes', 'days_in_waiting_list',
    'arrival_date_year' 
]

print(f"\n[Leakage] A remover {len(cols_to_drop)} variáveis viciadas/pós-evento...")
df_leaked = df.drop(columns=cols_to_drop, errors='ignore') #remove as colunas especificadas do dataframe

#ausencia de childrens na reserva
print("[Imputação] A preencher nulos em 'children' com 0 (Justificativa: missing=none).")
df_leaked['children'] = df_leaked['children'].fillna(0)  #substituição dos valores nulos por 0

#identifica e remove registos com ADR anomalos com o intervalo interquartil
print("\n[Outliers] A aplicar regra IQR para a variável ADR...")
Q1 = df_leaked['adr'].quantile(0.25) #calcula o primeiro quartil-25
Q3 = df_leaked['adr'].quantile(0.75) #calcula o terceiro quartil-75
IQR = Q3 - Q1  #calculo da amplitude interquartil
#limite inferior e superioe
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#limite inferior é de pelo menso 0.01
final_lower = max(0.01, lower_bound)

#identifica registos fora dos limites
mask_outliers = (df_leaked['adr'] < final_lower) | (df_leaked['adr'] > upper_bound)

df_clean = df_leaked[~mask_outliers].copy() #novo dataframe sem os outliers identificados
n_removed = n_original - len(df_clean) #quantas linhas foram removidas

print(f"-> Limites IQR: {final_lower:.2f}€ - {upper_bound:.2f}€")
print(f"-> Reservas removidas: {n_removed} ({ (n_removed/n_original)*100:.2f}%)")

#verifica se as variaveis permanecem no dataset
context_vars = ['hotel', 'arrival_date_month', 'distribution_channel', 'market_segment', 'customer_type']
missing_context = [v for v in context_vars if v not in df_clean.columns] #lista variaveis que nao foram encontradas

if not missing_context:
    print("\n[Contexto] Variáveis confirmadas")
else:
    print(f"\n[AVISO] Faltam variáveis: {missing_context}")

output_path = '../hotel_bookings_clean.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n[Sucesso] Dataset guardado em: {output_path}")
print(f"Dimensões finais: {df_clean.shape[0]} linhas, {df_clean.shape[1]} colunas.")