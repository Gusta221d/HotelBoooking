import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

print("--------- EDA --------- ")

try:
    df_original = pd.read_csv('../hotel_bookings_course_release_v1.csv')
    df_clean = pd.read_csv('../hotel_bookings_clean.csv') 
    print("[Sucesso] Ficheiros encontrados e carregados")
except FileNotFoundError:
    print("ERRO: Não foram encontrados os ficheiros")
    sys.exit(1)

output_dir = 'graficos_relatorio'
#verifica se a pasta não existe, caso contrario vai a criar
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sns.set_theme(style="whitegrid")

#boxplot
print("A gerar Boxplot do Preço (ADR)...")
plt.figure(figsize=(12, 6)) #dimensão total da figura- largura 12 e altura 6

#cria 1 subgrafico-1linha 2 colunas na 1 posicao
plt.subplot(1, 2, 1)
sns.boxplot(y=df_original['adr'], color='salmon')#cria o boxplot do adr original 
plt.title('ADR Original - 5400€ Outlier)')
plt.ylabel('Average Daily Rate (ADR)') #eixo do y

#cria 2 subgrafico-1linha 2 colunas na 2 posicao
plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['adr'], color='lightgreen')#cria o boxplot do adr o clean 
plt.title('ADR Cleaned (After IQR Rule)')
plt.ylabel('Average Daily Rate (ADR)')

plt.tight_layout() #evita sobreposicoes
plt.savefig(f'{output_dir}/1_boxplot_adr_comparacao.png', dpi=300)
plt.close()

#histograma
print("A gerar Histogramas de Distribuição")
fig, axes = plt.subplots(1, 2, figsize=(14, 5)) #cria uma figura com 2 subgraficos lado a lado

#histograma com antecedencia da reserva
sns.histplot(df_clean['lead_time'], bins=40, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Lead Time Distribution')
axes[0].set_xlabel('Days of Lead Time')
axes[0].set_ylabel('Frequency')

#histograma do aadr apos retirar os outliers
sns.histplot(df_clean['adr'], bins=40, kde=True, ax=axes[1], color='lightgreen')
axes[1].set_title('ADR Distribution')
axes[1].set_xlabel('Average Daily Rate (€)') 
axes[1].set_ylabel('Frequency')

plt.tight_layout()#evita sobreposicoes
plt.savefig(f'{output_dir}/2_histogramas_distribuicao.png', dpi=300) 
plt.close()

print(f"\n[Sucesso] Os gráficos foram gerados e guardados na pasta '{output_dir}'!")