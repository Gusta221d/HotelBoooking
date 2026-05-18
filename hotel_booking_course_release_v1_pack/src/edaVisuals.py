import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pipeline_utils import plots_dir

print("--------- EDA visual --------- ")

#carrega original e clean
try:
    df_original = pd.read_csv("../hotel_bookings_course_release_v1.csv")
    df_clean = pd.read_csv("../hotel_bookings_clean.csv")
except FileNotFoundError:
    print("ERRO: CSV em falta")
    sys.exit(1)

out_dir = plots_dir()
sns.set_theme(style="whitegrid")


def save_plot(name):
    #guarda figura na pasta unica graficos_relatorio/
    plt.savefig(os.path.join(out_dir, name), dpi=300)


#boxplot ADR antes vs depois do clean de qualidade
print("Boxplot ADR...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) #cria o grafico
sns.boxplot(y=df_original["adr"], ax=axes[0], color="salmon") #cria o boxplot
axes[0].set_title("ADR original") 
sns.boxplot(y=df_clean["adr"], ax=axes[1], color="lightgreen") #cria o boxplot
axes[1].set_title("ADR apos clean de qualidade") 
plt.tight_layout() 
save_plot("1_boxplot_adr.png")
plt.close()

#boxplots de variaveis numericas com potencial outlier
cols_plot = ["lead_time", "stays_in_week_nights", "adults", "total_of_special_requests"] #lista de variaveis que iram aparecer no boxplot
cols_plot = [c for c in cols_plot if c in df_clean.columns] #filtra as variaveis que estao no dataframe
print("Boxplots numericos...") 
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, cols_plot): #ciclo para criar o boxplot
    sns.boxplot(y=df_clean[col], ax=ax, color="skyblue") #cria o boxplot
    ax.set_title(col) #define o titulo do boxplot
plt.suptitle("Outliers numericos (base SemADR)")
plt.tight_layout()
save_plot("2_boxplots_numeric_outliers.png")
plt.close()

#histogramas de lead_time e ADR
fig, axes = plt.subplots(1, 2, figsize=(14, 5)) 
sns.histplot(df_clean["lead_time"], bins=40, kde=True, ax=axes[0], color="skyblue") 
axes[0].set_title("Lead time")
sns.histplot(df_clean["adr"], bins=40, kde=True, ax=axes[1], color="lightgreen")
axes[1].set_title("ADR")
plt.tight_layout() 
save_plot("3_histogramas.png")
plt.close()

print(f"[Sucesso] Figuras em {out_dir}/")
