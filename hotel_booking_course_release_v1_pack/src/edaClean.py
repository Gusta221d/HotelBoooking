import hashlib
import os

import numpy as np
import pandas as pd

from pipeline_utils import (
    CATEGORICAL_FEATURES,
    OUTLIER_NUM_COLS,
    adr_iqr_mask,
    apply_adr_quality_filter,
    group_rare_categories,
)

print("---------  Clean dados --------- ")
#caminhos do dataset bruto e pasta de tabelas do relatorio
file_path = "../hotel_bookings_course_release_v1.csv"
tables_dir = os.path.join("..", "tables")
os.makedirs(tables_dir, exist_ok=True)

#calcula SHA-256 para provar que usamos o course release v1
with open(file_path, "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
print(f"\n[Integrity] hotel_bookings_course_release_v1 | SHA-256: {file_hash}")

#carrega os dados
df = pd.read_csv(file_path)
n_original = len(df)

#lista de variaveis a remover (leakage, IDs, pos-evento)
cols_to_drop = [
    "agent", "company", "country",
    "is_canceled", "reservation_status", "reservation_status_date",
    "assigned_room_type", "booking_changes", "days_in_waiting_list",
    "arrival_date_year",
]
print(f"\n[Leakage] A remover {len(cols_to_drop)} variáveis (outcome/ID)")
df_work = df.drop(columns=cols_to_drop, errors="ignore")

#relatorio de missingness por coluna (numerico vs categorico)
num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_work.select_dtypes(include=["str", "category"]).columns.tolist()
miss_rows = []
#ciclo para contar os nulos por coluna
for col in num_cols + cat_cols:
    n_miss = int(df_work[col].isna().sum())
    miss_rows.append({
        "column": col,
        "type": "numeric" if col in num_cols else "categorical",
        "n_missing": n_miss,
        "pct_missing": round(100 * n_miss / len(df_work), 4),
    })
df_miss = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False) #ordena por percentagem de nulos
miss_path = os.path.join(tables_dir, "missingness_report.csv") #guarda o relatorio de nulos
df_miss.to_csv(miss_path, index=False)

print(f"\n[Missingness] Guardado: {miss_path}") 
print(df_miss[df_miss["n_missing"] > 0].head(10).to_string(index=False) if (df_miss["n_missing"] > 0).any() else "  Sem missingness relevante além de children")

#imputa children: NaN significa 0 criancas
print("[Imputação] children: NaN -> 0 (missing = nenhuma criança)") 
df_work["children"] = df_work["children"].fillna(0) 

#resumo IQR de outliers
outlier_rows = []
for col in OUTLIER_NUM_COLS: #ciclo para calcular os outliers IQR
    if col not in df_work.columns:
        continue
    s = df_work[col].dropna() #remove os nulos
    q1, q3 = s.quantile(0.25), s.quantile(0.75) #calcula o quartil 1 e 3
    iqr = q3 - q1 #calcula o IQR
    lo = q1 - 1.5 * iqr #calcula o limite inferior
    hi = q3 + 1.5 * iqr #calcula o limite superior
    if col == "adr": #adr nao pode ser negativo no limite inferior
        lo = max(0.01, lo)
    n_out = int(((s < lo) | (s > hi)).sum())
    outlier_rows.append({ #adiciona os outliers ao dataframe
        "column": col,
        "iqr_lower": round(float(lo), 4),
        "iqr_upper": round(float(hi), 4),
        "n_iqr_outliers": n_out,
        "pct_iqr_outliers": round(100 * n_out / len(df_work), 2),
        "removed_in_base_clean": col == "adr" and False,  #IQR ADR nao remove na base SemADR
    })

#linha extra: quantas linhas o filtro IQR de ADR removeria-> ~5753
adr_mask, adr_lo, adr_hi = adr_iqr_mask(df_work["adr"])
n_adr_iqr = int(adr_mask.sum())
outlier_rows.append({ #adiciona os outliers ao dataframe
    "column": "adr_iqr_would_remove",
    "iqr_lower": round(float(adr_lo), 4),
    "iqr_upper": round(float(adr_hi), 4),
    "n_iqr_outliers": n_adr_iqr,
    "pct_iqr_outliers": round(100 * n_adr_iqr / len(df_work), 2),
    "removed_in_base_clean": False,
})

df_outliers = pd.DataFrame(outlier_rows) #cria o dataframe de outliers
out_path = os.path.join(tables_dir, "outlier_summary.csv") #guarda o relatorio de outliers
df_outliers.to_csv(out_path, index=False) #guarda o dataframe de outliers
print(f"\n[Outliers] Resumo IQR guardado: {out_path}")
print(df_outliers[["column", "n_iqr_outliers", "pct_iqr_outliers"]].to_string(index=False))

#base clean: so remove ADR invalido (<=0 ou >5000); IQR ADR fica para variante ComADR
print("\n[ADR] Base clean: remove so ADR invalido (<=0 ou >5000); IQR ADR so na variante ComADR") 
df_clean = apply_adr_quality_filter(df_work) #remove os ADR invalidos
n_quality_removed = n_original - len(df_clean) #calcula o numero de ADR invalidos removidos
print(f"-> Removidos por qualidade ADR: {n_quality_removed} ({100 * n_quality_removed / n_original:.2f}%)") #percentagem de ADR invalidos removidos
print(f"-> IQR ADR afetaria mais {n_adr_iqr} linhas (aplicado apenas em ComADR no train_models)") #numero de linhas afetadas pelo IQR ADR

#agrupa categorias raras (<1%) em Other
df_clean = group_rare_categories(df_clean, CATEGORICAL_FEATURES)
print(f"[Categóricas] Níveis com freq < {100*0.01:.0f}% agrupados em 'Other'")

#guarda dataset base para os scripts seguintes
output_path = "../hotel_bookings_clean.csv"
df_clean.to_csv(output_path, index=False)
print(f"\n[Sucesso] Dataset base guardado: {output_path}")
print(f"Dimensões: {df_clean.shape[0]} linhas, {df_clean.shape[1]} colunas.")
