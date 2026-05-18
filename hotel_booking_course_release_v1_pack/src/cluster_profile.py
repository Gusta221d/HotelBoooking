import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

from pipeline_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_BASE,
    apply_adr_iqr_filter,
    build_aligned_profile_comparison,
    build_preprocessor,
    group_rare_categories,
    load_selected_k,
    min_cluster_pct,
    profile_agreement,
)

print("--------- PERFIS E SENSIBILIDADE (Std/Robust, SemADR/ComADR) ---------")

#carrega base clean e k escolhido no train_models
df = pd.read_csv("../hotel_bookings_clean.csv")
df = group_rare_categories(df, CATEGORICAL_FEATURES)
k_sel = load_selected_k("selected_k.txt", default=4)
seed = 42
print(f"[K] k={k_sel}")


def profile_table(df_in, labels, rep_label):
    #tabela descritiva por cluster (modais e medianas)
    d = df_in.copy()
    d["cluster"] = labels
    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    rows = []
    n = len(d)
    for cid, g in d.groupby("cluster"):
        rows.append({
            "representation": rep_label,
            "cluster": int(cid),
            "n_bookings": len(g),
            "pct_bookings": round(100 * len(g) / n, 2),
            "lead_time_median": round(float(g["lead_time"].median()), 1),
            "total_nights_median": round(float(g["total_nights"].median()), 1),
            "distribution_channel_mode": g["distribution_channel"].mode().iat[0],
            "market_segment_mode": g["market_segment"].mode().iat[0],
            "deposit_type_mode": g["deposit_type"].mode().iat[0],
        })
    return pd.DataFrame(rows).sort_values("cluster")


def fit_labels(df_v, numeric_cols, scaler, k):
    #treina K-Means e devolve etiquetas de cluster
    X = build_preprocessor(numeric_cols, scaler).fit_transform(df_v)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)


#perfil final: Std-SemADR em todo o dataset
labels_main = fit_labels(df, NUMERIC_BASE, StandardScaler(), k_sel)
tbl_main = profile_table(df, labels_main, "Std-SemADR")
tbl_main.to_csv(f"cluster_profile_k{k_sel}.csv", index=False)
print(f"\n--- Perfil final Std-SemADR (min_cl%={min_cluster_pct(labels_main)}%) ---")
print(tbl_main.to_string(index=False))

#sensibilidade Std vs Robust (clusters alinhados por Hungarian)
labels_std = fit_labels(df, NUMERIC_BASE, StandardScaler(), k_sel)
labels_rob = fit_labels(df, NUMERIC_BASE, RobustScaler(), k_sel)
agr_scaler = profile_agreement(df, labels_std, labels_rob)
cmp_scaler = build_aligned_profile_comparison(
    df, labels_std, labels_rob, "Std-SemADR", "Robust-SemADR"
)
cmp_scaler["profile_agreement"] = agr_scaler
cmp_scaler.to_csv("cluster_profile_scaler_sensitivity.csv", index=False)
print(f"\n[Std vs Robust] profile_agreement={agr_scaler} (clusters alinhados)")
print(cmp_scaler[["cluster_ref", "Std-SemADR_channel", "Robust-SemADR_channel",
                  "channel_match", "segment_match"]].to_string(index=False))

#sensibilidade SemADR vs ComADR (mesmo subset apos IQR de ADR)
df_com = apply_adr_iqr_filter(df)[0] 
labels_sem = fit_labels(df_com, NUMERIC_BASE, StandardScaler(), k_sel) #K-Means sem ADR nas features
labels_com = fit_labels(df_com, NUMERIC_BASE + ["adr"], StandardScaler(), k_sel) #K-Means com ADR nas features
agr_adr = profile_agreement(df_com, labels_sem, labels_com) #concordancia entre as duas segmentacoes (alinhamento Hungarian)
cmp_adr = build_aligned_profile_comparison( #tabela por par de clusters alinhados
    df_com, labels_sem, labels_com, "Std-SemADR", "Std-ComADR"
)
cmp_adr["profile_agreement"] = agr_adr #mesmo escalar agr_adr em todas as linhas (metadado do CSV)
cmp_adr.to_csv("cluster_profile_adr_sensitivity.csv", index=False)

#perfis individuais para o relatorio
pd.concat([
    profile_table(df_com, labels_sem, "Std-SemADR"), 
    profile_table(df_com, labels_com, "Std-ComADR"),
], ignore_index=True).to_csv("cluster_profile_sensitivity.csv", index=False) 

print(f"\n[SemADR vs ComADR] linhas partilhadas={len(df_com)} | profile_agreement={agr_adr}")
print(cmp_adr[["cluster_ref", "Std-SemADR_channel", "Std-ComADR_channel","channel_match", "segment_match"]].to_string(index=False))

#avisos automaticos (modais duplicados, micro-clusters no Robust)
flags = []
pairs = tbl_main[["cluster", "distribution_channel_mode", "market_segment_mode"]].values
for i in range(len(pairs)): #ciclo para comparar os perfis
    for j in range(i + 1, len(pairs)): 
        if pairs[i][1] == pairs[j][1] and pairs[i][2] == pairs[j][2]: #se o canal e o segmento sao iguais
            flags.append({ #acrescenta dict a lista flags (depois vira profile_quality_flags.csv)
                "flag": "duplicate_modal_channel_segment",
                "cluster_a": int(pairs[i][0]),
                "cluster_b": int(pairs[j][0]),
                "channel": pairs[i][1],
                "segment": pairs[i][2],
                "note": "interpret clusters with caution; modals not unique",
            })
sweep = pd.read_csv("k_sweep_summary.csv") #carrega o resumo de k
rob_sub = sweep[(sweep["rep"] == "Robust-SemADR") & (sweep["k"] == k_sel)] #filtra o resumo de k para o robust
if not rob_sub.empty and rob_sub.iloc[0]["min_cluster_pct"] < 1.0: #se o minimo de cluster percentual e menor que 1%
    flags.append({
        "flag": "robust_micro_cluster",
        "cluster_a": None,
        "cluster_b": None,
        "channel": "",
        "segment": "",
        "note": f"Robust-SemADR k={k_sel} min_cluster_pct={rob_sub.iloc[0]['min_cluster_pct']}% (sensitivity only; final profile uses Std)",
    })
pd.DataFrame(flags).to_csv("profile_quality_flags.csv", index=False) 
print(f"\n[Qualidade] {len(flags)} avisos-> profile_quality_flags.csv") 
print("\n[Sucesso] cluster_profile_k{k_sel}.csv, *_sensitivity.csv".format(k_sel=k_sel))
