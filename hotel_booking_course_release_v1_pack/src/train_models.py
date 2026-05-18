import os
import sys
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from pipeline_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_BASE,
    NUMERIC_CALENDAR_RAW,
    apply_adr_iqr_filter,
    build_preprocessor,
    build_rep_id,
    group_rare_categories,
    log_row,
    min_cluster_pct,
    sample_rule_all,
    save_selected_k,
    select_k_from_sweep,
    short_rep_label,
)
sys.path.append(os.path.dirname(__file__))
from ikmeans import iKMeans  # noqa: E402

print("--------- MODELACAO, IK-MEANS E LOGGING ---------")

#carrega base clean e reaplica governanca de raros
df = pd.read_csv("../hotel_bookings_clean.csv")
df = group_rare_categories(df, CATEGORICAL_FEATURES)
print(f"[Info] Base clean: {df.shape[0]} reservas")

k_grid = [3, 4, 5, 6, 7, 8]
seed = 42
results = []
sweep_rows = []

#quatro representacoes: Std/Robust, SemADR/ComADR, sensibilidade calendario
variants = [
    {"scaler_name": "Std", "scaler": StandardScaler(), "adr_status": "no",
     "numeric": NUMERIC_BASE, "apply_adr_iqr": False, "calendar_mode": "month-OHE"},
    {"scaler_name": "Robust", "scaler": RobustScaler(), "adr_status": "no",
     "numeric": NUMERIC_BASE, "apply_adr_iqr": False, "calendar_mode": "month-OHE"},
    {"scaler_name": "Std", "scaler": StandardScaler(), "adr_status": "yes",
     "numeric": NUMERIC_BASE + ["adr"], "apply_adr_iqr": True, "calendar_mode": "month-OHE"},
    {"scaler_name": "Std", "scaler": StandardScaler(), "adr_status": "no",
     "numeric": NUMERIC_BASE + NUMERIC_CALENDAR_RAW, "apply_adr_iqr": False,
     "calendar_mode": "month-OHE+week-day-raw"},
]

#inicia o ciclo por representacao
for variant in variants:
    df_v = df.copy()
    iqr_params = ""
    if variant["apply_adr_iqr"]:
        #ComADR: remove linhas fora do IQR de ADR (filtro de preco para clustering)
        df_v, lo, hi, n_drop = apply_adr_iqr_filter(df_v)
        iqr_params = f"adr_iqr=[{lo:.1f},{hi:.1f}];n_drop={n_drop}"

    #constroi matriz X (escala numerica + one-hot categoricas)
    preprocessor = build_preprocessor(variant["numeric"], variant["scaler"]) #constroi o preprocessor
    X = preprocessor.fit_transform(df_v) #transforma os dados
    n_feat = X.shape[1] #numero de features
    rep_id = build_rep_id( #constroi o id da representacao
        variant["scaler_name"], variant["adr_status"], n_feat, variant["calendar_mode"]
    )
    rep = short_rep_label( #constroi o label da representacao
        variant["scaler_name"], variant["adr_status"], variant["calendar_mode"]
    )
    print(f"\n--> {rep} | {rep_id} | n={len(df_v)}")

    #iK-Means: descobre K automatico e corre K-Means com esses centroides
    print("[iK-Means]...", end=" ", flush=True)
    t0 = time.time()
    ikm = iKMeans(min_cluster_size=11, random_state=seed)
    ikm_centroids = ikm.find_anomalous_patterns(X) #encontra os centroides anomalos
    k_ikm = len(ikm_centroids) #numero de clusters descobertos
    labels_ikm = KMeans(n_clusters=k_ikm, init=ikm_centroids, n_init=1, random_state=seed).fit_predict(X) #treina o modelo e devolve as labels de cluster
    rt_ikm = round(time.time() - t0, 2) #tempo decorrido (wall time)
    sil_ikm = silhouette_score(X, labels_ikm, sample_size=min(30000, len(X)), random_state=seed) #silhouette do modelo
    ik_params = f"min_cluster_size=11;random_state={seed};k_auto={k_ikm}" #parametros do modelo
    if iqr_params: #se o iqr_params nao for vazio, adiciona os parametros do modelo
        ik_params += f";{iqr_params}"
    results.append(log_row( #acrescenta dict a lista results (depois vira experiments.csv)
        rep, rep_id, "iK-Means", k_ikm, seed, rt_ikm, #parametros da linha
        {"silhouette": round(sil_ikm, 4), 
         "calinski_harabasz": round(calinski_harabasz_score(X, labels_ikm), 2), #calinski_harabasz do modelo
         "davies_bouldin": round(davies_bouldin_score(X, labels_ikm), 4)}, #davies_bouldin do modelo
        len(df_v), sample_rule_all(), parameters=ik_params, #parametros do modelo
        diagnostics={"min_cluster_pct": min_cluster_pct(labels_ikm)}, #percentagem do cluster mais pequeno
    ))
    print(f"K={k_ikm} Sil={sil_ikm:.4f} ({rt_ikm}s)")

    km_params = f"n_init=10;init=k-means++;random_state={seed}"
    if iqr_params:
        km_params += f";{iqr_params}"

    #K-Means baseline: sweep de k no intervalo predefinido
    for k in k_grid:
        t0 = time.time()
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X) #treina o modelo e devolve as labels de cluster
        rt = round(time.time() - t0, 2) #tempo decorrido (wall time)
        sil = silhouette_score(X, labels, sample_size=min(30000, len(X)), random_state=seed) #silhouette do modelo
        mcp = min_cluster_pct(labels) #percentagem do cluster mais pequeno
        results.append(log_row( #acrescenta dict a lista results (depois vira experiments.csv)
            rep, rep_id, "K-Means", k, seed, rt,
            {"silhouette": round(sil, 4),
             "calinski_harabasz": round(calinski_harabasz_score(X, labels), 2), #calinski_harabasz do modelo
             "davies_bouldin": round(davies_bouldin_score(X, labels), 4)}, #davies_bouldin do modelo
            len(df_v), sample_rule_all(), parameters=km_params, #parametros do modelo
            diagnostics={"min_cluster_pct": mcp}, #percentagem do cluster mais pequeno
        ))
        sweep_rows.append({ #acrescenta dict a lista sweep_rows (depois vira k_sweep_summary.csv)
            "rep": rep, "rep_id": rep_id, "k": k,
            "silhouette": round(sil, 4), "davies_bouldin": round(davies_bouldin_score(X, labels), 4),
            "min_cluster_pct": mcp,
        })
        print(f"K={k} Sil={sil:.4f} min_cl%={mcp}% ({rt}s)") 

#guarda log e resumo do sweep
pd.DataFrame(results).to_csv("experiments.csv", index=False)
pd.DataFrame(sweep_rows).to_csv("k_sweep_summary.csv", index=False)
print("\n[Sucesso] experiments.csv, k_sweep_summary.csv")

#seleciona k final na rep principal (Std-SemADR): max Silhouette com min_cluster_pct>=1%
primary_rep = "Std-SemADR"
df_sweep = pd.DataFrame(sweep_rows) #cria o dataframe de resultados
sub_primary = df_sweep[df_sweep["rep"] == primary_rep] #filtra o dataframe de resultados para a representacao principal
k_sel = select_k_from_sweep(sub_primary, min_pct=1.0) #seleciona o k final
save_selected_k(k_sel)
#linhas k=8 e k=k_sel so para comparar metricas no texto (a escolha e so k_sel)
row_k8 = sub_primary[sub_primary["k"] == 8].iloc[0]
row_sel = sub_primary[sub_primary["k"] == k_sel].iloc[0]
rep_id_primary = sub_primary.iloc[0]["rep_id"]
sel_note = ( 
    f"k_selected={k_sel} (sil={row_sel['silhouette']}, min_cl%={row_sel['min_cluster_pct']}); " #silhouette e percentagem do cluster mais pequeno
    f"k=8 not final (sil={row_k8['silhouette']}, min_cl%={row_k8['min_cluster_pct']}); " #silhouette e percentagem do cluster mais pequeno
    f"rule=min_cluster_pct>=1%" #regra para selecionar o k
)
with open("selection_notes.txt", "w", encoding="utf-8") as f: 
    f.write(sel_note + "\n") #escreve a nota no ficheiro

#nota sobre Robust
rob_sub = df_sweep[(df_sweep["rep"] == "Robust-SemADR") & (df_sweep["k"] == k_sel)]
rob_note = "" 
if not rob_sub.empty: 
    rob_note = f"; Robust-SemADR k={k_sel} min_cl%={rob_sub.iloc[0]['min_cluster_pct']}% (sensitivity only)" #nota sobre Robust (so sensibilidade; perfil final usa Std)

#linha extra no experiments.csv a documentar a escolha do k final
results.append(log_row( 
    primary_rep, rep_id_primary, "K-selection", k_sel, seed, 0, #parametros da linha
    {"silhouette": row_sel["silhouette"], "calinski_harabasz": None, "davies_bouldin": None}, #so silhouette preenchida; CH/DB None (linha de decisao, nao treino)
    len(df), sample_rule_all(), #parametros do modelo
    parameters="rule: max silhouette with min_cluster_pct>=1%", #regra para selecionar o k
    diagnostics={"min_cluster_pct": row_sel["min_cluster_pct"]}, #percentagem do cluster mais pequeno
    notes=sel_note + rob_note, #notas textuais (escolha de k + sensibilidade Robust)
))
pd.DataFrame(results).to_csv("experiments.csv", index=False)
print(f"\n[K selecionado] {sel_note}")
