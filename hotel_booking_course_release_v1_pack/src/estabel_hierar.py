import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_BASE,
    build_preprocessor,
    build_rep_id,
    plots_dir,
    group_rare_categories,
    load_selected_k,
    log_row,
    profile_agreement,
    sample_rule_subsample,
    short_rep_label,
)

sys.path.append(os.path.dirname(__file__)) 
from ikmeans import iKMeans 

print("--------- ESTABILIDADE, PERFIS-SEED E WARD/IK 20k ---------")

#carrega dados e representacao principal Std-SemADR
df = pd.read_csv("../hotel_bookings_clean.csv")
df = group_rare_categories(df, CATEGORICAL_FEATURES)
k_sel = load_selected_k("selected_k.txt", default=4) #carrega o k escolhido
rep = short_rep_label("Std", "no") #define a representacao
preprocessor = build_preprocessor(NUMERIC_BASE, StandardScaler())
X = preprocessor.fit_transform(df)
rep_id = build_rep_id("Std", "no", X.shape[1], "month-OHE")
print(f"[K] k={k_sel}, rep={rep}")

seeds_10 = [42, 67, 123, 2026, 71, 7, 99, 314, 555, 1001] #lista de seeds
labels_by_seed = {} #dicionario para armazenar as labels de cluster por seed
sil_per_seed = [] #silhouette por seed
profile_agree = [] #perfil por seed

#estabilidade: 10 seeds com o k escolhido
print("\n[Estabilidade] 10 seeds...")
ref_seed = 42
for s in seeds_10:
    labels = KMeans(n_clusters=k_sel, random_state=s, n_init=10).fit_predict(X) #treina o modelo e devolve as labels de cluster
    labels_by_seed[s] = labels 
    sil = silhouette_score(X, labels, sample_size=min(30000, len(X)), random_state=s) #calcula o silhouette
    sil_per_seed.append(round(sil, 4))
    if s != ref_seed:
        #compara perfis (canal, segmento, lead_time) vs seed de referencia
        profile_agree.append(profile_agreement(df, labels_by_seed[ref_seed], labels)) #concordancia de perfis entre seed 42 e seed s
    print(f"seed {s}: Sil={sil:.4f}")

#ARI entre todos os pares de seeds
ari_vals = [
    adjusted_rand_score(labels_by_seed[a], labels_by_seed[b]) #calcula o ARI
    for a, b in itertools.combinations(seeds_10, 2) #ciclo para calcular o ARI
]

#amostra fixa 20k para comparar familias de clustering
SAMPLE_N, SAMPLE_SEED = 20000, 42
np.random.seed(SAMPLE_SEED)
sample_idx = np.random.choice(X.shape[0], SAMPLE_N, replace=False)
np.savetxt("sample_indices_20k_seed42.txt", sample_idx, fmt="%d")  #indices para reproducao
print(f"[Sample] indices guardados: sample_indices_20k_seed42.txt (n={SAMPLE_N}, seed={SAMPLE_SEED})")

X_sample = X[sample_idx]
sr = sample_rule_subsample(SAMPLE_N, SAMPLE_SEED)
rep_id_20k = rep_id + f"-sample{SAMPLE_N}"
note_20k = f"same subsample n={SAMPLE_N} seed={SAMPLE_SEED}; same k={k_sel} for Sil/DB comparability"

comparison = []

#K-Means na amostra 20k
t0 = time.time()
lab_km = KMeans(n_clusters=k_sel, random_state=SAMPLE_SEED, n_init=10).fit_predict(X_sample)
km_rt = round(time.time() - t0, 2)
comparison.append(log_row(
    rep, rep_id_20k, "K-Means", k_sel, SAMPLE_SEED, km_rt,
    {"silhouette": round(silhouette_score(X_sample, lab_km, random_state=SAMPLE_SEED), 4),
     "calinski_harabasz": round(calinski_harabasz_score(X_sample, lab_km), 2),
     "davies_bouldin": round(davies_bouldin_score(X_sample, lab_km), 4)},
    SAMPLE_N, sr, parameters=f"n_init=10;seed={SAMPLE_SEED}",
    diagnostics={"min_cluster_pct": round(100 * np.bincount(lab_km).min() / len(lab_km), 2)},
    notes=note_20k,
))

#Ward hierarquico na mesma amostra 20k
t0 = time.time()
lab_w = AgglomerativeClustering(n_clusters=k_sel, linkage="ward").fit_predict(X_sample)
w_rt = round(time.time() - t0, 2)
comparison.append(log_row(
    rep, rep_id_20k, "Ward", k_sel, SAMPLE_SEED, w_rt,
    {"silhouette": round(silhouette_score(X_sample, lab_w, random_state=SAMPLE_SEED), 4),
     "calinski_harabasz": round(calinski_harabasz_score(X_sample, lab_w), 2),
     "davies_bouldin": round(davies_bouldin_score(X_sample, lab_w), 4)},
    SAMPLE_N, sr, parameters=f"linkage=ward;seed={SAMPLE_SEED}",
    diagnostics={"min_cluster_pct": round(100 * np.bincount(lab_w).min() / len(lab_w), 2)},
    notes=note_20k,
))

#iK-Means na amostra 20k (metricas sempre com k_sel para comparacao justa)
print("[iK-Means 20k]...", end=" ", flush=True)
t0 = time.time() #inicia o cronometro
ikm = iKMeans(min_cluster_size=11, random_state=SAMPLE_SEED) #cria o objeto iKMeans
ik_centroids = ikm.find_anomalous_patterns(X_sample) #encontra os centroides anomalos
k_ik_disc = len(ik_centroids) #numero de clusters descobertos
if k_ik_disc == k_sel: #se o numero de clusters descobertos for igual ao k escolhido
    lab_ik = KMeans(n_clusters=k_sel, init=ik_centroids, n_init=1, random_state=SAMPLE_SEED).fit_predict(X_sample) #treina o modelo e devolve as labels de cluster
    ik_params = f"min_cluster_size=11;ik_discovered_k={k_ik_disc};k_eval={k_sel};init=iK-centroids" #parametros do modelo
    ik_note = note_20k + "; iK init at k_sel" #nota do modelo
else: #se iK descobrir outro K, avaliamos Sil/DB com k_sel e init k-means++
    lab_ik = KMeans(n_clusters=k_sel, random_state=SAMPLE_SEED, n_init=10).fit_predict(X_sample) #treina o modelo e devolve as labels de cluster com k-means++
    ik_params = f"min_cluster_size=11;ik_discovered_k={k_ik_disc};k_eval={k_sel};init=k-means++" #parametros do modelo
    ik_note = note_20k + f"; iK discovered k={k_ik_disc}, metrics at k={k_sel} for fair comparison" #nota do modelo
ik_rt = round(time.time() - t0, 2) #calcula o tempo de execucao
comparison.append(log_row( #acrescenta dict a lista comparison (depois concat a experiments.csv)
    rep, rep_id_20k, "iK-Means", k_sel, SAMPLE_SEED, ik_rt, #parametros da linha
    {"silhouette": round(silhouette_score(X_sample, lab_ik, random_state=SAMPLE_SEED), 4), #silhouette do modelo
     "calinski_harabasz": round(calinski_harabasz_score(X_sample, lab_ik), 2), #calinski_harabasz do modelo
     "davies_bouldin": round(davies_bouldin_score(X_sample, lab_ik), 4)}, #davies_bouldin do modelo
    SAMPLE_N, sr, parameters=ik_params,
    diagnostics={"min_cluster_pct": round(100 * np.bincount(lab_ik).min() / len(lab_ik), 2)}, #percentagem do cluster mais pequeno
    notes=ik_note,
))
print(f"K_disc={k_ik_disc} K_eval={k_sel} Sil={comparison[-1]['silhouette']}")

print(f"\n[20k] K-Means={comparison[0]['silhouette']} Ward={comparison[1]['silhouette']} "f"iK-Means={comparison[2]['silhouette']}")#silhouette do modelo

#dendrograma Ward subamostra de 1500 
fig_dir = plots_dir()
np.random.seed(SAMPLE_SEED)
dend_idx = np.random.choice(X.shape[0], 1500, replace=False)
Z = linkage(X[dend_idx], method="ward")
cut_idx = max(0, len(dend_idx) - k_sel)
cut_h = (Z[cut_idx - 1, 2] + Z[cut_idx, 2]) / 2 if cut_idx > 0 else Z[0, 2]
plt.figure(figsize=(14, 6))
dendrogram(Z, no_labels=True)
plt.axhline(y=cut_h, color="k", linestyle="--", label=f"k={k_sel}")
plt.legend()
plt.title(f"Dendrograma Ward (N=1500, k={k_sel})")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "dendrograma_ward_n1500.png"), dpi=200)
plt.close()

#junta linhas 20k ao experiments.csv
experiments = pd.read_csv("experiments.csv")
experiments = pd.concat([experiments, pd.DataFrame(comparison)], ignore_index=True) 
experiments.to_csv("experiments.csv", index=False) 

#relatorios de estabilidade (ARI + concordancia de perfil)
pd.DataFrame([{
    "rep": rep, "k": k_sel, "n_seeds": 10,
    "ari_mean": round(float(np.mean(ari_vals)), 4),
    "ari_std": round(float(np.std(ari_vals)), 4),
    "silhouette_mean": round(float(np.mean(sil_per_seed)), 4),
    "silhouette_std": round(float(np.std(sil_per_seed)), 4),
    "profile_agreement_mean": round(float(np.mean(profile_agree)), 4),
    "profile_agreement_std": round(float(np.std(profile_agree)), 4), 
    "interpretation": "moderately unstable" if np.mean(ari_vals) < 0.7 else "more stable",
    "sample_indices_file": "sample_indices_20k_seed42.txt",
}]).to_csv("stability_report.csv", index=False) 

pd.DataFrame([{
    "metric": "profile_agreement",
    "description": "channel+segment+lead_time aligned vs seed 42",
    "value_mean": round(float(np.mean(profile_agree)), 4),
}]).to_csv("profile_stability_report.csv", index=False)

print(f"\n[Estabilidade] ARI={np.mean(ari_vals):.4f} | profile_agree={np.mean(profile_agree):.4f}")
print("[Sucesso] +3 linhas 20k em experiments.csv")
