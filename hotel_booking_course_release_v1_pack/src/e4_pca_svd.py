import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline_utils import plots_dir

#carrega base clean
try:
    df = pd.read_csv("../hotel_bookings_clean.csv")
except FileNotFoundError:
    df = pd.read_csv("hotel_bookings_clean.csv")

print(f"[Info] Dados carregados: {df.shape[0]} reservas, {df.shape[1]} atributos")

#features para E4 (inclui semana/dia brutos; core principal usa so mes em OHE)
categorical_features = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type",
]
numeric_features = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests",
]

#pipeline de pre-processamento (escala + one-hot)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

X_euclid = preprocessor.fit_transform(df)
print(f"[Info] Dimensão da representação original (EUCLID-SemADR): {X_euclid.shape}")

fig_dir = plots_dir()

#scree plot: variancia explicada para escolher n. de componentes PCA
print("\n[PCA] A calcular variância explicada (scree plot)...")

pca_full = PCA(random_state=42)
pca_full.fit(X_euclid)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

#limiares 80/90/95% de variancia acumulada
thresholds = [0.80, 0.90, 0.95]
n_components_per_threshold = {t: int(np.searchsorted(cumvar, t) + 1) for t in thresholds}
for t, n in n_components_per_threshold.items():
    print(f"   -> {int(t*100)}% variância explicada com {n} componentes")

#regra: 90% de variancia explicada
n_components_chosen = n_components_per_threshold[0.90]
print(f"\n[PCA] Dimensionalidade escolhida: {n_components_chosen} componentes (>=90% variancia)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#variancia por componente (ate 60 ou todas se houver menos features)
n_feats = len(pca_full.explained_variance_ratio_)
n_show = min(60, n_feats)
axes[0].bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show], color="steelblue", alpha=0.7)
axes[0].set_title(f"Variancia Explicada por Componente (top {n_show})")
axes[0].set_xlabel("Componente Principal")
axes[0].set_ylabel("Proporcao de Variancia Explicada")
axes[0].set_xlim(0, n_show + 1)

#variancia acumulada com linhas de referencia
axes[1].plot(range(1, len(cumvar) + 1), cumvar, color="steelblue", linewidth=1.5)
for t, n in n_components_per_threshold.items():
    axes[1].axhline(y=t, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axvline(x=n, color="gray", linestyle="--", linewidth=0.8)
    axes[1].annotate(
        f"{int(t*100)}% ({n} comp.)",
        xy=(n, t), xytext=(n + 5, t - 0.03),
        fontsize=8, color="dimgray",
    )
axes[1].set_title("Variância Acumulada Explicada (PCA)")
axes[1].set_xlabel("Número de Componentes")
axes[1].set_ylabel("Variância Acumulada")
axes[1].set_xlim(0, X_euclid.shape[1])
axes[1].set_ylim(0.5, 1.01)

plt.tight_layout()
scree_path = os.path.join(fig_dir, "e4_scree_plot.png")
plt.savefig(scree_path, dpi=300)
plt.close()
print(f"[Figura] Scree plot guardado em: {scree_path}")

#aplica PCA com dimensionalidade escolhida
print(f"\n[PCA] A reduzir de {X_euclid.shape[1]} para {n_components_chosen} dimensões...")
pca = PCA(n_components=n_components_chosen, random_state=42)
X_pca = pca.fit_transform(X_euclid)
print(f"[Info] Dimensão reduzida (PCA): {X_pca.shape}")
print(f"[Info] Variância total explicada: {pca.explained_variance_ratio_.sum():.4f}")

k_grid = [3, 4, 5, 6, 7, 8]
seeds = [42, 67, 123, 2026, 71]
results = []

representations = {
    "EUCLID-SemADR-Original": X_euclid,
    f"EUCLID-SemADR-PCA{n_components_chosen}": X_pca,
}

print("\n[Protocolo] A correr K-Means em ambas as representações (original vs PCA)...")

#compara K-Means no espaco original vs espaco PCA
for rep_name, X in representations.items():
    print(f"\n  Representação: {rep_name} | shape={X.shape}")
    for k in k_grid:
        sil_vals, ch_vals, db_vals, rt_vals = [], [], [], []
        for s in seeds:
            start = time.time()
            km = KMeans(n_clusters=k, random_state=s, n_init=10)
            labels = km.fit_predict(X)
            rt = round(time.time() - start, 2)

            sil = silhouette_score(X, labels, sample_size=30000, random_state=s)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)

            sil_vals.append(sil)
            ch_vals.append(ch)
            db_vals.append(db)
            rt_vals.append(rt)

        results.append({
            "representation_id": rep_name,
            "algorithm": "K-Means",
            "k": k,
            "seed": "mean_5seeds",
            "runtime_sec": round(np.mean(rt_vals), 2),
            "silhouette": round(np.mean(sil_vals), 4),
            "silhouette_std": round(np.std(sil_vals), 4),
            "calinski_harabasz": round(np.mean(ch_vals), 2),
            "davies_bouldin": round(np.mean(db_vals), 4),
            "davies_bouldin_std": round(np.std(db_vals), 4),
            "n_components": X.shape[1],
        })
        print(f"    K={k} | Sil={np.mean(sil_vals):.4f}±{np.std(sil_vals):.4f} "
              f"| DB={np.mean(db_vals):.4f} | CH={np.mean(ch_vals):.1f}")

df_res = pd.DataFrame(results)

#graficos Silhouette e Davies-Bouldin: original vs PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {
    "EUCLID-SemADR-Original": "steelblue",
    f"EUCLID-SemADR-PCA{n_components_chosen}": "darkorange",
}

for rep_name, color in colors.items():
    sub = df_res[df_res["representation_id"] == rep_name].sort_values("k")
    label = "Original" if "Original" in rep_name else f"PCA ({n_components_chosen} comp.)"

    axes[0].errorbar(sub["k"], sub["silhouette"], yerr=sub["silhouette_std"],
                     marker="o", label=label, color=color, capsize=4)
    axes[1].errorbar(sub["k"], sub["davies_bouldin"], yerr=sub["davies_bouldin_std"],
                     marker="o", label=label, color=color, capsize=4)

axes[0].set_title("Silhouette Score: Original vs PCA")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Silhouette (médio ± std)")
axes[0].legend()
axes[0].set_xticks(k_grid)

axes[1].set_title("Davies-Bouldin: Original vs PCA")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Davies-Bouldin (médio ± std)")
axes[1].legend()
axes[1].set_xticks(k_grid)

plt.tight_layout()
comp_path = os.path.join(fig_dir, "e4_comparacao_original_vs_pca.png")
plt.savefig(comp_path, dpi=300)
plt.close()
print(f"\n[Figura] Comparação guardada em: {comp_path}")

#caveat: variancia preservada nao garante separacao de clusters
best_orig = df_res[df_res["representation_id"] == "EUCLID-SemADR-Original"].sort_values("silhouette", ascending=False).iloc[0]
best_pca = df_res[df_res["representation_id"] == f"EUCLID-SemADR-PCA{n_components_chosen}"].sort_values("silhouette", ascending=False).iloc[0]

print("\n[Caveat metodológico] Comparação do melhor K em cada representação:")
print(f"Original  -> K={int(best_orig['k'])} | Sil={best_orig['silhouette']:.4f} | DB={best_orig['davies_bouldin']:.4f}")
print(f"PCA ({n_components_chosen}cp) -> K={int(best_pca['k'])}  | Sil={best_pca['silhouette']:.4f} | DB={best_pca['davies_bouldin']:.4f}")

if best_pca["silhouette"] >= best_orig["silhouette"]:
    print("-> PCA mantém ou melhora a separação de clusters (variância preservada é informativa).")
else:
    diff = best_orig["silhouette"] - best_pca["silhouette"]
    print(f"-> PCA reduz Silhouette em {diff:.4f}: a variância preservada não garante preservação das direções separadoras de clusters (conforme caveat metodológico do E4)")

#ficheiro proprio do E4 (nao mistura schema com experiments.csv)
df_res.to_csv("e4_pca_results.csv", index=False)
print("\n[Sucesso] e4_pca_results.csv")

print("\n[E4 Concluído] Ficheiros gerados:")
print(f"- {scree_path}")
print(f"- {comp_path}")
print("- e4_pca_results.csv")
