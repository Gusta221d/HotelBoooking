import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- Carregamento de dados ---
try:
    df = pd.read_csv('../hotel_bookings_clean.csv')
except FileNotFoundError:
    df = pd.read_csv('hotel_bookings_clean.csv')

print(f"[Info] Dados carregados: {df.shape[0]} reservas, {df.shape[1]} atributos")

# --- Features (igual ao protocolo core) ---
categorical_features = ['hotel', 'arrival_date_month', 'meal', 'market_segment',
                        'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']
numeric_features = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
                    'babies', 'is_repeated_guest', 'previous_cancellations',
                    'previous_bookings_not_canceled', 'required_car_parking_spaces',
                    'total_of_special_requests']

# --- Pipeline de pré-processamento (idêntico ao core) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_euclid = preprocessor.fit_transform(df)
print(f"[Info] Dimensão da representação original (EUCLID-SemADR): {X_euclid.shape}")

fig_dir = 'graficos_relatorio'
os.makedirs(fig_dir, exist_ok=True)

# -------------------------------------------------------
# 1. SCREE PLOT — justificação da dimensionalidade retida
# -------------------------------------------------------
print("\n[PCA] A calcular variância explicada (scree plot)...")

pca_full = PCA(random_state=42)
pca_full.fit(X_euclid)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Limiares standard de variância
thresholds = [0.80, 0.90, 0.95]
n_components_per_threshold = {t: int(np.searchsorted(cumvar, t) + 1) for t in thresholds}
for t, n in n_components_per_threshold.items():
    print(f"   -> {int(t*100)}% variância explicada com {n} componentes")

# Escolha justificada: 90% é o limiar habitual para equilibrar compressão e informação
n_components_chosen = n_components_per_threshold[0.90]
print(f"\n[PCA] Dimensionalidade escolhida: {n_components_chosen} componentes (≥90% variância)")

# Gráfico scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Variância por componente (primeiras 60)
axes[0].bar(range(1, 61), pca_full.explained_variance_ratio_[:60], color='steelblue', alpha=0.7)
axes[0].set_title('Variância Explicada por Componente (Top 60)')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Proporção de Variância Explicada')
axes[0].set_xlim(0, 61)

# Variância acumulada
axes[1].plot(range(1, len(cumvar) + 1), cumvar, color='steelblue', linewidth=1.5)
for t, n in n_components_per_threshold.items():
    axes[1].axhline(y=t, color='gray', linestyle='--', linewidth=0.8)
    axes[1].axvline(x=n, color='gray', linestyle='--', linewidth=0.8)
    axes[1].annotate(f'{int(t*100)}% ({n} comp.)',
                     xy=(n, t), xytext=(n + 5, t - 0.03),
                     fontsize=8, color='dimgray')
axes[1].set_title('Variância Acumulada Explicada (PCA)')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Variância Acumulada')
axes[1].set_xlim(0, X_euclid.shape[1])
axes[1].set_ylim(0.5, 1.01)

plt.tight_layout()
scree_path = os.path.join(fig_dir, 'e4_scree_plot.png')
plt.savefig(scree_path, dpi=300)
plt.close()
print(f"[Figura] Scree plot guardado em: {scree_path}")

# -------------------------------------------------------
# 2. REDUÇÃO — aplicar PCA com dimensionalidade escolhida
# -------------------------------------------------------
print(f"\n[PCA] A reduzir de {X_euclid.shape[1]} para {n_components_chosen} dimensões...")
pca = PCA(n_components=n_components_chosen, random_state=42)
X_pca = pca.fit_transform(X_euclid)
print(f"[Info] Dimensão reduzida (PCA): {X_pca.shape}")
print(f"[Info] Variância total explicada: {pca.explained_variance_ratio_.sum():.4f}")

# -------------------------------------------------------
# 3. PROTOCOLO DE CLUSTERING — igual ao core (k_grid + seeds)
# -------------------------------------------------------
k_grid = [3, 4, 5, 6, 7, 8]
seeds = [42, 67, 123, 2026, 71]  
results = []

representations = {
    'EUCLID-SemADR-Original': X_euclid,
    f'EUCLID-SemADR-PCA{n_components_chosen}': X_pca
}

print("\n[Protocolo] A correr K-Means em ambas as representações (original vs PCA)...")

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
            ch  = calinski_harabasz_score(X, labels)
            db  = davies_bouldin_score(X, labels)

            sil_vals.append(sil)
            ch_vals.append(ch)
            db_vals.append(db)
            rt_vals.append(rt)

        results.append({
            'representation_id': rep_name,
            'algorithm': 'K-Means',
            'k': k,
            'seed': 'mean_5seeds',
            'runtime_sec': round(np.mean(rt_vals), 2),
            'silhouette': round(np.mean(sil_vals), 4),
            'silhouette_std': round(np.std(sil_vals), 4),
            'calinski_harabasz': round(np.mean(ch_vals), 2),
            'davies_bouldin': round(np.mean(db_vals), 4),
            'davies_bouldin_std': round(np.std(db_vals), 4),
            'n_components': X.shape[1]
        })
        print(f"    K={k} | Sil={np.mean(sil_vals):.4f}±{np.std(sil_vals):.4f} "
              f"| DB={np.mean(db_vals):.4f} | CH={np.mean(ch_vals):.1f}")

# -------------------------------------------------------
# 4. COMPARAÇÃO VISUAL — Silhouette e Davies-Bouldin
# -------------------------------------------------------
df_res = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'EUCLID-SemADR-Original': 'steelblue',
          f'EUCLID-SemADR-PCA{n_components_chosen}': 'darkorange'}

for rep_name, color in colors.items():
    sub = df_res[df_res['representation_id'] == rep_name].sort_values('k')
    label = 'Original' if 'Original' in rep_name else f'PCA ({n_components_chosen} comp.)'

    axes[0].errorbar(sub['k'], sub['silhouette'], yerr=sub['silhouette_std'],
                     marker='o', label=label, color=color, capsize=4)
    axes[1].errorbar(sub['k'], sub['davies_bouldin'], yerr=sub['davies_bouldin_std'],
                     marker='o', label=label, color=color, capsize=4)

axes[0].set_title('Silhouette Score: Original vs PCA')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Silhouette (médio ± std)')
axes[0].legend()
axes[0].set_xticks(k_grid)

axes[1].set_title('Davies-Bouldin: Original vs PCA')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Davies-Bouldin (médio ± std)')
axes[1].legend()
axes[1].set_xticks(k_grid)

plt.tight_layout()
comp_path = os.path.join(fig_dir, 'e4_comparacao_original_vs_pca.png')
plt.savefig(comp_path, dpi=300)
plt.close()
print(f"\n[Figura] Comparação guardada em: {comp_path}")

# -------------------------------------------------------
# 5. CAVEAT METODOLÓGICO — variância ≠ separação de clusters
# -------------------------------------------------------
# Compara silhouette do melhor K em ambas as representações
best_orig = df_res[df_res['representation_id'] == 'EUCLID-SemADR-Original'].sort_values('silhouette', ascending=False).iloc[0]
best_pca  = df_res[df_res['representation_id'] == f'EUCLID-SemADR-PCA{n_components_chosen}'].sort_values('silhouette', ascending=False).iloc[0]

print("\n[Caveat metodológico] Comparação do melhor K em cada representação:")
print(f"  Original  -> K={int(best_orig['k'])} | Sil={best_orig['silhouette']:.4f} | DB={best_orig['davies_bouldin']:.4f}")
print(f"  PCA ({n_components_chosen}cp) -> K={int(best_pca['k'])}  | Sil={best_pca['silhouette']:.4f} | DB={best_pca['davies_bouldin']:.4f}")

if best_pca['silhouette'] >= best_orig['silhouette']:
    print("  -> PCA mantém ou melhora a separação de clusters (variância preservada é informativa).")
else:
    diff = best_orig['silhouette'] - best_pca['silhouette']
    print(f"  -> PCA reduz Silhouette em {diff:.4f}: a variância preservada não garante preservação")
    print("     das direções separadoras de clusters (conforme caveat metodológico do E4).")

# -------------------------------------------------------
# 6. LOGGING — adiciona ao experiments.csv
# -------------------------------------------------------
log_path = 'experiments.csv' if os.path.exists('experiments.csv') else '../experiments.csv'

# Guardar apenas colunas standard do experiments.csv
df_log = df_res[['representation_id', 'algorithm', 'k', 'seed',
                 'runtime_sec', 'silhouette', 'calinski_harabasz', 'davies_bouldin']].copy()
df_log.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
print(f"\n[Sucesso] Resultados E4 adicionados a: {log_path}")

# Guardar tabela completa E4
e4_csv = 'e4_pca_results.csv' if os.path.exists('experiments.csv') else '../e4_pca_results.csv'
df_res.to_csv(e4_csv, index=False)
print(f"[Sucesso] Tabela completa E4 guardada em: {e4_csv}")

print("\n[E4 Concluído] Ficheiros gerados:")
print(f"  - {scree_path}")
print(f"  - {comp_path}")
print(f"  - {e4_csv}")
