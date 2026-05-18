import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

METRIC_SHORT = "Euclid+OHE" 
RUN_META_FILE = os.path.join(os.path.dirname(__file__), "pipeline_run.json")

def get_run_context():
    if os.path.isfile(RUN_META_FILE): #se o ficheiro de meta existe, le o run_id e data
        with open(RUN_META_FILE, encoding="utf-8") as f: 
            meta = json.load(f) 
        return meta["run_id"], meta["date"][:10] #run_id e data
    now = datetime.now(timezone.utc) #data e hora atuais
    return now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y-%m-%d") 


def metric_sentence():
    #texto longo que explica o espaco metrico (para o relatorio / experiments.csv)
    return (
        "Euclidean distance on scaled numeric features "
        "+ one-hot encoded categorical features."
    )


# Colunas do preprocessor, OUTLIER_NUM_COLS 
MIN_CATEGORY_FREQ = 0.01  #niveis categoricos com menos de 1% passam a chamar-se "Other"
CATEGORICAL_FEATURES = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type",
]
NUMERIC_BASE = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "required_car_parking_spaces", "total_of_special_requests",
]
#semana/dia brutos: so na variante de sensibilidade (calendario); no core principal usa-se mes em OHE
NUMERIC_CALENDAR_RAW = ["arrival_date_week_number", "arrival_date_day_of_month"]
#colunas para o relatorio IQR de outliers (inclui adr e calendario)
OUTLIER_NUM_COLS = NUMERIC_BASE + ["adr"] + NUMERIC_CALENDAR_RAW


def group_rare_categories(df, cat_cols, min_freq=MIN_CATEGORY_FREQ):
    #evita categorias com pouquissimas linhas (sparse); junta tudo em "Other"
    out = df.copy()
    for col in cat_cols: #ciclo para agrupar as categorias raras
        if col not in out.columns: #se a coluna nao estiver no dataframe, passa para a proxima
            continue
        freq = out[col].value_counts(normalize=True) #normaliza as frequencias
        rare = freq[freq < min_freq].index #indices das categorias raras
        if len(rare) > 0:
            out[col] = out[col].where(~out[col].isin(rare), "Other")
    return out

def adr_iqr_mask(series):
    #True onde o ADR esta fora do intervalo [lower, upper] do metodo IQR classico
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = max(0.01, q1 - 1.5 * iqr)  #ADR nao pode ser negativo
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper), lower, upper


def apply_adr_quality_filter(df):
    #filtro de qualidade de dados (nao e o IQR); base SemADR usa isto
    return df[(df["adr"] > 0) & (df["adr"] <= 5000)].copy()


def apply_adr_iqr_filter(df):
    #remove linhas que o IQR de ADR marcar como extremas; devolve tambem limites e quantas linhas cairam
    mask, lo, hi = adr_iqr_mask(df["adr"])
    return df[~mask].copy(), lo, hi, int(mask.sum())


def build_rep_id(scaler_name, adr_status, n_features, calendar_mode="month-OHE"):
    #string fixa tipo R-EUCLID-d52-... para saberes scaler, dimensao final, ADR e calendario
    return (
        f"R-EUCLID-d{n_features}-{scaler_name}-ADR-{adr_status}"
        f"-country-excl-rare-gov-{calendar_mode}"
    )


def build_preprocessor(numeric_features, scaler):
    #uma matriz: bloco numerico escalado + bloco categorico em one-hot (unknown vira zeros)
    return ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def short_rep_label(scaler_name, adr_status, calendar_mode="month-OHE"):
    #nome curto para prints e coluna "rep" (ex: Std-SemADR, Std-ComADR+weekday)
    adr = "SemADR" if adr_status == "no" else "ComADR"
    suffix = "+weekday" if "week-day" in calendar_mode else ""
    return f"{scaler_name}-{adr}{suffix}"


def sample_rule_all():
    return "all rows"


def sample_rule_subsample(n, seed):
    #texto fixo para o CSV: quantas linhas e que seed de aleatorio
    return f"n={n};seed={seed}"


# experiments.csv - um dicionario = uma linha
def log_row(rep, rep_id, algorithm, k, seed, runtime_sec, metrics, n_rows, sample_rule, parameters="", diagnostics="", notes=""): #parametros da linha
    run_id, run_date = get_run_context() #run_id e data
    return {
        "run_id": run_id,
        "date": run_date,
        "rep": rep,
        "rep_id": rep_id,
        "metric": METRIC_SHORT,
        "metric_sentence": metric_sentence(),
        "algorithm": algorithm,
        "k": k,
        "seed": seed,
        "n_rows": n_rows,
        "sample_rule": sample_rule,
        "silhouette": metrics.get("silhouette"),
        "calinski_harabasz": metrics.get("calinski_harabasz"),
        "davies_bouldin": metrics.get("davies_bouldin"),
        "min_cluster_pct": (diagnostics.get("min_cluster_pct") if isinstance(diagnostics, dict) else None),
        "runtime_s": runtime_sec,
        "parameters": parameters,
        "diagnostics": (diagnostics if isinstance(diagnostics, str) else _fmt_diag(diagnostics)),
        "notes": notes or "",
    }


def _fmt_diag(d):
    #transforma por exemplo: {"min_cluster_pct": 2.5} em "min_cluster_pct=2.5"
    if not d:
        return ""
    return ";".join(f"{k}={v}" for k, v in d.items())


def min_cluster_pct(labels):
    #percentagem de linhas no cluster mais pequeno
    _, counts = np.unique(labels, return_counts=True)
    return round(100.0 * counts.min() / len(labels), 2)


def select_k_from_sweep(sweep_df, min_pct=1.0, default=6):
    #percorre k por ordem decrescente de Silhouette e escolhe o primeiro com min_cluster_pct >= min_pct
    if sweep_df.empty:
        return default
    sub = sweep_df.sort_values("silhouette", ascending=False)
    for _, row in sub.iterrows():
        if row.get("min_cluster_pct", 0) >= min_pct:
            return int(row["k"])
    #se nenhum k cumprir o minimo, fica o melhor Silhouette mesmo assim
    return int(sub.iloc[0]["k"])


def save_selected_k(k, path="selected_k.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(k))


def load_selected_k(path="selected_k.txt", default=6):
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return default


def plots_dir():
    d = "graficos_relatorio"
    os.makedirs(d, exist_ok=True)
    return d


#comparar duas segmentacoes (Std vs Robust, SemADR vs ComADR)
def align_cluster_pairs(labels_ref, labels_other):
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment

    cm = contingency_matrix(labels_ref, labels_other) #matriz de contingencia
    #linear_sum_assignment minimiza custo; com -cm minimiza = maximizar sobreposicao
    r_ind, c_ind = linear_sum_assignment(-cm)
    return list(zip(r_ind.tolist(), c_ind.tolist())) #indices alinhados


def profile_agreement(df, labels_ref, labels_other, lead_time_tol=30):
    #depois de alinhar pares (r,c), ve se canal modal, segmento modal e mediana de lead_time batem
    lr, lo = np.asarray(labels_ref), np.asarray(labels_other)
    scores = []
    for r, c in align_cluster_pairs(labels_ref, labels_other):
        g0 = df.loc[lr == r] #grupo de referencia
        g1 = df.loc[lo == c] #grupo alinhado
        if len(g0) == 0 or len(g1) == 0: #se o grupo de referencia ou o grupo alinhado for vazio, passa para o proximo
            continue
        ch = g0["distribution_channel"].mode().iat[0] == g1["distribution_channel"].mode().iat[0] #canal modal
        seg = g0["market_segment"].mode().iat[0] == g1["market_segment"].mode().iat[0] #segmento modal
        lt = abs(g0["lead_time"].median() - g1["lead_time"].median()) <= lead_time_tol #mediana de lead_time
        scores.append((int(ch) + int(seg) + int(lt)) / 3.0) #media das 3 concordancias (canal, segmento, lead_time) por par alinhado
    return round(float(np.mean(scores)) if scores else 0.0, 4)


def cluster_profile_row(g):
    #resumo de um unico cluster (um grupo de linhas do dataframe)
    return {
        "n_bookings": len(g),
        "pct_bookings": round(100 * len(g) / 1, 4),  #preenchido com % real em build_aligned_profile_comparison
        "lead_time_median": round(float(g["lead_time"].median()), 1), #mediana de lead_time
        "total_nights_median": round( #mediana de total_nights
            float((g["stays_in_weekend_nights"] + g["stays_in_week_nights"]).median()), 1
        ),
        "distribution_channel_mode": g["distribution_channel"].mode().iat[0], #canal modal
        "market_segment_mode": g["market_segment"].mode().iat[0], #segmento modal
        "deposit_type_mode": g["deposit_type"].mode().iat[0], #tipo de deposit modal
    }


def build_aligned_profile_comparison(df, labels_ref, labels_other, name_ref, name_other):
    #tabela com uma linha por par (cluster_ref, cluster_other) ja alinhados
    n = len(df)
    lr, lo = np.asarray(labels_ref), np.asarray(labels_other)
    rows = []
    for r, c in align_cluster_pairs(labels_ref, labels_other):
        g0 = df.loc[lr == r]
        g1 = df.loc[lo == c]
        if len(g0) == 0 or len(g1) == 0:
            continue
        p0 = cluster_profile_row(g0)
        p1 = cluster_profile_row(g1)
        p0["pct_bookings"] = round(100 * len(g0) / n, 2)
        p1["pct_bookings"] = round(100 * len(g1) / n, 2)
        rows.append({
            "cluster_ref": int(r),
            "cluster_other": int(c),
            "rep_ref": name_ref,
            "rep_other": name_other,
            f"{name_ref}_pct": p0["pct_bookings"],
            f"{name_other}_pct": p1["pct_bookings"],
            f"{name_ref}_lead_time_med": p0["lead_time_median"],
            f"{name_other}_lead_time_med": p1["lead_time_median"],
            f"{name_ref}_channel": p0["distribution_channel_mode"],
            f"{name_other}_channel": p1["distribution_channel_mode"],
            f"{name_ref}_segment": p0["market_segment_mode"],
            f"{name_other}_segment": p1["market_segment_mode"],
            "channel_match": p0["distribution_channel_mode"] == p1["distribution_channel_mode"],
            "segment_match": p0["market_segment_mode"] == p1["market_segment_mode"],
            "lead_time_close": abs(p0["lead_time_median"] - p1["lead_time_median"]) <= 30,
        })
    return pd.DataFrame(rows)
