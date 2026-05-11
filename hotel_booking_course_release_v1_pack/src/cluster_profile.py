import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

print("--------- PERFIL INICIAL DE CLUSTERS (K=8) ---------")

file_path = "../hotel_bookings_clean.csv"
df = pd.read_csv(file_path)

categorical_features = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

numeric_features = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X_euclid = preprocessor.fit_transform(df)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_euclid)

df_profile = df.copy()
df_profile["cluster"] = labels
df_profile["total_nights"] = (
    df_profile["stays_in_weekend_nights"] + df_profile["stays_in_week_nights"]
)

rows = []
n = len(df_profile)

for cluster_id, group in df_profile.groupby("cluster"):
    rows.append(
        {
            "cluster": int(cluster_id),
            "n_bookings": int(len(group)),
            "pct_bookings": round(100 * len(group) / n, 2),
            "lead_time_median": round(float(group["lead_time"].median()), 1),
            "total_nights_median": round(float(group["total_nights"].median()), 1),
            "distribution_channel_mode": group["distribution_channel"].mode().iat[0],
            "market_segment_mode": group["market_segment"].mode().iat[0],
        }
    )

output_path = "cluster_profile_k8.csv"
df_out = pd.DataFrame(rows).sort_values("cluster")
df_out.to_csv(output_path, index=False)

print(f"[Sucesso] Perfil inicial guardado em: {output_path}")
print(df_out.to_string(index=False))
