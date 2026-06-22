"""
Customer Segmentation
=====================
Groups customers by spending behaviour using K-Means and DBSCAN.
Segments feed into risk engine and dashboard analytics.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─────────────────────────────────────────
# 1. Build customer-level feature matrix
# ─────────────────────────────────────────

SEGMENT_FEATURES = [
    "avg_txn_amount", "std_txn_amount", "total_spend_30d",
    "txn_frequency_30d", "night_txn_ratio", "weekend_txn_ratio",
    "unique_merchants", "unique_categories", "avg_geo_distance",
    "high_risk_category_ratio", "max_single_txn",
]


def build_customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level df into one row per customer.
    Returns customer profile matrix.
    """
    g = df.groupby("customer_id")

    profiles = pd.DataFrame({
        "customer_id": list(g.groups.keys()),
        "avg_txn_amount":        g["amount"].mean().values,
        "std_txn_amount":        g["amount"].std().fillna(0).values,
        "total_spend_30d":       g["amount"].sum().values,
        "txn_frequency_30d":     g["amount"].count().values,
        "night_txn_ratio":       g["is_night"].mean().values,
        "weekend_txn_ratio":     g["is_weekend"].mean().values,
        "unique_merchants":      g["merchant_id"].nunique().values,
        "unique_categories":     g["merchant_category"].nunique().values,
        "avg_geo_distance":      g["geo_distance_km"].mean().values        if "geo_distance_km" in df.columns else 0,
        "high_risk_category_ratio": g["is_high_risk_merchant"].mean().values if "is_high_risk_merchant" in df.columns else 0,
        "max_single_txn":        g["amount"].max().values,
    })
    return profiles.reset_index(drop=True)


# ─────────────────────────────────────────
# 2. Optimal K selection (Elbow + Silhouette)
# ─────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range=(2, 12),
                   output_dir: str = "reports") -> int:
    """Plot elbow and silhouette; return best K by silhouette."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ks = list(range(k_range[0], k_range[1] + 1))
    inertias, silhouettes = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = ks[np.argmax(silhouettes)]
    print(f"Best K (silhouette): {best_k}  (score={max(silhouettes):.3f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertias, "o-", color="#378ADD")
    ax1.set_title("Elbow Curve"); ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")
    ax2.plot(ks, silhouettes, "o-", color="#1D9E75")
    ax2.axvline(best_k, color="#E24B4A", linestyle="--", lw=1.5)
    ax2.set_title("Silhouette Score"); ax2.set_xlabel("K"); ax2.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/kmeans_elbow.png", dpi=150)
    plt.close()

    return best_k


# ─────────────────────────────────────────
# 3. K-Means Segmentation
# ─────────────────────────────────────────

SEGMENT_LABELS = {
    0: "Low-value standard",
    1: "High-frequency everyday",
    2: "High-value premium",
    3: "Dormant / infrequent",
    4: "High-risk irregular",
}


def train_kmeans(profiles: pd.DataFrame, k: int = 5,
                 model_dir: str = "models/saved") -> tuple:
    """
    Fit K-Means, return (labelled_profiles, scaler, model).
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    avail = [c for c in SEGMENT_FEATURES if c in profiles.columns]
    X = profiles[avail].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-select K if not specified
    if k == 0:
        k = find_optimal_k(X_scaled)

    print(f"Fitting K-Means with K={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=500)
    labels = km.fit_predict(X_scaled)

    profiles = profiles.copy()
    profiles["segment"] = labels
    profiles["segment_label"] = profiles["segment"].map(
        lambda x: SEGMENT_LABELS.get(x, f"Segment {x}")
    )

    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {sil:.4f}")

    joblib.dump(km, f"{model_dir}/kmeans.pkl")
    joblib.dump(scaler, f"{model_dir}/segment_scaler.pkl")

    return profiles, scaler, km


# ─────────────────────────────────────────
# 4. DBSCAN (density-based outlier detection)
# ─────────────────────────────────────────

def run_dbscan(profiles: pd.DataFrame, eps: float = 0.8,
               min_samples: int = 10) -> pd.DataFrame:
    """
    DBSCAN assigns -1 to outlier customers (suspicious/unusual behaviour).
    """
    avail = [c for c in SEGMENT_FEATURES if c in profiles.columns]
    X = StandardScaler().fit_transform(profiles[avail].fillna(0))

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    profiles = profiles.copy()
    profiles["dbscan_cluster"] = db.fit_predict(X)
    profiles["is_outlier_customer"] = (profiles["dbscan_cluster"] == -1).astype(int)

    n_outliers = profiles["is_outlier_customer"].sum()
    n_clusters = profiles["dbscan_cluster"].nunique() - (1 if -1 in profiles["dbscan_cluster"].values else 0)
    print(f"DBSCAN: {n_clusters} clusters, {n_outliers} outlier customers")

    return profiles


# ─────────────────────────────────────────
# 5. Visualisation (PCA 2D)
# ─────────────────────────────────────────

def plot_segments(profiles: pd.DataFrame, output_dir: str = "reports"):
    """2D PCA visualisation of customer segments."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    avail = [c for c in SEGMENT_FEATURES if c in profiles.columns]
    X_scaled = StandardScaler().fit_transform(profiles[avail].fillna(0))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A", "#8B5CF6", "#EC4899"]

    for seg_id in sorted(profiles["segment"].unique()):
        mask = profiles["segment"] == seg_id
        label = profiles.loc[mask, "segment_label"].iloc[0] if "segment_label" in profiles.columns else f"Seg {seg_id}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.6, color=palette[seg_id % len(palette)],
                   label=label)

    # Highlight outlier customers
    if "is_outlier_customer" in profiles.columns:
        out_mask = profiles["is_outlier_customer"] == 1
        ax.scatter(coords[out_mask, 0], coords[out_mask, 1],
                   s=30, marker="x", color="black", alpha=0.8, label="Outlier (DBSCAN)")

    ax.set_title("Customer Segmentation (PCA 2D)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/customer_segments.png", dpi=150)
    plt.close()
    print(f"Segment plot saved → {output_dir}/customer_segments.png")
