"""
Main Training Pipeline
======================
Orchestrates: data generation → feature engineering →
model training → evaluation → report generation.

Run:  python train_pipeline.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.feature_engineering import build_features, FEATURE_COLUMNS
from models.xgboost_model import prepare_data, train as xgb_train, evaluate as xgb_evaluate
from models.isolation_forest import (
    train as if_train, evaluate as if_evaluate,
    get_anomaly_scores
)
from models.segmentation import (
    build_customer_profiles, train_kmeans, run_dbscan, plot_segments
)
from risk_engine.scorer import FraudRiskEngine, calibrate_thresholds


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

DATA_PATH   = "data/transactions_raw.csv"
MODEL_DIR   = "models/saved"
REPORT_DIR  = "reports"
TEST_SIZE   = 0.20
RANDOM_SEED = 42


def banner(text):
    print(f"\n{'═'*55}")
    print(f"  {text}")
    print(f"{'═'*55}")


# ─────────────────────────────────────────
# Step 0: Generate data if not present
# ─────────────────────────────────────────

def maybe_generate_data():
    if not Path(DATA_PATH).exists():
        banner("Step 0 — Generating synthetic dataset")
        sys.path.insert(0, "data")
        from generate_data import generate_customers, generate_merchants, \
                                   generate_transactions, inject_fraud_burst
        customers = generate_customers(n=2000)
        merchants = generate_merchants(n=500)
        df = generate_transactions(customers, merchants, n_days=60)
        df = inject_fraud_burst(df, n_bursts=20)
        Path("data").mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"  Saved → {DATA_PATH}")
    else:
        print(f"  Using existing dataset: {DATA_PATH}")


# ─────────────────────────────────────────
# Step 1: Feature engineering
# ─────────────────────────────────────────

def run_feature_engineering():
    banner("Step 1 — Feature Engineering")
    df = build_features(DATA_PATH)
    feat_path = "data/transactions_features.parquet"
    df.to_parquet(feat_path, index=False)
    print(f"  Features saved → {feat_path}")
    return df


# ─────────────────────────────────────────
# Step 2: Train/test split
# ─────────────────────────────────────────

def split_data(df: pd.DataFrame):
    banner("Step 2 — Train/Test Split")
    df_labelled = df.dropna(subset=["is_fraud"])
    train_df, test_df = train_test_split(
        df_labelled, test_size=TEST_SIZE,
        stratify=df_labelled["is_fraud"],
        random_state=RANDOM_SEED
    )
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"  Fraud in train: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean()*100:.2f}%)")
    print(f"  Fraud in test : {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean()*100:.2f}%)")
    return train_df, test_df


# ─────────────────────────────────────────
# Step 3: XGBoost
# ─────────────────────────────────────────

def run_xgboost(train_df, test_df):
    banner("Step 3 — XGBoost Training & Evaluation")
    X_train, y_train = prepare_data(train_df)
    X_test,  y_test  = prepare_data(test_df)
    model = xgb_train(X_train, y_train, model_dir=MODEL_DIR)
    metrics = xgb_evaluate(model, X_test, y_test, output_dir=REPORT_DIR)
    return metrics


# ─────────────────────────────────────────
# Step 4: Isolation Forest
# ─────────────────────────────────────────

def run_isolation_forest(train_df, test_df):
    banner("Step 4 — Isolation Forest Training & Evaluation")
    avail = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    model = if_train(train_df[avail], model_dir=MODEL_DIR)
    X_test, y_test = prepare_data(test_df)
    metrics = if_evaluate(model, X_test, y_test, output_dir=REPORT_DIR)
    return metrics


# ─────────────────────────────────────────
# Step 5: Customer segmentation
# ─────────────────────────────────────────

def run_segmentation(df: pd.DataFrame):
    banner("Step 5 — Customer Segmentation")
    profiles = build_customer_profiles(df)
    profiles, scaler, km = train_kmeans(profiles, k=5, model_dir=MODEL_DIR)
    profiles = run_dbscan(profiles)
    plot_segments(profiles, output_dir=REPORT_DIR)

    seg_summary = profiles.groupby("segment_label").agg(
        count=("customer_id", "count"),
        avg_spend=("avg_txn_amount", "mean"),
        avg_freq=("txn_frequency_30d", "mean"),
        outlier_rate=("is_outlier_customer", "mean"),
    ).round(2)
    print(seg_summary.to_string())
    profiles.to_csv(f"{REPORT_DIR}/customer_segments.csv", index=False)
    return profiles


# ─────────────────────────────────────────
# Step 6: Risk engine calibration
# ─────────────────────────────────────────

def run_calibration(test_df: pd.DataFrame):
    banner("Step 6 — Risk Engine Calibration")
    try:
        engine = FraudRiskEngine(model_dir=MODEL_DIR)
        sample = test_df.head(500)
        risk_scores = []
        for _, row in sample.iterrows():
            d = engine.score(row.to_dict())
            risk_scores.append(d.risk_score)

        y_true = sample["is_fraud"].values
        calibrate_thresholds(np.array(risk_scores), y_true, output_dir=REPORT_DIR)
    except Exception as e:
        print(f"  Calibration skipped (models not yet saved): {e}")


# ─────────────────────────────────────────
# Step 7: Summary report
# ─────────────────────────────────────────

def print_summary(xgb_metrics, if_metrics):
    banner("Final Summary")
    print(f"  XGBoost   ROC-AUC : {xgb_metrics.get('roc_auc', 'N/A')}")
    print(f"  XGBoost   PR-AUC  : {xgb_metrics.get('pr_auc', 'N/A')}")
    print(f"  XGBoost   F1      : {xgb_metrics.get('fraud_f1', 'N/A')}")
    print(f"  IForest   ROC-AUC : {if_metrics.get('roc_auc', 'N/A')}")
    print(f"  Reports saved → {REPORT_DIR}/")
    print(f"  Models  saved → {MODEL_DIR}/")
    print("\n  To start the API:  uvicorn src.api.main:app --port 8000")
    print("  To open dashboard: python dashboard/app.py\n")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    maybe_generate_data()
    df = run_feature_engineering()
    train_df, test_df = split_data(df)
    xgb_metrics = run_xgboost(train_df, test_df)
    if_metrics  = run_isolation_forest(train_df, test_df)
    _profiles   = run_segmentation(df)
    run_calibration(test_df)
    print_summary(xgb_metrics, if_metrics)
