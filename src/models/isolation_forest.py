"""
Isolation Forest — Unsupervised Anomaly Detector
=================================================
Detects novel fraud patterns WITHOUT needing labels.
Complements XGBoost by catching zero-day fraud types.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from features.feature_engineering import FEATURE_COLUMNS


# ─────────────────────────────────────────
# 1. Build & Train
# ─────────────────────────────────────────

def build_isolation_forest(contamination: float = 0.002) -> Pipeline:
    """
    contamination = expected fraction of outliers in training data.
    For credit card data, ~0.17-0.2% is fraud → set 0.002.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
        ))
    ])
    return model


def train(X: pd.DataFrame, contamination: float = 0.002,
          model_dir: str = "models/saved") -> Pipeline:
    """
    Train on ALL data (no labels needed).
    Saves the model for inference.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    available = [c for c in FEATURE_COLUMNS if c in X.columns]
    X_feat = X[available].fillna(0)

    print(f"Training Isolation Forest on {len(X_feat):,} samples, {len(available)} features...")
    model = build_isolation_forest(contamination=contamination)
    model.fit(X_feat)

    path = Path(model_dir) / "isolation_forest.pkl"
    joblib.dump(model, path)
    print(f"Model saved → {path}")
    return model


# ─────────────────────────────────────────
# 2. Scoring
# ─────────────────────────────────────────

def get_anomaly_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Returns anomaly score in [0, 1] where 1 = most anomalous.
    Isolation Forest natively returns negative scores; we flip and normalise.
    """
    available = [c for c in FEATURE_COLUMNS if c in X.columns]
    X_feat = X[available].fillna(0)

    raw_scores = model.named_steps["iforest"].score_samples(
        model.named_steps["scaler"].transform(X_feat)
    )
    # More negative = more anomalous; flip and scale to [0,1]
    scores = -raw_scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores


def predict(model: Pipeline, X: pd.DataFrame,
            threshold: float = 0.65) -> pd.Series:
    """
    Returns binary predictions: 1 = anomaly (potential fraud).
    threshold on the normalised [0,1] anomaly score.
    """
    scores = get_anomaly_scores(model, X)
    return (scores >= threshold).astype(int)


# ─────────────────────────────────────────
# 3. Evaluation (when labels exist)
# ─────────────────────────────────────────

def evaluate(model: Pipeline, X_test: pd.DataFrame,
             y_test: pd.Series, output_dir: str = "reports") -> dict:
    """Evaluate using anomaly scores against ground-truth labels."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scores = get_anomaly_scores(model, X_test)

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc  = average_precision_score(y_test, scores)

    print("\n── Isolation Forest Evaluation ─────────────────")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print("─────────────────────────────────────────────────")

    _plot_score_distribution(scores, y_test, output_dir)
    return {"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4)}


def _plot_score_distribution(scores, y_test, out):
    """Visualise score separation between fraud and legit."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[y_test == 0], bins=80, alpha=0.6,
            color="#378ADD", label="Legitimate", density=True)
    ax.hist(scores[y_test == 1], bins=80, alpha=0.7,
            color="#E24B4A", label="Fraud", density=True)
    ax.axvline(0.65, color="black", linestyle="--", lw=1.5, label="Threshold (0.65)")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
    ax.set_title("Isolation Forest — Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/if_score_distribution.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────
# 4. Inference helper
# ─────────────────────────────────────────

def score_single(model: Pipeline, txn: dict) -> dict:
    """Score a single transaction. Returns anomaly_score (0-100)."""
    row = pd.DataFrame([{c: txn.get(c, 0) for c in FEATURE_COLUMNS}])
    score = float(get_anomaly_scores(model, row)[0])
    return {
        "anomaly_score": round(score * 100, 1),
        "is_anomalous": score >= 0.65,
    }
