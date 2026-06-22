"""
XGBoost Fraud Classifier
========================
Supervised binary classifier with SMOTE oversampling,
Bayesian hyperparameter tuning, and full evaluation suite.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import feature list from engineering module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from features.feature_engineering import FEATURE_COLUMNS


# ─────────────────────────────────────────
# 1. Data split
# ─────────────────────────────────────────

def prepare_data(df: pd.DataFrame, label_col: str = "is_fraud"):
    """
    Returns X, y with only engineered features.
    Drops rows where label is missing.
    """
    df = df.dropna(subset=[label_col])
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].fillna(0)
    y = df[label_col].astype(int)
    print(f"Dataset: {len(X):,} rows | {y.sum():,} fraud ({y.mean()*100:.2f}%)")
    return X, y


# ─────────────────────────────────────────
# 2. Model definition
# ─────────────────────────────────────────

def build_model(scale_pos_weight: float = 100.0) -> ImbPipeline:
    """
    SMOTE → StandardScaler → XGBoost pipeline.
    scale_pos_weight handles residual imbalance after SMOTE.
    """
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",          # fast on large datasets
    )

    pipeline = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.2, random_state=42, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("clf", xgb_clf),
    ])
    return pipeline


# ─────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series,
          model_dir: str = "models/saved") -> ImbPipeline:
    """Train with 5-fold CV, print scores, save model."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Compute class weight for XGBoost
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos if pos > 0 else 100.0

    print(f"\nBuilding pipeline  (scale_pos_weight={spw:.1f})")
    model = build_model(scale_pos_weight=spw)

    # ── Cross-validation ──────────────────
    print("Running 5-fold stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    pr_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"  ROC-AUC : {roc_scores.mean():.4f} ± {roc_scores.std():.4f}")
    print(f"  PR-AUC  : {pr_scores.mean():.4f}  ± {pr_scores.std():.4f}")

    # ── Full fit ──────────────────────────
    print("Fitting on full training set...")
    model.fit(X, y)

    # Save
    model_path = Path(model_dir) / "xgboost_fraud.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    return model


# ─────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────

def evaluate(model: ImbPipeline, X_test: pd.DataFrame,
             y_test: pd.Series, output_dir: str = "reports") -> dict:
    """Full evaluation: metrics, curves, feature importance."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Core metrics ─────────────────────
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=["Legit", "Fraud"],
                                    output_dict=True)

    print("\n── Evaluation Results ──────────────────────────")
    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  PR-AUC             : {pr_auc:.4f}")
    print(f"  Fraud Precision    : {report['Fraud']['precision']:.4f}")
    print(f"  Fraud Recall       : {report['Fraud']['recall']:.4f}")
    print(f"  Fraud F1           : {report['Fraud']['f1-score']:.4f}")
    print(f"  False Positive Rate: {cm[0,1]/(cm[0,0]+cm[0,1]):.4f}")
    print("────────────────────────────────────────────────")

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc":  round(pr_auc, 4),
        "fraud_precision": round(report["Fraud"]["precision"], 4),
        "fraud_recall":    round(report["Fraud"]["recall"], 4),
        "fraud_f1":        round(report["Fraud"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────
    _plot_roc(y_test, y_proba, roc_auc, output_dir)
    _plot_pr(y_test, y_proba, pr_auc, output_dir)
    _plot_feature_importance(model, X_test.columns.tolist(), output_dir)
    _plot_confusion(cm, output_dir)

    return metrics


def _plot_roc(y_test, y_proba, roc_auc, out):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#E24B4A", label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#888")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fraud Detector"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/roc_curve.png", dpi=150); plt.close()


def _plot_pr(y_test, y_proba, pr_auc, out):
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, color="#378ADD", label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/pr_curve.png", dpi=150); plt.close()


def _plot_feature_importance(model, feature_names, out, top_n=20):
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], imp[idx], color="#1D9E75")
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
    ax.set_xlabel("Gain")
    fig.tight_layout(); fig.savefig(f"{out}/feature_importance.png", dpi=150); plt.close()


def _plot_confusion(cm, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Legit", "Pred Fraud"])
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)
    ax.set_title("Confusion Matrix"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(f"{out}/confusion_matrix.png", dpi=150); plt.close()


# ─────────────────────────────────────────
# 5. Inference helper
# ─────────────────────────────────────────

def predict_single(model: ImbPipeline, txn: dict) -> dict:
    """
    Score a single transaction dict.
    Returns risk_score (0-100) and decision.
    """
    available = [c for c in FEATURE_COLUMNS if c in txn]
    row = pd.DataFrame([{c: txn.get(c, 0) for c in available}])
    proba = model.predict_proba(row)[0][1]
    risk_score = int(round(proba * 100))

    if risk_score >= 70:
        decision = "BLOCK"
    elif risk_score >= 35:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    return {"risk_score": risk_score, "decision": decision, "fraud_probability": round(proba, 4)}
