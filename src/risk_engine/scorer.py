"""
Risk Scoring Engine
===================
Combines XGBoost, Isolation Forest, and business rules
into a single 0-100 risk score with actionable decisions.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


# ─────────────────────────────────────────
# 1. Risk score decision tiers
# ─────────────────────────────────────────

@dataclass
class RiskDecision:
    transaction_id: str
    risk_score: int                    # 0–100
    decision: str                      # ALLOW / REVIEW / BLOCK
    xgb_probability: float             # XGBoost fraud probability
    anomaly_score: float               # Isolation Forest (0–1)
    rule_flags: list                   # triggered business rules
    timestamp: str

    def to_dict(self):
        return asdict(self)


THRESHOLDS = {
    "ALLOW":  (0, 34),
    "REVIEW": (35, 69),
    "BLOCK":  (70, 100),
}


def score_to_decision(score: int) -> str:
    for decision, (lo, hi) in THRESHOLDS.items():
        if lo <= score <= hi:
            return decision
    return "REVIEW"


# ─────────────────────────────────────────
# 2. Business rule engine
# ─────────────────────────────────────────

def evaluate_business_rules(txn: dict) -> tuple[list, int]:
    """
    Hard-coded domain rules that override ML scores.
    Returns (list of triggered rules, additional risk points).
    """
    flags = []
    extra_risk = 0

    # Rule 1: Impossible travel
    if txn.get("impossible_travel", 0) == 1:
        flags.append("IMPOSSIBLE_TRAVEL")
        extra_risk += 30

    # Rule 2: Rapid succession (multiple txns within 2 mins)
    if txn.get("is_rapid_succession", 0) == 1:
        flags.append("RAPID_SUCCESSION")
        extra_risk += 15

    # Rule 3: High-risk merchant category
    if txn.get("is_high_risk_merchant", 0) == 1:
        flags.append("HIGH_RISK_MERCHANT")
        extra_risk += 10

    # Rule 4: Unusually large transaction (z-score > 5)
    if txn.get("amount_zscore", 0) > 5:
        flags.append("AMOUNT_ZSCORE_EXTREME")
        extra_risk += 20

    # Rule 5: First-ever transaction above ₹50,000
    if txn.get("is_new_merchant", 0) == 1 and txn.get("amount", 0) > 50000:
        flags.append("LARGE_NEW_MERCHANT_TXN")
        extra_risk += 20

    # Rule 6: Velocity burst — >10 transactions in 1 hour
    if txn.get("txn_count_1h", 0) > 10:
        flags.append("VELOCITY_BURST_1H")
        extra_risk += 25

    return flags, extra_risk


# ─────────────────────────────────────────
# 3. Ensemble scorer
# ─────────────────────────────────────────

class FraudRiskEngine:
    """
    Loads all saved models and scores incoming transactions.

    Scoring formula:
        base_score = 0.60 × xgb_score + 0.40 × anomaly_score
        final_score = min(100, base_score + rule_boost)
    """

    XGB_WEIGHT     = 0.60
    IF_WEIGHT      = 0.40
    RULE_CAP_BOOST = 40    # rules can add at most this much

    def __init__(self, model_dir: str = "models/saved"):
        model_dir = Path(model_dir)
        self.xgb_model = joblib.load(model_dir / "xgboost_fraud.pkl")
        self.if_model  = joblib.load(model_dir / "isolation_forest.pkl")
        print("Risk engine loaded (XGBoost + Isolation Forest)")

    def score(self, txn: dict) -> RiskDecision:
        """Score a single transaction dict."""
        from features.feature_engineering import FEATURE_COLUMNS
        # Build row with 0 defaults, then scrub any NaN/inf
        row = pd.DataFrame([{c: txn.get(c, 0) for c in FEATURE_COLUMNS}])
        row = row.fillna(0).replace([float("inf"), float("-inf")], 0)

        # XGBoost probability
        xgb_proba = float(self.xgb_model.predict_proba(row)[0][1])

        # Isolation Forest anomaly score (normalised 0-1)
        row_scaled = self.if_model.named_steps["scaler"].transform(row)
        import numpy as np_inner; row_scaled = np_inner.nan_to_num(row_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        raw_if = self.if_model.named_steps["iforest"].score_samples(row_scaled)[0]
        # Flip: more negative → more anomalous
        anomaly_score = float(np.clip(-raw_if / 0.5, 0, 1))

        # Weighted ensemble base score
        base_score = (
            self.XGB_WEIGHT * xgb_proba +
            self.IF_WEIGHT  * anomaly_score
        ) * 100

        # Business rules
        rule_flags, extra_risk = evaluate_business_rules(txn)
        extra_risk = min(extra_risk, self.RULE_CAP_BOOST)

        final_score = int(min(100, round(base_score + extra_risk)))
        decision    = score_to_decision(final_score)

        return RiskDecision(
            transaction_id  = str(txn.get("transaction_id", "unknown")),
            risk_score      = final_score,
            decision        = decision,
            xgb_probability = round(xgb_proba, 4),
            anomaly_score   = round(anomaly_score, 4),
            rule_flags      = rule_flags,
            timestamp       = datetime.utcnow().isoformat() + "Z",
        )

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of transactions, return with risk columns."""
        results = [self.score(row.to_dict()) for _, row in df.iterrows()]
        risk_df = pd.DataFrame([r.to_dict() for r in results])
        return pd.concat([df.reset_index(drop=True), risk_df], axis=1)


# ─────────────────────────────────────────
# 4. Threshold calibration
# ─────────────────────────────────────────

def calibrate_thresholds(risk_scores: np.ndarray, y_true: np.ndarray,
                          output_dir: str = "reports") -> dict:
    """
    Find optimal REVIEW and BLOCK thresholds for target recall.
    Useful when deploying to different risk appetites.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    thresholds = list(range(10, 90, 2))
    metrics = []

    for t in thresholds:
        y_pred = (risk_scores >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        metrics.append({"threshold": t, "precision": p, "recall": r, "f1": f})

    mdf = pd.DataFrame(metrics)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mdf["threshold"], mdf["precision"], label="Precision", color="#378ADD", lw=2)
    ax.plot(mdf["threshold"], mdf["recall"],    label="Recall",    color="#E24B4A", lw=2)
    ax.plot(mdf["threshold"], mdf["f1"],        label="F1",        color="#1D9E75", lw=2)
    ax.axvline(35, color="gray", linestyle="--", lw=1, label="Review threshold")
    ax.axvline(70, color="black", linestyle="--", lw=1, label="Block threshold")
    ax.set_xlabel("Risk Score Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold Calibration Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/threshold_calibration.png", dpi=150)
    plt.close()

    best_row = mdf.loc[mdf["f1"].idxmax()]
    print(f"Best F1 threshold: {int(best_row['threshold'])} → "
          f"P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, F1={best_row['f1']:.3f}")

    return mdf.to_dict(orient="records")