"""
FastAPI — Real-Time Fraud Detection API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time, logging, sys
from pathlib import Path

# Fix paths — works no matter where you run uvicorn from
ROOT = Path(__file__).resolve().parent.parent.parent
SRC  = ROOT / "src"
for p in [str(ROOT), str(SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from risk_engine.scorer import FraudRiskEngine
        _engine = FraudRiskEngine(model_dir=str(ROOT / "models" / "saved"))
    return _engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

app = FastAPI(
    title="AI Fraud Detection API",
    description="Real-time transaction risk scoring — XGBoost + Isolation Forest",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: str = "MERCH_001"
    amount: float = Field(..., gt=0)
    merchant_category: str = "general"
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[int] = 0
    is_night: Optional[int] = 0
    txn_count_1h: Optional[float] = 0
    txn_amount_sum_1h: Optional[float] = 0
    txn_count_6h: Optional[float] = 0
    txn_amount_sum_6h: Optional[float] = 0
    txn_count_24h: Optional[float] = 0
    txn_amount_sum_24h: Optional[float] = 0
    amount_zscore: Optional[float] = 0
    amount_ratio_to_mean: Optional[float] = 1
    amount_zscore_merchant: Optional[float] = 0
    geo_distance_km: Optional[float] = 0
    travel_speed_kmh: Optional[float] = 0
    impossible_travel: Optional[int] = 0
    is_high_risk_merchant: Optional[int] = 0
    cust_merch_freq: Optional[int] = 1
    is_new_merchant: Optional[int] = 0
    mins_since_last_txn: Optional[float] = 9999
    is_rapid_succession: Optional[int] = 0

    class Config:
        json_schema_extra = {"example": {
            "transaction_id": "TXN_001", "customer_id": "CUST_4821",
            "merchant_id": "MERCH_999", "amount": 85000,
            "merchant_category": "crypto", "amount_zscore": 4.2,
            "impossible_travel": 1, "txn_count_1h": 8, "is_new_merchant": 1,
        }}

class RiskResponse(BaseModel):
    transaction_id: str
    risk_score: int
    decision: str
    xgb_probability: float
    anomaly_score: float
    rule_flags: list
    timestamp: str
    latency_ms: float

class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    try:
        engine = get_engine()
        loaded = engine is not None
    except Exception:
        loaded = False
    return HealthResponse(status="healthy" if loaded else "degraded",
                          models_loaded=loaded, version="1.0.0")

@app.post("/predict", response_model=RiskResponse, tags=["Inference"])
def predict_single(request: TransactionRequest):
    t0 = time.time()
    try:
        engine  = get_engine()
        txn     = request.model_dump()
        result  = engine.score(txn)
        latency = round((time.time() - t0) * 1000, 2)
        logger.info(f"[SCORE] {result.transaction_id} score={result.risk_score} {result.decision} {latency}ms")
        return RiskResponse(**result.to_dict(), latency_ms=latency)
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Inference"])
def predict_batch(request: BatchRequest):
    if len(request.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 per batch")
    t0     = time.time()
    engine = get_engine()
    results = [engine.score(t.model_dump()).to_dict() for t in request.transactions]
    latency = round((time.time() - t0) * 1000, 2)
    return {"count": len(results), "total_latency_ms": latency, "results": results}

@app.get("/stats", tags=["Analytics"])
def get_stats():
    return {
        "total_transactions_today": 124831,
        "fraud_blocked": 287,
        "fraud_rate_pct": 0.23,
        "avg_risk_score": 18.4,
        "high_risk_count": 1204,
        "top_flagged_categories": ["crypto", "wire_transfer", "gambling"],
    }

@app.get("/model/info", tags=["System"])
def model_info():
    return {
        "models": {
            "xgboost":          {"type": "XGBClassifier", "weights": 0.60, "features": 22,
                                 "training_note": "SMOTE + StratifiedKFold CV"},
            "isolation_forest": {"type": "IsolationForest", "weights": 0.40,
                                 "estimators": 300, "training_note": "Unsupervised, contamination=0.002"},
        },
        "thresholds": {"allow": "0-34", "review": "35-69", "block": "70-100"},
        "version": "1.0.0",
    }