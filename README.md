# 🛡️ FraudShield AI — Financial Fraud Detection System

An end-to-end machine learning system that detects fraudulent bank transactions in real time using an ensemble of XGBoost and Isolation Forest models, deployed via a FastAPI backend and interactive Streamlit dashboard.

## 🎯 What it does
Every transaction is scored 0–100 in under 100ms.
- 🟢 Score 0–34 → ALLOW
- 🟡 Score 35–69 → REVIEW
- 🔴 Score 70–100 → BLOCK

## 🧠 ML Models
- **XGBoost** — Supervised classifier trained on labelled fraud data with SMOTE oversampling
- **Isolation Forest** — Unsupervised anomaly detector that catches new unknown fraud patterns
- **Ensemble** — 60% XGBoost + 40% Isolation Forest = final risk score

## 📊 Results
| Metric | Score |
|--------|-------|
| XGBoost ROC-AUC | 0.9999 |
| XGBoost Accuracy | 99.8% |
| Isolation Forest Accuracy | 98.2% |
| API Response Time | < 100ms |

## 🛠️ Tech Stack
Python · XGBoost · Scikit-learn · FastAPI · Streamlit · Pandas · Plotly

## 🚀 How to run
```bash
pip install -r requirements.txt
python data.py
python model.py
streamlit run app.py
```

## ✨ Features
- 22 behavioural features — velocity, z-score, geo distance, impossible travel
- Real-time REST API at localhost:8000
- Live transaction stream simulator
- Interactive dashboard — score any transaction instantly
- Customer segmentation using K-Means + DBSCAN

## 👨‍💻 Built by
Sujal Khedkar — Final Year Data Science Project
