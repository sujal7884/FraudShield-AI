"""
Synthetic Fraud Dataset Generator
===================================
Creates a realistic labelled dataset for training and demo purposes.
Mirrors the Kaggle credit card fraud distribution (~0.17% fraud).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string

np.random.seed(42)
random.seed(42)


MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "online_retail", "pharmacy",
    "hotel", "airline", "utilities", "subscription", "electronics",
    "clothing", "entertainment", "gambling", "crypto", "wire_transfer",
    "prepaid_card", "money_order", "atm_cash",
]

HIGH_RISK_CATS = {"gambling", "crypto", "wire_transfer", "prepaid_card", "money_order"}

CITIES = [
    (28.6139, 77.2090, "Delhi"),
    (19.0760, 72.8777, "Mumbai"),
    (12.9716, 77.5946, "Bangalore"),
    (22.5726, 88.3639, "Kolkata"),
    (17.3850, 78.4867, "Hyderabad"),
    (13.0827, 80.2707, "Chennai"),
    (23.0225, 72.5714, "Ahmedabad"),
    (18.5204, 73.8567, "Pune"),
]


def random_id(prefix="", length=8):
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def generate_customers(n=2000):
    customers = []
    for _ in range(n):
        city = random.choice(CITIES)
        customers.append({
            "customer_id": random_id("CUST_"),
            "base_lat": city[0] + np.random.normal(0, 0.05),
            "base_lon": city[1] + np.random.normal(0, 0.05),
            "avg_spend":   np.random.lognormal(mean=8, sigma=1.2),   # ₹~3K avg
            "txn_per_day": np.random.poisson(lam=2) + 1,
            "home_city":   city[2],
        })
    return pd.DataFrame(customers)


def generate_merchants(n=500):
    merchants = []
    for _ in range(n):
        city = random.choice(CITIES)
        cat = random.choices(
            MERCHANT_CATEGORIES,
            weights=[10,8,7,12,5,4,4,3,6,5,5,4,1,1,1,1,1,2],
            k=1
        )[0]
        merchants.append({
            "merchant_id": random_id("MERCH_"),
            "merchant_category": cat,
            "lat": city[0] + np.random.normal(0, 0.1),
            "lon": city[1] + np.random.normal(0, 0.1),
        })
    return pd.DataFrame(merchants)


def generate_transactions(customers: pd.DataFrame, merchants: pd.DataFrame,
                           n_days: int = 30, fraud_rate: float = 0.0017) -> pd.DataFrame:
    """
    Simulate transactions with realistic fraud injection patterns:
    - Normal: near home location, typical spend
    - Fraud: unusual location, high amounts, rapid bursts, high-risk merchants
    """
    records = []
    start_date = datetime(2024, 1, 1)

    for _, cust in customers.iterrows():
        n_txns = int(cust["txn_per_day"] * n_days * np.random.uniform(0.8, 1.2))

        for _ in range(n_txns):
            is_fraud = np.random.random() < fraud_rate
            merch = merchants.sample(1).iloc[0]
            ts = start_date + timedelta(
                days=int(np.random.randint(0, n_days)),
                hours=int(np.random.choice(range(24), p=_hour_weights())),
                minutes=int(np.random.randint(0, 60)),
            )

            if is_fraud:
                amount = np.random.lognormal(mean=10.5, sigma=1.5)  # much higher
                lat = cust["base_lat"] + np.random.uniform(-5, 5)   # far away
                lon = cust["base_lon"] + np.random.uniform(-5, 5)
                merch = merchants[merchants["merchant_category"].isin(HIGH_RISK_CATS)].sample(1).iloc[0] \
                        if random.random() > 0.4 else merch
            else:
                amount = np.random.lognormal(mean=np.log(cust["avg_spend"]), sigma=0.6)
                lat = cust["base_lat"] + np.random.normal(0, 0.02)
                lon = cust["base_lon"] + np.random.normal(0, 0.02)

            records.append({
                "transaction_id":    random_id("TXN_"),
                "customer_id":       cust["customer_id"],
                "merchant_id":       merch["merchant_id"],
                "amount":            round(max(1, amount), 2),
                "transaction_time":  ts,
                "merchant_category": merch["merchant_category"],
                "latitude":          round(lat, 4),
                "longitude":         round(lon, 4),
                "is_fraud":          int(is_fraud),
            })

    df = pd.DataFrame(records).sort_values("transaction_time").reset_index(drop=True)
    fraud_count = df["is_fraud"].sum()
    print(f"Generated {len(df):,} transactions | {fraud_count} fraud ({fraud_count/len(df)*100:.3f}%)")
    return df


def _hour_weights():
    """Weight hours so daytime transactions are more common."""
    weights = np.array([
        0.5,0.3,0.2,0.2,0.2,0.3,  # 0-5 AM
        0.8,1.5,2.5,3.0,3.5,3.5,  # 6-11 AM
        3.5,3.0,3.0,3.0,3.5,4.0,  # noon-5 PM
        4.0,3.5,2.5,2.0,1.5,1.0,  # 6-11 PM
    ], dtype=float)
    return weights / weights.sum()


def inject_fraud_burst(df: pd.DataFrame, n_bursts: int = 10) -> pd.DataFrame:
    """
    Add rapid-succession fraud bursts:
    Same customer, 5–12 transactions within 5 minutes.
    """
    df = df.copy()
    new_rows = []
    customers = df["customer_id"].unique()

    for _ in range(n_bursts):
        cust_id = np.random.choice(customers)
        base_row = df[df["customer_id"] == cust_id].sample(1).iloc[0]
        base_time = base_row["transaction_time"]

        for j in range(np.random.randint(5, 13)):
            r = base_row.copy()
            r["transaction_id"]   = random_id("TXN_BURST_")
            r["transaction_time"] = base_time + timedelta(seconds=j * 20)
            r["amount"]           = np.random.uniform(500, 5000)
            r["is_fraud"]         = 1
            new_rows.append(r)

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.sort_values("transaction_time").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    customers = generate_customers(n=2000)
    merchants = generate_merchants(n=500)
    df = generate_transactions(customers, merchants, n_days=60)
    df = inject_fraud_burst(df, n_bursts=20)

    out_path = "data/transactions_raw.csv"
    import os; os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved → {out_path}")
    print(df.head())