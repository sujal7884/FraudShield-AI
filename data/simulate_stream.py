"""
Live Transaction Stream Simulator
===================================
Simulates a real bank feed — sends transactions to your API
automatically every 2 seconds. Watch fraud get caught in real time.

Run: python data/simulate_stream.py
"""

import requests
import random
import time
import json
from datetime import datetime

API = "http://localhost:8000/predict"

# ── Colours for terminal output ──
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Data pools ──
CUSTOMERS = [f"CUST_{i:04d}" for i in range(1, 201)]
MERCHANTS = [f"MERCH_{i:03d}" for i in range(1, 51)]

NORMAL_CATEGORIES    = ["grocery", "restaurant", "fuel", "pharmacy", "clothing", "electronics", "online_retail", "utilities"]
HIGH_RISK_CATEGORIES = ["crypto", "wire_transfer", "gambling", "prepaid_card", "money_order"]

MERCHANT_NAMES = {
    "grocery":       ["BigBazaar", "DMart", "Reliance Fresh", "More Supermarket"],
    "restaurant":    ["Zomato", "Swiggy", "McDonald's", "Dominos"],
    "fuel":          ["HP Petrol", "Indian Oil", "BPCL"],
    "crypto":        ["Binance", "CoinDCX", "WazirX", "Crypto.com"],
    "wire_transfer": ["Western Union", "MoneyGram", "SWIFT Transfer"],
    "gambling":      ["Betway", "Dream11", "My11Circle"],
    "online_retail": ["Amazon IN", "Flipkart", "Myntra", "Nykaa"],
    "electronics":   ["Croma", "Vijay Sales", "Apple Store"],
    "pharmacy":      ["Apollo Pharmacy", "MedPlus", "Netmeds"],
    "clothing":      ["H&M", "Zara", "Westside", "Pantaloons"],
    "utilities":     ["MSEB", "Tata Power", "Airtel", "Jio"],
    "prepaid_card":  ["Paytm Wallet", "Amazon Pay", "PhonePe"],
    "money_order":   ["India Post", "MoneyOrder Express"],
}


def make_normal_transaction(customer_id):
    """Generate a completely normal, legitimate transaction."""
    cat     = random.choice(NORMAL_CATEGORIES)
    amount  = round(random.uniform(100, 4000), 2)
    names   = MERCHANT_NAMES.get(cat, ["Unknown Merchant"])
    return {
        "transaction_id":      f"TXN_{int(time.time()*1000)}",
        "customer_id":         customer_id,
        "merchant_id":         random.choice(MERCHANTS),
        "amount":              amount,
        "merchant_category":   cat,
        "_merchant_name":      random.choice(names),
        # Features
        "hour":                datetime.now().hour,
        "is_night":            1 if datetime.now().hour >= 22 or datetime.now().hour <= 5 else 0,
        "is_weekend":          1 if datetime.now().weekday() >= 5 else 0,
        "amount_zscore":       round(random.uniform(-0.5, 1.5), 2),
        "amount_ratio_to_mean":round(random.uniform(0.5, 1.8), 2),
        "txn_count_1h":        random.randint(0, 2),
        "txn_amount_sum_1h":   round(random.uniform(0, 3000), 2),
        "txn_count_6h":        random.randint(1, 5),
        "txn_count_24h":       random.randint(2, 10),
        "geo_distance_km":     round(random.uniform(0, 15), 2),
        "travel_speed_kmh":    round(random.uniform(0, 60), 2),
        "impossible_travel":   0,
        "is_high_risk_merchant": 0,
        "is_new_merchant":     random.randint(0, 1),
        "cust_merch_freq":     random.randint(1, 20),
        "mins_since_last_txn": round(random.uniform(30, 800), 1),
        "is_rapid_succession": 0,
        "amount_zscore_merchant": round(random.uniform(-0.5, 1.2), 2),
    }


def make_fraud_transaction(customer_id, fraud_type):
    """Generate a fraudulent transaction of a specific type."""
    base = make_normal_transaction(customer_id)

    if fraud_type == "impossible_travel":
        base.update({
            "amount":              round(random.uniform(20000, 90000), 2),
            "merchant_category":   random.choice(HIGH_RISK_CATEGORIES),
            "_merchant_name":      "Suspicious Merchant",
            "impossible_travel":   1,
            "travel_speed_kmh":    round(random.uniform(900, 1500), 1),
            "geo_distance_km":     round(random.uniform(800, 2000), 1),
            "is_high_risk_merchant": 1,
            "amount_zscore":       round(random.uniform(3.5, 6.0), 2),
            "is_new_merchant":     1,
        })

    elif fraud_type == "velocity_burst":
        base.update({
            "amount":              round(random.uniform(1000, 8000), 2),
            "txn_count_1h":        random.randint(11, 20),
            "txn_amount_sum_1h":   round(random.uniform(30000, 80000), 2),
            "is_rapid_succession": 1,
            "mins_since_last_txn": round(random.uniform(0.5, 2.0), 1),
            "amount_zscore":       round(random.uniform(2.5, 4.5), 2),
        })

    elif fraud_type == "large_crypto":
        base.update({
            "amount":              round(random.uniform(50000, 95000), 2),
            "merchant_category":   "crypto",
            "_merchant_name":      random.choice(MERCHANT_NAMES["crypto"]),
            "is_high_risk_merchant": 1,
            "is_new_merchant":     1,
            "amount_zscore":       round(random.uniform(4.0, 7.0), 2),
            "amount_ratio_to_mean":round(random.uniform(8, 20), 2),
            "is_night":            1,
        })

    elif fraud_type == "wire_transfer":
        base.update({
            "amount":              round(random.uniform(30000, 80000), 2),
            "merchant_category":   "wire_transfer",
            "_merchant_name":      random.choice(MERCHANT_NAMES["wire_transfer"]),
            "is_high_risk_merchant": 1,
            "amount_zscore":       round(random.uniform(3.0, 5.5), 2),
            "txn_count_1h":        random.randint(3, 8),
            "is_rapid_succession": 1,
            "is_new_merchant":     1,
        })

    base["transaction_id"] = f"TXN_FRAUD_{int(time.time()*1000)}"
    return base


def send_transaction(txn):
    """Send transaction to API, return result."""
    merchant_name = txn.pop("_merchant_name", txn["merchant_category"])
    t0 = time.time()
    try:
        r       = requests.post(API, json=txn, timeout=10)
        latency = round((time.time() - t0) * 1000)
        result  = r.json()
        return result, merchant_name, latency
    except Exception as e:
        return {"error": str(e)}, merchant_name, 0


def print_result(txn, result, merchant_name, latency, fraud_type=None):
    """Pretty print the transaction result."""
    if "error" in result:
        print(f"{RED}  ✗ API Error: {result['error']}{RESET}")
        return

    score    = result.get("risk_score", 0)
    decision = result.get("decision", "?")
    xgb      = result.get("xgb_probability", 0)
    flags    = result.get("rule_flags", [])

    # Decision colour
    if decision == "BLOCK":
        dec_str = f"{RED}{BOLD}🔴 BLOCK{RESET}"
    elif decision == "REVIEW":
        dec_str = f"{YELLOW}{BOLD}🟡 REVIEW{RESET}"
    else:
        dec_str = f"{GREEN}🟢 ALLOW{RESET}"

    # Score bar
    filled  = int(score / 5)
    bar     = "█" * filled + "░" * (20 - filled)
    bar_col = RED if score >= 70 else YELLOW if score >= 35 else GREEN

    fraud_label = f" {YELLOW}[{fraud_type.upper().replace('_',' ')}]{RESET}" if fraud_type else ""

    print(f"\n{'─'*62}")
    print(f"  {BOLD}{txn['transaction_id'][-16:]}{RESET}{fraud_label}")
    print(f"  Customer : {txn['customer_id']}   Merchant: {merchant_name}")
    print(f"  Amount   : ₹{txn['amount']:,.2f}   Category: {txn['merchant_category']}")
    print(f"  Score    : {bar_col}[{bar}]{RESET} {BOLD}{score}/100{RESET}")
    print(f"  Decision : {dec_str}   XGBoost: {xgb*100:.1f}%   Latency: {latency}ms")
    if flags:
        print(f"  Flags    : {RED}{' · '.join(flags)}{RESET}")


def print_banner():
    print(f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════════════════╗
║         FraudShield AI — Live Transaction Stream         ║
║     Every transaction scored by YOUR XGBoost model       ║
╚══════════════════════════════════════════════════════════╝
{RESET}
  Sending transactions to {BOLD}localhost:8000/predict{RESET}
  Fraud injection rate: ~15%%
  Press {BOLD}Ctrl+C{RESET} to stop
""")


def main():
    print_banner()

    # Check API is up
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        h = r.json()
        if h.get("models_loaded"):
            print(f"  {GREEN}✅ API connected — models loaded{RESET}\n")
        else:
            print(f"  {YELLOW}⚠️  API up but models not loaded{RESET}\n")
    except:
        print(f"  {RED}✗ Cannot reach API at localhost:8000{RESET}")
        print(f"  Start it first: uvicorn src.api.main:app --port 8000 --reload\n")
        return

    # Stats counters
    stats = {"total": 0, "allow": 0, "review": 0, "block": 0}
    fraud_types = ["impossible_travel", "velocity_burst", "large_crypto", "wire_transfer"]

    print(f"  {'─'*60}")

    try:
        while True:
            customer = random.choice(CUSTOMERS)

            # 15% chance of fraud
            is_fraud = random.random() < 0.15

            if is_fraud:
                fraud_type = random.choice(fraud_types)
                txn        = make_fraud_transaction(customer, fraud_type)
            else:
                fraud_type = None
                txn        = make_normal_transaction(customer)

            result, merchant_name, latency = send_transaction(txn)
            print_result(txn, result, merchant_name, latency, fraud_type)

            # Update stats
            stats["total"] += 1
            dec = result.get("decision","")
            if dec == "ALLOW":  stats["allow"]  += 1
            if dec == "REVIEW": stats["review"] += 1
            if dec == "BLOCK":  stats["block"]  += 1

            # Print running stats every 10 transactions
            if stats["total"] % 10 == 0:
                fraud_rate = (stats["block"] / stats["total"]) * 100
                print(f"\n  {BLUE}{BOLD}── Stats ({stats['total']} transactions) ──────────────{RESET}")
                print(f"  🟢 Allowed: {stats['allow']}  🟡 Review: {stats['review']}  🔴 Blocked: {stats['block']}")
                print(f"  Fraud block rate: {fraud_rate:.1f}%")
                print(f"  {BLUE}{'─'*50}{RESET}")

            # Wait 2 seconds before next transaction
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n  {BOLD}Final Summary{RESET}")
        print(f"  {'─'*40}")
        print(f"  Total scored : {stats['total']}")
        print(f"  🟢 Allowed   : {stats['allow']}")
        print(f"  🟡 Review    : {stats['review']}")
        print(f"  🔴 Blocked   : {stats['block']}")
        if stats['total'] > 0:
            print(f"  Fraud rate   : {stats['block']/stats['total']*100:.1f}%")
        print(f"\n  Stream stopped. Goodbye!\n")


if __name__ == "__main__":
    main()