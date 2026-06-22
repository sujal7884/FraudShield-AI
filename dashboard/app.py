import streamlit as st, requests, json, time, random, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from datetime import datetime

API = "http://localhost:8000"

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.fraud-alert{background:#FCEBEB;border-left:4px solid #E24B4A;padding:10px 16px;border-radius:6px;margin:4px 0}
.review-alert{background:#FAEEDA;border-left:4px solid #EF9F27;padding:10px 16px;border-radius:6px;margin:4px 0}
.allow-alert{background:#EAF3DE;border-left:4px solid #1D9E75;padding:10px 16px;border-radius:6px;margin:4px 0}
.stMetric{background:var(--background-color);border-radius:10px}
</style>""", unsafe_allow_html=True)

# ── API helpers ──
def check_api():
    try: return requests.get(f"{API}/health", timeout=3).json()
    except: return None

def score_txn(txn):
    try: return requests.post(f"{API}/predict", json=txn, timeout=10).json()
    except Exception as e: return {"error": str(e)}

def get_stats():
    try: return requests.get(f"{API}/stats", timeout=3).json()
    except: return None

def get_model_info():
    try: return requests.get(f"{API}/model/info", timeout=3).json()
    except: return None

# ── Transaction generators ──
CUSTOMERS = [f"CUST_{i:04d}" for i in range(1, 201)]
MERCHANTS_MAP = {
    "grocery":["BigBazaar","DMart","Reliance Fresh"],"restaurant":["Zomato","Swiggy","Dominos"],
    "fuel":["HP Petrol","Indian Oil","BPCL"],"crypto":["Binance","CoinDCX","WazirX"],
    "wire_transfer":["Western Union","MoneyGram","SWIFT"],"gambling":["Betway","Dream11"],
    "online_retail":["Amazon IN","Flipkart","Myntra"],"electronics":["Croma","Apple Store"],
    "pharmacy":["Apollo Pharmacy","MedPlus"],"clothing":["H&M","Zara","Westside"],
    "utilities":["MSEB","Tata Power","Airtel"],"prepaid_card":["Paytm","Amazon Pay"],
}
NORMAL_CATS   = ["grocery","restaurant","fuel","pharmacy","clothing","online_retail","utilities","electronics"]
HIGH_RISK_CATS= ["crypto","wire_transfer","gambling","prepaid_card"]

def make_txn(force_fraud=False):
    is_fraud = force_fraud or random.random() < 0.15
    customer = random.choice(CUSTOMERS)
    if is_fraud:
        fraud_type = random.choice(["impossible_travel","velocity_burst","large_crypto","wire_transfer"])
        cat = random.choice(HIGH_RISK_CATS)
        amount = round(random.uniform(20000, 95000), 2)
        txn = {
            "transaction_id": f"TXN_{int(time.time()*1000)}",
            "customer_id": customer, "merchant_id": f"MERCH_{random.randint(1,50)}",
            "amount": amount, "merchant_category": cat,
            "amount_zscore": round(random.uniform(3.5, 7.0), 2),
            "impossible_travel": 1 if fraud_type=="impossible_travel" else 0,
            "travel_speed_kmh": round(random.uniform(900,1500),1) if fraud_type=="impossible_travel" else 0,
            "txn_count_1h": random.randint(8,15) if fraud_type=="velocity_burst" else random.randint(0,2),
            "is_rapid_succession": 1 if fraud_type in ["velocity_burst","wire_transfer"] else 0,
            "is_high_risk_merchant": 1, "is_new_merchant": 1, "is_night": 1,
            "amount_ratio_to_mean": round(random.uniform(8,20),2),
            "geo_distance_km": round(random.uniform(500,2000),1) if fraud_type=="impossible_travel" else 0,
            "mins_since_last_txn": round(random.uniform(0.5,2.0),1),
        }
        merchant = random.choice(MERCHANTS_MAP.get(cat, ["Unknown"]))
        return txn, merchant, fraud_type
    else:
        cat    = random.choice(NORMAL_CATS)
        amount = round(random.uniform(100, 4000), 2)
        txn = {
            "transaction_id": f"TXN_{int(time.time()*1000)}",
            "customer_id": customer, "merchant_id": f"MERCH_{random.randint(1,50)}",
            "amount": amount, "merchant_category": cat,
            "amount_zscore": round(random.uniform(-0.5, 1.2), 2),
            "impossible_travel": 0, "travel_speed_kmh": 0,
            "txn_count_1h": random.randint(0,2), "is_rapid_succession": 0,
            "is_high_risk_merchant": 0, "is_new_merchant": random.randint(0,1),
            "is_night": 1 if datetime.now().hour >= 22 or datetime.now().hour <= 5 else 0,
            "amount_ratio_to_mean": round(random.uniform(0.5, 1.8), 2),
            "geo_distance_km": round(random.uniform(0,15),1),
            "mins_since_last_txn": round(random.uniform(30,800),1),
        }
        merchant = random.choice(MERCHANTS_MAP.get(cat, ["Unknown"]))
        return txn, merchant, None

# ── Session state init ──
if "feed" not in st.session_state:        st.session_state.feed = []
if "stats" not in st.session_state:       st.session_state.stats = {"total":0,"allow":0,"review":0,"block":0}
if "stream_on" not in st.session_state:   st.session_state.stream_on = False
if "score_history" not in st.session_state: st.session_state.score_history = []

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("---")
    health = check_api()
    api_ok = bool(health and health.get("models_loaded"))
    if api_ok: st.success("✅ API Connected")
    else:      st.error("❌ API Offline — run uvicorn first")
    st.markdown("---")
    page = st.radio("Navigation", ["⚡ Live Stream","🎯 Score Transaction","📊 Overview","🧠 Model Info"])
    st.markdown("---")
    st.caption("XGBoost 60% + Isolation Forest 40%")
    st.caption("ALLOW:0-34 · REVIEW:35-69 · BLOCK:70-100")

# ════════════════════════════════════════
# PAGE — LIVE STREAM
# ════════════════════════════════════════
if "Live Stream" in page:
    st.markdown("# ⚡ Live Transaction Stream")
    st.caption("Transactions scored by your XGBoost model in real time")

    if not api_ok:
        st.error("Start your API first: `uvicorn src.api.main:app --port 8000 --reload`")
        st.stop()

    # Controls row
    col1, col2, col3, col4 = st.columns([1,1,1,2])
    with col1:
        start = st.button("▶ START", type="primary",  disabled=st.session_state.stream_on)
    with col2:
        stop  = st.button("⏹ STOP",  type="secondary", disabled=not st.session_state.stream_on)
    with col3:
        inject = st.button("💉 INJECT FRAUD", disabled=not st.session_state.stream_on)
    with col4:
        speed = st.select_slider("Speed", options=["Slow (3s)","Normal (1.5s)","Fast (0.5s)"], value="Normal (1.5s)")

    delay_map = {"Slow (3s)": 3.0, "Normal (1.5s)": 1.5, "Fast (0.5s)": 0.5}
    delay = delay_map[speed]

    if start: st.session_state.stream_on = True
    if stop:  st.session_state.stream_on = False
    if st.session_state.stream_on:
        st.markdown("🔴 **STREAMING LIVE** — transactions scoring every few seconds")
    else:
        st.info("Press ▶ START to begin the live stream")

    # KPI metrics
    s = st.session_state.stats
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Scored", s["total"])
    m2.metric("🟢 Allowed",   s["allow"])
    m3.metric("🟡 Review",    s["review"])
    m4.metric("🔴 Blocked",   s["block"])
    m5.metric("Fraud Rate",   f"{s['block']/max(s['total'],1)*100:.1f}%")

    st.markdown("---")

    # Charts + Feed side by side
    chart_col, feed_col = st.columns([1, 1])

    with chart_col:
        st.markdown("### 📊 Risk Score History")
        if st.session_state.score_history:
            df_hist = pd.DataFrame(st.session_state.score_history[-40:])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df_hist))), y=df_hist["score"],
                mode="lines+markers",
                marker=dict(color=df_hist["score"].apply(
                    lambda x: "#E24B4A" if x>=70 else "#EF9F27" if x>=35 else "#1D9E75"
                ), size=8),
                line=dict(color="#888", width=1)
            ))
            fig.add_hline(y=70, line_dash="dash", line_color="#E24B4A", annotation_text="BLOCK")
            fig.add_hline(y=35, line_dash="dash", line_color="#EF9F27", annotation_text="REVIEW")
            fig.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10),
                              yaxis=dict(range=[0,105]), showlegend=False,
                              xaxis_title="Transaction #", yaxis_title="Risk Score")
            st.plotly_chart(fig, use_container_width=True)

            # Decision pie
            if s["total"] > 0:
                pie_df = pd.DataFrame({
                    "Decision": ["Allow","Review","Block"],
                    "Count":    [s["allow"], s["review"], s["block"]]
                })
                fig2 = px.pie(pie_df, values="Count", names="Decision",
                              color="Decision",
                              color_discrete_map={"Allow":"#1D9E75","Review":"#EF9F27","Block":"#E24B4A"},
                              hole=0.4)
                fig2.update_layout(height=200, margin=dict(t=10,b=10,l=10,r=10),
                                   showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Start the stream to see charts")

    with feed_col:
        st.markdown("### 📋 Transaction Feed")
        feed_placeholder = st.empty()

        def render_feed():
            if not st.session_state.feed:
                feed_placeholder.info("No transactions yet")
                return
            html = ""
            for t in st.session_state.feed[:15]:
                dec   = t["decision"]
                score = t["score"]
                css   = "fraud-alert" if dec=="BLOCK" else "review-alert" if dec=="REVIEW" else "allow-alert"
                icon  = "🔴" if dec=="BLOCK" else "🟡" if dec=="REVIEW" else "🟢"
                flags = " · ".join(t["flags"]) if t["flags"] else ""
                fraud_badge = f'<span style="background:#854F0B;color:#fff;padding:1px 6px;border-radius:10px;font-size:10px;margin-left:6px">{t["fraud_type"].upper().replace("_"," ")}</span>' if t.get("fraud_type") else ""
                html += f"""
                <div class="{css}">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:600;font-size:13px">{icon} {dec} &nbsp; <span style="font-size:18px;font-weight:700">{score}</span>/100 {fraud_badge}</span>
                    <span style="font-size:11px;color:#666">{t['latency']}ms</span>
                  </div>
                  <div style="font-size:12px;margin-top:4px">
                    💳 <b>{t['merchant']}</b> &nbsp;·&nbsp; ₹{t['amount']:,.0f} &nbsp;·&nbsp; {t['category']} &nbsp;·&nbsp; {t['customer']}
                  </div>
                  {f'<div style="font-size:11px;color:#A32D2D;margin-top:2px">⚑ {flags}</div>' if flags else ''}
                </div>"""
            feed_placeholder.markdown(html, unsafe_allow_html=True)

        render_feed()

    # Stream loop
    if st.session_state.stream_on:
        force_fraud = inject
        txn, merchant, fraud_type = make_txn(force_fraud=force_fraud)
        t0     = time.time()
        result = score_txn(txn)
        latency= round((time.time()-t0)*1000)

        if "risk_score" in result:
            entry = {
                "decision":   result["decision"],
                "score":      result["risk_score"],
                "xgb":        result["xgb_probability"],
                "flags":      result.get("rule_flags",[]),
                "merchant":   merchant,
                "amount":     txn["amount"],
                "category":   txn["merchant_category"],
                "customer":   txn["customer_id"],
                "latency":    latency,
                "fraud_type": fraud_type,
            }
            st.session_state.feed.insert(0, entry)
            st.session_state.feed = st.session_state.feed[:50]
            st.session_state.score_history.append({"score": result["risk_score"], "decision": result["decision"]})
            st.session_state.score_history = st.session_state.score_history[-100:]

            d = result["decision"]
            st.session_state.stats["total"] += 1
            st.session_state.stats[d.lower()] += 1

        time.sleep(delay)
        st.rerun()

# ════════════════════════════════════════
# PAGE — SCORE TRANSACTION
# ════════════════════════════════════════
elif "Score Transaction" in page:
    st.markdown("# 🎯 Score a Transaction")
    st.caption("Sends directly to your XGBoost model at localhost:8000/predict")

    PRESETS = {
        "🔴 High risk — crypto + impossible travel": {
            "transaction_id":"TXN001","customer_id":"CUST4821","merchant_id":"M999",
            "amount":85000,"merchant_category":"crypto","amount_zscore":4.2,
            "impossible_travel":1,"txn_count_1h":8,"is_new_merchant":1,
            "is_night":1,"is_high_risk_merchant":1,"is_rapid_succession":1,"travel_speed_kmh":1200,
        },
        "🟢 Normal — grocery purchase": {
            "transaction_id":"TXN002","customer_id":"CUST1234","merchant_id":"M001",
            "amount":850,"merchant_category":"grocery","amount_zscore":0.3,
            "impossible_travel":0,"txn_count_1h":1,"is_new_merchant":0,"is_high_risk_merchant":0,
        },
        "🟡 Suspicious wire transfer": {
            "transaction_id":"TXN003","customer_id":"CUST7777","merchant_id":"MWIRE",
            "amount":45000,"merchant_category":"wire_transfer","amount_zscore":3.1,
            "txn_count_1h":5,"is_new_merchant":1,"is_high_risk_merchant":1,"is_rapid_succession":1,
        },
    }

    preset = st.selectbox("Load preset:", ["Custom"]+list(PRESETS.keys()))
    default = json.dumps(PRESETS.get(preset,{"transaction_id":"TXN001","customer_id":"CUST001","merchant_id":"M001","amount":1500,"merchant_category":"online_retail"}),indent=2)

    col1, col2 = st.columns(2)
    with col1:
        txt = st.text_area("Transaction JSON", value=default, height=320)
        btn = st.button("▶ SCORE WITH YOUR MODEL", type="primary", disabled=not api_ok)
    with col2:
        if btn:
            try:
                txn = json.loads(txt)
                with st.spinner("Running XGBoost + Isolation Forest..."):
                    t0  = time.time()
                    res = score_txn(txn)
                    lat = round((time.time()-t0)*1000)
                if "risk_score" not in res:
                    st.error(f"API Error: {json.dumps(res)}")
                else:
                    sc=res["risk_score"]; dec=res["decision"]
                    emoji={"BLOCK":"🔴","REVIEW":"🟡","ALLOW":"🟢"}[dec]
                    st.markdown(f"## {emoji} {dec}")
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Risk Score",f"{sc}/100")
                    c2.metric("XGBoost",f"{res['xgb_probability']*100:.1f}%")
                    c3.metric("Anomaly",f"{res['anomaly_score']*100:.1f}%")
                    c4.metric("Latency",f"{lat}ms")
                    color="#E24B4A" if sc>=70 else "#EF9F27" if sc>=35 else "#1D9E75"
                    fig=go.Figure(go.Indicator(mode="gauge+number",value=sc,
                        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                               "steps":[{"range":[0,34],"color":"#EAF3DE"},{"range":[35,69],"color":"#FAEEDA"},{"range":[70,100],"color":"#FCEBEB"}]}))
                    fig.update_layout(height=220,margin=dict(t=30,b=0,l=20,r=20))
                    st.plotly_chart(fig,use_container_width=True)
                    flags=res.get("rule_flags",[])
                    if flags:
                        st.markdown("**Triggered rules:**")
                        cols=st.columns(min(len(flags),3))
                        for i,f in enumerate(flags): cols[i%3].error(f.replace("_"," "))
                    else: st.success("No rules triggered")
            except json.JSONDecodeError: st.error("Invalid JSON")
        elif not api_ok: st.warning("Start API: `uvicorn src.api.main:app --port 8000 --reload`")
        else: st.info("Click the button to score using your trained model")

# ════════════════════════════════════════
# PAGE — OVERVIEW
# ════════════════════════════════════════
elif "Overview" in page:
    st.markdown("# 📊 Overview")
    stats = get_stats()
    if stats:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Fraud Blocked",stats["fraud_blocked"])
        c2.metric("Fraud Rate",f"{stats['fraud_rate_pct']}%")
        c3.metric("Avg Risk Score",stats["avg_risk_score"])
        c4.metric("High Risk",stats["high_risk_count"])
        if stats.get("top_flagged_categories"):
            st.markdown("### 🚨 Top Flagged Categories")
            cols=st.columns(len(stats["top_flagged_categories"]))
            for i,cat in enumerate(stats["top_flagged_categories"]):
                cols[i].error(f"#{i+1} {cat.replace('_',' ').title()}")
    col1,col2=st.columns(2)
    with col1:
        hours=[f"{h:02d}:00" for h in range(24)]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hours,y=[random.randint(2500,4500) for _ in range(24)],fill="tozeroy",name="Legit",line=dict(color="#378ADD")))
        fig.add_trace(go.Scatter(x=hours,y=[random.randint(2,20) for _ in range(24)],fill="tozeroy",name="Fraud",line=dict(color="#E24B4A")))
        fig.update_layout(title="24h Volume",height=300,margin=dict(t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        cats=["Crypto","Wire Transfer","Gambling","Prepaid Card","Online Retail","ATM"]
        rates=[8.3,5.4,5.8,2.5,0.29,0.82]
        df2=pd.DataFrame({"Category":cats,"Rate":rates}).sort_values("Rate")
        fig2=px.bar(df2,x="Rate",y="Category",orientation="h",title="Fraud Rate % by Category",
                    color="Rate",color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"])
        fig2.update_layout(height=300,margin=dict(t=40,b=0),coloraxis_showscale=False)
        st.plotly_chart(fig2,use_container_width=True)

# ════════════════════════════════════════
# PAGE — MODEL INFO
# ════════════════════════════════════════
elif "Model Info" in page:
    st.markdown("# 🧠 Model Information")
    info=get_model_info()
    if not info: st.error("API offline"); st.stop()
    col1,col2=st.columns(2)
    for i,(name,det) in enumerate(info["models"].items()):
        c=col1 if i==0 else col2
        with c:
            st.markdown(f"### {name.replace('_',' ').title()}")
            for k,v in det.items(): st.markdown(f"**{k.replace('_',' ').title()}:** `{v}`")
    st.markdown("### Thresholds")
    c1,c2,c3=st.columns(3)
    c1.success(f"✅ ALLOW: {info['thresholds']['allow']}")
    c2.warning(f"🟡 REVIEW: {info['thresholds']['review']}")
    c3.error(f"🔴 BLOCK: {info['thresholds']['block']}")
    st.markdown("### Ensemble Formula")
    st.code("risk_score = (0.60 × XGBoost_prob + 0.40 × IForest_score) × 100\n           + business_rule_boost (max +40)\n\nBLOCK  if score >= 70\nREVIEW if score >= 35\nALLOW  otherwise")