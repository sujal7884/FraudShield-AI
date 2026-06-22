import streamlit as st, requests, json, time, random, pandas as pd
import plotly.express as px, plotly.graph_objects as go

API = "http://localhost:8000"

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-badge{background:#FCEBEB;color:#A32D2D;padding:4px 12px;border-radius:20px;font-weight:700}
.allow-badge{background:#EAF3DE;color:#3B6D11;padding:4px 12px;border-radius:20px;font-weight:700}
</style>""", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("---")
    health = check_api()
    if health and health.get("models_loaded"):
        st.success("✅ API Connected · models loaded")
    else:
        st.error("❌ API Offline")
        st.caption("Run: uvicorn src.api.main:app --port 8000 --reload")
    st.markdown("---")
    page = st.radio("Navigation", ["🎯 Score Transaction","⚡ Live Feed","📊 Overview","🧠 Model Info"])
    st.markdown("---")
    st.caption("XGBoost 60% + Isolation Forest 40%")
    st.caption("ALLOW:0-34 · REVIEW:35-69 · BLOCK:70-100")

api_online = bool(health and health.get("models_loaded"))
PRESETS = {
    "🔴 High risk":{"transaction_id":"TXN001","customer_id":"CUST4821","merchant_id":"M999","amount":85000,"merchant_category":"crypto","amount_zscore":4.2,"impossible_travel":1,"txn_count_1h":8,"is_new_merchant":1,"is_night":1,"is_high_risk_merchant":1,"is_rapid_succession":1,"travel_speed_kmh":1200},
    "🟢 Normal":{"transaction_id":"TXN002","customer_id":"CUST1234","merchant_id":"M001","amount":850,"merchant_category":"grocery","amount_zscore":0.3,"impossible_travel":0,"txn_count_1h":1,"is_new_merchant":0,"is_night":0,"is_high_risk_merchant":0},
    "🟡 Suspicious wire":{"transaction_id":"TXN003","customer_id":"CUST7777","merchant_id":"MWIRE","amount":45000,"merchant_category":"wire_transfer","amount_zscore":3.1,"txn_count_1h":5,"is_new_merchant":1,"is_high_risk_merchant":1,"is_rapid_succession":1},
}

if "Score Transaction" in page:
    st.markdown("# 🎯 Score a Transaction")
    st.caption("Sends JSON directly to your XGBoost model at localhost:8000/predict")
    preset = st.selectbox("Load preset:", ["Custom"]+list(PRESETS.keys()))
    default = json.dumps(PRESETS.get(preset,{"transaction_id":"TXN001","customer_id":"CUST001","merchant_id":"M001","amount":1500,"merchant_category":"online_retail"}), indent=2)
    col1, col2 = st.columns(2)
    with col1:
        txt = st.text_area("Transaction JSON", value=default, height=340)
        btn = st.button("▶ SCORE WITH YOUR MODEL", type="primary", disabled=not api_online)
    with col2:
        if btn:
            try:
                txn = json.loads(txt)
                with st.spinner("Running XGBoost + Isolation Forest..."):
                    res = score_txn(txn)
                if "error" in res:
                    st.error(res["error"])
                else:
                    sc = res["risk_score"]; dec = res["decision"]
                    emoji = {"BLOCK":"🔴","REVIEW":"🟡","ALLOW":"🟢"}[dec]
                    st.markdown(f"## {emoji} {dec}")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Risk Score", f"{sc}/100")
                    c2.metric("XGBoost", f"{res['xgb_probability']*100:.1f}%")
                    c3.metric("Anomaly", f"{res['anomaly_score']*100:.1f}%")
                    c4.metric("Latency", f"{res['latency_ms']}ms")
                    color = "#E24B4A" if sc>=70 else "#EF9F27" if sc>=35 else "#1D9E75"
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=sc,
                        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                               "steps":[{"range":[0,34],"color":"#EAF3DE"},{"range":[35,69],"color":"#FAEEDA"},{"range":[70,100],"color":"#FCEBEB"}]}))
                    fig.update_layout(height=220, margin=dict(t=30,b=0,l=20,r=20))
                    st.plotly_chart(fig, use_container_width=True)
                    flags = res.get("rule_flags",[])
                    if flags:
                        st.markdown("**Triggered rules:**")
                        for f in flags: st.error(f.replace("_"," "))
                    else:
                        st.success("No rules triggered")
                    st.caption(f"Scored at {res['timestamp']}")
            except json.JSONDecodeError:
                st.error("Invalid JSON")
        elif not api_online:
            st.warning("Start API:\n```\nuvicorn src.api.main:app --port 8000 --reload\n```")
        else:
            st.info("Click the button to score using your trained model")

elif "Live Feed" in page:
    st.markdown("# ⚡ Live Transaction Feed")
    st.caption("Scores random transactions through your real XGBoost model")
    if not api_online: st.error("API offline"); st.stop()
    CATS = ["online_retail","crypto","gambling","restaurant","grocery","wire_transfer","prepaid_card"]
    HR   = {"crypto","gambling","wire_transfer","prepaid_card"}
    MERCHANTS = ["Amazon IN","Binance","PayTM","Zomato","PhonePe","ICICI","Swiggy","MakeMyTrip","Cred","Zerodha"]
    n = st.slider("How many transactions to score", 5, 30, 10)
    if st.button("▶ SCORE BATCH NOW", type="primary"):
        results = []
        bar = st.progress(0)
        for i in range(n):
            is_hr = random.random() < 0.15
            cat   = random.choice(list(HR)) if is_hr else random.choice([c for c in CATS if c not in HR])
            fraud = is_hr and random.random() < 0.5
            txn = {"transaction_id":f"LIVE_{i}","customer_id":f"CUST_{random.randint(1,2000)}","merchant_id":f"M_{random.randint(1,500)}",
                   "amount":random.randint(10000,90000) if fraud else random.randint(100,5000),
                   "merchant_category":cat,"amount_zscore":round(random.uniform(2,5),1) if fraud else round(random.uniform(0,1.5),1),
                   "is_high_risk_merchant":1 if is_hr else 0,"impossible_travel":1 if fraud and random.random()<0.3 else 0,
                   "txn_count_1h":random.randint(5,12) if fraud else random.randint(0,2),"is_new_merchant":1 if random.random()<0.2 else 0,
                   "is_rapid_succession":1 if fraud and random.random()<0.4 else 0,
                   "mins_since_last_txn":random.randint(0,3) if fraud else random.randint(30,600)}
            r = score_txn(txn)
            if "error" not in r:
                results.append({"Merchant":random.choice(MERCHANTS),"Amount":f"₹{txn['amount']:,}",
                                 "Score":r["risk_score"],"Decision":r["decision"],
                                 "XGBoost %":f"{r['xgb_probability']*100:.1f}%","Flags":", ".join(r.get("rule_flags",[]) or ["—"])})
            bar.progress((i+1)/n)
            time.sleep(0.05)
        bar.empty()
        if results:
            df = pd.DataFrame(results)
            blocked = sum(1 for r in results if r["Decision"]=="BLOCK")
            review  = sum(1 for r in results if r["Decision"]=="REVIEW")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total",len(results)); c2.metric("🔴 Blocked",blocked)
            c3.metric("🟡 Review",review);   c4.metric("🟢 Allowed",len(results)-blocked-review)
            fig = px.histogram(x=[r["Score"] for r in results], nbins=15, title="Risk Score Distribution", color_discrete_sequence=["#378ADD"])
            fig.add_vline(x=35,line_dash="dash",line_color="#EF9F27",annotation_text="Review threshold")
            fig.add_vline(x=70,line_dash="dash",line_color="#E24B4A",annotation_text="Block threshold")
            fig.update_layout(height=260,margin=dict(t=40,b=0),showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)

elif "Overview" in page:
    st.markdown("# 📊 Overview")
    stats = get_stats()
    if stats:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Fraud Blocked",stats["fraud_blocked"]); c2.metric("Fraud Rate",f"{stats['fraud_rate_pct']}%")
        c3.metric("Avg Risk Score",stats["avg_risk_score"]); c4.metric("High Risk",stats["high_risk_count"])
        if stats.get("top_flagged_categories"):
            st.markdown("### 🚨 Top Flagged Categories")
            cols = st.columns(len(stats["top_flagged_categories"]))
            for i,cat in enumerate(stats["top_flagged_categories"]):
                cols[i].error(f"#{i+1} {cat.replace('_',' ').title()}")
    col1,col2 = st.columns(2)
    with col1:
        hours=[f"{h:02d}:00" for h in range(24)]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hours,y=[random.randint(2500,4500) for _ in range(24)],fill="tozeroy",name="Legit",line=dict(color="#378ADD")))
        fig.add_trace(go.Scatter(x=hours,y=[random.randint(2,20) for _ in range(24)],fill="tozeroy",name="Fraud",line=dict(color="#E24B4A")))
        fig.update_layout(title="24h Volume",height=280,margin=dict(t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        cats=["Crypto","Wire Transfer","Gambling","Prepaid Card","Online Retail","ATM"]
        rates=[8.3,5.4,5.8,2.5,0.29,0.82]
        df2=pd.DataFrame({"Category":cats,"Rate":rates}).sort_values("Rate")
        fig2=px.bar(df2,x="Rate",y="Category",orientation="h",title="Fraud Rate % by Category",color="Rate",color_continuous_scale=["#1D9E75","#EF9F27","#E24B4A"])
        fig2.update_layout(height=280,margin=dict(t=40,b=0),coloraxis_showscale=False)
        st.plotly_chart(fig2,use_container_width=True)

elif "Model Info" in page:
    st.markdown("# 🧠 Model Information")
    info = get_model_info()
    if not info: st.error("API offline"); st.stop()
    col1,col2 = st.columns(2)
    for i,(name,det) in enumerate(info["models"].items()):
        c = col1 if i==0 else col2
        with c:
            st.markdown(f"### {name.replace('_',' ').title()}")
            for k,v in det.items(): st.markdown(f"**{k.replace('_',' ').title()}:** `{v}`")
    st.markdown("### Thresholds")
    c1,c2,c3 = st.columns(3)
    c1.success(f"✅ ALLOW: {info['thresholds']['allow']}")
    c2.warning(f"🟡 REVIEW: {info['thresholds']['review']}")
    c3.error(f"🔴 BLOCK: {info['thresholds']['block']}")
    st.markdown("### Ensemble Formula")
    st.code("risk_score = (0.60 × XGBoost_prob + 0.40 × IForest_score) × 100\n           + business_rule_boost (max +40)\n\nBLOCK  if score >= 70\nREVIEW if score >= 35\nALLOW  otherwise")
