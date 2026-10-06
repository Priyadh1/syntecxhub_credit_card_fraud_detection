import random

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Fraud Investigation Console", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1300px;}
    .hero {background: linear-gradient(120deg, #1e3a8a 0%, #6d28d9 100%); padding: 20px 28px;
           border-radius: 16px; color: #fff; margin-bottom: 14px;}
    .hero h1 {margin: 0; font-size: 1.7rem; color: #fff;}
    .hero p {margin: 4px 0 0 0; opacity: .9; font-size: .95rem;}
    .card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 14px 18px;
           background: rgba(128,128,128,.07);}
    .card .lbl {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65;}
    .card .val {font-size: 1.45rem; font-weight: 700; margin-top: 2px;}
    .card .sub {font-size: .8rem; opacity: .7; margin-top: 2px;}
    .pill {display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: .9rem;}
    .p-low {background: rgba(34,197,94,.18); color: #16a34a;}
    .p-med {background: rgba(245,158,11,.20); color: #d97706;}
    .p-high {background: rgba(239,68,68,.18); color: #dc2626;}
    .p-crit {background: rgba(190,18,60,.22); color: #be123c;}
    .rec {border-left: 6px solid; border-radius: 10px; padding: 14px 18px; margin-top: 10px;
          background: rgba(128,128,128,.08);}
    .footer {text-align: center; opacity: .65; font-size: .85rem; margin-top: 30px;
             padding-top: 12px; border-top: 1px solid rgba(128,128,128,.25);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ load assets
@st.cache_resource
def load_assets():
    model = joblib.load("fraud_xgb_smote_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("creditcard_small.csv")
    feats = [c for c in df.columns if c != "Class"]
    mu = df[feats].mean()
    sd = df[feats].std().replace(0, 1)
    z_all = ((df[feats] - mu) / sd).values
    nn = NearestNeighbors(n_neighbors=11).fit(z_all)
    return model, scaler, df, feats, mu, sd, nn


model, scaler, df, FEATURES, ALL_MU, ALL_SD, NN = load_assets()

normal_df = df[df["Class"] == 0]
N_MEAN = normal_df[FEATURES].mean()
N_STD = normal_df[FEATURES].std().replace(0, 1)
TOP = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)


# ------------------------------------------------------------ agent tools
def risk_scoring(txn):
    row = pd.DataFrame([txn])[FEATURES].astype(float)
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
    return float(model.predict_proba(row)[0, 1])


def behaviour_check(txn, z_thr):
    flags = []
    for f in dict.fromkeys(TOP + ["Amount"]):
        z = (txn[f] - N_MEAN[f]) / N_STD[f]
        if abs(z) >= z_thr:
            flags.append((f, round(float(z), 1)))
    pct = float((normal_df["Amount"] < txn["Amount"]).mean())
    return flags, pct


def similarity_check(txn, self_idx, k):
    z = ((pd.Series(txn)[FEATURES].astype(float) - ALL_MU) / ALL_SD).values.reshape(1, -1)
    dist, ind = NN.kneighbors(z, n_neighbors=11)
    pairs = [(i, d) for i, d in zip(ind[0], dist[0]) if i != self_idx][:k]
    labels = df["Class"].values[[i for i, _ in pairs]]
    return int(labels.sum()), len(pairs), float(np.mean([d for _, d in pairs]))


def risk_level(p, review_t, block_t):
    if p >= 0.9:
        return "CRITICAL", "p-crit"
    if p >= block_t:
        return "HIGH", "p-high"
    if p >= review_t:
        return "MEDIUM", "p-med"
    return "LOW", "p-low"


def orchestrate(txn, self_idx, cfg):
    """Runs the multi-agent pipeline and returns a log of every agent."""
    log = []
    review_t, block_t = cfg["review_t"], cfg["block_t"]

    hour = int((txn["Time"] % 86400) // 3600)
    day = int(txn["Time"] // 86400) + 1
    log.append(dict(
        agent="Intake Agent", icon="📥", status="completed", color="#2563eb",
        summary=f"Record validated: amount ${txn['Amount']:.2f}, day {day}, hour {hour:02d}:00.",
        detail=[f"{len(FEATURES)} features received (Time, Amount, V1-V28).",
                "Derived day / hour from the Time offset."]))

    prob = risk_scoring(txn)
    log.append(dict(
        agent="Risk Scoring Agent", icon="🧠", status="completed", color="#7c3aed",
        summary=f"Fraud probability = {prob:.1%} (XGBoost + SMOTE).",
        detail=["Model trained on SMOTE-balanced data, tested on an untouched test set.",
                f"Score {prob:.3f} against thresholds review ≥ {review_t:.2f}, block ≥ {block_t:.2f}."]))

    flags, pct = [], None
    if prob < 0.05:
        log.append(dict(agent="Behavioural Baseline Agent", icon="📈", status="skipped", color="#9ca3af",
                        summary="Skipped: score is very low, deeper checks not needed.", detail=[]))
        log.append(dict(agent="Similarity Agent", icon="🕸️", status="skipped", color="#9ca3af",
                        summary="Skipped: score is very low.", detail=[]))
        votes, kk = 0, 0
    else:
        if cfg["use_behaviour"]:
            flags, pct = behaviour_check(txn, cfg["z_thr"])
            names = ", ".join(f"{f} (z={z})" for f, z in flags) or "none"
            log.append(dict(
                agent="Behavioural Baseline Agent", icon="📈", status="completed", color="#0891b2",
                summary=f"{len(flags)} feature(s) deviate ≥ {cfg['z_thr']:.0f}σ from normal behaviour.",
                detail=[f"Unusual: {names}.",
                        f"Amount is higher than {pct:.0%} of normal transactions."]))
        else:
            log.append(dict(agent="Behavioural Baseline Agent", icon="📈", status="disabled",
                            color="#9ca3af", summary="Disabled in sidebar.", detail=[]))
        if cfg["use_similarity"]:
            votes, kk, avg_d = similarity_check(txn, self_idx, cfg["k"])
            log.append(dict(
                agent="Similarity Agent", icon="🕸️", status="completed", color="#059669",
                summary=f"{votes} of {kk} most similar past cases were fraud.",
                detail=[f"Nearest-neighbour search over {len(df):,} historical transactions.",
                        f"Average distance to neighbours: {avg_d:.2f}."]))
        else:
            votes, kk = 0, 0
            log.append(dict(agent="Similarity Agent", icon="🕸️", status="disabled",
                            color="#9ca3af", summary="Disabled in sidebar.", detail=[]))

    corroborated = len(flags) >= 3 or (kk > 0 and votes >= max(1, int(0.8 * kk)))
    if prob >= block_t:
        action, state = "BLOCK", "⚡ Autonomously Blocked"
    elif prob >= review_t:
        action = "MANUAL REVIEW (high priority)" if corroborated else "MANUAL REVIEW"
        state = "⏳ Escalated to Human"
    else:
        action, state = "APPROVE", "✅ Auto-approved"
    log.append(dict(
        agent="Policy Agent", icon="⚖️", status="completed", color="#d97706",
        summary=f"Policy result: {action}.",
        detail=[f"Probability {prob:.1%} vs thresholds {review_t:.2f} / {block_t:.2f}.",
                "Grey-zone case is prioritised when other agents corroborate the risk."
                if review_t <= prob < block_t else "Clear-cut case: no human needed for the decision."]))

    lvl, lvl_cls = risk_level(prob, review_t, block_t)
    report = (f"{state}. The model estimates a {prob:.1%} chance of fraud ({lvl} risk). "
              f"{len(flags)} unusual feature(s) found"
              + (f" ({', '.join(f for f, _ in flags)})" if flags else "")
              + (f"; {votes} of {kk} similar past cases were fraud." if kk else "."))
    log.append(dict(agent="Case Manager Agent", icon="🗂️", status="completed", color="#dc2626",
                    summary=f"Ticket state: {state}.", detail=[report]))

    return dict(prob=prob, action=action, state=state, level=lvl, level_cls=lvl_cls,
                flags=flags, pct=pct, votes=votes, k=kk, log=log, report=report,
                hour=hour, day=day)


def sim_telemetry(seed):
    r = random.Random(seed)
    return pd.DataFrame({
        "Signal": ["Channel", "Device type", "Operating system", "IP reputation", "Geo-velocity check"],
        "Value": [r.choice(["Online", "In-store (chip)", "Mobile wallet"]),
                  r.choice(["Android phone", "iPhone", "Windows laptop", "POS terminal"]),
                  r.choice(["Android 14", "iOS 17", "Windows 11", "Embedded"]),
                  r.choice(["Clean", "Clean", "Clean", "Proxy detected"]),
                  r.choice(["Normal", "Normal", "Normal", "Impossible travel"])],
    })


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🛡️ Context & Inputs")
    mode = st.radio("Input mode", ["Case queue", "Search by ID", "Manual entry"])

    fraud_ids = df[df["Class"] == 1].sample(15, random_state=7).index.tolist()
    normal_ids = df[df["Class"] == 0].sample(15, random_state=7).index.tolist()
    queue = pd.Series(fraud_ids + normal_ids).sample(frac=1, random_state=3).tolist()

    self_idx = None
    if mode == "Case queue":
        self_idx = st.selectbox(
            "Open case", queue,
            format_func=lambda i: f"CASE-{i:06d} · ${df.loc[i, 'Amount']:.2f}")
        txn = df.loc[self_idx, FEATURES].to_dict()
        case_id = f"CASE-{self_idx:06d}"
    elif mode == "Search by ID":
        self_idx = int(st.number_input("Transaction ID (row number)", 0, len(df) - 1, 0))
        txn = df.loc[self_idx, FEATURES].to_dict()
        case_id = f"CASE-{self_idx:06d}"
    else:
        st.caption("Other features are filled with the typical normal value.")
        txn = N_MEAN.to_dict()
        txn["Amount"] = st.number_input("Amount ($)", 0.0, 30000.0, 120.0)
        txn["Time"] = st.number_input("Time offset (seconds)", 0.0, 200000.0, 50000.0)
        for f in [f for f in TOP if f not in ("Amount", "Time")]:
            txn[f] = st.number_input(f, value=float(N_MEAN[f]), format="%.3f")
        case_id = "CASE-MANUAL"

    st.divider()
    st.markdown("### ⚙️ System parameters")
    review_t = st.slider("Manual review at or above", 0.05, 0.95, 0.30, 0.05)
    block_t = st.slider("Auto-block at or above", 0.05, 0.99, 0.70, 0.05)
    if review_t >= block_t:
        st.warning("Review threshold should be below the block threshold.")

    st.markdown("### 🤖 Active agents")
    st.checkbox("Intake Agent", True, disabled=True)
    st.checkbox("Risk Scoring Agent", True, disabled=True)
    use_behaviour = st.checkbox("Behavioural Baseline Agent", True)
    use_similarity = st.checkbox("Similarity Agent", True)
    st.checkbox("Policy Agent", True, disabled=True)
    st.checkbox("Case Manager Agent", True, disabled=True)

    st.markdown("### 🔧 Tool configuration")
    z_thr = st.slider("Unusual-feature cut-off (σ)", 2.0, 5.0, 3.0, 0.5)
    k = st.slider("Similar cases to compare (k)", 3, 10, 5)

cfg = dict(review_t=review_t, block_t=block_t, use_behaviour=use_behaviour,
           use_similarity=use_similarity, z_thr=z_thr, k=k)
res = orchestrate(txn, self_idx, cfg)

# ------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <h1>🛡️ Credit-Card Fraud Investigation Console</h1>
      <p>Multi-agent decision pipeline on top of an XGBoost + SMOTE model · Built by Priyadharshini Murugan</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔎 Case investigation", "📊 Model performance", "🧩 Capabilities"])

# ------------------------------------------------------ tab 1: investigation
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="card"><div class="lbl">Case ID</div><div class="val">{case_id}</div>'
                f'<div class="sub">Day {res["day"]} · {res["hour"]:02d}:00</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="lbl">Overall risk level</div><div class="val">'
                f'<span class="pill {res["level_cls"]}">{res["level"]}</span></div>'
                f'<div class="sub">Fraud probability {res["prob"]:.1%}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="lbl">Transaction amount</div>'
                f'<div class="val">${txn["Amount"]:,.2f}</div><div class="sub">&nbsp;</div></div>',
                unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div class="lbl">Ticket state</div>'
                f'<div class="val" style="font-size:1.1rem">{res["state"]}</div>'
                f'<div class="sub">Action: {res["action"]}</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown("#### 🧾 Transaction profile")
        prof = {"Case ID": case_id, "Amount": f"${txn['Amount']:,.2f}",
                "Time offset": f"{int(txn['Time']):,} s (day {res['day']}, {res['hour']:02d}:00)"}
        for f in [f for f in TOP if f not in ("Amount", "Time")]:
            prof[f"{f} (key signal)"] = f"{txn[f]:.3f}"
        st.table(pd.DataFrame({"Field": list(prof), "Value": list(prof.values())}).set_index("Field"))

        st.markdown("#### 📈 Consumer behaviour baseline")
        pct = res["pct"] if res["pct"] is not None else float((normal_df["Amount"] < txn["Amount"]).mean())
        base = pd.DataFrame({
            "Metric": ["Typical normal amount (mean)", "Typical normal amount (median)",
                       "95th percentile of normal amounts", "This transaction vs normal amounts"],
            "Value": [f"${normal_df['Amount'].mean():,.2f}", f"${normal_df['Amount'].median():,.2f}",
                      f"${normal_df['Amount'].quantile(.95):,.2f}", f"higher than {pct:.0%}"],
        }).set_index("Metric")
        st.table(base)
        st.caption("Baseline is built from all normal transactions in the dataset "
                   "(the data is anonymised, so there is no per-cardholder history).")

        st.markdown("#### 📡 Device telemetry")
        st.table(sim_telemetry(case_id).set_index("Signal"))
        st.caption("⚠️ Simulated for demonstration. Device and location data are not part of "
                   "the dataset and do not affect the model's score.")

    with right:
        st.markdown("#### 🤖 Agent hub · orchestration timeline")
        for i, step in enumerate(res["log"], 1):
            icon = {"completed": "🟢", "skipped": "⚪", "disabled": "⚫"}[step["status"]]
            with st.expander(f"{icon}  {i}. {step['icon']} {step['agent']} — {step['status']}",
                             expanded=step["status"] == "completed"):
                st.markdown(f"**{step['summary']}**")
                for d in step["detail"]:
                    st.write("• " + d)
            if i < len(res["log"]):
                st.markdown("<div style='text-align:center; opacity:.45; line-height:.6'>│</div>",
                            unsafe_allow_html=True)

        color = {"p-low": "#16a34a", "p-med": "#d97706", "p-high": "#dc2626", "p-crit": "#be123c"}[res["level_cls"]]
        st.markdown(
            f'<div class="rec" style="border-color:{color}"><b>Final recommendation: {res["action"]}</b><br>'
            f'{res["report"]}</div>', unsafe_allow_html=True)
        if mode != "Manual entry":
            actual = int(df.loc[self_idx, "Class"])
            st.caption(f"Ground truth for this record: {'FRAUD' if actual == 1 else 'NORMAL'}")

# ------------------------------------------------- tab 2: model performance
with tab2:
    st.markdown("#### Results of all models (fraud class, 98 frauds in the test set)")
    perf = pd.DataFrame({
        "Model": ["RF baseline (no SMOTE)", "RF + SMOTE", "XGBoost + SMOTE"],
        "Precision": [1.000, 0.955, 0.905],
        "Recall": [0.837, 0.867, 0.878],
        "F1-score": [0.911, 0.909, 0.891],
        "ROC-AUC": [0.969, 0.969, 0.973],
    })
    long = perf.melt("Model", var_name="Metric", value_name="Score")
    chart = (alt.Chart(long).mark_bar()
             .encode(x=alt.X("Metric:N", title=None), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.7, 1.0])),
                     color="Model:N", xOffset="Model:N", tooltip=["Model", "Metric", "Score"])
             .properties(height=330, width="container"))
    st.altair_chart(chart)
    st.dataframe(perf.set_index("Model"))

    a, b = st.columns(2)
    with a:
        st.markdown("#### Confusion-matrix counts")
        st.table(pd.DataFrame({
            "Model": ["RF baseline", "RF + SMOTE", "XGBoost + SMOTE"],
            "True normal": [5001, 4997, 4992], "False alarms": [0, 4, 9],
            "Frauds missed": [16, 13, 12], "Frauds caught": [82, 85, 86]}).set_index("Model"))
    with b:
        st.markdown("#### Precision vs recall by threshold (XGBoost + SMOTE)")
        thr = pd.DataFrame({"Threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
                            "Precision": [0.604, 0.838, 0.905, 0.935, 0.988],
                            "Recall": [0.918, 0.898, 0.878, 0.878, 0.847]}).set_index("Threshold")
        st.line_chart(thr)
    st.caption("SMOTE shifts the tradeoff toward catching more fraud at the cost of a few false alarms. "
               "Differences are small because the test set has only 98 frauds.")

# --------------------------------------------------- tab 3: capabilities
with tab3:
    st.markdown("#### Modern fraud-detection capabilities and what this project covers")
    st.table(pd.DataFrame({
        "Capability": ["Real-time risk scoring", "Behavioural baselines", "Advanced machine learning",
                       "Graph AI and networks", "Adaptability"],
        "In this project": ["✅ Implemented", "🟡 Partly", "✅ Implemented", "🔜 Roadmap", "🔜 Roadmap"],
        "How": ["Each transaction is scored instantly from its amount, time offset and 28 anonymised features.",
                "Compared with the typical normal transaction in the dataset (no per-cardholder history exists).",
                "Random Forest and XGBoost trained on SMOTE-balanced data and compared with precision, recall, ROC-AUC.",
                "Needs card and merchant IDs, which this dataset does not have. The Similarity Agent is a nearest-neighbour proxy.",
                "The model is static. Retraining on new fraud patterns and drift monitoring are planned."],
    }).set_index("Capability"))
    st.info("The agents are rule-based and use the trained model as their main tool. "
            "No external LLM or API key is used.")

st.markdown('<div class="footer">Built by <b>Priyadharshini Murugan</b> · '
            'Syntecxhub Machine Learning Internship · Project 2: Credit-Card Fraud Detection</div>',
            unsafe_allow_html=True)
