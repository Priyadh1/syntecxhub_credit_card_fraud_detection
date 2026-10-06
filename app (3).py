import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Investigation Agent", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .hero {background: linear-gradient(120deg, #1e3a8a, #6d28d9); padding: 26px 32px;
           border-radius: 16px; color: white; margin-bottom: 18px;}
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: 6px 0 0 0; opacity: 0.9;}
    </style>
    <div class="hero">
      <h1>🛡️ Credit-Card Fraud Investigation Agent</h1>
      <p>XGBoost + SMOTE model with a tool-using decision agent: predict, check, decide, explain.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- load assets
@st.cache_resource
def load_assets():
    model = joblib.load("fraud_xgb_smote_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("creditcard_small.csv")
    return model, scaler, df


model, scaler, df = load_assets()
FEATURES = [c for c in df.columns if c != "Class"]

normal_df = df[df["Class"] == 0]
normal_mean = normal_df[FEATURES].mean()
normal_std = normal_df[FEATURES].std()
top_features = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)


# -------------------------------------------------------------- agent tools
def predict_fraud(txn):
    """TOOL 1: fraud probability from the trained XGBoost model."""
    row = pd.DataFrame([txn])[FEATURES].astype(float)
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
    return float(model.predict_proba(row)[0, 1])


def check_unusual(txn):
    """TOOL 2: features far (3+ std) from normal behaviour."""
    flags = []
    for f in dict.fromkeys(top_features + ["Amount"]):
        z = (txn[f] - normal_mean[f]) / normal_std[f]
        if abs(z) >= 3:
            flags.append((f, round(float(z), 1)))
    return flags


def decide_action(prob, flags, review_t, block_t):
    """TOOL 3: business decision using the thresholds."""
    if prob >= block_t:
        return "BLOCK"
    if prob >= review_t:
        return "BLOCK (escalated)" if len(flags) >= 3 else "MANUAL REVIEW"
    return "APPROVE"


def fraud_agent(txn, review_t, block_t):
    steps = []
    prob = predict_fraud(txn)
    steps.append(f"Tool 1 · predict_fraud → fraud probability = {prob:.3f}")

    if prob < 0.05:
        flags = []
        steps.append("Very low risk, so deeper checks were skipped.")
    else:
        flags = check_unusual(txn)
        names = ", ".join(f"{f} (z={z})" for f, z in flags) or "none"
        steps.append(f"Tool 2 · check_unusual → {len(flags)} unusual feature(s): {names}")

    action = decide_action(prob, flags, review_t, block_t)
    steps.append(f"Tool 3 · decide_action → {action}")

    report = (
        f"Decision: {action}. The model estimates a {prob:.1%} chance of fraud. "
        f"{len(flags)} feature(s) differ strongly from normal transactions"
        + (f" ({', '.join(f for f, _ in flags)})." if flags else ".")
    )
    return {"action": action, "prob": prob, "steps": steps, "report": report}


def show_result(result, actual=None):
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{result['prob']:.1%}")
    c2.metric("Agent decision", result["action"])
    if actual is not None:
        c3.metric("Actual label", "FRAUD" if actual == 1 else "NORMAL")
    st.progress(min(max(result["prob"], 0.0), 1.0))

    if result["action"].startswith("BLOCK"):
        st.error(result["report"])
    elif result["action"] == "MANUAL REVIEW":
        st.warning(result["report"])
    else:
        st.success(result["report"])

    with st.expander("See how the agent worked (step by step)", expanded=True):
        for s in result["steps"]:
            st.write("• " + s)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Business thresholds")
    review_t = st.slider("Send to manual review at or above", 0.05, 0.95, 0.30, 0.05)
    block_t = st.slider("Block at or above", 0.05, 0.99, 0.70, 0.05)
    if review_t >= block_t:
        st.warning("Review threshold should be lower than the block threshold.")
    st.caption(
        "Lower thresholds catch more fraud but flag more genuine customers. "
        "Defaults come from the precision-recall analysis (0.3 / 0.7)."
    )
    st.divider()
    st.caption(
        "Demo built on a sampled version of the Credit Card Fraud Detection dataset. "
        "The agent is rule-based and uses the trained model as its main tool."
    )

# ------------------------------------------------------------------- tabs
tab1, tab2, tab3 = st.tabs(["🔎 Check a transaction", "✍️ Enter your own values", "📊 About the model"])

with tab1:
    st.subheader("Pick a transaction from the dataset")
    fraud_ids = df[df["Class"] == 1].sample(15, random_state=7).index
    normal_ids = df[df["Class"] == 0].sample(15, random_state=7).index
    ids = pd.Index(fraud_ids.tolist() + normal_ids.tolist()).to_series().sample(frac=1, random_state=3).tolist()

    choice = st.selectbox(
        "Transaction",
        ids,
        format_func=lambda i: f"Transaction #{i}  ·  Amount ${df.loc[i, 'Amount']:.2f}",
    )
    txn = df.loc[choice, FEATURES].to_dict()
    st.caption("The actual label is revealed only after you run the check.")

    if st.button("Check transaction", type="primary"):
        result = fraud_agent(txn, review_t, block_t)
        show_result(result, actual=int(df.loc[choice, "Class"]))

with tab2:
    st.subheader("Enter a transaction manually")
    st.caption("Remaining features are filled with the typical normal-transaction value.")
    cols = st.columns(3)
    manual = normal_mean.to_dict()
    manual["Amount"] = cols[0].number_input("Amount ($)", 0.0, 30000.0, 100.0)
    manual["Time"] = cols[1].number_input("Time (seconds)", 0.0, 200000.0, float(normal_mean["Time"]))
    for i, f in enumerate([f for f in top_features if f not in ("Amount", "Time")]):
        manual[f] = cols[i % 3].number_input(f, value=float(normal_mean[f]), format="%.3f")

    if st.button("Check my transaction", type="primary"):
        show_result(fraud_agent(manual, review_t, block_t))

with tab3:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Data:** sampled Credit Card Fraud Detection dataset (about 1.9% fraud).
        2. **Imbalance fix:** SMOTE creates synthetic fraud examples in the training data only.
        3. **Model:** XGBoost trained on the balanced data, evaluated on an untouched test set.
        4. **Agent:** predicts the fraud probability, checks for unusual features, applies the
           business thresholds and explains its decision.
        """
    )
    st.markdown("**Test-set results (fraud class, 98 frauds):**")
    st.table(
        pd.DataFrame(
            {
                "Precision": [0.905],
                "Recall": [0.878],
                "F1": [0.891],
                "ROC-AUC": [0.973],
            },
            index=["XGBoost + SMOTE"],
        )
    )
