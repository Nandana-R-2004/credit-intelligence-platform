import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Intelligence Platform",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
  .stApp {
    background-color: #f2fbf6;
}

body {
    color: #0b1f3a;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #111111;
}

.subtitle {
    text-align: center;
    color: #4a6fa5;
    font-size: 16px;
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #d6e6f5;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #000000;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")
)

metrics = joblib.load(
    os.path.join(BASE_DIR, "models", "model_metrics.pkl")
)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🏦 Credit Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered loan approval & risk decision system</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- LEFT PANEL ----------------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">👤 Applicant Profile</div>',
        unsafe_allow_html=True
    )

    age = st.number_input("Age (Years)", value=30)
    employment_years = st.number_input("Work Experience (Years)", value=5)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">💰 Financial Information</div>',
        unsafe_allow_html=True
    )

    income = st.number_input("Annual Income (₹)", value=300000)
    credit = st.number_input("Loan Amount Requested (₹)", value=200000)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT PANEL ----------------
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📊 Credit Intelligence Engine</div>',
        unsafe_allow_html=True
    )

    credit_history = st.number_input(
        "Credit History Score",
        value=0.5
    )

    financial_stability = st.number_input(
        "Financial Stability Score",
        value=0.5
    )

    external_rating = st.number_input(
        "External Credit Rating",
        value=0.5
    )

    st.caption(
        "These are AI-derived risk signals used by financial institutions"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.divider()

if st.button("🔍 Evaluate Credit Risk", use_container_width=True):

    input_data = pd.DataFrame([[
        credit_history,
        financial_stability,
        external_rating,
        income,
        credit,
        age,
        employment_years
    ]], columns=[
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AGE",
        "YEARS_EMPLOYED"
    ])

    prob = model.predict_proba(input_data)[0][1]

    st.markdown("## 📌 Decision Report")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Default Probability", f"{prob:.2%}")

    with result_col2:

        if prob < 0.35:
            st.success("🟢 APPROVED")
            risk_level = "LOW"

        elif prob < 0.65:
            st.warning("🟡 MANUAL REVIEW REQUIRED")
            risk_level = "MEDIUM"

        else:
            st.error("🔴 SPECIAL REVIEW REQUIRED")
            risk_level = "HIGH"

    with result_col3:
        st.info("AI Model: XGBoost Classifier")

    st.progress(float(prob))

    st.markdown("### 🧠 Decision Insight")

    if risk_level == "LOW":

        st.write(
            "Customer shows strong repayment capability based on financial and credit behavior patterns."
        )

    elif risk_level == "MEDIUM":

        st.write(
            "Moderate risk detected. Additional verification is recommended before final approval."
        )

    else:

        st.write(
            "The applicant has a higher predicted default risk. However, this does not automatically mean rejection."
        )

        st.info(
            """
Recommended Actions:

• Verify additional financial documents

• Consider collateral-backed lending

• Evaluate guarantor availability

• Check eligibility for government-supported loan schemes

• Conduct manual credit assessment
"""
        )

    st.caption(
        "This platform provides a risk assessment only. Final lending decisions should also consider human review, collateral, guarantors, regulatory requirements, and financial inclusion policies."
    )

    # ---------------- MODEL PERFORMANCE ----------------

    