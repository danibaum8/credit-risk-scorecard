import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="",
    layout="wide",
)

BLUE   = "#1B4F72"   # Option A — Navy
ORANGE = "#E74C3C"   # Option A — Alert Red
MID    = "#5D8AA8"   # mid-tone blue for moderate risk

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1  { color: #111111; font-weight: 700; font-size: 1.8rem; }
    h3  { color: #333333; font-weight: 600; }
    .score-band { border-radius: 8px; padding: 1rem 1.4rem; margin-top: 0.5rem; }
    .band-low   { background: #EAF0F6; border-left: 4px solid #1B4F72; }
    .band-mid   { background: #FDF3EC; border-left: 4px solid #5D8AA8; }
    .band-high  { background: #FCEAE3; border-left: 4px solid #E74C3C; }
    .stSelectbox label, .stNumberInput label, .stSlider label { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("model/inference.pkl", "rb") as f:
        return pickle.load(f)

bundle        = load_bundle()
lr            = bundle["lr"]
scaler        = bundle["scaler"]
woe_maps      = bundle["woe_maps"]
feature_names = bundle["feature_names"]
mappings      = bundle["mappings"]

PDO        = bundle["pdo"]
BASE_SCORE = bundle["base_score"]
BASE_ODDS  = bundle["base_odds"]
FACTOR     = PDO / np.log(2)
OFFSET     = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

CAT_COLS = list(woe_maps.keys())

# ── Score function ─────────────────────────────────────────────────────────────
def compute_score(inputs: dict) -> int:
    row = {}
    for col in feature_names:
        if col in CAT_COLS:
            row[col] = woe_maps[col].get(inputs[col], 0.0)
        else:
            row[col] = inputs[col]
    X = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X)
    p_bad = lr.predict_proba(X_scaled)[0, 1]
    p_bad = np.clip(p_bad, 1e-6, 1 - 1e-6)
    log_odds = np.log(p_bad / (1 - p_bad))
    score = OFFSET - FACTOR * log_odds
    return int(np.clip(round(score), 0, 1000))

# ── Score band helper ──────────────────────────────────────────────────────────
def get_band(score):
    if score >= 700:
        return "Low Risk", "0%", "band-low", BLUE
    elif score >= 600:
        return "Moderate-Low Risk", "~10-14%", "band-low", BLUE
    elif score >= 500:
        return "Moderate Risk", "~9-10%", "band-mid", "#E8A87C"
    elif score >= 400:
        return "Moderate-High Risk", "~40-70%", "band-high", ORANGE
    else:
        return "High Risk", ">70%", "band-high", ORANGE

# ── Gauge chart ────────────────────────────────────────────────────────────────
def make_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 52, "color": "#111111"}},
        gauge={
            "axis": {
                "range": [0, 1000],
                "tickwidth": 1,
                "tickcolor": "#CCCCCC",
                "tickvals": [0, 200, 400, 500, 600, 700, 800, 1000],
                "ticktext": ["0", "200", "400", "500", "600", "700", "800", "1000"],
            },
            "bar": {"color": BLUE if score >= 500 else ORANGE, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   400], "color": "#FCEAE3"},
                {"range": [400, 500], "color": "#FDF3EC"},
                {"range": [500, 700], "color": "#EAF0F6"},
                {"range": [700, 1000],"color": "#D6E8F2"},
            ],
            "threshold": {
                "line": {"color": "#111111", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        margin=dict(t=20, b=10, l=30, r=30),
        height=280,
        paper_bgcolor="white",
        font={"family": "sans-serif"},
    )
    return fig

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Credit Risk Scorecard")
st.markdown("Enter applicant details to generate a credit score (0-1000). Higher score = lower risk.")
st.divider()

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown("### Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Loan**")
        purpose = st.selectbox(
            "Purpose",
            list(mappings["purpose"].values()),
            index=3,
        )
        credit_amount = st.number_input(
            "Loan amount (DM)", min_value=100, max_value=20000, value=3000, step=100
        )
        duration = st.slider("Duration (months)", 4, 72, 18)
        installment_rate = st.selectbox(
            "Installment rate (% of income)", [1, 2, 3, 4], index=1
        )

    with col2:
        st.markdown("**Financial**")
        checking_account = st.selectbox(
            "Checking account",
            list(mappings["checking_account"].values()),
        )
        savings_account = st.selectbox(
            "Savings account",
            list(mappings["savings_account"].values()),
        )
        existing_credits = st.selectbox("Existing credits at bank", [1, 2, 3, 4])
        other_installments = st.selectbox(
            "Other installment plans",
            list(mappings["other_installments"].values()),
            index=2,
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Personal**")
        age = st.slider("Age", 18, 75, 35)
        employment = st.selectbox(
            "Employment length",
            list(mappings["employment"].values()),
            index=2,
        )
        personal_status = st.selectbox(
            "Personal status",
            list(mappings["personal_status"].values()),
            index=2,
        )
        dependents = st.selectbox("Number of dependents", [1, 2])
        foreign_worker = st.selectbox(
            "Foreign worker",
            list(mappings["foreign_worker"].values()),
        )

    with col4:
        st.markdown("**Property & History**")
        property_ = st.selectbox(
            "Property",
            list(mappings["property"].values()),
        )
        housing = st.selectbox(
            "Housing",
            list(mappings["housing"].values()),
            index=1,
        )
        residence_since = st.selectbox("Years at current residence", [1, 2, 3, 4], index=1)
        credit_history = st.selectbox(
            "Credit history",
            list(mappings["credit_history"].values()),
            index=2,
        )
        other_debtors = st.selectbox(
            "Other debtors / guarantors",
            list(mappings["other_debtors"].values()),
        )

# ── Compute score ──────────────────────────────────────────────────────────────
inputs = {
    "checking_account"  : checking_account,
    "duration"          : duration,
    "credit_history"    : credit_history,
    "purpose"           : purpose,
    "credit_amount"     : credit_amount,
    "savings_account"   : savings_account,
    "employment"        : employment,
    "installment_rate"  : installment_rate,
    "personal_status"   : personal_status,
    "other_debtors"     : other_debtors,
    "residence_since"   : residence_since,
    "property"          : property_,
    "age"               : age,
    "other_installments": other_installments,
    "housing"           : housing,
    "existing_credits"  : existing_credits,
    "dependents"        : dependents,
    "foreign_worker"    : foreign_worker,
}

score = compute_score(inputs)
band_label, bad_rate, band_css, band_colour = get_band(score)

with right:
    st.markdown("### Credit Score")
    st.plotly_chart(make_gauge(score), use_container_width=True)

    st.markdown(f"""
    <div class="score-band {band_css}">
        <strong style="font-size:1.1rem; color:{band_colour};">{band_label}</strong><br>
        <span style="font-size:0.9rem; color:#555;">
            Estimated default rate for this score band: <strong>{bad_rate}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Score breakdown**")
    bands = {
        "700+":    ("Low Risk",           "0%",     BLUE),
        "600-699": ("Moderate-Low Risk",  "10-14%", "#5D8AA8"),
        "500-599": ("Moderate Risk",      "9-10%",  "#5D8AA8"),
        "400-499": ("Moderate-High Risk", "40-70%", "#E74C3C"),
        "< 400":   ("High Risk",          ">70%",   ORANGE),
    }
    band_df = pd.DataFrame(
        [(b, v[0], v[1]) for b, v in bands.items()],
        columns=["Score Band", "Risk Level", "Typical Bad Rate"],
    )
    st.dataframe(band_df, hide_index=True, use_container_width=True)

st.divider()
st.caption("Built with the German Credit Dataset. Model: Logistic Regression | AUC 0.826 | Gini 0.652 | KS 0.600")
