import io
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (load_dataset, get_data_summary, filter_dataframe, FEATURE_COLS,
                               FEATURE_LABELS, TARGET)
from utils.model_trainer import train_and_evaluate_models, generate_tree_plot, best_model_name
from utils import visualizations as viz
from utils.risk_analyzer import (score_cohort_risk, assign_risk_tiers, discover_at_risk_profiles,
                                 get_what_if_prediction, clinical_flags, TIER_ACTIONS)

st.set_page_config(page_title="IronGuard — Anemia Screening Dashboard", page_icon="🩸",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Plus Jakarta Sans', sans-serif;}
.stApp {background-color:#F8FAFC;}
.block-container {padding-top:1.6rem; max-width:1440px;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#1F0A0E 0%,#0F172A 100%) !important;}
[data-testid="stSidebar"] * {color:#E2E8F0 !important;}
.hero {background:linear-gradient(135deg,#7F1D1D,#DC2626 60%,#F87171); color:white; padding:26px 32px;
       border-radius:18px; margin-bottom:18px; box-shadow:0 10px 30px rgba(220,38,38,.25);}
.hero h1 {margin:0; font-size:2rem; font-weight:800; color:white;}
.hero p {margin:6px 0 0; opacity:.92;}
.kpi {background:white; border:1px solid #E2E8F0; border-radius:14px; padding:16px 18px;
      box-shadow:0 2px 8px rgba(15,23,42,.05);}
.kpi .v {font-size:1.7rem; font-weight:800; color:#0F172A;}
.kpi .l {font-size:.78rem; color:#64748B; text-transform:uppercase; letter-spacing:.05em;}
.insight {background:#FEF2F2; border-left:4px solid #DC2626; padding:12px 16px; border-radius:8px;
          margin:8px 0 16px; color:#450A0A; font-size:.92rem;}
.disclaimer {background:#FFFBEB; border-left:4px solid #F59E0B; padding:10px 14px; border-radius:8px;
             font-size:.85rem; color:#78350F;}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>',
                 unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🩸 IronGuard")
    st.caption("Iron-deficiency anemia screening from routine blood counts")
    up = st.file_uploader("Upload dataset (csv / xlsx)", type=["csv", "xlsx"])
    st.markdown("---")
    st.markdown("### Model settings")
    test_size = st.slider("Test split", 0.10, 0.40, 0.20, 0.05)
    max_depth = st.slider("Decision-tree depth", 2, 10, 5)
    tune = st.checkbox("Auto-tune tree (GridSearchCV, recall)", value=False)
    svm_c = st.select_slider("SVM C", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
    st.markdown("---")
    st.markdown("### Cohort filters")
    sex_f = st.radio("Sex", ["All", "Female", "Male"], horizontal=True)

try:
    df = load_dataset(up)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

with st.sidebar:
    hgb_rng = st.slider("Hemoglobin range (g/dL)", float(df.HGB.min()), float(df.HGB.max()),
                        (float(df.HGB.min()), float(df.HGB.max())))
    mcv_rng = st.slider("MCV range (fL)", float(df.MCV.min()), float(df.MCV.max()),
                        (float(df.MCV.min()), float(df.MCV.max())))

fdf = filter_dataframe(df, sex_f, hgb_rng, mcv_rng)
if len(fdf) < 50:
    st.warning("Filters leave too few patients. Widen the filters.")
    st.stop()
S = get_data_summary(fdf)
res = train_and_evaluate_models(df, test_size=test_size, max_depth=max_depth, tune=tune, svm_c=svm_c)

st.markdown("""<div class="hero"><h1>🩸 IronGuard — Anemia Screening Dashboard</h1>
<p>Predict iron-deficiency anemia from a routine complete blood count, prioritise patients for confirmatory
iron-panel testing, and explore haematological risk profiles.</p></div>""", unsafe_allow_html=True)

c = st.columns(5)
kpi(c[0], "Patients", f"{S['rows']:,}")
kpi(c[1], "Iron deficient", f"{S['iron_def']:,}")
kpi(c[2], "Prevalence", f"{S['prevalence']:.1f}%")
kpi(c[3], "Male share", f"{S['male_pct']:.1f}%")
kpi(c[4], "Missing values", S["missing"])
st.write("")

tabs = st.tabs(["📊 Overview", "🔍 EDA", "🤖 Models", "🌳 Decision Rules", "🎯 Risk Triage",
                "🧪 What-If Simulator", "📥 Data & Export"])

# ---------------- Overview ----------------
with tabs[0]:
    a, b = st.columns(2)
    a.plotly_chart(viz.create_status_donut(fdf), use_container_width=True)
    b.plotly_chart(viz.create_prevalence_by_hgb_band(fdf), use_container_width=True)
    low_h = fdf[fdf.HGB < 12][TARGET].mean() * 100 if (fdf.HGB < 12).any() else 0
    insight(f"{S['prevalence']:.1f}% of the filtered cohort is iron deficient; among patients with "
            f"hemoglobin below 12 g/dL the rate rises to {low_h:.1f}%.")
    st.markdown("""**Problem framing.** Ferritin and iron-panel tests are costly and not always ordered. A CBC is
cheap and universal. The model uses **only CBC values + sex**; ferritin, serum iron, TIBC, transferrin saturation,
folate and B12 are excluded to prevent target leakage.""")
    st.markdown('<div class="disclaimer">Educational project. Not a medical device and not a substitute for clinical diagnosis.</div>',
                unsafe_allow_html=True)

# ---------------- EDA ----------------
with tabs[1]:
    a, b = st.columns(2)
    a.plotly_chart(viz.create_hgb_histogram(fdf), use_container_width=True)
    b.plotly_chart(viz.create_mcv_boxplot(fdf), use_container_width=True)
    med = fdf.groupby("Iron_Status")[["HGB", "MCV", "MCH", "RDW"]].median().round(2)
    if {"Iron Deficient", "Normal"} <= set(med.index):
        insight(f"Iron-deficient patients show median MCV {med.loc['Iron Deficient','MCV']} fL vs "
                f"{med.loc['Normal','MCV']} fL, and RDW {med.loc['Iron Deficient','RDW']}% vs {med.loc['Normal','RDW']}%.")
    a, b = st.columns(2)
    a.plotly_chart(viz.create_rdw_by_sex(fdf), use_container_width=True)
    b.plotly_chart(viz.create_hgb_mcv_scatter(fdf), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(viz.create_feature_target_corr(fdf), use_container_width=True)
    b.plotly_chart(viz.create_correlation_heatmap(fdf), use_container_width=True)
    st.plotly_chart(viz.create_ferritin_check(fdf), use_container_width=True)
    insight("Ferritin separates the classes cleanly because it defines the label, which is exactly why it is kept out of the model.")

# ---------------- Models ----------------
with tabs[2]:
    best = best_model_name(res)
    st.success(f"Highest recall on iron-deficient patients: **{best}**")
    mt = pd.DataFrame({"Decision Tree": res["dt_metrics"], "SVM (RBF)": res["svm_metrics"]}).T.round(4)
    st.dataframe(mt, use_container_width=True)
    if res["best_params"]:
        st.caption(f"GridSearchCV best parameters: {res['best_params']}")
    st.plotly_chart(viz.create_metrics_comparison_bar(res), use_container_width=True)
    a, b = st.columns(2)
    cms = viz.create_confusion_matrix_charts(res)
    a.plotly_chart(cms[0], use_container_width=True)
    b.plotly_chart(cms[1], use_container_width=True)
    st.plotly_chart(viz.create_roc_curve_chart(res), use_container_width=True)
    insight("A false negative is a missed iron-deficient patient, so recall is the primary metric; class weights are balanced for this reason.")

# ---------------- Decision rules ----------------
with tabs[3]:
    st.plotly_chart(viz.create_feature_importance_bar(res), use_container_width=True)
    top = res["importance"].head(3)["Feature"].tolist()
    insight(f"Top predictors: {', '.join(top)}.")
    depth = st.slider("Tree levels to display", 1, 4, 3)
    st.pyplot(generate_tree_plot(res, depth))

# ---------------- Risk triage ----------------
with tabs[4]:
    mdl = st.radio("Scoring model", ["Decision Tree", "SVM (RBF)"], horizontal=True)
    scored = assign_risk_tiers(score_cohort_risk(fdf, res, "svm" if mdl.startswith("SVM") else "dt"))
    a, b = st.columns([1, 1])
    a.plotly_chart(viz.create_tier_distribution(scored), use_container_width=True)
    with b:
        st.markdown("#### Recommended actions")
        for t, act in TIER_ACTIONS.items():
            st.markdown(f"**{t}** — {act}")
    st.markdown("#### Discovered high-risk phenotypes")
    risky, summ = discover_at_risk_profiles(scored)
    if len(summ):
        st.dataframe(summ, use_container_width=True)
    else:
        st.info("Not enough high-risk patients in this filtered cohort to cluster.")
    st.markdown("#### Highest-risk patients")
    st.dataframe(scored.sort_values("Risk_Prob", ascending=False)
                 [["Sex", "HGB", "MCV", "MCH", "RDW", "RBC", "Risk_Prob", "Risk_Tier", "Iron_Status"]].head(25),
                 use_container_width=True)

# ---------------- What-if ----------------
with tabs[5]:
    st.markdown("Adjust a patient's CBC values and see the predicted iron-deficiency risk.")
    med_all = df[FEATURE_COLS].median()
    cols = st.columns(3)
    patient = {}
    patient["GENDER"] = 1 if cols[0].radio("Sex", ["Female", "Male"], horizontal=True) == "Male" else 0
    key = ["HGB", "MCV", "MCH", "RDW", "RBC", "PLT", "HCT", "MCHC", "WBC"]
    for i, f in enumerate(key):
        lo, hi = float(df[f].quantile(.005)), float(df[f].quantile(.995))
        patient[f] = cols[(i + 1) % 3].slider(FEATURE_LABELS[f], lo, hi, float(med_all[f]))
    for f in FEATURE_COLS:
        patient.setdefault(f, float(med_all[f]))
    m = st.radio("Model", ["Decision Tree", "SVM (RBF)"], horizontal=True, key="wm")
    out = get_what_if_prediction(res, patient, "svm" if m.startswith("SVM") else "dt")
    a, b = st.columns([1, 1])
    a.plotly_chart(viz.create_risk_gauge(out["prob"]), use_container_width=True)
    with b:
        st.markdown(f"### Tier: {out['tier']}")
        st.write(out["action"])
        st.markdown("**CBC flags**")
        for fl in clinical_flags(patient):
            st.markdown(f"- {fl}")

# ---------------- Data & Export ----------------
with tabs[6]:
    st.dataframe(fdf.head(200), use_container_width=True)
    st.markdown("**Summary statistics**")
    st.dataframe(fdf[FEATURE_COLS].describe().T.round(2), use_container_width=True)
    scored_all = assign_risk_tiers(score_cohort_risk(fdf, res))
    buf = io.StringIO()
    scored_all.to_csv(buf, index=False)
    st.download_button("⬇️ Download scored cohort (CSV)", buf.getvalue(), "scored_cohort.csv", "text/csv")
