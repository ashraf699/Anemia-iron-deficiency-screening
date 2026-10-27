import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import FEATURE_COLS, TARGET, CBC_FEATURES

COLOR_NORMAL = "#10B981"
COLOR_DEF = "#EF4444"
COLOR_PRIMARY = "#3B82F6"
TEMPLATE = "plotly_white"
CMAP = {"Normal": COLOR_NORMAL, "Iron Deficient": COLOR_DEF}


def _layout(fig, title, h=380):
    fig.update_layout(title=dict(text=title, font=dict(size=16)), template=TEMPLATE, height=h,
                      margin=dict(l=20, r=20, t=55, b=20), font=dict(family="Plus Jakarta Sans, Inter, sans-serif"))
    return fig


def create_status_donut(df):
    c = df["Iron_Status"].value_counts()
    fig = go.Figure(go.Pie(labels=c.index, values=c.values, hole=.6,
                           marker=dict(colors=[CMAP[l] for l in c.index]), textinfo="label+percent"))
    return _layout(fig, "Iron Status Distribution")


def create_hgb_histogram(df):
    fig = px.histogram(df, x="HGB", color="Iron_Status", nbins=60, barmode="overlay", opacity=.7,
                       color_discrete_map=CMAP, labels={"HGB": "Hemoglobin (g/dL)"})
    return _layout(fig, "Hemoglobin Distribution by Iron Status")


def create_mcv_boxplot(df):
    fig = px.box(df, x="Iron_Status", y="MCV", color="Iron_Status", color_discrete_map=CMAP, points=False)
    fig.update_layout(showlegend=False)
    return _layout(fig, "MCV (Cell Size) by Iron Status")


def create_rdw_by_sex(df):
    g = df.groupby(["Sex", "Iron_Status"])["RDW"].median().reset_index()
    fig = px.bar(g, x="Sex", y="RDW", color="Iron_Status", barmode="group", color_discrete_map=CMAP,
                 labels={"RDW": "Median RDW (%)"})
    return _layout(fig, "Median RDW by Sex & Iron Status")


def create_prevalence_by_hgb_band(df):
    bins = [0, 8, 10, 12, 13, 14, 16, 30]
    d = df.assign(Band=pd.cut(df["HGB"], bins))
    g = d.groupby("Band", observed=True)[TARGET].agg(["mean", "size"]).reset_index()
    g["Band"] = g["Band"].astype(str)
    fig = px.bar(g, x="Band", y="mean", text=g["size"].astype(str) + " pts",
                 labels={"mean": "Iron deficiency rate", "Band": "HGB band (g/dL)"},
                 color="mean", color_continuous_scale="Reds")
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "Iron Deficiency Rate by Hemoglobin Band")


def create_correlation_heatmap(df):
    cols = CBC_FEATURES + ["GENDER", TARGET]
    corr = df[cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return _layout(fig, "Correlation Matrix (CBC + Target)", h=620)


def create_feature_target_corr(df):
    c = df[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig = px.bar(x=c.values, y=c.index, orientation="h", color=c.values, color_continuous_scale="RdBu_r",
                 labels={"x": "Correlation with iron deficiency", "y": ""})
    return _layout(fig, "Feature Correlation with Iron Deficiency", h=520)


def create_hgb_mcv_scatter(df, n=3000):
    d = df.sample(min(n, len(df)), random_state=1)
    fig = px.scatter(d, x="HGB", y="MCV", color="Iron_Status", color_discrete_map=CMAP, opacity=.55)
    return _layout(fig, "Hemoglobin vs MCV (sampled)", h=430)


def create_ferritin_check(df):
    d = df[df["FERRITIN"] < df["FERRITIN"].quantile(.99)]
    fig = px.box(d, x="Iron_Status", y="FERRITIN", color="Iron_Status", color_discrete_map=CMAP)
    fig.update_layout(showlegend=False)
    return _layout(fig, "Ferritin by Iron Status (label validation only, not a model input)")


def create_confusion_matrix_charts(res):
    figs = []
    for key, name in (("dt_cm", "Decision Tree"), ("svm_cm", "SVM (RBF)")):
        fig = px.imshow(res[key], text_auto=True, color_continuous_scale="Blues",
                        x=["Normal", "Iron Def."], y=["Normal", "Iron Def."],
                        labels=dict(x="Predicted", y="Actual"))
        figs.append(_layout(fig, f"{name} Confusion Matrix", h=340))
    return figs


def create_roc_curve_chart(res):
    fig = go.Figure()
    for k, n in (("dt", "Decision Tree"), ("svm", "SVM (RBF)")):
        f, t = res[f"{k}_roc"]
        auc = res[f"{k}_metrics"]["ROC-AUC"]
        fig.add_trace(go.Scatter(x=f, y=t, name=f"{n} (AUC={auc:.3f})", mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="Chance"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return _layout(fig, "ROC Curves", h=430)


def create_metrics_comparison_bar(res):
    rows = []
    for k, n in (("dt_metrics", "Decision Tree"), ("svm_metrics", "SVM (RBF)")):
        for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            rows.append({"Model": n, "Metric": m, "Score": res[k][m]})
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f")
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, "Model Comparison on Hold-out Test Set")


def create_feature_importance_bar(res, top=12):
    d = res["importance"].head(top).iloc[::-1]
    fig = px.bar(d, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    return _layout(fig, "Decision Tree Feature Importance", h=420)


def create_risk_gauge(prob):
    color = COLOR_NORMAL if prob < .25 else "#F59E0B" if prob < .5 else "#F97316" if prob < .75 else COLOR_DEF
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color),
                   steps=[dict(range=[0, 25], color="#D1FAE5"), dict(range=[25, 50], color="#FEF3C7"),
                          dict(range=[50, 75], color="#FFEDD5"), dict(range=[75, 100], color="#FEE2E2")])))
    return _layout(fig, "Iron Deficiency Probability", h=300)


def create_tier_distribution(df_scored):
    c = df_scored["Risk_Tier"].value_counts().reindex(["Low", "Moderate", "High", "Critical"]).fillna(0)
    fig = px.bar(x=c.index, y=c.values, color=c.index,
                 color_discrete_map={"Low": COLOR_NORMAL, "Moderate": "#F59E0B", "High": "#F97316", "Critical": COLOR_DEF},
                 labels={"x": "Risk tier", "y": "Patients"})
    fig.update_layout(showlegend=False)
    return _layout(fig, "Patients per Risk Tier")
