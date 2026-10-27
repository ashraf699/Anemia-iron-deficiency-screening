import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TIERS = [(0.75, "Critical"), (0.50, "High"), (0.25, "Moderate"), (0.0, "Low")]

TIER_ACTIONS = {
    "Low": "Routine follow-up. Re-screen at the next scheduled check-up.",
    "Moderate": "Order serum ferritin; review diet and recent blood loss; repeat CBC in 3 months.",
    "High": "Order full iron panel (ferritin, serum iron, TIBC, transferrin saturation); clinician review; start iron-rich diet counselling.",
    "Critical": "Priority clinical evaluation: confirm with iron panel, investigate source of blood loss/malabsorption, consider supplementation.",
}


def tier_for(p: float) -> str:
    for thr, name in TIERS:
        if p >= thr:
            return name
    return "Low"


def score_cohort_risk(df, res, model="dt"):
    """Score every patient with the chosen model and return df with Risk_Prob."""
    X = df[res["features"]]
    if model == "svm":
        prob = res["svm"].predict_proba(res["scaler"].transform(X))[:, 1]
    else:
        prob = res["dt"].predict_proba(X)[:, 1]
    out = df.copy()
    out["Risk_Prob"] = prob
    return out


def assign_risk_tiers(df_scored):
    out = df_scored.copy()
    out["Risk_Tier"] = out["Risk_Prob"].apply(tier_for)
    return out


def discover_at_risk_profiles(df_scored, k=3):
    """Cluster patients with Risk_Prob >= 0.5 into haematological phenotypes."""
    risky = df_scored[df_scored["Risk_Prob"] >= 0.5].copy()
    if len(risky) < k * 10:
        return risky.assign(Profile="Insufficient data"), pd.DataFrame()
    cols = ["HGB", "MCV", "MCH", "RDW", "RBC", "PLT"]
    Z = StandardScaler().fit_transform(risky[cols])
    risky["Cluster"] = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    med = risky.groupby("Cluster")[cols].median()
    names = {}
    for c, r in med.iterrows():
        tags = []
        tags.append("Severe anaemia" if r["HGB"] < 10 else "Mild anaemia" if r["HGB"] < 12 else "Near-normal Hb")
        tags.append("microcytic" if r["MCV"] < 80 else "normocytic")
        if r["RDW"] > 15:
            tags.append("high anisocytosis")
        if r["PLT"] > 400:
            tags.append("reactive thrombocytosis")
        names[c] = ", ".join(tags)
    risky["Profile"] = risky["Cluster"].map(names)
    summary = risky.groupby("Profile").agg(Patients=("HGB", "size"), HGB=("HGB", "median"), MCV=("MCV", "median"),
                                           MCH=("MCH", "median"), RDW=("RDW", "median"), PLT=("PLT", "median"),
                                           Female_pct=("GENDER", lambda s: (1 - s.mean()) * 100)).round(2).reset_index()
    return risky, summary


def get_what_if_prediction(res, patient: dict, model="dt"):
    row = pd.DataFrame([patient])[res["features"]]
    if model == "svm":
        p = float(res["svm"].predict_proba(res["scaler"].transform(row))[:, 1][0])
    else:
        p = float(res["dt"].predict_proba(row)[:, 1][0])
    tier = tier_for(p)
    return {"prob": p, "tier": tier, "action": TIER_ACTIONS[tier]}


def clinical_flags(p: dict) -> list:
    """Rule-of-thumb flags (WHO-style thresholds) to accompany model output."""
    f = []
    hgb_cut = 13.0 if p.get("GENDER", 0) == 1 else 12.0
    if p["HGB"] < hgb_cut:
        f.append(f"Hemoglobin {p['HGB']:.1f} g/dL below {hgb_cut} g/dL (anaemia range)")
    if p["MCV"] < 80:
        f.append("Microcytosis (MCV < 80 fL)")
    if p["MCH"] < 27:
        f.append("Hypochromia (MCH < 27 pg)")
    if p["RDW"] > 14.5:
        f.append("Raised RDW (> 14.5%): mixed red-cell sizes")
    return f or ["No classic CBC red flags"]
