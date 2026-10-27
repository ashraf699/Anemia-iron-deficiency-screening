import os
import io
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_PATH = os.path.join("data", "anemia_dataset.csv")

RENAME_MAP = {
    "NE#": "NEUT", "LY#": "LYMPH", "MO#": "MONO", "EO#": "EOS", "BA#": "BASO",
    "FERRITTE": "FERRITIN", "SD": "SERUM_IRON", "SDTSD": "TIBC", "TSD": "TRANSFERRIN_SAT",
    "Iron_anemia_Class": "IRON_CLASS", "HGB_Anemia_Class": "HGB_CLASS",
    "Folate_anemia_class": "FOLATE_CLASS", "B12_Anemia_class": "B12_CLASS",
    "All_Class": "ALL_CLASS",
}

# Routine complete-blood-count (CBC) parameters available at first contact.
CBC_FEATURES = [
    "WBC", "NEUT", "LYMPH", "MONO", "EOS", "BASO", "RBC", "HGB", "HCT",
    "MCV", "MCH", "MCHC", "RDW", "PLT", "MPV", "PCT", "PDW",
]
FEATURE_COLS = ["GENDER"] + CBC_FEATURES

# Biochemistry that DEFINES the label (iron panel) or is another label -> excluded (leakage).
LEAKAGE_COLS = [
    "FERRITIN", "SERUM_IRON", "TIBC", "TRANSFERRIN_SAT",
    "HGB_CLASS", "FOLATE_CLASS", "B12_CLASS", "ALL_CLASS", "FOLATE", "B12",
]
TARGET = "IRON_CLASS"

NUMERIC_COLS = CBC_FEATURES + ["FERRITIN", "SERUM_IRON", "TIBC", "TRANSFERRIN_SAT", "FOLATE", "B12"]

FEATURE_LABELS = {
    "GENDER": "Sex (0=F, 1=M)", "WBC": "White Blood Cells (x10³/µL)", "NEUT": "Neutrophils (x10³/µL)",
    "LYMPH": "Lymphocytes (x10³/µL)", "MONO": "Monocytes (x10³/µL)", "EOS": "Eosinophils (x10³/µL)",
    "BASO": "Basophils (x10³/µL)", "RBC": "Red Blood Cells (x10⁶/µL)", "HGB": "Hemoglobin (g/dL)",
    "HCT": "Hematocrit (%)", "MCV": "Mean Corpuscular Volume (fL)", "MCH": "Mean Corpuscular Hb (pg)",
    "MCHC": "MCHC (g/dL)", "RDW": "Red Cell Distribution Width (%)", "PLT": "Platelets (x10³/µL)",
    "MPV": "Mean Platelet Volume (fL)", "PCT": "Plateletcrit (%)", "PDW": "Platelet Dist. Width (%)",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP)
    df.columns = [str(c).strip() for c in df.columns]
    for c in NUMERIC_COLS + ["GENDER", TARGET]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET, "GENDER"]).copy()
    for c in NUMERIC_COLS:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df[TARGET] = df[TARGET].astype(int)
    df["GENDER"] = df["GENDER"].astype(int)
    df["Sex"] = np.where(df["GENDER"] == 1, "Male", "Female")
    df["Iron_Status"] = np.where(df[TARGET] == 1, "Iron Deficient", "Normal")
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dataset(file_source=None):
    """Load default CSV, a path, or an uploaded buffer (csv / xlsx)."""
    if file_source is None:
        if not os.path.exists(DEFAULT_PATH):
            raise FileNotFoundError(f"Default dataset '{DEFAULT_PATH}' not found.")
        df = pd.read_csv(DEFAULT_PATH)
    elif isinstance(file_source, (str, os.PathLike)):
        df = pd.read_excel(file_source) if str(file_source).endswith("xlsx") else pd.read_csv(file_source)
    else:
        content = file_source.getvalue()
        name = getattr(file_source, "name", "")
        df = pd.read_excel(io.BytesIO(content)) if name.endswith("xlsx") else pd.read_csv(io.BytesIO(content))
    df = _normalise(df)
    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    pos = int(df[TARGET].sum())
    return {
        "rows": n, "cols": df.shape[1], "missing": int(df[FEATURE_COLS].isna().sum().sum()),
        "iron_def": pos, "normal": n - pos, "prevalence": pos / n * 100,
        "male_pct": df["GENDER"].mean() * 100, "duplicates": int(df.duplicated().sum()),
    }


def prepare_features(df: pd.DataFrame):
    """Return X (CBC only, leakage removed) and y."""
    return df[FEATURE_COLS].copy(), df[TARGET].copy()


def filter_dataframe(df, sex="All", hgb_range=None, mcv_range=None):
    out = df
    if sex != "All":
        out = out[out["Sex"] == sex]
    if hgb_range:
        out = out[out["HGB"].between(*hgb_range)]
    if mcv_range:
        out = out[out["MCV"].between(*mcv_range)]
    return out
