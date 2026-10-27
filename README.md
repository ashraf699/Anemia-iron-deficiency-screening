# 🩸 IronGuard — Iron-Deficiency Anemia Screening & Risk Triage Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F51B5.svg)](https://plotly.com/)

An end-to-end machine-learning screening tool and interactive dashboard that predicts **iron-deficiency anemia** from a
routine **complete blood count (CBC)**, ranks patients into four risk tiers, discovers high-risk haematological
phenotypes, and lets clinicians run what-if scenarios.

> Educational project. Not a medical device; not a substitute for clinical diagnosis.

---

## 📌 Executive Summary

Ferritin and iron-panel tests are the diagnostic standard but are costly and often ordered late. A CBC is cheap and
nearly universal. This project:

1. **Prevents target leakage** – ferritin, serum iron, TIBC, transferrin saturation, folate, B12 and all other class labels are excluded; only CBC values + sex are used.
2. **Prioritises recall on the iron-deficient class** – a missed patient costs more than an extra confirmatory test.
3. **Stays explainable** – a Decision Tree gives auditable rules; an RBF-SVM serves as the non-linear benchmark.
4. **Triages patients** – 4-tier risk framework (Low / Moderate / High / Critical) with recommended actions.
5. **Profiles high-risk patients** – K-Means phenotypes (e.g. microcytic, high-RDW groups).
6. **Simulates scenarios** – real-time what-if predictor with CBC red-flag explanations.

---

## 📁 Repository Structure

```
anemia-iron-deficiency-screening/
│
├── app.py                      # Streamlit dashboard (7 tabs)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Ingestion, renaming, leakage prevention, filtering
│   ├── model_trainer.py        # Decision Tree & SVM, 5-fold CV, GridSearchCV
│   ├── visualizations.py       # Plotly charts with unified theme
│   └── risk_analyzer.py        # Risk tiers, phenotype clustering, what-if simulator
│
├── data/
│   └── anemia_dataset.csv      # 15,300 patient records, 29 columns
├── Anemia_Analysis.ipynb       # Research notebook (EDA + model prototyping)
├── .streamlit/config.toml      # Theme configuration
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Overview (`data/anemia_dataset.csv`)

15,300 patient records with CBC parameters, iron/folate/B12 biochemistry and anemia class labels
(source: SKILICARSLAN Anemia Dataset, Kaggle).

| Group | Columns |
|-------|---------|
| Demographic | `GENDER` (0 = female, 1 = male) |
| CBC (model inputs) | WBC, NEUT, LYMPH, MONO, EOS, BASO, RBC, HGB, HCT, MCV, MCH, MCHC, RDW, PLT, MPV, PCT, PDW |
| Iron panel (leakage – excluded) | FERRITIN, SERUM_IRON, TIBC, TRANSFERRIN_SAT |
| Other labs / labels (excluded) | FOLATE, B12, ALL_CLASS, HGB_CLASS, FOLATE_CLASS, B12_CLASS |

### Target
`IRON_CLASS` = 1 (iron deficient): **4,182 patients (27.3%)** · 0 (normal): **11,118 (72.7%)**

### Data quality notes
* No missing values; 88 exact duplicate rows are retained (identical CBC values can occur in routine labs).
* Extreme values (e.g. WBC up to 246.7, ferritin up to 27,332) are clinically possible and are kept; the SVM uses standard scaling and the tree is outlier-robust.
* Loader applies median imputation and type coercion so uploaded files with gaps still work.

---

## 🤖 Results (20% stratified hold-out, 3,060 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV Acc. |
|-------|---------:|----------:|-------:|---:|--------:|---------------:|
| Decision Tree (depth 5, balanced) | 0.916 | 0.769 | **0.992** | 0.866 | 0.972 | 0.916 |
| SVM (RBF, balanced) | 0.897 | 0.742 | 0.958 | 0.836 | 0.963 | 0.903 |

The Decision Tree catches 99% of iron-deficient patients; the trade-off is more false alarms
(precision 77%), which only cost a confirmatory ferritin test.

---

## 🖥️ Dashboard Tabs

| Tab | Content |
|-----|---------|
| Overview | KPIs, class balance, deficiency rate by hemoglobin band |
| EDA | Distributions, correlation heatmap, HGB-vs-MCV scatter, ferritin label check |
| Models | Metric table, confusion matrices, ROC, comparison chart |
| Decision Rules | Feature importance and visual tree |
| Risk Triage | Tier distribution, recommended actions, phenotypes, top-risk patients |
| What-If Simulator | Adjust CBC values → probability gauge, tier and clinical flags |
| Data & Export | Data preview, statistics, scored-cohort CSV download |

### Risk tiers
| Tier | Probability | Action |
|------|-------------|--------|
| Low | < 25% | Routine follow-up |
| Moderate | 25–50% | Ferritin test, repeat CBC in 3 months |
| High | 50–75% | Full iron panel, clinician review |
| Critical | ≥ 75% | Priority evaluation, investigate blood loss / malabsorption |

---

## 🚀 Quick Start

```bash
cd anemia-iron-deficiency-screening
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. Optionally run `jupyter notebook Anemia_Analysis.ipynb` for the research workflow.

---

## ⚠️ Limitations
* Single-source dataset; labels come from laboratory thresholds, not clinician adjudication.
* No history (diet, menstruation, medication, comorbidity) is available.
* Thalassaemia trait and chronic disease can mimic microcytic patterns and cause false positives.
* Requires external validation before any clinical use.

## 📄 License
MIT
