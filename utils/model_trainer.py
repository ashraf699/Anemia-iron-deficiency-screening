import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_loader import prepare_features

RANDOM_STATE = 42


def _metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


@st.cache_resource(show_spinner="Training Decision Tree and SVM models...")
def train_and_evaluate_models(df: pd.DataFrame, test_size=0.20, max_depth=5, tune=False, svm_c=1.0):
    """Train DT + SVM predicting iron-deficiency from CBC only. Recall on the deficient class is prioritised."""
    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    best_params = None
    if tune:
        grid = GridSearchCV(
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [5, 10, 20, 40]},
            cv=cv, scoring="recall", n_jobs=-1)
        grid.fit(X_tr, y_tr)
        dt = grid.best_estimator_
        best_params = grid.best_params_
    else:
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, class_weight="balanced",
                                    random_state=RANDOM_STATE).fit(X_tr, y_tr)
    svm = SVC(kernel="rbf", C=svm_c, class_weight="balanced", probability=True,
              random_state=RANDOM_STATE).fit(X_tr_s, y_tr)

    dt_pred, dt_prob = dt.predict(X_te), dt.predict_proba(X_te)[:, 1]
    sv_pred, sv_prob = svm.predict(X_te_s), svm.predict_proba(X_te_s)[:, 1]
    m_dt, m_sv = _metrics(y_te, dt_pred, dt_prob), _metrics(y_te, sv_pred, sv_prob)
    m_dt["CV Accuracy"] = cross_val_score(dt, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    m_sv["CV Accuracy"] = cross_val_score(SVC(kernel="rbf", C=svm_c, class_weight="balanced"),
                                          X_tr_s, y_tr, cv=cv, scoring="accuracy").mean()

    fpr_dt, tpr_dt, _ = roc_curve(y_te, dt_prob)
    fpr_sv, tpr_sv, _ = roc_curve(y_te, sv_prob)
    imp = pd.DataFrame({"Feature": X.columns, "Importance": dt.feature_importances_}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "dt": dt, "svm": svm, "scaler": scaler, "features": list(X.columns),
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "dt_metrics": m_dt, "svm_metrics": m_sv,
        "dt_cm": confusion_matrix(y_te, dt_pred), "svm_cm": confusion_matrix(y_te, sv_pred),
        "dt_roc": (fpr_dt, tpr_dt), "svm_roc": (fpr_sv, tpr_sv),
        "dt_prob_test": dt_prob, "svm_prob_test": sv_prob,
        "importance": imp, "best_params": best_params,
    }


def generate_tree_plot(res, max_depth=3):
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(res["dt"], feature_names=res["features"], class_names=["Normal", "Iron Deficient"],
              filled=True, rounded=True, fontsize=9, max_depth=max_depth, ax=ax)
    ax.set_title("Decision Tree - Iron Deficiency Screening Rules (top levels)")
    return fig


def best_model_name(res):
    return "Decision Tree" if res["dt_metrics"]["Recall"] >= res["svm_metrics"]["Recall"] else "SVM (RBF)"
