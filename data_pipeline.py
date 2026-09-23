"""
data_pipeline.py
================
Backend data pipeline and machine learning engines for the Healthcare Analytics
web application.

Responsibilities:
  - Load and clean the CSV
  - Engineer datetime and anomaly features
  - Train / cache classification model  (Test Results)
  - Train / cache regression model      (Billing Amount & Daily Cost)
  - Expose helper utilities used by app.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str = "cleaned_healthcare_data.csv") -> pd.DataFrame:
    """Load the cleaned CSV and apply all feature-engineering steps."""
    df = pd.read_csv(path)

    # ── Datetime parsing ──────────────────────────────────────────────────────
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"]    = pd.to_datetime(df["Discharge Date"],    errors="coerce")

    df["Admission_Year"]    = df["Date of Admission"].dt.year
    df["Admission_Month"]   = df["Date of Admission"].dt.month
    df["Admission_Quarter"] = df["Date of Admission"].dt.quarter
    df["Admission_DayOfWeek"] = df["Date of Admission"].dt.dayofweek   # 0 = Monday

    # ── LOS & Cost anomaly flags ───────────────────────────────────────────────
    los_thresh         = df["Length_of_Stay"].quantile(0.90)
    cost_thresh        = df["Daily_Cost"].quantile(0.90)
    df["LOS_Outlier"]  = df["Length_of_Stay"] > los_thresh
    df["HighCost_Flag"] = df["Daily_Cost"]    > cost_thresh

    # ── Derived convenience columns ───────────────────────────────────────────
    df["Admission_MonthName"] = df["Date of Admission"].dt.strftime("%b %Y")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FILTERING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    date_range=None,
    insurance: list = None,
    hospital: list = None,
    admission_type: list = None,
    condition: list = None,
) -> pd.DataFrame:
    """Return a copy of *df* after applying sidebar filter selections."""
    fdf = df.copy()

    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        fdf = fdf[(fdf["Date of Admission"] >= start) & (fdf["Date of Admission"] <= end)]

    if insurance:
        fdf = fdf[fdf["Insurance Provider"].isin(insurance)]

    if hospital:
        fdf = fdf[fdf["Hospital"].isin(hospital)]

    if admission_type:
        fdf = fdf[fdf["Admission Type"].isin(admission_type)]

    if condition:
        fdf = fdf[fdf["Medical Condition"].isin(condition)]

    return fdf


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ML FEATURE SETS
# ──────────────────────────────────────────────────────────────────────────────

# Features used by the classification engine
CLF_NUMERIC  = ["Age", "Length_of_Stay"]
CLF_ORDINAL  = ["Age Group"]             # known ordering
CLF_NOMINAL  = ["Gender", "Medical Condition", "Admission Type", "Medication"]

# Features used by the regression engines
REG_NUMERIC  = ["Age", "Length_of_Stay"]
REG_NOMINAL  = ["Gender", "Medical Condition", "Admission Type",
                "Medication", "Insurance Provider", "Age Group"]

AGE_GROUP_ORDER = [["Child", "Adult", "Middle-Aged", "Senior"]]


def _build_clf_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough",                         CLF_NUMERIC),
            ("ord", OrdinalEncoder(categories=AGE_GROUP_ORDER), CLF_ORDINAL),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CLF_NOMINAL),
        ],
        remainder="drop",
    )


def _build_reg_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough",                                              REG_NUMERIC),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), REG_NOMINAL),
        ],
        remainder="drop",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ENGINE 1 – TEST OUTCOME CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame):
    """
    Train a Gradient Boosting multi-class classifier to predict Test Results.

    Returns
    -------
    pipeline   : fitted sklearn Pipeline
    le         : fitted LabelEncoder for the target
    metrics    : dict with confusion_matrix, classification_report, feature_names
    X_test     : held-out feature DataFrame (for SHAP / further analysis)
    y_test_enc : held-out encoded labels
    """
    feature_cols = CLF_NUMERIC + CLF_ORDINAL + CLF_NOMINAL
    sub = df[feature_cols + ["Test Results"]].dropna()

    le = LabelEncoder()
    y  = le.fit_transform(sub["Test Results"])
    X  = sub[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("prep",  _build_clf_preprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            random_state=42, subsample=0.8
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    # Recover feature names after preprocessing
    ohe         = pipeline.named_steps["prep"].named_transformers_["nom"]
    nom_names   = ohe.get_feature_names_out(CLF_NOMINAL).tolist()
    feat_names  = CLF_NUMERIC + CLF_ORDINAL + nom_names

    importances = pipeline.named_steps["model"].feature_importances_
    feat_imp_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    metrics = {
        "confusion_matrix":        cm,
        "classification_report":   report,
        "feature_importance":      feat_imp_df,
        "label_encoder":           le,
        "classes":                 le.classes_,
    }

    return pipeline, le, metrics, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ENGINE 2 – BILLING / DAILY COST REGRESSOR
# ──────────────────────────────────────────────────────────────────────────────

def train_regressor(df: pd.DataFrame):
    """
    Train Gradient Boosting regression models for Billing Amount and Daily Cost.

    Returns
    -------
    bill_pipe  : fitted pipeline for Billing Amount
    cost_pipe  : fitted pipeline for Daily Cost
    metrics    : dict with RMSE, MAE, R2 for both targets + feature importance
    """
    feature_cols = REG_NUMERIC + REG_NOMINAL
    sub = df[feature_cols + ["Billing Amount", "Daily_Cost"]].dropna()
    X   = sub[feature_cols]

    results = {}

    for target in ["Billing Amount", "Daily_Cost"]:
        y = sub[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        pipe = Pipeline([
            ("prep",  _build_reg_preprocessor()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.08, max_depth=4,
                random_state=42, subsample=0.8
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        ohe        = pipe.named_steps["prep"].named_transformers_["nom"]
        nom_names  = ohe.get_feature_names_out(REG_NOMINAL).tolist()
        feat_names = REG_NUMERIC + nom_names
        importances = pipe.named_steps["model"].feature_importances_

        feat_imp_df = (
            pd.DataFrame({"Feature": feat_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(15)
        )

        results[target] = {
            "pipeline":          pipe,
            "rmse":              rmse,
            "mae":               mae,
            "r2":                r2,
            "feature_importance": feat_imp_df,
        }

    return results["Billing Amount"]["pipeline"], results["Daily_Cost"]["pipeline"], results


# ──────────────────────────────────────────────────────────────────────────────
# 6.  INFERENCE HELPER
# ──────────────────────────────────────────────────────────────────────────────

def predict_patient(
    clf_pipeline, clf_le,
    bill_pipeline, cost_pipeline,
    df: pd.DataFrame,
    age: int,
    gender: str,
    admission_type: str,
    condition: str,
    insurance: str,
    medication: str,
    planned_days: int,
) -> dict:
    """
    Run both ML engines for a single patient input and return predictions + anomaly flags.
    """
    # Determine Age Group from age
    if age < 18:
        age_group = "Child"
    elif age < 45:
        age_group = "Adult"
    elif age < 65:
        age_group = "Middle-Aged"
    else:
        age_group = "Senior"

    # ── Classification ────────────────────────────────────────────────────────
    clf_input = pd.DataFrame([{
        "Age":               age,
        "Length_of_Stay":    planned_days,
        "Age Group":         age_group,
        "Gender":            gender,
        "Medical Condition": condition,
        "Admission Type":    admission_type,
        "Medication":        medication,
    }])
    proba      = clf_pipeline.predict_proba(clf_input)[0]
    classes    = clf_le.classes_
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    # ── Regression ────────────────────────────────────────────────────────────
    reg_input = pd.DataFrame([{
        "Age":               age,
        "Length_of_Stay":    planned_days,
        "Gender":            gender,
        "Medical Condition": condition,
        "Admission Type":    admission_type,
        "Medication":        medication,
        "Insurance Provider": insurance,
        "Age Group":         age_group,
    }])
    pred_billing = float(bill_pipeline.predict(reg_input)[0])
    pred_cost    = float(cost_pipeline.predict(reg_input)[0])

    # ── Anomaly / Risk flags ──────────────────────────────────────────────────
    los_thresh  = df["Length_of_Stay"].quantile(0.90)
    cost_thresh = df["Daily_Cost"].quantile(0.90)

    # Condition-level baselines
    cond_df     = df[df["Medical Condition"] == condition]
    cond_los    = cond_df["Length_of_Stay"].mean() if not cond_df.empty else df["Length_of_Stay"].mean()
    cond_cost   = cond_df["Daily_Cost"].mean()     if not cond_df.empty else df["Daily_Cost"].mean()

    los_risk    = planned_days > los_thresh
    cost_risk   = pred_cost   > cost_thresh
    above_cond_los  = planned_days > (cond_los  * 1.5)
    above_cond_cost = pred_cost    > (cond_cost * 1.5)

    return {
        "proba_dict":       proba_dict,
        "predicted_class":  classes[proba.argmax()],
        "pred_billing":     pred_billing,
        "pred_daily_cost":  pred_cost,
        "los_risk":         los_risk,
        "cost_risk":        cost_risk,
        "above_cond_los":   above_cond_los,
        "above_cond_cost":  above_cond_cost,
        "cond_los_mean":    cond_los,
        "cond_cost_mean":   cond_cost,
    }
