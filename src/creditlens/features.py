"""Feature engineering shared by training and inference, so both always see identical inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_NUMERIC = [
    "age",
    "annual_income",
    "employment_years",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "existing_monthly_debt",
    "num_delinquencies_12m",
    "max_dpd_24m",
]

ENGINEERED = ["emi", "dti", "ltv", "loan_to_income", "is_personal", "is_home", "is_vehicle"]

# EMI, loan amount and existing debt are already expressed through DTI and loan-to-income.
# Feeding them in as well makes coefficients and SHAP signs unstable (e.g. "higher income
# increases risk"), which would confuse a credit officer reading the reason codes.
REDUNDANT = {"emi", "loan_amount", "existing_monthly_debt"}
MODEL_FEATURES = [f for f in RAW_NUMERIC + ENGINEERED if f not in REDUNDANT]

# Columns that only exist after the loan is disbursed. Using them would leak the outcome
# into training, so they are dropped before any model sees the data (policy MRM-4).
LEAKAGE_COLUMNS = ["post_disbursal_collection_flag"]

FEATURE_LABELS = {
    "age": "Applicant age",
    "annual_income": "Annual income",
    "employment_years": "Years in current employment",
    "credit_score": "Bureau credit score",
    "loan_amount": "Requested loan amount",
    "loan_term_months": "Loan tenure (months)",
    "interest_rate": "Interest rate",
    "existing_monthly_debt": "Existing monthly debt",
    "num_delinquencies_12m": "Delinquencies in last 12 months",
    "max_dpd_24m": "Worst days-past-due in last 24 months",
    "emi": "Proposed EMI",
    "dti": "Debt-to-income ratio",
    "ltv": "Loan-to-value ratio",
    "loan_to_income": "Loan-to-income ratio",
    "is_personal": "Unsecured personal loan",
    "is_home": "Home loan",
    "is_vehicle": "Vehicle loan",
}


def compute_emi(principal, annual_rate_pct, months):
    """Standard reducing-balance EMI. Works on scalars and numpy arrays."""
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    n = np.asarray(months, dtype=float)
    p = np.asarray(principal, dtype=float)
    growth = np.power(1 + r, n)
    return p * r * growth / (growth - 1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered credit-risk features added."""
    out = df.copy()
    monthly_income = out["annual_income"] / 12.0
    out["emi"] = compute_emi(out["loan_amount"], out["interest_rate"], out["loan_term_months"])
    out["dti"] = (out["existing_monthly_debt"] + out["emi"]) / monthly_income
    collateral = out.get("collateral_value", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["ltv"] = np.where(collateral > 0, out["loan_amount"] / collateral.replace(0, np.nan), 0.0)
    out["ltv"] = out["ltv"].fillna(0.0)
    out["loan_to_income"] = out["loan_amount"] / out["annual_income"]
    loan_type = out.get("loan_type", pd.Series("personal", index=out.index))
    out["is_personal"] = (loan_type == "personal").astype(int)
    out["is_home"] = (loan_type == "home").astype(int)
    out["is_vehicle"] = (loan_type == "vehicle").astype(int)
    return out


def to_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and keep only the approved model inputs (no leakage columns)."""
    return engineer_features(df)[MODEL_FEATURES].astype(float)
