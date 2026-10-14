"""Synthetic retail loan portfolio.

Real bank data cannot be published, so this generator produces a realistic portfolio whose
default behaviour follows well-known credit-risk relationships (DTI, bureau score,
delinquency history, employment stability, LTV). It also plants a post-disbursal leakage
column so the training pipeline can demonstrate leakage detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import compute_emi

PURPOSES = {
    "personal": ["Medical expenses", "Wedding", "Home renovation", "Education", "Debt consolidation", "Travel"],
    "home": ["Purchase of first home", "Purchase of flat", "Home construction"],
    "vehicle": ["New car purchase", "Two-wheeler purchase", "Used car purchase"],
}


def generate_portfolio(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 65, n)
    employment_years = np.minimum(rng.gamma(2.0, 3.0, n), age - 20).round(1)
    annual_income = np.round(rng.lognormal(13.3, 0.5, n), -3)
    credit_score = np.clip(rng.normal(705, 60, n), 300, 900).astype(int)
    loan_type = rng.choice(["personal", "home", "vehicle"], n, p=[0.5, 0.3, 0.2])

    multiplier = np.select(
        [loan_type == "personal", loan_type == "home"],
        [rng.uniform(0.1, 0.9, n), rng.uniform(1.5, 3.5, n)],
        rng.uniform(0.3, 1.0, n),
    )
    loan_amount = np.round(annual_income * multiplier, -3)

    loan_term_months = np.select(
        [loan_type == "personal", loan_type == "home"],
        [rng.choice([24, 36, 48, 60], n, p=[0.15, 0.3, 0.25, 0.3]), rng.choice([180, 240, 300], n)],
        rng.choice([36, 48, 60, 84], n),
    )
    base_rate = np.select([loan_type == "personal", loan_type == "home"], [13.0, 8.5], 10.0)
    interest_rate = np.round(base_rate + (750 - credit_score) / 50 + rng.normal(0, 0.6, n), 2).clip(6.5, 30)

    monthly_income = annual_income / 12
    existing_monthly_debt = np.round(monthly_income * rng.beta(1.5, 9, n), -2)

    delinquency_rate = np.clip((720 - credit_score) / 120, 0.05, 3.0)
    num_delinquencies_12m = rng.poisson(delinquency_rate)
    severe = rng.random(n) < np.clip((660 - credit_score) / 400, 0.01, 0.5)
    max_dpd_24m = np.where(severe, 90, np.where(num_delinquencies_12m > 0, rng.choice([30, 60], n), 0))

    secured_ltv = np.where(loan_type == "home", rng.uniform(0.5, 0.84, n), rng.uniform(0.6, 0.89, n))
    collateral_value = np.where(loan_type == "personal", 0.0, np.round(loan_amount / secured_ltv, -3))

    emi = compute_emi(loan_amount, interest_rate, loan_term_months)
    dti = (existing_monthly_debt + emi) / monthly_income
    ltv = np.where(collateral_value > 0, loan_amount / np.maximum(collateral_value, 1), 0)

    logit = (
        -3.1
        + 3.2 * (dti - 0.4)
        - 0.013 * (credit_score - 700)
        + 0.45 * num_delinquencies_12m
        + 1.0 * (max_dpd_24m >= 90)
        - 0.07 * np.minimum(employment_years, 10)
        + 0.8 * (ltv > 0.85)
        + 0.35 * (loan_type == "personal")
        # Non-linear effects seen in real books: payment shock above 55% DTI and the
        # compounding of a thin bureau score with a stretched budget.
        + 0.9 * (dti > 0.55)
        + 0.7 * ((credit_score < 640) & (dti > 0.45))
        + rng.normal(0, 0.45, n)
    )
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Post-disbursal information: almost a copy of the target. Must never be a model input.
    collection_flag = np.where(default == 1, rng.random(n) < 0.85, rng.random(n) < 0.03).astype(int)

    purpose = np.array([rng.choice(PURPOSES[t]) for t in loan_type])

    return pd.DataFrame(
        {
            "applicant_id": [f"APP-{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "credit_score": credit_score,
            "loan_type": loan_type,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "existing_monthly_debt": existing_monthly_debt,
            "num_delinquencies_12m": num_delinquencies_12m,
            "max_dpd_24m": max_dpd_24m,
            "collateral_value": collateral_value,
            "loan_purpose": purpose,
            "post_disbursal_collection_flag": collection_flag,
            "default": default,
        }
    )


SAMPLE_APPLICANTS = {
    "Prime salaried - home loan": dict(
        applicant_id="APP-DEMO-1", age=34, annual_income=1800000, employment_years=8, credit_score=782,
        loan_type="home", loan_amount=5500000, loan_term_months=240, interest_rate=8.6,
        existing_monthly_debt=12000, num_delinquencies_12m=0, max_dpd_24m=0, collateral_value=7800000,
        loan_purpose="Purchase of first home",
    ),
    "Near-prime - personal loan": dict(
        applicant_id="APP-DEMO-2", age=27, annual_income=540000, employment_years=1.5, credit_score=648,
        loan_type="personal", loan_amount=450000, loan_term_months=36, interest_rate=15.2,
        existing_monthly_debt=6500, num_delinquencies_12m=1, max_dpd_24m=30, collateral_value=0,
        loan_purpose="Debt consolidation",
    ),
    "High risk - stretched borrower": dict(
        applicant_id="APP-DEMO-3", age=41, annual_income=420000, employment_years=0.6, credit_score=588,
        loan_type="personal", loan_amount=600000, loan_term_months=24, interest_rate=18.5,
        existing_monthly_debt=9000, num_delinquencies_12m=3, max_dpd_24m=90, collateral_value=0,
        loan_purpose="Travel",
    ),
}
