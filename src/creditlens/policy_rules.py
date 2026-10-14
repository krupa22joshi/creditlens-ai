"""Deterministic policy engine.

The decision itself is rule-based and auditable; the LLM only explains it. This keeps the
generative component out of the approval path, in line with MRM-2 (human in the loop).
"""
from __future__ import annotations

from .schemas import Applicant, PolicyCheck

UNSECURED_CAP_INR = 2_500_000

GRADE_BANDS = [  # (upper PD bound, grade, approval authority) - policy RLP-6
    (0.05, "A", "Branch credit officer"),
    (0.12, "B", "Regional credit manager"),
    (0.25, "C", "Credit Committee"),
    (1.01, "D", "Chief Risk Officer (exception only)"),
]


def risk_grade(pd_value: float) -> tuple[str, str]:
    for upper, grade, authority in GRADE_BANDS:
        if pd_value < upper:
            return grade, authority
    return "D", GRADE_BANDS[-1][2]


def run_policy_checks(a: Applicant, derived: dict[str, float]) -> list[PolicyCheck]:
    dti, ltv = derived["dti"], derived["ltv"]
    monthly_income = derived["monthly_income"]
    unsecured = a.loan_type == "personal"
    checks: list[PolicyCheck] = []

    # RLP-2 Debt-to-income
    if dti > 0.50:
        checks.append(PolicyCheck(rule_id="RLP-2", name="Debt-to-income limit", status="FAIL",
                                  detail=f"DTI of {dti:.1%} exceeds the 50% ceiling."))
    elif dti > 0.40:
        mitigant = a.employment_years > 5 or not unsecured
        checks.append(PolicyCheck(
            rule_id="RLP-2", name="Debt-to-income limit", status="PASS" if mitigant else "REVIEW",
            detail=f"DTI of {dti:.1%} is in the 40-50% band; "
                   + ("mitigant present (stable employment or collateral)." if mitigant else "a documented mitigant is required."),
        ))
    else:
        checks.append(PolicyCheck(rule_id="RLP-2", name="Debt-to-income limit", status="PASS",
                                  detail=f"DTI of {dti:.1%} is within the 40% comfort level."))

    # RLP-3 Credit score
    if a.credit_score < 600 and unsecured:
        checks.append(PolicyCheck(rule_id="RLP-3", name="Credit score threshold", status="FAIL",
                                  detail=f"Score {a.credit_score} is below 600, the minimum for unsecured loans."))
    elif a.credit_score < 680:
        checks.append(PolicyCheck(rule_id="RLP-3", name="Credit score threshold", status="REVIEW",
                                  detail=f"Score {a.credit_score} is near-prime; enhanced income verification needed (KYC-2)."))
    else:
        checks.append(PolicyCheck(rule_id="RLP-3", name="Credit score threshold", status="PASS",
                                  detail=f"Score {a.credit_score} is prime."))

    # RLP-4 Delinquency history
    if a.max_dpd_24m >= 90 and unsecured:
        checks.append(PolicyCheck(rule_id="RLP-4", name="Delinquency history", status="FAIL",
                                  detail=f"A {a.max_dpd_24m}+ DPD event in the last 24 months is a hard decline for unsecured loans."))
    elif a.num_delinquencies_12m >= 2 or a.max_dpd_24m >= 90:
        checks.append(PolicyCheck(rule_id="RLP-4", name="Delinquency history", status="REVIEW",
                                  detail=f"{a.num_delinquencies_12m} delinquencies in 12 months, worst DPD {a.max_dpd_24m}; Credit Committee review."))
    else:
        checks.append(PolicyCheck(rule_id="RLP-4", name="Delinquency history", status="PASS",
                                  detail=f"{a.num_delinquencies_12m} delinquencies in 12 months, worst DPD {a.max_dpd_24m}."))

    # RLP-5 Loan-to-value
    if unsecured:
        checks.append(PolicyCheck(rule_id="RLP-5", name="Loan-to-value limit", status="N/A",
                                  detail="Unsecured loan - LTV not applicable."))
    else:
        cap = 0.80 if a.loan_type == "home" else 0.85
        if a.collateral_value <= 0:
            checks.append(PolicyCheck(rule_id="RLP-5", name="Loan-to-value limit", status="FAIL",
                                      detail="Secured loan submitted without a collateral valuation."))
        else:
            ok = ltv <= cap
            checks.append(PolicyCheck(rule_id="RLP-5", name="Loan-to-value limit", status="PASS" if ok else "FAIL",
                                      detail=f"LTV of {ltv:.1%} against a {cap:.0%} cap for {a.loan_type} loans."))

    # RLP-7 Unsecured loan amount caps
    if unsecured:
        income_cap = 20 * monthly_income
        if a.loan_amount > min(income_cap, UNSECURED_CAP_INR):
            checks.append(PolicyCheck(rule_id="RLP-7", name="Unsecured amount cap", status="FAIL",
                                      detail="Requested amount exceeds 20x monthly income or the INR 25,00,000 cap."))
        else:
            checks.append(PolicyCheck(rule_id="RLP-7", name="Unsecured amount cap", status="PASS",
                                      detail="Requested amount is within 20x monthly income and the absolute cap."))

    # KYC-3 Employment stability
    if a.employment_years < 1:
        checks.append(PolicyCheck(rule_id="KYC-3", name="Employment stability", status="REVIEW",
                                  detail="Less than one year in current employment - note as a risk factor."))
    else:
        checks.append(PolicyCheck(rule_id="KYC-3", name="Employment stability", status="PASS",
                                  detail="At least one year in current employment."))

    # AML-1 Amount unusually high relative to income
    lti_limit = 1.5 if unsecured else 8.0
    if derived["loan_to_income"] > lti_limit:
        checks.append(PolicyCheck(rule_id="AML-1", name="Amount vs income profile", status="REVIEW",
                                  detail="Requested amount is unusually high relative to income - enhanced due diligence."))
    else:
        checks.append(PolicyCheck(rule_id="AML-1", name="Amount vs income profile", status="PASS",
                                  detail="Requested amount is consistent with the income profile."))
    return checks


def decide(grade: str, checks: list[PolicyCheck]) -> str:
    statuses = {c.status for c in checks}
    if "FAIL" in statuses or grade == "D":
        return "DECLINE"
    if "REVIEW" in statuses or grade == "C":
        return "REFER TO CREDIT COMMITTEE"
    return "APPROVE"
