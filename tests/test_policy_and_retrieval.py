from creditlens.memo import format_inr
from creditlens.policy_rules import decide, risk_grade, run_policy_checks
from creditlens.schemas import PolicyCheck


def _checks(risk_model, applicant):
    return {c.rule_id: c for c in run_policy_checks(applicant, risk_model.derived_metrics(applicant))}


def test_risk_grade_bands():
    assert risk_grade(0.03)[0] == "A"
    assert risk_grade(0.05)[0] == "B"
    assert risk_grade(0.20)[0] == "C"
    assert risk_grade(0.40)[0] == "D"


def test_hard_declines_for_stretched_borrower(risk_model, samples):
    checks = _checks(risk_model, samples["High risk - stretched borrower"])
    assert checks["RLP-3"].status == "FAIL"   # score below 600 on unsecured
    assert checks["RLP-4"].status == "FAIL"   # 90+ DPD
    assert checks["KYC-3"].status == "REVIEW"  # under one year in job
    assert decide("B", list(checks.values())) == "DECLINE"


def test_prime_home_loan_passes(risk_model, samples):
    checks = _checks(risk_model, samples["Prime salaried - home loan"])
    assert all(c.status in ("PASS", "N/A") for c in checks.values())


def test_decision_logic():
    ok = [PolicyCheck(rule_id="RLP-2", name="x", status="PASS", detail="")]
    review = ok + [PolicyCheck(rule_id="RLP-3", name="y", status="REVIEW", detail="")]
    assert decide("A", ok) == "APPROVE"
    assert decide("C", ok) == "REFER TO CREDIT COMMITTEE"
    assert decide("A", review) == "REFER TO CREDIT COMMITTEE"
    assert decide("D", ok) == "DECLINE"


def test_retriever_finds_the_right_clause(retriever):
    assert retriever.search("maximum debt to income ratio monthly obligations", k=1)[0].clause_id == "RLP-2"
    assert retriever.search("loan to value for home loans", k=1)[0].clause_id == "RLP-5"
    assert retriever.search("SHAP reason codes explainability", k=1)[0].clause_id == "MRM-1"


def test_retriever_lookup_by_id(retriever):
    assert retriever.get("KYC-3").title == "Employment Stability"
    assert retriever.get("XYZ-1") is None


def test_indian_number_format():
    assert format_inr(450000) == "INR 4,50,000"
    assert format_inr(2500000) == "INR 25,00,000"
    assert format_inr(999) == "INR 999"
