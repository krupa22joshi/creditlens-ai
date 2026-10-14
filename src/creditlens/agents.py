"""Multi-agent credit appraisal workflow.

    Applicant
       |
  [1] Risk Analyst      PD model + SHAP reason codes
       |
  [2] Compliance Officer deterministic policy checks, risk grade, decision
       |
  [3] Policy Researcher  retrieves the clauses that govern this specific case
       |
  [4] Memo Writer       LLM drafts the credit memo, grounded in facts + clauses
       |
  [5] Validator  ------> fails? feed issues back to [4] (bounded retries)
       |                  still failing? fall back to the deterministic template
  Assessment (memo + evidence + audit trace)

Each agent is a small, testable function over a shared state object, so the flow is easy to
audit and to port to an orchestration framework such as LangGraph if needed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

from . import config
from .llm import LLMClient, LLMError, get_llm
from .memo import SYSTEM_PROMPT, build_prompt, format_inr, render_template_memo
from .model import RiskModel
from .policy_rules import decide, risk_grade, run_policy_checks
from .retriever import PolicyRetriever
from .schemas import Applicant, Assessment, PolicyCheck, RetrievedClause, RiskDriver, ValidationReport
from .validator import validate_memo

log = logging.getLogger(__name__)

ALWAYS_RETRIEVE = ["RLP-6", "RLP-8", "MRM-1", "MRM-2"]  # grading, memo content, explainability, human review


@dataclass
class CaseState:
    applicant: Applicant
    derived: dict = field(default_factory=dict)
    pd: float = 0.0
    grade: str = ""
    authority: str = ""
    drivers: list[RiskDriver] = field(default_factory=list)
    checks: list[PolicyCheck] = field(default_factory=list)
    decision: str = ""
    clauses: list[RetrievedClause] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    memo: str = ""
    memo_source: str = "llm"
    attempts: int = 0
    rejected_draft_issues: list[list[str]] = field(default_factory=list)
    llm_error: str | None = None
    validation: ValidationReport | None = None
    trace: list[str] = field(default_factory=list)

    def log(self, agent: str, message: str) -> None:
        self.trace.append(f"[{agent}] {message}")


def _display_value(feature: str, value: float) -> str:
    if feature in ("dti", "ltv"):
        return f"{value * 100:.1f}%"
    if feature == "loan_to_income":
        return f"{value:.2f}x income"
    if feature in ("annual_income", "loan_amount", "existing_monthly_debt", "emi"):
        return format_inr(value)
    if feature.startswith("is_"):
        return "yes" if value else "no"
    if feature == "interest_rate":
        return f"{value}%"
    return f"{value:g}"


class CreditAppraisalPipeline:
    def __init__(self, model: RiskModel | None = None, retriever: PolicyRetriever | None = None,
                 llm: LLMClient | None = None, max_attempts: int | None = None):
        self.model = model or RiskModel.load()
        self.retriever = retriever or PolicyRetriever()
        self.llm = llm or get_llm()
        self.max_attempts = max_attempts or config.MAX_MEMO_ATTEMPTS

    # [1] Risk Analyst
    def risk_analyst(self, s: CaseState) -> None:
        s.derived = self.model.derived_metrics(s.applicant)
        s.pd = self.model.predict_pd(s.applicant)
        s.drivers = self.model.explain(s.applicant)
        s.log("RiskAnalyst", f"PD={s.pd:.4f} using {self.model.name} {self.model.version}; "
                             f"top driver: {s.drivers[0].label if s.drivers else 'n/a'}")

    # [2] Compliance Officer
    def compliance_officer(self, s: CaseState) -> None:
        s.grade, s.authority = risk_grade(s.pd)
        s.checks = run_policy_checks(s.applicant, s.derived)
        s.decision = decide(s.grade, s.checks)
        flagged = [f"{c.rule_id}:{c.status}" for c in s.checks if c.status in ("FAIL", "REVIEW")]
        s.log("ComplianceOfficer", f"grade={s.grade}, decision={s.decision}, flagged={flagged or 'none'}")

    # [3] Policy Researcher
    def policy_researcher(self, s: CaseState) -> None:
        wanted: dict[str, RetrievedClause] = {}
        # Clauses behind every check that was actually run are mandatory evidence.
        for cid in [c.rule_id for c in s.checks if c.status != "N/A"] + ALWAYS_RETRIEVE:
            clause = self.retriever.get(cid)
            if clause:
                wanted[cid] = clause
        # Semantic search adds context for the specific risk drivers and open issues. One best
        # clause per query keeps loosely related policy text out of the prompt.
        queries = [f"{d.label} {d.direction}" for d in s.drivers[:3]]
        queries += [f"{c.name} {c.detail}" for c in s.checks if c.status in ("FAIL", "REVIEW")]
        for q in queries:
            for hit in self.retriever.search(q, k=1):
                wanted.setdefault(hit.clause_id, hit)
        s.clauses = sorted(wanted.values(), key=lambda c: c.clause_id)
        s.log("PolicyResearcher", f"retrieved {len(s.clauses)} clauses: {[c.clause_id for c in s.clauses]}")

    def _build_facts(self, s: CaseState) -> dict:
        a, d = s.applicant, s.derived
        return {
            "applicant_id": a.applicant_id,
            "date": date.today().isoformat(),
            "loan_type": a.loan_type,
            "loan_purpose": a.loan_purpose,
            "loan_amount": format_inr(a.loan_amount),
            "loan_term_months": a.loan_term_months,
            "interest_rate": a.interest_rate,
            "annual_income": format_inr(a.annual_income),
            "monthly_income": format_inr(d["monthly_income"]),
            "emi": format_inr(d["emi"]),
            "existing_monthly_debt": format_inr(a.existing_monthly_debt),
            "credit_score": a.credit_score,
            "employment_years": a.employment_years,
            "num_delinquencies_12m": a.num_delinquencies_12m,
            "max_dpd_24m": a.max_dpd_24m,
            "dti_percent": round(d["dti"] * 100, 1),
            "ltv_percent": round(d["ltv"] * 100, 1),
            "pd_percent": round(s.pd * 100, 1),
            "risk_grade": s.grade,
            "approval_authority": s.authority,
            "decision": s.decision,
            "model": self.model.name,
            "model_version": self.model.version,
            "risk_drivers": [
                {"factor": r.label, "value": _display_value(r.feature, r.value), "effect": r.direction}
                for r in s.drivers
            ],
            "policy_checks": [c.model_dump() for c in s.checks],
        }

    # [4] Memo Writer + [5] Validator, with a bounded self-correction loop
    def write_and_validate(self, s: CaseState) -> None:
        s.facts = self._build_facts(s)
        feedback: list[str] | None = None
        for attempt in range(1, self.max_attempts + 1):
            s.attempts = attempt
            try:
                draft = self.llm.generate(SYSTEM_PROMPT, build_prompt(s.facts, s.clauses, feedback))
            except LLMError as exc:
                s.llm_error = str(exc)
                s.log("MemoWriter", f"attempt {attempt}: LLM call failed ({exc})")
                break
            report = validate_memo(draft, s.facts, s.clauses)
            s.log("Validator", f"attempt {attempt}: {'passed' if report.passed else f'{len(report.issues)} issue(s)'}")
            if report.passed:
                s.memo, s.validation, s.memo_source = draft, report, "llm"
                return
            s.rejected_draft_issues.append(report.issues)
            feedback = report.issues

        # Never surface an unvalidated draft (MRM-3): fall back to the deterministic template.
        s.memo = render_template_memo(s.facts)
        s.validation = validate_memo(s.memo, s.facts, s.clauses)
        s.memo_source = "fallback-template"
        s.log("Validator", "LLM draft rejected - using validated template memo")

    def run(self, applicant: Applicant) -> Assessment:
        started = time.perf_counter()
        s = CaseState(applicant=applicant)
        for step in (self.risk_analyst, self.compliance_officer, self.policy_researcher, self.write_and_validate):
            step(s)
        s.log("Pipeline", f"completed in {time.perf_counter() - started:.2f}s with provider '{self.llm.name}'")
        return Assessment(
            applicant=applicant,
            probability_of_default=round(s.pd, 4),
            risk_grade=s.grade,
            decision=s.decision,
            approval_authority=s.authority,
            derived_metrics=s.derived,
            risk_drivers=s.drivers,
            policy_checks=s.checks,
            retrieved_clauses=s.clauses,
            credit_memo=s.memo,
            validation=s.validation,
            memo_attempts=s.attempts,
            rejected_draft_issues=s.rejected_draft_issues,
            llm_error=s.llm_error,
            memo_source=s.memo_source,
            llm_provider=self.llm.name,
            model_name=self.model.name,
            model_version=self.model.version,
            trace=s.trace,
        )
