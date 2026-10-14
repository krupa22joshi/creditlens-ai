"""Typed data contracts shared by the pipeline, the API and the dashboard."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LoanType = Literal["personal", "home", "vehicle"]
CheckStatus = Literal["PASS", "REVIEW", "FAIL", "N/A"]
Decision = Literal["APPROVE", "REFER TO CREDIT COMMITTEE", "DECLINE"]


class Applicant(BaseModel):
    applicant_id: str = Field(default="APP-0001", description="Reference used in the credit memo")
    age: int = Field(ge=18, le=80)
    annual_income: float = Field(gt=0, description="Verified gross annual income in INR")
    employment_years: float = Field(ge=0, le=50)
    credit_score: int = Field(ge=300, le=900)
    loan_type: LoanType = "personal"
    loan_amount: float = Field(gt=0, description="Requested amount in INR")
    loan_term_months: int = Field(ge=6, le=360)
    interest_rate: float = Field(gt=0, le=40, description="Annual interest rate in percent")
    existing_monthly_debt: float = Field(ge=0, description="Existing EMIs and card dues per month in INR")
    num_delinquencies_12m: int = Field(ge=0, le=24, description="Number of 30+ DPD events in last 12 months")
    max_dpd_24m: int = Field(ge=0, le=180, description="Worst days-past-due in last 24 months")
    collateral_value: float = Field(default=0, ge=0, description="Market value of collateral in INR (secured loans)")
    loan_purpose: str = Field(default="Personal expenses", max_length=120)


class RiskDriver(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    direction: Literal["increases risk", "decreases risk"]


class PolicyCheck(BaseModel):
    rule_id: str
    name: str
    status: CheckStatus
    detail: str


class RetrievedClause(BaseModel):
    clause_id: str
    title: str
    source: str
    text: str
    score: float


class ValidationReport(BaseModel):
    passed: bool
    issues: list[str] = []
    unsupported_numbers: list[str] = []
    invalid_citations: list[str] = []
    citations_found: list[str] = []


class Assessment(BaseModel):
    applicant: Applicant
    probability_of_default: float
    risk_grade: Literal["A", "B", "C", "D"]
    decision: Decision
    approval_authority: str
    derived_metrics: dict[str, float]
    risk_drivers: list[RiskDriver]
    policy_checks: list[PolicyCheck]
    retrieved_clauses: list[RetrievedClause]
    credit_memo: str
    validation: ValidationReport
    memo_attempts: int
    memo_source: Literal["llm", "fallback-template"]
    llm_provider: str
    model_name: str
    model_version: str
    trace: list[str]
