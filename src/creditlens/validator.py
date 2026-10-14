"""Guardrails for generated credit memos (policy MRM-3).

The validator checks a draft against the system of record before anyone sees it:
* every number in the memo must be traceable to the facts or the cited policy text,
* every cited clause must be one of the clauses supplied to the writer,
* every FAIL / REVIEW policy check must be discussed with its citation,
* the recommendation must match the policy engine's decision exactly,
* the memo must carry the AI-generated label and the required sections.
"""
from __future__ import annotations

import json
import re

from .memo import REQUIRED_SECTIONS
from .schemas import RetrievedClause, ValidationReport

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Any bracketed reference such as [RLP-2], [RLP-3, KYC-2] or an invented [RBI-99] counts as a citation.
BRACKET = re.compile(r"\[([^\[\]]{1,80})\]")
CLAUSE_REF = re.compile(r"\b[A-Z]{2,5}-\d{1,3}\b")  # short numbers only, so "[APP-00012]" is not a citation
DECISIONS = ["APPROVE", "REFER TO CREDIT COMMITTEE", "DECLINE"]


def _numbers(text: str) -> list[tuple[str, float]]:
    found = []
    for raw in NUMBER.findall(text):
        cleaned = raw.rstrip(",").replace(",", "")
        try:
            found.append((raw.rstrip(","), float(cleaned)))
        except ValueError:
            continue
    return found


def _supported(value: float, allowed: set[float]) -> bool:
    if value <= 12 and value.is_integer():  # list counts, small ordinals
        return True
    for v in allowed:
        if abs(value - v) <= max(0.051, 0.001 * abs(v)):
            return True
        if value.is_integer() and abs(value - v) < 0.5:  # legitimate rounding to whole number
            return True
    return False


def _section(memo: str, heading: str) -> str:
    start = memo.find(heading)
    if start == -1:
        return ""
    nxt = memo.find("\n## ", start + len(heading))
    return memo[start: nxt if nxt != -1 else len(memo)]


def validate_memo(memo: str, facts: dict, clauses: list[RetrievedClause]) -> ValidationReport:
    issues: list[str] = []

    for heading in REQUIRED_SECTIONS:
        if heading not in memo:
            issues.append(f"Missing required section '{heading}'.")
    if "ai-generated" not in memo.lower():
        issues.append("Memo must be labelled as an AI-generated draft (MRM-2).")

    # Numerical grounding
    reference_text = json.dumps(facts, ensure_ascii=False) + " " + " ".join(f"{c.clause_id} {c.text}" for c in clauses)
    allowed = {v for _, v in _numbers(reference_text)}
    unsupported = sorted({raw for raw, v in _numbers(memo) if not _supported(v, allowed)})
    if unsupported:
        issues.append(f"Numbers not found in the facts or policy text: {', '.join(unsupported)}.")

    # Citation grounding and coverage
    provided = {c.clause_id for c in clauses}
    cited = sorted({ref for group in BRACKET.findall(memo) for ref in CLAUSE_REF.findall(group)})
    invalid = [c for c in cited if c not in provided]
    if not cited:
        issues.append("Memo cites no policy clauses.")
    if invalid:
        issues.append(f"Cited clauses that were not provided as evidence: {', '.join(invalid)}.")
    for check in facts.get("policy_checks", []):
        if check["status"] in ("FAIL", "REVIEW") and check["rule_id"] not in cited:
            issues.append(f"{check['status']} check {check['rule_id']} ({check['name']}) is not discussed with a citation.")

    # Decision consistency
    recommendation = _section(memo, "## Recommendation")
    decision = facts["decision"]
    if decision not in recommendation:
        issues.append(f"Recommendation section must state the decision '{decision}' verbatim.")
    # Decisions are written in capitals, so a case-sensitive match avoids flagging prose like "hard decline".
    contradicting = [
        d for d in DECISIONS
        if d != decision and d not in decision and re.search(rf"\b{d}\b", recommendation)
    ]
    if contradicting:
        issues.append(f"Recommendation contradicts the policy decision (mentions {', '.join(contradicting)}).")

    return ValidationReport(
        passed=not issues,
        issues=issues,
        unsupported_numbers=unsupported,
        invalid_citations=invalid,
        citations_found=cited,
    )
