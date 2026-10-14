"""Prompting and templating for the credit memo."""
from __future__ import annotations

import json

from .schemas import RetrievedClause

REQUIRED_SECTIONS = ["## Summary", "## Risk Drivers", "## Policy Assessment", "## Recommendation"]

SYSTEM_PROMPT = """You are a senior credit analyst at a retail bank. You draft credit memos that a
credit officer will review before any decision is final.

Hard rules:
1. Use ONLY the facts in <facts> and the clauses in <policy_clauses>. Never invent numbers,
   thresholds, names or regulations.
2. Copy every number exactly as it appears in <facts> (same unit and rounding).
3. The decision has already been made by the policy engine. State it verbatim; do not change it.
4. Cite policy clauses inline using their IDs in square brackets, e.g. [RLP-2]. Only cite IDs
   that appear in <policy_clauses>. Every FAIL or REVIEW check must be discussed and cited.
5. Write in clear, professional English for a business reader. Be concise.

Output format (Markdown, exactly these headings, nothing before the title):
# Credit Memo - <applicant_id>
_AI-generated draft for credit officer review [MRM-2]._
## Summary
## Risk Drivers
## Policy Assessment
## Recommendation"""


def build_prompt(facts: dict, clauses: list[RetrievedClause], feedback: list[str] | None = None) -> str:
    clause_block = "\n".join(f"[{c.clause_id}] {c.title} ({c.source}): {c.text}" for c in clauses)
    prompt = (
        "Draft the credit memo for this application.\n\n"
        f"<facts>\n{json.dumps(facts, indent=2, ensure_ascii=False)}\n</facts>\n\n"
        f"<policy_clauses>\n{clause_block}\n</policy_clauses>\n"
    )
    if feedback:
        issues = "\n".join(f"- {issue}" for issue in feedback)
        prompt += (
            "\n<validation_feedback>\nYour previous draft was rejected by the automated validator for these "
            f"reasons. Fix every one of them:\n{issues}\n</validation_feedback>\n"
        )
    return prompt


def format_inr(amount: float) -> str:
    """Format a number in the Indian numbering system, e.g. 450000 -> 'INR 4,50,000'."""
    amount = round(float(amount))
    sign, digits = ("-" if amount < 0 else ""), str(abs(int(amount)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"INR {sign}{digits}"


def render_template_memo(f: dict) -> str:
    """Deterministic memo built straight from the facts. Used offline and as a safe fallback."""
    drivers = "\n".join(
        f"- **{d['factor']}** ({d['value']}) {d['effect']}." for d in f["risk_drivers"]
    ) or "- No material drivers identified."

    # Checks that did not apply are listed for completeness but not cited as evidence.
    check_lines = "\n".join(
        f"| {c['rule_id'] if c['status'] == 'N/A' else '[' + c['rule_id'] + ']'} | {c['name']} | {c['status']} | {c['detail']} |"
        for c in f["policy_checks"]
    )
    open_items = [c for c in f["policy_checks"] if c["status"] in ("FAIL", "REVIEW")]
    if open_items:
        actions = "\n".join(
            f"- {'Blocking' if c['status'] == 'FAIL' else 'To resolve'}: {c['name']} [{c['rule_id']}] - {c['detail']}"
            for c in open_items
        )
    else:
        actions = "- No exceptions. Proceed with standard documentation [RLP-8]."

    return f"""# Credit Memo - {f['applicant_id']}
_AI-generated draft for credit officer review [MRM-2]. Model {f['model']} ({f['model_version']}), prepared {f['date']}._

## Summary
The applicant has requested a {f['loan_type']} loan of {f['loan_amount']} over {f['loan_term_months']} months at {f['interest_rate']}% for "{f['loan_purpose']}". Verified annual income is {f['annual_income']}, and the proposed EMI of {f['emi']} takes the debt-to-income ratio to {f['dti_percent']}%. The PD model estimates a probability of default of {f['pd_percent']}%, which maps to risk grade {f['risk_grade']} [RLP-6].

## Risk Drivers
The factors with the largest influence on this applicant's score, from SHAP attributions [MRM-1]:
{drivers}

## Policy Assessment
| Clause | Check | Status | Detail |
|---|---|---|---|
{check_lines}

## Recommendation
**{f['decision']}**. Approval authority: {f['approval_authority']} [RLP-6].
{actions}

This memo is decision support only; the final decision rests with the credit officer [MRM-2].
"""
