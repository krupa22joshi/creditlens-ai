"""Batch evaluation of the end-to-end pipeline on unseen synthetic applicants.

Reports guardrail metrics that matter to a risk function, not just model accuracy:
validation pass rate, fallback rate, citation coverage, decision mix and latency.

Usage: python scripts/evaluate.py [--n 50] [--provider mock]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from creditlens import config
from creditlens.agents import CreditAppraisalPipeline
from creditlens.data import generate_portfolio
from creditlens.llm import get_llm
from creditlens.schemas import Applicant

APPLICANT_FIELDS = set(Applicant.model_fields)

ISSUE_CATEGORIES = [
    ("Numbers not found", "unsupported number"),
    ("not provided as evidence", "invalid citation"),
    ("is not discussed with a citation", "flagged check not cited"),
    ("verbatim", "decision not stated"),
    ("contradicts", "decision contradicted"),
    ("Missing required section", "missing section"),
    ("AI-generated", "missing AI label"),
    ("cites no policy", "no citations"),
]


def issue_category(issue: str) -> str:
    return next((label for marker, label in ISSUE_CATEGORIES if marker in issue), "other")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--save-report", action="store_true", help="Also write results and a sample memo to reports/")
    args = parser.parse_args()

    pipeline = CreditAppraisalPipeline(llm=get_llm(args.provider))
    # A different seed from training, so these applicants were never seen by the model.
    portfolio = generate_portfolio(args.n, seed=config.RANDOM_SEED + 1000)

    results, latencies = [], []
    for i, row in enumerate(portfolio.to_dict("records"), 1):
        applicant = Applicant(**{k: v for k, v in row.items() if k in APPLICANT_FIELDS})
        t0 = time.perf_counter()
        result = pipeline.run(applicant)
        if result.llm_error:
            # An outage or quota error says nothing about memo quality - stop rather than skew the metrics.
            print(f"Stopping early: LLM unavailable ({result.llm_error}). Reporting {len(results)} applicants.")
            break
        results.append(result)
        latencies.append(time.perf_counter() - t0)
        print(f"[{i}/{args.n}] {applicant.applicant_id}: {result.decision:<26} attempts={result.memo_attempts} "
              f"source={result.memo_source} ({latencies[-1]:.1f}s)", flush=True)

    if not results:
        sys.exit("No applicants were evaluated.")
    flagged = [(r, c) for r in results for c in r.policy_checks if c.status in ("FAIL", "REVIEW")]
    covered = sum(c.rule_id in r.validation.citations_found for r, c in flagged)
    rejected = [issues for r in results for issues in r.rejected_draft_issues]
    n = len(results)
    report = {
        "provider": pipeline.llm.name,
        "model": getattr(pipeline.llm, "model", None) or pipeline.llm.name,
        "applicants": n,
        "final_validation_pass_rate": round(sum(r.validation.passed for r in results) / n, 3),
        "first_draft_pass_rate": round(sum(r.memo_source == "llm" and r.memo_attempts == 1 for r in results) / n, 3),
        "fixed_by_self_correction_rate": round(sum(r.memo_source == "llm" and r.memo_attempts > 1 for r in results) / n, 3),
        "fallback_rate": round(sum(r.memo_source == "fallback-template" for r in results) / n, 3),
        "drafts_rejected_by_validator": len(rejected),
        "rejection_reasons": dict(Counter(issue_category(i) for issues in rejected for i in issues).most_common()),
        "example_rejections": [issues for issues in rejected[:5]],
        "flagged_check_citation_coverage": round(covered / len(flagged), 3) if flagged else 1.0,
        "avg_clauses_retrieved": round(statistics.mean(len(r.retrieved_clauses) for r in results), 2),
        "decision_mix": dict(Counter(r.decision for r in results)),
        "grade_mix": dict(sorted(Counter(r.risk_grade for r in results).items())),
        "latency_seconds": {"mean": round(statistics.mean(latencies), 3), "p95": round(sorted(latencies)[int(0.95 * (n - 1))], 3)},
    }
    out = config.ARTIFACTS_DIR / f"eval_report_{pipeline.llm.name}.json"
    out.write_text(json.dumps(report, indent=2))
    if args.save_report:
        reports = config.PROJECT_ROOT / "reports"
        reports.mkdir(exist_ok=True)
        (reports / f"eval_{pipeline.llm.name}.json").write_text(json.dumps(report, indent=2))
        sample = next((r for r in results if r.memo_source == "llm"), results[0])
        (reports / f"sample_memo_{pipeline.llm.name}.md").write_text(sample.credit_memo, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
