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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    pipeline = CreditAppraisalPipeline(llm=get_llm(args.provider))
    # A different seed from training, so these applicants were never seen by the model.
    portfolio = generate_portfolio(args.n, seed=config.RANDOM_SEED + 1000)

    results, latencies = [], []
    for row in portfolio.to_dict("records"):
        applicant = Applicant(**{k: v for k, v in row.items() if k in APPLICANT_FIELDS})
        t0 = time.perf_counter()
        results.append(pipeline.run(applicant))
        latencies.append(time.perf_counter() - t0)

    flagged = [(r, c) for r in results for c in r.policy_checks if c.status in ("FAIL", "REVIEW")]
    covered = sum(c.rule_id in r.validation.citations_found for r, c in flagged)
    report = {
        "provider": pipeline.llm.name,
        "applicants": len(results),
        "validation_pass_rate": round(sum(r.validation.passed for r in results) / len(results), 3),
        "first_attempt_pass_rate": round(sum(r.memo_source == "llm" and r.memo_attempts == 1 for r in results) / len(results), 3),
        "fallback_rate": round(sum(r.memo_source == "fallback-template" for r in results) / len(results), 3),
        "flagged_check_citation_coverage": round(covered / len(flagged), 3) if flagged else 1.0,
        "avg_clauses_retrieved": round(statistics.mean(len(r.retrieved_clauses) for r in results), 2),
        "decision_mix": dict(Counter(r.decision for r in results)),
        "grade_mix": dict(sorted(Counter(r.risk_grade for r in results).items())),
        "latency_seconds": {"mean": round(statistics.mean(latencies), 3), "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3)},
    }
    out = config.ARTIFACTS_DIR / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
