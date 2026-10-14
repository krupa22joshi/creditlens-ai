"""Benchmark lexical, dense and hybrid retrieval on a labelled question set.

Each question in evaluation/retrieval_questions.json names the single clause that answers it.
Questions are tagged "keyword" (shares vocabulary with the policy) or "paraphrase" (how a
credit officer or customer would actually ask), which is where dense retrieval should help.

Usage: python scripts/evaluate_retrieval.py [--save-report]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from creditlens import config
from creditlens.retriever import PolicyRetriever

QUESTIONS = config.PROJECT_ROOT / "evaluation" / "retrieval_questions.json"


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = PolicyRetriever()
    modes = ["lexical"] + (["dense", "hybrid"] if retriever.dense_matrix is not None else [])

    report = {"questions": len(questions), "clauses": len(retriever.clauses), "results": {}, "misses": {}}
    for mode in modes:
        ranks, misses = {"all": [], "keyword": [], "paraphrase": []}, []
        for q in questions:
            ids = [c.clause_id for c in retriever.search(q["question"], k=len(retriever.clauses), mode=mode)]
            rank = ids.index(q["expected"]) + 1 if q["expected"] in ids else None
            ranks["all"].append(rank)
            ranks[q["style"]].append(rank)
            if rank != 1:
                misses.append({"question": q["question"], "expected": q["expected"], "got": ids[:3]})
        report["results"][mode] = {style: score(r) for style, r in ranks.items() if r}
        report["misses"][mode] = misses

    print(f"{len(questions)} questions over {len(retriever.clauses)} clauses\n")
    print(f"{'mode':<9}{'subset':<12}{'hit@1':>7}{'hit@3':>7}{'MRR':>7}")
    for mode, subsets in report["results"].items():
        for subset, m in subsets.items():
            print(f"{mode:<9}{subset:<12}{m['hit@1']:>7}{m['hit@3']:>7}{m['mrr']:>7}")
    if args.save_report:
        out = config.PROJECT_ROOT / "reports" / "retrieval_eval.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
