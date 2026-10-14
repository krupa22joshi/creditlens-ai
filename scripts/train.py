"""Train and persist the PD model.  Usage: python scripts/train.py [--rows 20000] [--folds 5]"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from creditlens.model import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None, help="Synthetic portfolio size")
    parser.add_argument("--folds", type=int, default=None, help="Stratified CV folds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics = train(n_rows=args.rows, cv_folds=args.folds)
    print(json.dumps({k: metrics[k] for k in ("model_name", "model_version", "holdout_roc_auc", "holdout_pr_auc")}, indent=2))
    print("\nModel comparison:")
    for row in metrics["model_comparison"]:
        print(f"  {row['model']:<20} ROC-AUC {row['cv_roc_auc']:.4f} ± {row['cv_roc_auc_std']:.4f}   PR-AUC {row['cv_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
