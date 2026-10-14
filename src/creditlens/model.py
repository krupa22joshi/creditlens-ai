"""Probability-of-default model: leakage audit, model comparison, training and explanations."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .data import generate_portfolio
from .features import FEATURE_LABELS, LEAKAGE_COLUMNS, MODEL_FEATURES, engineer_features, to_model_matrix
from .schemas import Applicant, RiskDriver

log = logging.getLogger(__name__)


def candidate_models(seed: int) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
        ),
    }


def leakage_audit(df: pd.DataFrame, target: str = "default", threshold: float = 0.85) -> list[dict]:
    """Flag columns that separate the target suspiciously well on their own.

    A single raw feature with ROC-AUC above ``threshold`` is almost always a sign of
    information from the future (e.g. collection status) rather than genuine signal.
    """
    engineered = engineer_features(df)
    candidates = [c for c in engineered.columns if c not in (target, "applicant_id") and engineered[c].dtype.kind in "if"]
    findings = []
    for col in candidates:
        values = engineered[col].fillna(0)
        if values.nunique() < 2:
            continue
        auc = roc_auc_score(engineered[target], values)
        auc = max(auc, 1 - auc)
        findings.append({"column": col, "single_feature_auc": round(float(auc), 4), "suspected_leakage": auc >= threshold})
    return sorted(findings, key=lambda f: f["single_feature_auc"], reverse=True)


def train(n_rows: int | None = None, cv_folds: int | None = None, seed: int | None = None,
          save_to: Path | None = None) -> dict:
    """Compare four models with stratified CV, refit the best one and persist it."""
    n_rows = n_rows or config.TRAINING_ROWS
    cv_folds = cv_folds or config.CV_FOLDS
    seed = config.RANDOM_SEED if seed is None else seed
    save_to = save_to or config.MODEL_PATH

    df = generate_portfolio(n_rows, seed)
    audit = leakage_audit(df)
    flagged = [f["column"] for f in audit if f["suspected_leakage"]]
    unexpected = sorted(set(flagged) - set(LEAKAGE_COLUMNS))
    if unexpected:
        log.warning("Leakage audit flagged columns not in the drop list: %s", unexpected)

    X = to_model_matrix(df.drop(columns=LEAKAGE_COLUMNS))
    y = df["default"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    comparison = []
    for name, est in candidate_models(seed).items():
        scores = cross_validate(est, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
        comparison.append({
            "model": name,
            "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
            "cv_roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
            "cv_pr_auc": round(float(scores["test_average_precision"].mean()), 4),
        })
        log.info("%-20s ROC-AUC %.4f", name, comparison[-1]["cv_roc_auc"])

    best = max(comparison, key=lambda r: r["cv_roc_auc"])
    estimator = candidate_models(seed)[best["model"]]
    estimator.fit(X_train, y_train)
    test_scores = estimator.predict_proba(X_test)[:, 1]

    trained_at = datetime.now(timezone.utc)
    metrics = {
        "model_name": best["model"],
        "model_version": f"pd-{trained_at:%Y%m%d%H%M%S}",
        "trained_at": trained_at.isoformat(),
        "training_rows": n_rows,
        "default_rate": round(float(y.mean()), 4),
        "holdout_roc_auc": round(float(roc_auc_score(y_test, test_scores)), 4),
        "holdout_pr_auc": round(float(average_precision_score(y_test, test_scores)), 4),
        "model_comparison": comparison,
        "leakage_audit": audit[:8],
        "dropped_leakage_columns": LEAKAGE_COLUMNS,
        "features": MODEL_FEATURES,
    }

    bundle = {
        "estimator": estimator,
        "features": MODEL_FEATURES,
        "background": X_train.sample(min(200, len(X_train)), random_state=seed),
        "metrics": metrics,
    }
    save_to.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_to)
    save_to.with_name("metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


class RiskModel:
    """Loads the trained bundle and serves PD scores with per-applicant explanations."""

    def __init__(self, bundle: dict):
        self.estimator = bundle["estimator"]
        self.features: list[str] = bundle["features"]
        self.background: pd.DataFrame = bundle["background"]
        self.metrics: dict = bundle["metrics"]

    @classmethod
    def load(cls, path: Path | None = None, train_if_missing: bool = True) -> RiskModel:
        path = path or config.MODEL_PATH
        if not path.exists():
            if not train_if_missing:
                raise FileNotFoundError(f"No trained model at {path}. Run `python scripts/train.py`.")
            log.info("No model found at %s - training a new one", path)
            train(save_to=path)
        return cls(joblib.load(path))

    @property
    def name(self) -> str:
        return self.metrics["model_name"]

    @property
    def version(self) -> str:
        return self.metrics["model_version"]

    def _matrix(self, applicant: Applicant) -> pd.DataFrame:
        return to_model_matrix(pd.DataFrame([applicant.model_dump()]))

    def derived_metrics(self, applicant: Applicant) -> dict[str, float]:
        row = engineer_features(pd.DataFrame([applicant.model_dump()])).iloc[0]
        return {
            "monthly_income": round(applicant.annual_income / 12, 2),
            "emi": round(float(row["emi"]), 2),
            "dti": round(float(row["dti"]), 4),
            "ltv": round(float(row["ltv"]), 4),
            "loan_to_income": round(float(row["loan_to_income"]), 4),
        }

    def predict_pd(self, applicant: Applicant) -> float:
        return float(self.estimator.predict_proba(self._matrix(applicant))[0, 1])

    def explain(self, applicant: Applicant, top_k: int = 6) -> list[RiskDriver]:
        x = self._matrix(applicant)
        contributions = self._contributions(x)
        order = np.argsort(-np.abs(contributions))[:top_k]
        return [
            RiskDriver(
                feature=self.features[i],
                label=FEATURE_LABELS.get(self.features[i], self.features[i]),
                value=round(float(x.iloc[0, i]), 4),
                contribution=round(float(contributions[i]), 4),
                direction="increases risk" if contributions[i] > 0 else "decreases risk",
            )
            for i in order
            if abs(contributions[i]) > 1e-6
        ]

    def _contributions(self, x: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions in log-odds space (SHAP for trees, exact for linear)."""
        est = self.estimator
        if isinstance(est, Pipeline):
            scaler, lr = est[0], est[-1]
            centred = scaler.transform(x) - scaler.transform(self.background).mean(axis=0)
            return (lr.coef_[0] * centred)[0]
        if isinstance(est, XGBClassifier):
            import xgboost as xgb

            contribs = est.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
            return contribs[0, :-1]  # last column is the bias term
        try:
            import shap

            values = shap.TreeExplainer(est).shap_values(x)
            values = np.asarray(values[1] if isinstance(values, list) else values)
            return values[0, :, 1] if values.ndim == 3 else values[0]
        except Exception:  # pragma: no cover - fallback if shap is unavailable
            return self._mean_substitution(x)

    def _mean_substitution(self, x: pd.DataFrame) -> np.ndarray:
        base = self.estimator.predict_proba(x)[0, 1]
        means = self.background.mean()
        out = np.zeros(len(self.features))
        for i, col in enumerate(self.features):
            probe = x.copy()
            probe[col] = means[col]
            out[i] = base - self.estimator.predict_proba(probe)[0, 1]
        return out
