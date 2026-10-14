from creditlens.data import generate_portfolio
from creditlens.features import LEAKAGE_COLUMNS, MODEL_FEATURES
from creditlens.model import leakage_audit


def test_portfolio_has_realistic_default_rate():
    df = generate_portfolio(5000, seed=1)
    assert 0.05 < df["default"].mean() < 0.30


def test_leakage_audit_flags_post_disbursal_column():
    audit = leakage_audit(generate_portfolio(4000, seed=3))
    flagged = {f["column"] for f in audit if f["suspected_leakage"]}
    assert set(LEAKAGE_COLUMNS) <= flagged
    assert not (flagged - set(LEAKAGE_COLUMNS)), f"genuine features flagged as leakage: {flagged}"


def test_leakage_columns_never_reach_the_model(risk_model):
    assert not set(LEAKAGE_COLUMNS) & set(risk_model.features)
    assert risk_model.features == MODEL_FEATURES


def test_model_compares_four_candidates_and_beats_random(risk_model):
    metrics = risk_model.metrics
    assert len(metrics["model_comparison"]) == 4
    assert metrics["holdout_roc_auc"] > 0.7


def test_risk_ordering_matches_intuition(risk_model, samples):
    prime = risk_model.predict_pd(samples["Prime salaried - home loan"])
    risky = risk_model.predict_pd(samples["High risk - stretched borrower"])
    assert prime < risky


def test_explanations_are_ranked_and_labelled(risk_model, samples):
    drivers = risk_model.explain(samples["High risk - stretched borrower"])
    assert drivers
    magnitudes = [abs(d.contribution) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(d.label and d.direction in ("increases risk", "decreases risk") for d in drivers)
