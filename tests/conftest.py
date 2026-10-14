import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from creditlens.data import SAMPLE_APPLICANTS
from creditlens.model import RiskModel, train
from creditlens.retriever import PolicyRetriever
from creditlens.schemas import Applicant


@pytest.fixture(scope="session")
def risk_model(tmp_path_factory) -> RiskModel:
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    train(n_rows=3000, cv_folds=3, seed=7, save_to=path)
    return RiskModel.load(path, train_if_missing=False)


@pytest.fixture(scope="session")
def retriever() -> PolicyRetriever:
    return PolicyRetriever()


@pytest.fixture
def samples() -> dict[str, Applicant]:
    return {name: Applicant(**values) for name, values in SAMPLE_APPLICANTS.items()}
