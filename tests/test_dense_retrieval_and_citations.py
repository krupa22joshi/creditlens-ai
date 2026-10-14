import json

import httpx
import numpy as np
import pytest

from creditlens import llm as llm_module
from creditlens.llm import LLMError, post_with_retry
from creditlens.retriever import PolicyRetriever
from creditlens.validator import validate_memo

SYNONYMS = {"wife": "co-applicant", "salary": "income", "spouse": "co-applicant"}


class FakeEmbedder:
    """Deterministic stand-in for the embedding API: hashed bag of words with a synonym map."""

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        out = np.zeros((len(texts), 256))
        for row, text in enumerate(texts):
            for word in text.lower().replace(",", " ").replace(".", " ").split():
                out[row, hash(SYNONYMS.get(word, word)) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture(scope="module")
def dense_retriever():
    return PolicyRetriever(embedder=FakeEmbedder(), min_dense=0.0)


def test_all_modes_return_ranked_clauses(dense_retriever):
    for mode in ("lexical", "dense", "hybrid"):
        hits = dense_retriever.search("co-applicant income family member", k=3, mode=mode)
        assert hits and hits[0].clause_id == "COL-4", mode


def test_dense_is_default_when_embedder_present(dense_retriever, retriever):
    assert dense_retriever.default_mode == "dense"
    assert retriever.default_mode == "lexical"  # offline fallback


def test_dense_mode_requires_embedder(retriever):
    with pytest.raises(RuntimeError):
        retriever.dense_scores("anything")


def test_knowledge_base_covers_all_policy_areas(retriever):
    prefixes = {cid.split("-")[0] for cid in retriever.clause_ids}
    assert {"RLP", "MRM", "KYC", "AML", "COL", "FPC", "COLL", "FRD", "DIG"} <= prefixes


def _facts():
    return {"decision": "APPROVE", "policy_checks": [
        {"rule_id": "RLP-3", "name": "Credit score threshold", "status": "REVIEW", "detail": ""}
    ]}


def _memo(citation):
    return (f"# Credit Memo\n_AI-generated draft._\n## Summary\nok\n## Risk Drivers\nok\n"
            f"## Policy Assessment\nScore is near-prime {citation}.\n## Recommendation\n**APPROVE**\n")


def test_validator_parses_grouped_citations(retriever):
    clauses = [retriever.get("RLP-3"), retriever.get("KYC-2")]
    assert validate_memo(_memo("[RLP-3, KYC-2]"), _facts(), clauses).passed
    report = validate_memo(_memo("[RLP-3, RBI-99]"), _facts(), clauses)
    assert report.invalid_citations == ["RBI-99"]


def test_applicant_ids_in_brackets_are_not_citations(retriever):
    report = validate_memo(_memo("[RLP-3] for [APP-00012]"), _facts(), [retriever.get("RLP-3")])
    assert report.passed, report.issues


def test_post_with_retry_backs_off_on_rate_limit(monkeypatch):
    responses = iter([
        httpx.Response(429, text=json.dumps({"error": {"details": [{"retryDelay": "0s"}]}}).replace(" ", "")),
        httpx.Response(200, json={"ok": True}),
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *a, **k: next(responses))
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    assert post_with_retry("https://x", {}, {}, "test") == {"ok": True}


def test_daily_quota_fails_fast_without_sleeping(monkeypatch):
    body = '{"error":{"details":[{"violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},{"retryDelay":"5029s"}]}}'
    monkeypatch.setattr(llm_module.httpx, "post", lambda *a, **k: httpx.Response(429, text=body))
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: pytest.fail("should not wait on a daily quota"))
    with pytest.raises(llm_module.QuotaExceededError):
        post_with_retry("https://x", {}, {}, "test")


def test_post_with_retry_fails_fast_on_client_error(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "post", lambda *a, **k: httpx.Response(400, text="bad request"))
    with pytest.raises(LLMError, match="400"):
        post_with_retry("https://x", {}, {}, "test")
