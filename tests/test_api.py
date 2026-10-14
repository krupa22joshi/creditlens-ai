from fastapi.testclient import TestClient

from creditlens import api
from creditlens.agents import CreditAppraisalPipeline
from creditlens.data import SAMPLE_APPLICANTS
from creditlens.llm import MockLLM


def test_api_endpoints(risk_model, retriever, monkeypatch):
    pipeline = CreditAppraisalPipeline(model=risk_model, retriever=retriever, llm=MockLLM())
    monkeypatch.setattr(api, "pipeline", lambda: pipeline)
    client = TestClient(api.app)

    assert client.get("/health").json()["status"] == "ok"

    resp = client.post("/assess", json=SAMPLE_APPLICANTS["Prime salaried - home loan"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["validation"]["passed"]
    assert body["risk_grade"] in "ABCD"

    assert client.get("/policies/search", params={"q": "loan to value"}).json()[0]["clause_id"] == "RLP-5"
    assert client.get("/policies/rlp-2").status_code == 200
    assert client.get("/policies/NOPE-1").status_code == 404
    assert client.post("/assess", json={"age": 10}).status_code == 422
