"""REST API. Run with: uvicorn creditlens.api:app --reload"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .agents import CreditAppraisalPipeline
from .data import SAMPLE_APPLICANTS
from .schemas import Applicant, Assessment, RetrievedClause

app = FastAPI(
    title="CreditLens AI",
    version=__version__,
    description="Explainable PD scoring, policy checks and validated GenAI credit memos.",
)


@lru_cache(maxsize=1)
def pipeline() -> CreditAppraisalPipeline:
    return CreditAppraisalPipeline()


@app.get("/health")
def health() -> dict:
    p = pipeline()
    return {"status": "ok", "model": p.model.name, "model_version": p.model.version, "llm_provider": p.llm.name}


@app.get("/model/metrics")
def model_metrics() -> dict:
    return pipeline().model.metrics


@app.get("/samples")
def samples() -> dict:
    return SAMPLE_APPLICANTS


@app.post("/assess", response_model=Assessment)
def assess(applicant: Applicant) -> Assessment:
    return pipeline().run(applicant)


@app.get("/policies/search", response_model=list[RetrievedClause])
def search_policies(q: str = Query(min_length=3), k: int = Query(3, ge=1, le=10)) -> list[RetrievedClause]:
    return pipeline().retriever.search(q, k=k)


@app.get("/policies/{clause_id}", response_model=RetrievedClause)
def get_policy(clause_id: str) -> RetrievedClause:
    clause = pipeline().retriever.get(clause_id.upper())
    if not clause:
        raise HTTPException(status_code=404, detail=f"Clause {clause_id} not found")
    return clause
