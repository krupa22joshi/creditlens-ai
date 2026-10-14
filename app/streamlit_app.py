"""CreditLens AI dashboard. Run with: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from creditlens.agents import CreditAppraisalPipeline
from creditlens.data import SAMPLE_APPLICANTS
from creditlens.llm import PROVIDERS, LLMError, get_llm
from creditlens.memo import format_inr
from creditlens.model import RiskModel
from creditlens.retriever import PolicyRetriever
from creditlens.schemas import Applicant

st.set_page_config(page_title="CreditLens AI", page_icon=":bank:", layout="wide")

STATUS_ICON = {"PASS": "✅", "REVIEW": "🟠", "FAIL": "❌", "N/A": "➖"}
DECISION_COLOR = {"APPROVE": "green", "REFER TO CREDIT COMMITTEE": "orange", "DECLINE": "red"}


@st.cache_resource(show_spinner="Loading PD model...")
def load_model() -> RiskModel:
    return RiskModel.load()


@st.cache_resource
def load_retriever() -> PolicyRetriever:
    return PolicyRetriever()


def applicant_form(defaults: dict) -> Applicant | None:
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_id = st.text_input("Applicant ID", defaults["applicant_id"])
            age = st.number_input("Age", 18, 80, int(defaults["age"]))
            annual_income = st.number_input("Annual income (INR)", 50_000, 50_000_000, int(defaults["annual_income"]), step=10_000)
            employment_years = st.number_input("Years in current job", 0.0, 50.0, float(defaults["employment_years"]), step=0.5)
            credit_score = st.slider("Bureau credit score", 300, 900, int(defaults["credit_score"]))
        with c2:
            types = ["personal", "home", "vehicle"]
            loan_type = st.selectbox("Loan type", types, index=types.index(defaults["loan_type"]))
            loan_amount = st.number_input("Loan amount (INR)", 10_000, 100_000_000, int(defaults["loan_amount"]), step=10_000)
            loan_term_months = st.number_input("Tenure (months)", 6, 360, int(defaults["loan_term_months"]))
            interest_rate = st.number_input("Interest rate (%)", 1.0, 40.0, float(defaults["interest_rate"]), step=0.1)
            collateral_value = st.number_input("Collateral value (INR)", 0, 200_000_000, int(defaults["collateral_value"]), step=10_000)
        with c3:
            existing_monthly_debt = st.number_input("Existing monthly debt (INR)", 0, 5_000_000, int(defaults["existing_monthly_debt"]), step=500)
            num_delinquencies_12m = st.number_input("30+ DPD events, last 12 months", 0, 24, int(defaults["num_delinquencies_12m"]))
            max_dpd_24m = st.selectbox("Worst DPD, last 24 months", [0, 30, 60, 90, 120], index=[0, 30, 60, 90, 120].index(defaults["max_dpd_24m"]))
            loan_purpose = st.text_input("Loan purpose", defaults["loan_purpose"])
        submitted = st.form_submit_button("Run credit appraisal", type="primary", use_container_width=True)
    if not submitted:
        return None
    return Applicant(
        applicant_id=applicant_id, age=age, annual_income=annual_income, employment_years=employment_years,
        credit_score=credit_score, loan_type=loan_type, loan_amount=loan_amount, loan_term_months=loan_term_months,
        interest_rate=interest_rate, existing_monthly_debt=existing_monthly_debt,
        num_delinquencies_12m=num_delinquencies_12m, max_dpd_24m=max_dpd_24m,
        collateral_value=collateral_value, loan_purpose=loan_purpose,
    )


def pd_gauge(pd_value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pd_value * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probability of default"},
        gauge={
            "axis": {"range": [0, 50]},
            "bar": {"color": "#1f2937"},
            "steps": [
                {"range": [0, 5], "color": "#bbf7d0"}, {"range": [5, 12], "color": "#fef08a"},
                {"range": [12, 25], "color": "#fed7aa"}, {"range": [25, 50], "color": "#fecaca"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def drivers_chart(drivers) -> go.Figure:
    df = pd.DataFrame([d.model_dump() for d in drivers]).iloc[::-1]
    colors = ["#dc2626" if c > 0 else "#16a34a" for c in df["contribution"]]
    fig = go.Figure(go.Bar(x=df["contribution"], y=df["label"], orientation="h", marker_color=colors))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="SHAP contribution (log-odds)",
                      xaxis_title="← lowers risk   |   raises risk →")
    return fig


def render_assessment(result) -> None:
    color = DECISION_COLOR[result.decision]
    st.markdown(f"### Decision: :{color}[{result.decision}]  ·  Risk grade **{result.risk_grade}**  ·  "
                f"Authority: {result.approval_authority}")
    m = result.derived_metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("EMI", format_inr(m["emi"]))
    k2.metric("Debt-to-income", f"{m['dti']:.1%}")
    k3.metric("Loan-to-value", f"{m['ltv']:.1%}" if m["ltv"] else "n/a")
    k4.metric("Loan-to-income", f"{m['loan_to_income']:.2f}x")

    g1, g2 = st.columns([1, 1.4])
    g1.plotly_chart(pd_gauge(result.probability_of_default), use_container_width=True)
    g2.plotly_chart(drivers_chart(result.risk_drivers), use_container_width=True)

    memo_tab, checks_tab, evidence_tab, audit_tab = st.tabs(["Credit memo", "Policy checks", "Evidence", "Audit trail"])
    with memo_tab:
        badge = "✅ passed validation" if result.validation.passed else "⚠️ failed validation"
        st.caption(f"Source: {result.memo_source} · provider: {result.llm_provider} · attempts: {result.memo_attempts} · {badge}")
        st.markdown(result.credit_memo)
        st.download_button("Download memo (.md)", result.credit_memo, file_name=f"{result.applicant.applicant_id}_credit_memo.md")
    with checks_tab:
        st.dataframe(
            pd.DataFrame([{"": STATUS_ICON[c.status], "Clause": c.rule_id, "Check": c.name, "Status": c.status, "Detail": c.detail}
                          for c in result.policy_checks]),
            hide_index=True, use_container_width=True,
        )
    with evidence_tab:
        for c in result.retrieved_clauses:
            with st.expander(f"{c.clause_id} · {c.title}  ({c.source})"):
                st.write(c.text)
    with audit_tab:
        st.json(result.validation.model_dump())
        st.code("\n".join(result.trace), language="text")


def model_page(model: RiskModel) -> None:
    m = model.metrics
    st.subheader(f"Selected model: {m['model_name']}  ({m['model_version']})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hold-out ROC-AUC", m["holdout_roc_auc"])
    c2.metric("Hold-out PR-AUC", m["holdout_pr_auc"])
    c3.metric("Portfolio default rate", f"{m['default_rate']:.1%}")
    st.markdown("**Stratified cross-validation comparison**")
    st.dataframe(pd.DataFrame(m["model_comparison"]), hide_index=True, use_container_width=True)
    st.markdown("**Leakage audit** - single-feature ROC-AUC; anything above 0.85 is treated as post-outcome information")
    st.dataframe(pd.DataFrame(m["leakage_audit"]), hide_index=True, use_container_width=True)
    st.caption(f"Dropped before training: {', '.join(m['dropped_leakage_columns'])}")


def policy_page(retriever: PolicyRetriever) -> None:
    query = st.text_input("Ask the policy knowledge base", "Can my wife's salary be counted to help me qualify?")
    modes = ["lexical"] if retriever.dense_matrix is None else ["dense", "hybrid", "lexical"]
    mode = st.radio("Retrieval mode", modes, horizontal=True,
                    help="dense = Gemini embeddings · lexical = TF-IDF · hybrid = reciprocal rank fusion of both")
    if retriever.dense_matrix is None:
        st.caption("Set GEMINI_API_KEY to enable semantic (embedding) search.")
    for hit in retriever.search(query, k=4, mode=mode):
        st.markdown(f"**{hit.clause_id} · {hit.title}** — _{hit.source}, score {hit.score}_")
        st.write(hit.text)


def main() -> None:
    st.title("🏦 CreditLens AI")
    st.caption("Explainable credit appraisal: PD model + SHAP · policy engine · RAG · self-validating GenAI credit memo")

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("LLM provider", ["auto"] + sorted(PROVIDERS), help="'auto' picks the first provider with an API key, else mock")
        sample = st.selectbox("Load sample applicant", list(SAMPLE_APPLICANTS))
        page = st.radio("View", ["Credit appraisal", "Model governance", "Policy search"])
        st.info("All memos are AI-generated drafts for credit officer review.", icon="ℹ️")

    model, retriever = load_model(), load_retriever()
    st.sidebar.caption(f"Model: {model.name} ({model.version}) · Retrieval: {retriever.default_mode} · "
                       f"{len(retriever.clauses)} policy clauses")
    if page == "Model governance":
        return model_page(model)
    if page == "Policy search":
        return policy_page(retriever)

    applicant = applicant_form(SAMPLE_APPLICANTS[sample])
    if applicant:
        try:
            llm = get_llm(provider)
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
            return
        with st.spinner("Agents at work: scoring, checking policy, retrieving clauses, drafting and validating memo..."):
            result = CreditAppraisalPipeline(model=model, retriever=retriever, llm=llm).run(applicant)
        render_assessment(result)


main()
