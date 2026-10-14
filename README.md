# CreditLens AI

**Explainable, policy-grounded credit appraisal with self-validating Generative AI credit memos.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

Banks spend hours per application writing credit memos: pulling the risk score, checking the
lending policy clause by clause, explaining the decision and documenting it for audit.
CreditLens AI automates that workflow with a multi-agent pipeline. The key design choice is that
**the GenAI writes and explains, but never decides**. Every generated memo is checked against the
system of record before anyone sees it.

| What a credit team gets | How |
|---|---|
| A probability of default with reason codes | ML model chosen from four candidates with stratified CV; SHAP attributions per applicant |
| An auditable decision | Deterministic policy engine (DTI, bureau score, delinquency, LTV, amount caps, KYC, AML) |
| Answers grounded in policy | Clause-level semantic RAG (Gemini embeddings) over 36 clauses in 6 policy documents, with inline citations like `[RLP-2]` |
| A memo that a regulator would accept | LLM draft → validator → feedback loop → safe fallback; nothing unvalidated is ever shown |
| Freedom from vendor lock-in | Gemini, Azure OpenAI, OpenAI or an offline writer, chosen with one environment variable |

---

## Architecture

```mermaid
flowchart TD
    A[Loan application] --> R1
    subgraph Pipeline [Multi-agent appraisal pipeline]
        R1["1 · Risk Analyst<br/>PD model + SHAP reason codes"] --> R2
        R2["2 · Compliance Officer<br/>policy checks · risk grade · decision"] --> R3
        R3["3 · Policy Researcher<br/>clause-level retrieval (RAG)"] --> R4
        R4["4 · Memo Writer<br/>LLM drafts grounded memo"] --> R5
        R5{"5 · Validator<br/>numbers · citations · decision · labels"}
        R5 -- issues fed back --> R4
    end
    R5 -- passed --> OUT[Credit memo + evidence + audit trail]
    R5 -- still failing after N attempts --> FB[Deterministic template memo] --> OUT
    KB[(Policy knowledge base<br/>lending · model risk · KYC/AML<br/>collateral · fair practices · fraud)] --> R3
    M[(Versioned PD model)] --> R1
```

### The validator: guardrails for GenAI in a regulated process

Before a memo is accepted, `validator.py` checks that:

1. **Every number** (PD, DTI, income, EMI, thresholds) can be traced to the application facts or to the cited policy text. Invented figures are rejected.
2. **Every citation** refers to a clause that was actually retrieved as evidence. Invented references such as `[RBI-99]` are rejected.
3. **Every FAIL / REVIEW check** is discussed and cited.
4. **The recommendation matches** the policy engine's decision verbatim and does not contradict it.
5. **The memo is labelled** as an AI-generated draft for human review and contains all required sections.

If a draft fails, the issues go back to the LLM as feedback (self-correction). If it still fails
after the retry budget, the system falls back to a deterministic template that is itself
validated. Tests cover both paths with deliberately hallucinating LLM stubs.

---

## Results

Results on a 20,000-loan synthetic portfolio (8.1% default rate). Hold-out = 20% stratified split.

| Model | CV ROC-AUC (5-fold) | CV PR-AUC |
|---|---|---|
| **Logistic Regression** (selected) | **0.826 ± 0.012** | **0.405** |
| Random Forest | 0.822 ± 0.008 | 0.389 |
| Gradient Boosting | 0.820 ± 0.010 | 0.377 |
| XGBoost | 0.816 ± 0.011 | 0.385 |

Selected model hold-out ROC-AUC: **0.832**. Logistic regression wins because the remaining risk
signal is close to linear once DTI and LTV are engineered. This result is also the easiest one to
defend to a model risk committee: higher model complexity is not used when it does not pay off.

**Leakage audit.** The raw data includes `post_disbursal_collection_flag`, a field only known
after the loan is paid out. On its own it scores a single-feature ROC-AUC of about 0.91. The
audit flags it automatically and the pipeline drops it before training (policy MRM-4).

### Live-LLM guardrail evaluation

40 unseen applicants, memos written by **Gemini Flash-Lite** (`gemini-flash-lite-latest`) with
dense retrieval. Full output: [`reports/eval_gemini.json`](reports/eval_gemini.json); example
memo: [`reports/sample_memo_gemini.md`](reports/sample_memo_gemini.md).

| Metric | Result |
|---|---|
| First drafts that passed validation | **77.5%** (31 / 40) |
| Fixed by self-correction (validator feedback → redraft) | **20%** (8 / 40) |
| Fell back to the deterministic template | 2.5% (1 / 40) |
| **Final memos passing validation** | **100%** |
| Flagged policy checks discussed with a citation | 100% |
| Mean / p95 latency per application | 9.8 s / 22.3 s (free-tier rate limits included) |

**What the validator caught**: 10 rejected drafts out of 50 generated.

| Failure mode | Count | Example |
|---|---|---|
| A REVIEW/FAIL policy check not cited | 9 | the near-prime score check `[RLP-3]` was skipped |
| Citation to a clause that does not exist | 2 | `RLR-6`, a typo of `RLP-6` |
| A number not present in the facts | 1 | `10,000` |

These are the errors a credit officer could miss when reading quickly. Without the validator,
almost **1 in 4 memos** would have reached a reviewer with a gap or an unsupported figure.

### Retrieval benchmark

44 labelled questions over 36 policy clauses ([`evaluation/retrieval_questions.json`](evaluation/retrieval_questions.json)).
"Paraphrase" questions are phrased the way a credit officer or customer would ask, for example
*"Can my wife's salary be counted to help me qualify?"* → `COL-4`.

| Retrieval | Hit@1 (all) | Hit@1 (paraphrase) | MRR |
|---|---|---|---|
| Lexical (TF-IDF) | 61.4% | 50.0% | 0.665 |
| **Dense (Gemini `gemini-embedding-001`)** | **95.5%** | **92.9%** | **0.970** |
| Hybrid (reciprocal rank fusion) | 72.7% | 60.7% | 0.802 |

Dense retrieval is the default. Equal-weight hybrid fusion scored *below* dense alone, because
the lexical ranking adds noise on paraphrased questions, so it is not used by default. Full
results, including every miss: [`reports/retrieval_eval.json`](reports/retrieval_eval.json).

### Offline run

With no API key, the offline template writer gives a 100% validation pass rate by construction.
That run checks the pipeline and the validator wiring; it does not measure LLM quality.
CI runs it on every push.

---

## Quick start

```bash
git clone https://github.com/krupa22joshi/creditlens-ai.git
cd creditlens-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/train.py                 # trains and versions the PD model (~1 min)
streamlit run app/streamlit_app.py      # dashboard on http://localhost:8501
```

The app runs **with no API key** using the offline memo writer. To use a real LLM, copy
`.env.example` to `.env` and set one provider:

```bash
LLM_PROVIDER=gemini         GEMINI_API_KEY=...
LLM_PROVIDER=azure_openai   AZURE_OPENAI_ENDPOINT=...  AZURE_OPENAI_API_KEY=...  AZURE_OPENAI_DEPLOYMENT=...
LLM_PROVIDER=openai         OPENAI_API_KEY=...
```

### REST API

```bash
uvicorn creditlens.api:app --app-dir src --reload     # docs at http://localhost:8000/docs
```

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/assess` | Full appraisal: PD, grade, drivers, checks, clauses, memo, validation, trace |
| `GET` | `/policies/search?q=...` | Semantic search over policy clauses |
| `GET` | `/policies/{clause_id}` | Fetch a clause, e.g. `RLP-2` |
| `GET` | `/model/metrics` | Model card: comparison, hold-out scores, leakage audit |
| `GET` | `/health` | Model version and active LLM provider |

### Docker

```bash
docker compose up --build     # API on :8000, dashboard on :8501
```

### Tests and evaluation

```bash
pytest                                              # 28 tests: model, leakage, rules, RAG, guardrails, retries, API
python scripts/evaluate.py --n 40 --provider gemini  # live-LLM guardrail metrics (add --save-report for reports/)
python scripts/evaluate_retrieval.py                # lexical vs dense vs hybrid on 44 labelled questions
```

CI (GitHub Actions) runs lint, the test suite, training, the guardrail evaluation and the
lexical retrieval benchmark on every push, offline, with no API keys.

---

## Dashboard

* **Credit appraisal**: enter an application or load a sample. The page shows the decision, PD gauge, SHAP driver chart, the generated memo (downloadable), policy checks, retrieved evidence and the full agent audit trail.
* **Model governance**: model comparison, hold-out metrics and the leakage audit, acting as a lightweight model card.
* **Policy search**: ask questions in plain language and compare dense, hybrid and lexical retrieval side by side.

---

## Project structure

```
creditlens-ai/
├── app/streamlit_app.py        # dashboard
├── knowledge_base/             # 6 policy documents, 36 clauses, one clause per heading
├── evaluation/                 # labelled retrieval questions (keyword vs paraphrase)
├── reports/                    # saved results: live-LLM evaluation, retrieval benchmark, sample memo
├── scripts/
│   ├── train.py                # train + version the PD model
│   ├── evaluate.py             # batch guardrail evaluation (offline or live LLM)
│   └── evaluate_retrieval.py   # retrieval benchmark
├── src/creditlens/
│   ├── agents.py               # multi-agent pipeline and self-correction loop
│   ├── validator.py            # GenAI guardrails
│   ├── memo.py                 # system prompt, prompt builder, template memo
│   ├── llm.py                  # Gemini / Azure OpenAI / OpenAI / offline providers, retry + quota handling
│   ├── embeddings.py           # Gemini embeddings with on-disk cache
│   ├── retriever.py            # clause-level RAG: dense, lexical and hybrid (RRF)
│   ├── policy_rules.py         # deterministic policy engine and risk grading
│   ├── model.py                # leakage audit, model comparison, SHAP explanations
│   ├── features.py             # shared feature engineering
│   ├── data.py                 # synthetic portfolio generator and demo applicants
│   ├── schemas.py              # typed contracts (pydantic)
│   └── api.py                  # FastAPI service
└── tests/
```

---

## Design decisions

* **Decision and narration are separated.** Approve/refer/decline comes from transparent rules plus model grading, and the LLM only explains it. This keeps GenAI out of the approval path, which most model-risk and responsible-AI frameworks require.
* **Retrieval works at clause level, not page level.** Each chunk is one citable policy clause, so citations can be checked by a machine and audits are simple.
* **Clauses are retrieved two ways.** Clauses linked to triggered checks are always fetched by ID (reliable recall). Semantic search adds the single best clause for each top risk driver, so loosely related text stays out of the prompt.
* **Retrieval mode was chosen by measurement, not assumption.** Hybrid search is often recommended by default, but on this corpus it scored below pure embeddings, so dense is the default and lexical remains the offline fallback.
* **Failures are separated from quality.** A quota or network error stops the evaluation rather than being counted as a bad memo, and daily-quota errors fail fast instead of waiting.
* **The LLM layer has no vendor SDKs.** Providers are called over HTTPS, so moving a client from Gemini to Azure OpenAI is a configuration change, not a code change.
* **Synthetic data with real structure.** The generator encodes known credit-risk relationships, including nonlinear DTI payment shock and score × DTI interaction, plus a planted leakage column. The whole project can then be published and reproduced without customer data.
* **Explainability was chosen over raw complexity.** Features that duplicate DTI (EMI, loan amount, existing debt) are excluded so SHAP reason codes keep intuitive signs for credit officers.

## Roadmap

- [x] Semantic retrieval with embeddings, benchmarked against lexical and hybrid
- [x] Live-LLM evaluation with rejection-reason breakdown
- [ ] Move the in-memory vector index to a vector store (e.g. ChromaDB) for larger corpora
- [ ] Analyst Q&A agent with tool calling ("what loan amount would be approvable?")
- [ ] Fairness monitoring across segments (MRM-5) and drift alerts (MRM-6)
- [ ] Ingest real regulatory circulars (PDF) with clause-aware chunking
- [ ] LLM-as-judge scoring of memo readability alongside the rule-based validator

---

*The policy documents in `knowledge_base/` are illustrative policies for a fictional bank, written
for this project. They are not official regulatory text. All applicant data is synthetic.*

**Author:** Krupa Joshi · [LinkedIn](https://linkedin.com/in/krupa-joshi-99b832287) · [GitHub](https://github.com/krupa22joshi)
