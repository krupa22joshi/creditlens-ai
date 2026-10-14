from creditlens.agents import CreditAppraisalPipeline
from creditlens.llm import LLMClient, MockLLM
from creditlens.validator import validate_memo

# Well formatted, but invents numbers and a clause, and flips the decision.
BAD_MEMO = (
    "# Credit Memo - X\n_AI-generated draft._\n## Summary\nPD is 3.7% and income is INR 99,99,999.\n"
    "## Risk Drivers\n- none\n## Policy Assessment\nAll good under [RBI-99] and [RLP-2].\n"
    "## Recommendation\n**APPROVE**\n"
)


class HallucinatingLLM(LLMClient):
    name = "hallucinating"

    def __init__(self):
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return BAD_MEMO


class SelfCorrectingLLM(HallucinatingLLM):
    """Fails on the first attempt, then follows the validator feedback."""

    def generate(self, system, prompt):
        self.calls += 1
        if self.calls == 1:
            return BAD_MEMO
        assert "<validation_feedback>" in prompt
        return MockLLM().generate(system, prompt)


def _pipeline(risk_model, retriever, llm):
    return CreditAppraisalPipeline(model=risk_model, retriever=retriever, llm=llm, max_attempts=2)


def test_end_to_end_with_offline_llm(risk_model, retriever, samples):
    result = _pipeline(risk_model, retriever, MockLLM()).run(samples["Near-prime - personal loan"])
    assert result.validation.passed, result.validation.issues
    assert result.memo_source == "llm"
    assert result.decision in result.credit_memo
    assert {"RLP-2", "RLP-6", "MRM-2"} <= {c.clause_id for c in result.retrieved_clauses}
    assert any(line.startswith("[Validator]") for line in result.trace)


def test_every_sample_produces_a_valid_memo(risk_model, retriever, samples):
    pipeline = _pipeline(risk_model, retriever, MockLLM())
    for applicant in samples.values():
        result = pipeline.run(applicant)
        assert result.validation.passed, (applicant.applicant_id, result.validation.issues)


def test_hallucinated_memo_is_never_shown(risk_model, retriever, samples):
    llm = HallucinatingLLM()
    result = _pipeline(risk_model, retriever, llm).run(samples["High risk - stretched borrower"])
    assert llm.calls == 2                       # retried with feedback
    assert result.memo_source == "fallback-template"
    assert result.validation.passed
    assert "99,99,999" not in result.credit_memo


def test_validator_feedback_enables_self_correction(risk_model, retriever, samples):
    llm = SelfCorrectingLLM()
    result = _pipeline(risk_model, retriever, llm).run(samples["Near-prime - personal loan"])
    assert llm.calls == 2
    assert result.memo_source == "llm" and result.memo_attempts == 2
    assert result.validation.passed


def test_validator_catches_each_failure_mode(risk_model, retriever, samples):
    pipeline = _pipeline(risk_model, retriever, MockLLM())
    result = pipeline.run(samples["High risk - stretched borrower"])
    facts = {"decision": result.decision, "policy_checks": [c.model_dump() for c in result.policy_checks]}
    report = validate_memo(BAD_MEMO, facts, result.retrieved_clauses)
    assert not report.passed
    assert "RBI-99" in report.invalid_citations
    assert "99,99,999" in report.unsupported_numbers
    assert any("contradicts" in issue or "verbatim" in issue for issue in report.issues)
