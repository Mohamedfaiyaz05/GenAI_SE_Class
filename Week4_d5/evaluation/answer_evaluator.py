
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

SUFFICIENCY_THRESHOLD = 0.8
FAITHFULNESS_THRESHOLD = 0.9
ANSWER_RELEVANCY_THRESHOLD = 0.8
CONTEXT_RELEVANCY_THRESHOLD = 0.7


class AnswerEvaluation(BaseModel):
    evidence_sufficient: bool
    answer_supported: bool

    verdict: Literal[
        "PASS",
        "RETRY",
        "INSUFFICIENT_EVIDENCE"
    ]

    reason: str

    sufficiency_score: float = Field(ge=0, le=1)
    faithfulness_score: float = Field(ge=0, le=1)
    answer_relevancy_score: float = Field(ge=0, le=1)
    contextual_relevancy_score: float = Field(ge=0, le=1)


def check_evidence_sufficiency(
    question: str,
    context: list[str],
) -> tuple[float, str]:
    """Check whether the retrieved context fully answers the question."""

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    if not context:
        return 0.0, "No evidence was retrieved."

    metric = GEval(
        name="Evidence Sufficiency",
        evaluation_steps=[
            "Identify every distinct information requirement "
            "in the customer's question.",
            "Examine only the supplied retrieval context.",
            "Determine whether the context explicitly supports "
            "an answer to every requirement.",
            "Penalize missing, contradictory, ambiguous or "
            "incomplete information.",
            "Do not use external knowledge or assume missing facts.",
            "Give a high score only when all requested information "
            "can be answered from the context."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=SUFFICIENCY_THRESHOLD,
        model=MODEL_NAME,
        async_mode=False,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output="Evidence sufficiency assessment only.",
        retrieval_context=context,
    )

    metric.measure(test_case)

    if metric.score is None:
        raise RuntimeError("Evidence sufficiency returned no score")

    return float(metric.score), metric.reason or ""


def evaluate_answer(
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
) -> AnswerEvaluation:
    """Evaluate a Support Agent answer against FAISS evidence."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be non-empty")

    if not isinstance(retrieved_chunks, list):
        raise ValueError("retrieved_chunks must be a list")

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    context = []

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a dictionary")

        content = chunk.get("content", "")

        if not isinstance(content, str):
            raise ValueError("Chunk content must be a string")

        if content.strip():
            context.append(content.strip())

    if not context:
        return AnswerEvaluation(
            evidence_sufficient=False,
            answer_supported=False,
            verdict="INSUFFICIENT_EVIDENCE",
            reason="No usable handbook evidence was retrieved.",
            sufficiency_score=0.0,
            faithfulness_score=0.0,
            answer_relevancy_score=0.0,
            contextual_relevancy_score=0.0,
        )

    test_case = LLMTestCase(
        input=question.strip(),
        actual_output=answer.strip(),
        retrieval_context=context,
    )

    # 1. Check evidence sufficiency independently.
    sufficiency_score, sufficiency_reason = (
        check_evidence_sufficiency(question.strip(), context)
    )

    # 2. Check whether the answer is grounded in evidence.
    faithfulness = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=MODEL_NAME,
        include_reason=True,
        async_mode=False,
    )

    # 3. Check whether the answer addresses the question.
    answer_relevancy = AnswerRelevancyMetric(
        threshold=ANSWER_RELEVANCY_THRESHOLD,
        model=MODEL_NAME,
        include_reason=True,
        async_mode=False,
    )

    # 4. Check whether the retrieved chunks are relevant.
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=CONTEXT_RELEVANCY_THRESHOLD,
        model=MODEL_NAME,
        include_reason=True,
        async_mode=False,
    )

    faithfulness.measure(test_case)
    answer_relevancy.measure(test_case)
    contextual_relevancy.measure(test_case)

    if any(
        metric.score is None
        for metric in (
            faithfulness,
            answer_relevancy,
            contextual_relevancy,
        )
    ):
        raise RuntimeError("One or more DeepEval metrics returned no score")

    faithfulness_score = float(faithfulness.score)
    answer_score = float(answer_relevancy.score)
    context_score = float(contextual_relevancy.score)

    evidence_sufficient = (
        sufficiency_score >= SUFFICIENCY_THRESHOLD
    )

    answer_supported = (
        faithfulness_score >= FAITHFULNESS_THRESHOLD
        and answer_score >= ANSWER_RELEVANCY_THRESHOLD
    )

    # Routing decision — enforced by Python.
    if not evidence_sufficient:
        verdict = "INSUFFICIENT_EVIDENCE"

    elif not answer_supported:
        verdict = "RETRY"

    else:
        verdict = "PASS"

    reasons = [
        f"Evidence sufficiency: {sufficiency_reason}",
        f"Faithfulness: {faithfulness.reason or ''}",
        f"Answer relevancy: {answer_relevancy.reason or ''}",
        f"Context relevancy: {contextual_relevancy.reason or ''}",
    ]

    return AnswerEvaluation(
        evidence_sufficient=evidence_sufficient,
        answer_supported=answer_supported,
        verdict=verdict,
        reason=" | ".join(reasons),
        sufficiency_score=sufficiency_score,
        faithfulness_score=faithfulness_score,
        answer_relevancy_score=answer_score,
        contextual_relevancy_score=context_score,
    )
