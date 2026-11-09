"""Fast tests for safety screening and public-search eligibility."""

from types import SimpleNamespace

import pytest

import flows.support_flow as support_flow
from flows.support_flow import (
    MedicalSupportFlow,
    is_emergency_question,
    is_general_education_question,
    respond,
)


def test_emergency_question_routes_without_retrieval(monkeypatch):
    def unexpected_retrieval(self, k):
        pytest.fail("Emergency questions must not trigger handbook retrieval")

    monkeypatch.setattr(MedicalSupportFlow, "_retrieve", unexpected_retrieval)

    answer = respond("I cannot breathe")

    assert "emergency service now" in answer


def test_general_education_gate_rejects_personal_advice():
    assert is_general_education_question("What is asthma?")
    assert not is_general_education_question("What should I take for asthma?")
    assert not is_general_education_question("I have asthma, what do I do?")


def test_emergency_screen_flags_high_risk_phrases():
    assert is_emergency_question("Someone is having trouble breathing")
    assert is_emergency_question("There may have been an overdose")


def test_sufficient_handbook_evidence_returns_crew_answer(monkeypatch):
    monkeypatch.setattr(
        MedicalSupportFlow,
        "_retrieve",
        lambda self, k: [{"content": "Handbook evidence supports this answer."}],
    )
    monkeypatch.setattr(
        support_flow,
        "check_evidence_sufficiency",
        lambda question, context: (0.95, "Evidence is sufficient"),
    )
    monkeypatch.setattr(
        support_flow,
        "run_support_agent",
        lambda question, evidence: "A grounded general-information answer.",
    )
    monkeypatch.setattr(
        support_flow,
        "evaluate_answer",
        lambda question, answer, retrieved_chunks: SimpleNamespace(
            verdict="PASS"
        ),
    )

    answer = respond("Tell me about this handbook topic")

    assert answer == "A grounded general-information answer."


def test_refund_policy_question_uses_handbook_not_web_fallback(monkeypatch):
    evidence = [
        {
            "content": (
                "Refund policy. Cancel a prepaid appointment at least 24 "
                "hours before it starts to request a full refund."
            ),
            "source": "medical_support_handbook.md",
            "chunk_id": 10,
        },
        {
            "content": (
                "Late cancellations need staff review. Approved refunds are "
                "initiated within 7 business days."
            ),
            "source": "medical_support_handbook.md",
            "chunk_id": 11,
        },
    ]
    monkeypatch.setattr(
        MedicalSupportFlow,
        "_retrieve",
        lambda self, k: evidence,
    )
    monkeypatch.setattr(
        support_flow,
        "check_evidence_sufficiency",
        lambda question, context: (
            (0.95, "Refund terms are covered.")
            if all("refund" in chunk.lower() for chunk in context)
            else pytest.fail("Refund evidence was not passed to evaluation")
        ),
    )
    monkeypatch.setattr(
        support_flow,
        "run_support_agent",
        lambda question, evidence: (
            "The demo policy allows eligible cancellations 24 hours ahead "
            "to request a refund. Approved refunds are initiated within "
            "7 business days."
        ),
    )
    monkeypatch.setattr(
        support_flow,
        "evaluate_answer",
        lambda question, answer, retrieved_chunks: SimpleNamespace(
            verdict="PASS"
        ),
    )
    monkeypatch.setattr(
        support_flow,
        "run_fallback_agent",
        lambda **kwargs: pytest.fail(
            "A handbook policy question must not use web fallback"
        ),
    )

    answer = respond("What is the refund policy here?")

    assert "24 hours" in answer
    assert "7 business days" in answer
