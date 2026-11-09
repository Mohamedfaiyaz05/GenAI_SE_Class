"""Deterministic gate that sends manual-intervention cases to staff review."""
import re


_HANDOFF_REQUEST = re.compile(
    r"\b(?:speak|talk|connect|transfer|escalate)\b.{0,65}"
    r"\b(?:manager|supervisor|human|person|agent)\b"
    r"|\b(?:manager|supervisor)\b.{0,45}\b(?:please|now|immediately)\b",
    re.I,
)
_CURRENCY_AMOUNT = re.compile(
    r"(?:₹|INR\b|Rs\.?\s*)\s*([\d,]+(?:\.\d{1,2})?)"
    r"|([\d,]+(?:\.\d{1,2})?)\s*(?:rupees\b|INR\b|Rs\.?)",
    re.I,
)
_REFUND_REQUEST = re.compile(
    r"\b(?:request|want|would\s+like|need|apply\s+for|claim|process|approve|"
    r"issue|receive|get)\b.{0,55}\brefund\b"
    r"|\brefund\b.{0,55}\b(?:request|want|need|apply|claim|process|approve|"
    r"issue|me)\b",
    re.I,
)


class UserApprovalAgent:
    """Create review requests; a human, never this class, makes the decision."""

    REFUND_REVIEW_THRESHOLD = 500

    @classmethod
    def review_reason(
        cls,
        *,
        status: str,
        complaint: str,
        category: str,
        urgency: str,
    ) -> str | None:
        message = complaint or ""
        if urgency.casefold() == "high":
            return "High urgency requires human review"
        if category.casefold() == "payment":
            return "Payment issue requires specialist review"
        if _HANDOFF_REQUEST.search(message):
            return "Customer requested human support"
        if status == "Escalated":
            return "Escalated support request requires human review"

        if category.casefold() == "refund" and _REFUND_REQUEST.search(message):
            for match in _CURRENCY_AMOUNT.finditer(message):
                amount_text = next(group for group in match.groups() if group is not None)
                amount = float(amount_text.replace(",", ""))
                if amount >= cls.REFUND_REVIEW_THRESHOLD:
                    return (
                        f"Refund request of at least INR "
                        f"{cls.REFUND_REVIEW_THRESHOLD} requires human approval"
                    )
        return None
