"""Step 10: retry only correctable answer-grounding failures.

Security, missing evidence, evaluator outages, and mandatory escalation are
not correctable by rewriting an answer. The allowlist is deliberately narrow.
"""

RETRYABLE_REASONS = frozenset({
    'Semantic grounding evaluator did not confirm support',
})
STATUS_PREFIX = 'Order status contradiction:'


def should_retry_grounding(reasons: list[str]) -> bool:
    """Retry only when EVERY failure is a recognized grounding mismatch."""
    return bool(reasons) and all(
        reason in RETRYABLE_REASONS or reason.startswith(STATUS_PREFIX)
        for reason in reasons
    )
