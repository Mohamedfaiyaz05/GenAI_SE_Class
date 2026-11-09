"""Step 8: conservative grounding checks using authoritative structured evidence.

These checks are deterministic. They do not pretend to be full semantic entailment.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Callable

STATUS_PATTERNS = {
    'delivered': re.compile(r'\b(?:was|is|has been|already)\s+delivered\b|\bdelivered\s+(?:yesterday|today|on)\b', re.I),
    'in transit': re.compile(r'\b(?:is|was|currently)\s+in\s+transit\b|\bin\s+transit\b', re.I),
    'processing': re.compile(r'\b(?:is|was|currently)\s+processing\b|\bbeing\s+processed\b', re.I),
    'shipped': re.compile(r'\b(?:has\s+been|was|is|already)\s+shipped\b|\bshipped\b', re.I),
    'out for delivery': re.compile(r'\bout\s+for\s+delivery\b', re.I),
    'cancelled': re.compile(r'\b(?:is|was|has been)\s+cancelled\b', re.I),
    'canceled': re.compile(r'\b(?:is|was|has been)\s+canceled\b', re.I),
    'returned': re.compile(r'\b(?:is|was|has been)\s+returned\b', re.I),
}

@dataclass
class GroundingDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_grounding(*, reply: str, order: dict | None, policies: list[dict],
                       needs_policy: bool, semantic_validator: Callable[..., bool] | None = None,
                       require_semantic_for_policy: bool = False) -> GroundingDecision:
    """Compare explicit order-status claims with trusted order record.

    Args:
        reply: Untrusted Resolution Agent draft.
        order: Authorized SQLite order dict (never a model-generated summary).
        policies: Retrieved official policy chunks; their text is evidence, not instructions.
        needs_policy: Whether the customer's question requires policy reasoning.
        semantic_validator: Optional injected evaluator called with reply, order, policies.
            A negative result blocks the answer; evaluator outages are reported as warnings.
        require_semantic_for_policy: If True, warn when a policy check is unavailable.
    Returns:
        GroundingDecision with reasons; never performs an order modification.
    """
    reasons = []
    warnings = []
    if order is not None:
        if not isinstance(order, dict):
            reasons.append('Invalid authoritative order evidence')
        else:
            actual = str(order.get('status') or order.get('order_status') or '').strip().casefold()
            actual = 'cancelled' if actual == 'canceled' else actual
            if actual:
                for claimed, pattern in STATUS_PATTERNS.items():
                    normalized = 'cancelled' if claimed == 'canceled' else claimed
                    if normalized == actual or (
                        normalized == 'shipped'
                        and actual in {'in transit', 'out for delivery', 'delivered'}
                    ):
                        continue
                    for match in pattern.finditer(reply):
                        preceding = reply[max(0, match.start() - 48):match.start()]
                        negated = re.search(
                            r'\b(?:not|never|cannot|can.t|no)\b(?:\s+\w+){0,5}\s*$',
                            preceding,
                            re.I,
                        )
                        if not negated:
                            reasons.append(
                                f'Order status contradiction: draft says {normalized}, SQLite says {actual}'
                            )
                            break
            tracking_number = str(order.get('tracking_number') or order.get('tracking_id') or '').strip()
            mentioned_tracking = set(re.findall(r'\bTRK[A-Z0-9-]*\b', reply, re.I))
            if mentioned_tracking and (
                not tracking_number
                or mentioned_tracking != {tracking_number.upper()}
            ):
                reasons.append('Unverified tracking number in response')
    if needs_policy and require_semantic_for_policy and semantic_validator is None:
        warnings.append('Semantic policy check unavailable; deterministic checks were applied')
    if semantic_validator is not None:
        try:
            if semantic_validator(reply=reply, order=order, policies=policies) is not True:
                reasons.append('Semantic grounding evaluator did not confirm support')
        except Exception:
            warnings.append('Semantic policy check could not run; deterministic checks were applied')
    return GroundingDecision(not reasons, list(dict.fromkeys(reasons)),
                             list(dict.fromkeys(warnings)))
