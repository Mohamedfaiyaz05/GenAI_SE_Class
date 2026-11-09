"""One final validation stage: evidence plus drafted answer."""
from dataclasses import dataclass, field
import re
from typing import Callable
from guardrails.grounding import validate_grounding

ORDER_PATTERN = re.compile(r'\bORD\d{1,12}\b', re.I)
SECRET_PATTERN = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{12,}|(?:password|api[_ -]?key)\s*[:=]\s*\S+)\b', re.I)
REFUND_PROMISE = re.compile(r'\b(?:refund (?:has been|is|was) (?:approved|issued|processed)|(?:i|we) (?:have )?(?:issued|approved|processed) (?:your |the )?refund|you (?:will|are guaranteed to) (?:receive|get) (?:a |your )?refund)\b', re.I)
GUARANTEE_CLAIM = re.compile(
    r'\b(?:guarantee(?:d)?|promise(?:d)?)\b.{0,60}\b(?:refund|arriv\w*|deliver\w*|tomorrow)\b'
    r'|\b(?:refund|arrival|delivery)\b.{0,50}\bguaranteed\b',
    re.I,
)
ACTION_PROMISE = re.compile(r'\b(?:i|we) (?:have )?(?:cancelled|canceled|changed|updated|refunded) (?:your |the )?(?:order|payment)\b', re.I)

SAFE_RESPONSE = ('Your request needs additional verification or specialist review. '
                 'No changes have been made to your order or payment. '
                 'Please contact support through a verified channel.')
HUMAN_HANDOFF_REQUEST = re.compile(
    r'\b(?:speak|talk|connect|transfer|escalate)\b.{0,65}'
    r'\b(?:manager|supervisor|human|person|agent)\b'
    r'|\b(?:manager|supervisor)\b.{0,45}\b(?:please|now|immediately)\b',
    re.I,
)
VERY_ANGRY_SENTIMENT = re.compile(
    r'\b(?:furious|livid|enraged|outraged|extremely\s+angry|very\s+angry|'
    r'extremely\s+frustrated)\b',
    re.I,
)
CLARIFICATION_REQUEST = re.compile(
    r'\b(?:please\s+)?(?:share|provide|confirm)\b.{0,70}\b(?:order|details|information)\b'
    r'|\b(?:could|can)\s+you\b.{0,70}\b(?:order|details|information)\b'
    r'|\bneed\b.{0,35}\b(?:order\s+id|more\s+details)\b',
    re.I,
)
HUMAN_POLICY_REQUIREMENT = re.compile(
    r'\bhuman\s+confirmation\s+is\s+required\b'
    r'|\bmust\s+be\s+(?:referred|escalated)\s+to\s+human\s+support\b',
    re.I,
)
POLICY_GENERIC_TERMS = {
    'account', 'and', 'are', 'be', 'by', 'for', 'from', 'human', 'in', 'is',
    'it', 'delivery', 'login', 'must', 'not', 'of', 'or', 'password', 'policy',
    'reset', 'required', 'shipping', 'support', 'such', 'the', 'this', 'to',
    'with',
}


def _policy_requires_handoff(complaint: str, policies: list[dict]) -> bool:
    query_terms = set(re.findall(r'[a-z0-9]+', complaint.casefold()))
    for policy in policies:
        content = str(policy.get('content', ''))
        for sentence in re.split(r'[.!?\n]+', content):
            if not HUMAN_POLICY_REQUIREMENT.search(sentence):
                continue
            policy_terms = set(re.findall(r'[a-z0-9]+', sentence.casefold()))
            if (policy_terms - POLICY_GENERIC_TERMS) & query_terms:
                return True
    return False

@dataclass
class GuardrailDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_evidence(*, requested_order_id: str, order_access: str,
                      needs_policy: bool, policies: list[dict],
                      needs_history: bool, memory_checked: bool,
                      authenticated: bool, errors: list[str]) -> GuardrailDecision:
    """Enforce required evidence sources and authentication before answering.

    Args:
        requested_order_id: Query Agent order ID after Python format validation.
        order_access: Status returned by the authorized SQLite lookup.
        needs_policy: Whether the request needs official policy evidence.
        policies: Original policy passages captured from the retrieval tool.
        needs_history: Whether customer history was explicitly requested.
        memory_checked: Whether customer-scoped memory search was attempted.
        authenticated: Whether backend supplied a trusted customer identity.
        errors: Retrieval or Knowledge Agent execution errors.

    Returns:
        GuardrailDecision with allowed flag and deterministic escalation reasons.
    """
    reasons = []
    if requested_order_id and order_access != 'found':
        reasons.append('Authorized order evidence unavailable')
    if needs_policy and not policies:
        reasons.append('Required official policy evidence missing')
    if needs_history and not authenticated:
        reasons.append('Authentication required for customer history')
    elif needs_history and not memory_checked:
        reasons.append('Customer history search not completed')
    if errors:
        reasons.append('Knowledge retrieval or agent execution error')
    return GuardrailDecision(not reasons, reasons)


def validate_response(*, reply: str, requested_order_id: str,
                      order_access: str, policies: list[dict]) -> GuardrailDecision:
    """Reject unsupported or sensitive final responses before showing customers.

    Args:
        reply: Draft response from Resolution Agent (untrusted model output).
        requested_order_id: Python-approved order identifier, if any.
        order_access: Whether the authorized order was actually found.
        policies: Captured official policy passages, not LLM summaries.

    Returns:
        GuardrailDecision. Conservative lexical checks supplement, but do not
        replace, semantic evidence-sufficiency checks planned for Step 8.
    """
    reasons = []
    if not isinstance(reply, str) or not reply.strip():
        reasons.append('Empty response')
        return GuardrailDecision(False, reasons)
    if SECRET_PATTERN.search(reply):
        reasons.append('Potential credential disclosure')
    mentioned = {s.upper() for s in ORDER_PATTERN.findall(reply)}
    if mentioned and (order_access != 'found' or mentioned != {requested_order_id.upper()}):
        reasons.append('Unverified order identifier in response')
    if REFUND_PROMISE.search(reply):
        # Even with a policy passage, this application cannot execute refunds.
        reasons.append('Unverified refund approval or execution claim')
    for match in GUARANTEE_CLAIM.finditer(reply):
        context = reply[max(0, match.start() - 24):match.end()]
        if not re.search(r'\b(?:cannot|can.t|not|never|no)\b', context, re.I):
            reasons.append('Unverified delivery or refund guarantee claim')
            break
    if ACTION_PROMISE.search(reply):
        reasons.append('Unverified order or payment modification claim')
    return GuardrailDecision(not reasons, reasons)


def validate_final(*, reply: str, requested_order_id: str,
                   order_access: str, needs_policy: bool, policies: list[dict],
                   needs_history: bool, memory_checked: bool,
                   authenticated: bool, errors: list[str],
                   category: str, urgency: str, complaint: str,
                   has_order: bool, memories: list[dict],
                   needs_human: bool, confidence: float, human_reason: str = '',
                   order: dict | None = None, semantic_validator: Callable | None = None,
                   require_semantic_for_policy: bool = False,
                   sentiment: str = 'neutral') -> GuardrailDecision:
    """One post-Resolution gate for required evidence AND customer-facing claims.

    Args:
        reply: Resolution Agent draft (never sent before validation).
        requested_order_id: Validated ID from input gate.
        order_access: Authorized SQLite lookup result.
        needs_policy: Whether official policy retrieval is mandatory.
        policies: Raw official policy passages captured by Python.
        needs_history: Whether customer explicitly requested past complaints.
        memory_checked: Whether scoped ChromaDB retrieval actually ran.
        authenticated: Whether backend identity was supplied.
        errors: Tool/agent execution errors from Knowledge Agent.
        category: Query Agent category.
        urgency: Query Agent urgency classification; High urgency requires human follow-up.
        sentiment: Query Agent sentiment; very angry customers require human follow-up.
        complaint: Original untrusted customer message.
        has_order: Whether authorized order record was returned.
        memories: Customer-scoped memory results, possibly empty.
        needs_human: Resolution Agent recommendation; deterministic handoff conditions decide.
        confidence: Resolution Agent confidence in range 0..1.
        human_reason: Optional reason from Resolution Agent, retained for audit compatibility.

    Returns:
        GuardrailDecision; failure means replace draft with SAFE_RESPONSE.
    """
    evidence = validate_evidence(
        requested_order_id=requested_order_id, order_access=order_access,
        needs_policy=needs_policy, policies=policies,
        needs_history=needs_history, memory_checked=memory_checked,
        authenticated=authenticated, errors=errors)
    response = validate_response(
        reply=reply, requested_order_id=requested_order_id,
        order_access=order_access, policies=policies)
    grounding = validate_grounding(
        reply=reply, order=order, policies=policies, needs_policy=needs_policy,
        semantic_validator=semantic_validator,
        require_semantic_for_policy=require_semantic_for_policy)
    reasons = evidence.reasons + response.reasons + grounding.reasons
    warnings = grounding.warnings
    if needs_history and memory_checked and not memories:
        # No past memory is not a retrieval failure, but do not claim no complaints ever.
        if re.search(r'\b(?:never complained|no previous complaints|no past complaints)\b', reply, re.I):
            reasons.append('Unsupported assertion of no previous complaints')
    if category == 'payment':
        reasons.append('Payment requires specialist verification')
    if _policy_requires_handoff(complaint, policies):
        reasons.append('Official policy requires human confirmation')
    if urgency.casefold() == 'high':
        reasons.append('High urgency requires human follow-up')
    if sentiment.casefold() == 'angry' or VERY_ANGRY_SENTIMENT.search(complaint):
        reasons.append('Very angry customer requires human follow-up')
    if category in ('shipping', 'refund', 'product') and not has_order and not policies:
        if not CLARIFICATION_REQUEST.search(reply):
            reasons.append('No authoritative order or policy evidence')
    if any(t in complaint.casefold() for t in
           ('legal action', 'lawyer', 'sue you', 'fraud', 'charged twice', 'charged three times', 'chargeback')):
        reasons.append('Potential legal, fraud or payment dispute')
    if HUMAN_HANDOFF_REQUEST.search(complaint):
        reasons.append('Customer requested a human support handoff')
    if not 0 <= confidence <= 1 or confidence < 0.5:
        reasons.append('Low or invalid response confidence')
    reasons = list(dict.fromkeys(reasons))
    return GuardrailDecision(not reasons, reasons, list(dict.fromkeys(warnings)))
