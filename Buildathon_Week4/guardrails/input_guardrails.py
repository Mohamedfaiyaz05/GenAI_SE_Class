"""ShopEase request authorization and high-confidence prompt-injection screening.

This is a defense-in-depth filter, NOT a guarantee that prompt injection is impossible.
The authenticated customer identity must come from a trusted backend session.
"""
from dataclasses import dataclass, field
import re

ORDER_PATTERN = re.compile(r'\bORD[0-9]{1,12}\b', re.I)
CUSTOMER_PATTERN = re.compile(r'\bC[0-9]{3,12}\b', re.I)

@dataclass
class InputDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    requested_order_id: str = ''
    needs_policy: bool = False
    needs_history: bool = False

# Require an explicit command or security-sensitive request, rather than rejecting
# harmless mentions of words like "system", "SQL", "refund" or "prompt injection".
ATTACK_RULES = {
    'bulk_private_data_request': re.compile(
        r'\b(?:show|list|dump|export|reveal|retrieve|return|provide|print|give\s+me)\b.{0,130}'
        r'\b(?:all|every|complete)\b.{0,55}\b(?:customers?|users?|orders?|emails?|records?|personal\s+details|orders\s+table)\b'
        r'|\b(?:show|reveal|print|provide|return)\b.{0,110}\b(?:all\s+user\s+emails|every\s+customer.s\s+orders|complete\s+orders\s+table)\b', re.I | re.S),
    'authentication_bypass': re.compile(
        r'\b(?:skip|bypass|disable|turn\s+off|ignore|never\s+ask\s+for)\b.{0,100}'
        r'\b(?:authentication|authorization|verification|ownership|access\s+controls?|security\s+checks?|permissions?|eligibility\s+checks?)\b'
        r'|\bjust\s+trust\s+me\b.{0,90}\b(?:show|give|access|orders?)\b', re.I | re.S),
    'fake_system_directive': re.compile(r'\[\s*SYSTEM\s*\].{0,500}\[/\s*SYSTEM\s*\]', re.I | re.S),
    'role_escalation': re.compile(
        r'\b(?:i\s+am|you\s+are\s+now|act\s+as)\b.{0,90}'
        r'\b(?:administrator|admin|unrestricted\s+assistant|security\s+officer)\b.{0,160}'
        r'\b(?:override|provide|show|reveal|disable|access|all\s+user|all\s+customer)\b', re.I | re.S),
    'tool_or_sql_execution': re.compile(
        r'\b(?:call|execute|run|invoke)\b.{0,90}\b(?:database\s+tool|sql|lookup_order|search_customer_memory)\b'
        r'|\bselect\s+\*\s+from\s+(?:customers|orders)\b', re.I | re.S),
    'internal_instruction_extraction': re.compile(
        r'\b(?:show|print|reveal|expose|provide|dump)\b.{0,130}'
        r'\b(?:system\s+prompt|developer\s+instructions|hidden\s+(?:instructions|security\s+rules)|tool\s+schemas|api\s+keys?|secret\s+keys?)\b', re.I | re.S),
    'persistent_privilege_override': re.compile(
        r'\b(?:remember|assume)\b.{0,100}\b(?:administrator|admin|verification|authorization)\b'
        r'.{0,120}\b(?:passed|approved|verified|never|again)\b', re.I | re.S),
    'encoded_command_execution': re.compile(
        r'\b(?:decode|interpret)\b.{0,100}\b(?:base64|encoded)\b.{0,120}\b(?:follow|obey|execute|system\s+command|instruction)\b'
        r'|\b(?:decode|interpret)\s+and\s+(?:follow|obey|execute)\b.{0,100}\b(?:base64|instruction)\b', re.I | re.S),
    'untrusted_note_instruction': re.compile(
        r'\b(?:please\s+)?(?:follow|obey|execute)\b.{0,110}\b(?:tracking\s+note|system\s+override|document\s+instruction)\b', re.I | re.S),
    'unauthorized_refund_execution': re.compile(
        r'\b(?:mark|set)\b.{0,65}\brefund\b.{0,70}\b(?:approved|paid|processed)\b'
        r'|\b(?:approve|pay|process)\b.{0,60}\brefund\b.{0,90}\b(?:skip|without|already\s+authorized)\b', re.I | re.S),
}

POLICY_TERMS = (
    'policy', 'refund', 'return', 'exchange', 'reimburse', 'shipping',
    'delivery', 'warranty', 'cancel', 'replacement', 'eligib', 'damaged',
    'defective', 'broken', 'password', 'login', 'gift card', 'custom branding',
)
HISTORY_PATTERNS = (
    r'\b(?:previous|earlier|past|last)\s+(?:support\s+)?(?:request|ticket|complaint|interaction|conversation)\b',
    r'\b(?:my|our)\s+(?:previous|earlier|last)\s+(?:support\s+)?(?:issue|request|ticket|conversation)\b',
    r'\bwhat\s+happened\b.{0,60}\b(?:last\s+time|previously|earlier)\b',
    r'\b(?:what|how)\s+(?:did|was)\s+(?:we|i|you)\s+(?:discuss|say|do|resolve)\b.{0,60}\b(?:before|last\s+time|previously)\b',
    r'\b(?:contacted|talked\s+to|spoke\s+with)\s+(?:support|customer\s+service|you)\b.{0,80}\b(?:before|earlier|previously|last\s+time)\b',
    r'\b(?:remember|recall)\b.{0,70}\b(?:last\s+time|previous|earlier|our\s+last)\b',
    r'\b(?:remember|recall|find|look\s+up|search)\b.{0,60}\b(?:my|our|previous|last|earlier)\b.{0,45}\b(?:contact|request|issue|conversation|ticket|support)\b',
    r'\b(?:have\s+i|did\s+i)\s+(?:contacted|contact|complain(?:ed)?|asked|reported)\b.{0,60}\b(?:before|previously|earlier)\b',
    r'\b(?:support\s+)?history\b',
)


def check_prompt_injection(complaint: str) -> list[str]:
    """Screen explicit security-bypass requests in original customer text.

    Args:
        complaint (str): Original untrusted customer message.
    Returns:
        list[str]: Stable names of matched high-confidence attack categories.
        An empty list is NOT proof the message is free of prompt injection.
    """
    return [name for name, pattern in ATTACK_RULES.items() if pattern.search(complaint or '')]


def validate_input(*, complaint: str, extracted_order_id: str,
                   category: str, authenticated_customer_id: str | None) -> InputDecision:
    """Check a request before allowing the Knowledge Agent to use tools.

    Args:
        complaint (str): Raw customer message.
        extracted_order_id (str): Order ID proposed by Query Agent.
        category (str): Query Agent category, only a routing hint.
        authenticated_customer_id (str | None): Server-verified identity, NEVER
            an identity inferred from the message or an LLM.
    Returns:
        InputDecision: allow/block, reasons, validated ID and retrieval flags.
        Actual order ownership is checked separately inside SQLite lookup.
    """
    message = (complaint or '').strip()
    order_id = (extracted_order_id or '').strip().upper()
    category = (category or '').casefold()
    reasons = check_prompt_injection(message)
    if not message:
        reasons.append('empty_request')
    if len(message) > 4000:
        reasons.append('request_too_long')
    if order_id and not re.fullmatch(r'ORD[0-9]{1,12}', order_id):
        reasons.append('invalid_extracted_order_id')
    explicit_ids = {s.upper() for s in ORDER_PATTERN.findall(message)}
    if len(explicit_ids) > 1:
        reasons.append('multiple_order_ids_require_clarification')
    elif explicit_ids and order_id not in explicit_ids:
        reasons.append('extracted_order_id_mismatch')
    # Reject a malformed ORD token even if the Query Agent omitted it.
    malformed = re.findall(r'\bORD[0-9][A-Za-z0-9_\x27-]*', message, re.I)
    if any(not re.fullmatch(r'ORD[0-9]{1,12}', token, re.I) for token in malformed):
        reasons.append('malformed_order_id')
    # Reject order-like SQL injection payloads including trailing quote / SQL operators.
    if re.search(r'\bORD\d+\s*[\x27\"]\s*(?:OR|AND)\b', message, re.I):
        reasons.append('order_id_sql_injection_payload')
    # User-provided identity must not override trusted session identity.
    claimed = {s.upper() for s in CUSTOMER_PATTERN.findall(message)}
    if claimed and (not authenticated_customer_id or claimed != {authenticated_customer_id.upper()}):
        # Mentioning a different customer in a complaint can be legitimate.
        # Block only when explicitly claiming that identity or requesting their private records.
        if re.search(r'\b(?:i\s+am|i.m|logged\s+in\s+as|orders?\s+belonging\s+to|show\s+.*\bC\d+)\b', message, re.I):
            reasons.append('untrusted_customer_identity_claim')
    lowered = message.casefold()
    needs_history = any(re.search(pattern, lowered, re.I) for pattern in HISTORY_PATTERNS)
    has_policy_terms = any(term in lowered for term in POLICY_TERMS)
    explicit_policy_question = re.search(
        r'\b(?:policy|eligib(?:le|ility)|terms|guidelines)\b'
        r'|\b(?:can\s+i|am\s+i|do\s+i\s+qualify|how\s+(?:do|can|long)|'
        r'what\s+(?:is|are))\b.{0,80}\b(?:refund|return|exchange|reimburse|'
        r'shipping|delivery|warranty|cancel|replacement)\b'
        r'|\b(?:i\s+want\s+to|need\s+to|trying\s+to)\b.{0,40}\breturn\b',
        lowered,
    )
    history_lookup_only = needs_history and not explicit_policy_question
    vague_request = re.search(
        r'\b(?:has|have)\s+(?:a\s+)?problem\b.{0,50}\b(?:fix|help)\b'
        r'|\bplease\s+fix\s+it\b',
        lowered,
    )
    needs_policy = (
        not history_lookup_only
        and (has_policy_terms
             or category in {'account', 'payment'}
             or (category == 'other' and not needs_history and not vague_request))
    )
    if needs_history and not has_policy_terms:
        needs_policy = False
    if (order_id or needs_history) and not authenticated_customer_id:
        reasons.append('authentication_required')
    return InputDecision(not reasons, list(dict.fromkeys(reasons)), order_id,
                         needs_policy, needs_history)
