"""ShopEase Support AI API; demo authentication is never for production use."""
import csv
import io
import json
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from database.seed import CUSTOMERS, ORDERS, seed_demo_data
from database.human_review_repository import (
    list_review_audit,
    list_reviews,
    record_review_decision,
    register_pending_review,
)
from agents.user_approval_agent import UserApprovalAgent
from guardrails.final_guardrails import HUMAN_HANDOFF_REQUEST, SAFE_RESPONSE, validate_final
from guardrails.input_guardrails import validate_input
from services.faithfulness import (
    DeepEvalFaithfulnessValidator,
    faithfulness_timeline_event,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DB_PATH = Path(os.getenv('SHOPEASE_DB_PATH', str(ROOT / 'shopease.db')))
DEMO_MODE = os.getenv('DEMO_MODE', 'true').lower() == 'true'
DEMO_KEY = os.getenv('DEMO_API_KEY', 'local-demo-key-change-me')
HUMAN_REVIEW_API_KEY = os.getenv('HUMAN_REVIEW_API_KEY', '').strip()
SUPPORT_PIPELINE_MODE = os.getenv(
    'SUPPORT_PIPELINE_MODE',
    'graphflow' if os.getenv('OPENAI_API_KEY') else 'demo',
).casefold()
if SUPPORT_PIPELINE_MODE not in {'demo', 'graphflow'}:
    raise ValueError('SUPPORT_PIPELINE_MODE must be "demo" or "graphflow"')
DEEPEVAL_REQUESTED = os.getenv('ENABLE_DEEPEVAL', 'false').strip().casefold() == 'true'
DEEPEVAL_ENABLED = DEEPEVAL_REQUESTED and bool(os.getenv('OPENAI_API_KEY', '').strip())
DEEPEVAL_THRESHOLD = float(os.getenv('DEEPEVAL_THRESHOLD', '0.7'))
DEEPEVAL_MODEL = os.getenv('DEEPEVAL_MODEL', 'gpt-4o-mini').strip() or 'gpt-4o-mini'
if not 0 < DEEPEVAL_THRESHOLD <= 1:
    raise ValueError('DEEPEVAL_THRESHOLD must be greater than 0 and at most 1')

app = FastAPI(title='ShopEase Support AI', version='0.1.0')
_GRAPHFLOW_RUNTIME = None
_GRAPHFLOW_MODEL = None

@contextmanager
def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def initialize():
    with db() as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
            status TEXT NOT NULL, tracking_id TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            customer_id TEXT NOT NULL, customer_name TEXT NOT NULL,
            order_id TEXT, complaint TEXT NOT NULL, category TEXT NOT NULL,
            urgency TEXT NOT NULL, sentiment TEXT NOT NULL,
            response TEXT NOT NULL, status TEXT NOT NULL,
            reason TEXT, policy_sources TEXT NOT NULL,
            answer_attempts INTEGER NOT NULL, timeline TEXT NOT NULL
        );
        ''')
        if DEMO_MODE:
            conn.executemany(
                'INSERT OR IGNORE INTO customers (customer_id, name, email) VALUES (?,?,?)',
                CUSTOMERS,
            )
            conn.executemany(
                '''INSERT OR IGNORE INTO orders (order_id, customer_id, status, tracking_id)
                   VALUES (?,?,?,?)''',
                [(order_id, customer_id, status, tracking_number)
                 for (order_id, customer_id, status, _item_name, _estimated_delivery,
                      _delivered_at, tracking_number, _tracking_summary) in ORDERS],
            )

initialize()

class ComplaintRequest(BaseModel):
    complaint: str = Field(min_length=3, max_length=5000)
    order_id: str | None = Field(default=None, max_length=60)


class HumanReviewDecision(BaseModel):
    decision: Literal['Approved', 'Rejected']
    reviewer: str = Field(min_length=2, max_length=100)
    note: str = Field(default='', max_length=1000)


class TicketOut(BaseModel):
    ticket_id: str
    created_at: str
    customer_id: str
    customer_name: str
    order_id: str | None
    complaint: str
    category: str
    urgency: str
    sentiment: str
    response: str
    status: str
    reason: str | None
    policy_sources: list[str]
    answer_attempts: int
    timeline: list[dict[str, Any]]
    human_review_status: str | None = None
    human_reviewed_at: str | None = None

def identity(x_demo_api_key: str | None = Header(default=None), x_customer_id: str | None = Header(default=None)) -> dict:
    if not DEMO_MODE:
        # Replace with validated JWT/session identity before production deployment.
        raise HTTPException(503, 'Production authentication adapter is not configured')
    if x_demo_api_key != DEMO_KEY:
        raise HTTPException(401, 'Invalid demo API key')
    if not x_customer_id:
        raise HTTPException(401, 'A demo customer account is required')
    with db() as conn:
        row = conn.execute('SELECT * FROM customers WHERE customer_id=?', (x_customer_id,)).fetchone()
    if not row:
        raise HTTPException(401, 'Unknown demo customer')
    return dict(row)


def human_review_staff(x_human_review_key: str | None = Header(default=None)) -> None:
    if not HUMAN_REVIEW_API_KEY:
        raise HTTPException(503, 'Set HUMAN_REVIEW_API_KEY to enable staff review.')
    if not x_human_review_key or not secrets.compare_digest(
        x_human_review_key, HUMAN_REVIEW_API_KEY
    ):
        raise HTTPException(401, 'Invalid staff review credentials')


def _review_db_path() -> Path:
    return _graphflow_db_path() if SUPPORT_PIPELINE_MODE == 'graphflow' else DB_PATH


def _admin_tickets() -> list[dict]:
    path = _review_db_path()
    if SUPPORT_PIPELINE_MODE == 'graphflow':
        seed_demo_data(path)
    conn = sqlite3.connect(str(path), timeout=15)
    try:
        conn.row_factory = sqlite3.Row
        if SUPPORT_PIPELINE_MODE == 'graphflow':
            rows = conn.execute(
                """SELECT t.ticket_id, t.timestamp AS created_at, t.customer_id,
                          COALESCE(t.customer_name, c.display_name, '') AS customer_name,
                          COALESCE(c.email, '') AS email, t.order_id, t.complaint,
                          t.category, t.urgency, t.sentiment, t.response, t.status,
                          t.escalation_reason AS reason
                   FROM tickets t
                   LEFT JOIN customers c ON c.customer_id = t.customer_id
                   ORDER BY t.timestamp DESC"""
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT t.ticket_id, t.created_at, t.customer_id, t.customer_name,
                          COALESCE(c.email, '') AS email, t.order_id, t.complaint,
                          t.category, t.urgency, t.sentiment, t.response, t.status,
                          t.reason
                   FROM tickets t
                   LEFT JOIN customers c ON c.customer_id = t.customer_id
                   ORDER BY t.created_at DESC"""
            ).fetchall()
    finally:
        conn.close()
    tickets = [dict(row) for row in rows]
    for ticket in tickets:
        if ticket['status'] == 'Escalated':
            register_pending_review(
                path, ticket['ticket_id'],
                ticket.get('reason') or 'Escalated support request requires human review',
            )
    reviews = {review['ticket_id']: review for review in list_reviews(path)}
    for ticket in tickets:
        review = reviews.get(ticket['ticket_id'])
        ticket['human_review_status'] = review['status'] if review else None
        ticket['human_review_reason'] = review['trigger_reason'] if review else None
        ticket['reviewer'] = review['reviewer'] if review else None
        ticket['reviewer_note'] = review['reviewer_note'] if review else None
    return tickets


def decode(row):
    d = dict(row)
    d['policy_sources'] = json.loads(d['policy_sources'])
    d['timeline'] = json.loads(d['timeline'])
    return d


def _graphflow_db_path() -> Path:
    path = Path(os.getenv(
        'SHOPEASE_AGENT_DB_PATH',
        str(ROOT / 'data' / 'graphflow_support.db'),
    ))
    return path if path.is_absolute() else ROOT / path


def _graphflow_pipeline():
    """Create request-scoped agents over shared, initialized retrieval resources."""
    global _GRAPHFLOW_RUNTIME, _GRAPHFLOW_MODEL
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(
            503,
            'GraphFlow mode requires OPENAI_API_KEY in the process environment or .env file.',
        )
    if _GRAPHFLOW_RUNTIME is None:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from memory.chroma_memory import CustomerMemory
        from rag.hybrid_retriever import HybridRetriever

        agent_db_path = _graphflow_db_path()
        seed_demo_data(agent_db_path)
        retriever = HybridRetriever()
        retriever.build()
        memory = CustomerMemory(directory=str(ROOT / 'data' / 'chroma'))
        model_name = os.getenv('OPENAI_MODEL_NAME') or 'gpt-4o-mini'
        model = OpenAIChatCompletionClient(model=model_name)
        _GRAPHFLOW_MODEL = model
        _GRAPHFLOW_RUNTIME = (model, retriever, memory, agent_db_path)

    from services.graphflow_pipeline import ShopEaseGraphFlowPipeline
    from services.knowledge_service import KnowledgeService

    model, retriever, memory, agent_db_path = _GRAPHFLOW_RUNTIME
    return ShopEaseGraphFlowPipeline(
        model,
        KnowledgeService(retriever, memory, agent_db_path),
        db_path=agent_db_path,
        enable_deepeval=DEEPEVAL_ENABLED,
        deepeval_threshold=DEEPEVAL_THRESHOLD,
        deepeval_model=DEEPEVAL_MODEL,
    )


def _agent_ticket_out(record: dict, timeline: list[dict] | None = None) -> dict:
    return {
        'ticket_id': record['ticket_id'],
        'created_at': record['timestamp'],
        'customer_id': record['customer_id'],
        'customer_name': record.get('customer_name') or '',
        'order_id': record.get('order_id'),
        'complaint': record['complaint'],
        'category': record['category'],
        'urgency': record['urgency'],
        'sentiment': record['sentiment'],
        'response': record['response'],
        'status': record['status'],
        'reason': record.get('escalation_reason'),
        'policy_sources': record['kb_sources'],
        'answer_attempts': record['answer_attempts'],
        'timeline': timeline or [],
    }


def _include_customer_review_status(tickets: list[dict], path: Path) -> list[dict]:
    reviews = {
        review['ticket_id']: review
        for review in list_reviews(path)
    }
    return [
        {
            **ticket,
            'human_review_status': (
                reviews[ticket['ticket_id']]['status']
                if ticket['ticket_id'] in reviews else None
            ),
            'human_reviewed_at': (
                reviews[ticket['ticket_id']]['reviewed_at']
                if ticket['ticket_id'] in reviews else None
            ),
        }
        for ticket in tickets
    ]


def event(name, detail, **extra):
    return {'agent': name, 'detail': detail, 'timestamp': datetime.now(timezone.utc).isoformat(), **extra}


def _classify_request(complaint: str) -> str:
    text = complaint.casefold()
    if any(term in text for term in ('previously', 'previous complaint', 'last time',
                                     'complained before', 'support history', 'past complaint')):
        return 'account'
    if any(term in text for term in ('refund', 'return', 'exchange', 'reimburse')):
        return 'refund'
    if any(term in text for term in (
            'payment', 'billing', 'charged', 'chargeback', 'transaction', 'credit card')):
        return 'payment'
    if any(term in text for term in ('password', 'login', 'account', 'sign in')):
        return 'account'
    if any(term in text for term in (
            'damaged', 'defective', 'broken', 'unusable', 'unsafe',
            'warranty', 'product', 'replacement')):
        return 'product'
    if (re.search(r'\bord\d+\b', text)
            or any(term in text for term in (
                'order', 'tracking', 'delivery', 'delayed', 'late', 'package', 'shipped'
            ))):
        return 'shipping'
    return 'other'


def _is_high_urgency(complaint: str) -> bool:
    return bool(re.search(
        r'\b(?:immediately|urgent|extremely\s+(?:frustrated|angry|upset)|'
        r'speak\s+to\s+(?:a\s+)?(?:manager|human|person|agent)|'
        r'talk\s+to\s+(?:a\s+)?(?:manager|human|person|agent)|'
        r'legal\s+action|lawyer|fraud|chargeback)\b',
        complaint,
        flags=re.I,
    ))


def _needs_human_handoff(complaint: str) -> bool:
    return bool(HUMAN_HANDOFF_REQUEST.search(complaint) or re.search(
        r'\b(?:legal\s+action|lawyer|fraud|chargeback|charged\s+(?:twice|three\s+times))\b',
        complaint,
        flags=re.I,
    ))


def _search_demo_policies(question: str, category: str, limit: int = 2) -> list[dict]:
    """Return lexical matches from local policy sections for the offline demo.

    This deliberately does not claim to be FAISS/semantic RAG. The production
    GraphFlow pipeline owns hybrid retrieval and uses the same official files.
    """
    policy_dir = ROOT / 'knowledge_base'
    stop_words = {
        'about', 'after', 'also', 'can', 'could', 'does', 'for', 'from', 'get',
        'have', 'how', 'into', 'late', 'may', 'my', 'our', 'please', 'tell',
        'that', 'the', 'this', 'what', 'when', 'where', 'with', 'would', 'you',
    }
    terms = {
        token for token in re.findall(r'[a-z0-9]+', question.casefold())
        if len(token) > 2 and token not in stop_words
    }
    preferred_source = {
        'refund': 'refund_policy.md',
        'shipping': 'shipping_policy.md',
        'product': 'product_faq.md',
        'account': 'account_and_login.md',
    }.get(category)
    matches = []
    for path in sorted(policy_dir.glob('*.md')):
        if preferred_source and path.name != preferred_source:
            continue
        text = path.read_text(encoding='utf-8-sig')
        sections = re.split(r'(?m)(?=^#{2,3}\s+)', text)
        for section in sections:
            searchable = section.strip()
            if not searchable or searchable.startswith('**Policy ID:**'):
                continue
            heading_match = re.match(r'^#{2,3}\s+(.+)', searchable)
            heading = heading_match.group(1).strip() if heading_match else path.stem
            body_terms = re.findall(r'[a-z0-9]+', searchable.casefold())
            overlap = set(body_terms) & terms
            score = len(overlap)
            if category == 'refund' and re.search(r'\breturn\b', question, re.I):
                if re.search(r'return eligibility and time window|how to request a return', heading, re.I):
                    score += 5
            if score >= 3:
                matches.append({
                    'source': path.name,
                    'section': heading,
                    'content': searchable[:1600],
                    'score': score,
                })
    matches.sort(key=lambda hit: (-hit['score'], hit['source'], hit['section']))
    return [{key: value for key, value in hit.items() if key != 'score'}
            for hit in matches[:min(limit, 2)]]


def _policy_excerpt(hit: dict) -> str:
    content = hit.get('content', '')
    content = re.sub(r'(?m)^#{1,3}\s+.*$', '', content)
    content = re.sub(r'(?m)^\s*\*\*Policy ID:\*\*.*$', '', content)
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    content = re.sub(r'(?m)^\s*(?:[-*]\s+|>\s*)', '', content)
    content = re.sub(r'[*`_]', '', content)
    excerpt = ' '.join(content.split())
    if len(excerpt) > 500:
        excerpt = excerpt[:497].rsplit(' ', 1)[0] + '...'
    return excerpt


def _search_demo_history(customer_id: str, question: str) -> list[dict]:
    """Search only the authenticated customer's saved demo tickets."""
    stop_words = {
        'about', 'can', 'contacted', 'happened', 'have', 'i', 'last', 'me',
        'my', 'previously', 'remember', 'support', 'the', 'time', 'what',
    }
    terms = {
        token for token in re.findall(r'[a-z0-9]+', question.casefold())
        if len(token) > 3 and token not in stop_words
    }
    if 'delivery' in terms or 'shipping' in terms:
        terms.update({'delivery', 'shipping'})
    if 'delayed' in terms or 'late' in terms:
        terms.update({'delayed', 'delay', 'late'})
    with db() as conn:
        rows = conn.execute(
            '''SELECT ticket_id, complaint, response, created_at, status, category
               FROM tickets WHERE customer_id=? ORDER BY created_at DESC LIMIT 50''',
            (customer_id,),
        ).fetchall()
    ranked = []
    for row in rows:
        record = dict(row)
        searchable = f"{record['complaint']} {record['response']}".casefold()
        score = sum(searchable.count(term) for term in terms)
        if score >= 2:
            ranked.append((score, record))
    ranked.sort(key=lambda item: (-item[0], item[1]['created_at']))
    return [record for _, record in ranked[:3]]


def demo_process(complaint: str, order_id: str | None, user: dict) -> dict:
    """Deterministic local demo; live agents/RAG/memory run through GraphFlow."""
    category = _classify_request(complaint)
    if (category == 'shipping' and not order_id and not re.search(r'\bORD\d+\b', complaint, re.I)
            and re.search(r'\b(?:package|order)\s+(?:has|have)\s+(?:a\s+)?problem\b', complaint, re.I)):
        category = 'other'
    urgent = _is_high_urgency(complaint)
    urgency = 'High' if urgent else 'Medium'
    if re.search(r'\b(?:furious|livid|enraged|outraged|extremely\s+angry|very\s+angry|'
                 r'extremely\s+frustrated|charged.{0,30}(?:twice|three\s+times))\b',
                 complaint, re.I):
        sentiment = 'angry'
    elif re.search(r'\b(?:frustrated|annoyed|upset|unhappy)\b', complaint, re.I):
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    extracted = re.search(r'\bORD[- ]?\d+\b', complaint, flags=re.I)
    text_order = extracted.group().upper().replace(' ', '').replace('-', '') if extracted else ''
    requested = order_id.strip().upper() if order_id else text_order
    query = event('Query Agent Analysis',
                  f'Category: {category}; Urgency: {urgency}; Sentiment: {sentiment}',
                  category=category, urgency=urgency, sentiment=sentiment,
                  requested_order_id=requested)
    timeline = [query]

    input_check = validate_input(
        complaint=complaint,
        extracted_order_id=requested or '',
        category=category,
        authenticated_customer_id=user.get('customer_id'),
    )
    timeline.append(event('Input Guardrail', 'Request checked before private data lookup',
                          allowed=input_check.allowed, reasons=input_check.reasons))
    policy_hits: list[dict] = []
    memories: list[dict] = []
    memory_checked = False
    order = None
    order_access = 'not_requested'
    errors: list[str] = []

    if input_check.allowed and _needs_human_handoff(complaint):
        response = ('I understand this is urgent and have marked your request for human review. '
                    'A manager or support specialist will need to follow up; no order or payment '
                    'changes have been made.')
        status, reason, attempts = 'Escalated', 'High urgency or explicit human escalation request', 0
        timeline.append(event('Escalation', reason, requested=True))
    elif not input_check.allowed:
        response = SAFE_RESPONSE
        status = 'Escalated'
        reason = '; '.join(input_check.reasons)
        attempts = 0
        timeline.append(event('Escalation', 'Input guardrail blocked the request before private lookup',
                              reasons=input_check.reasons))
    else:
        if requested:
            with db() as conn:
                order_row = conn.execute(
                    '''SELECT order_id, status, tracking_id FROM orders
                       WHERE order_id=? AND customer_id=?''',
                    (requested, user['customer_id']),
                ).fetchone()
            order = dict(order_row) if order_row else None
            order_access = 'found' if order else 'not_found_or_not_owned'
            timeline.append(event('Knowledge Base Search',
                                  'Authorized SQLite order lookup completed',
                                  order_access=order_access,
                                  order_id=requested if order else None))

        if input_check.needs_policy:
            policy_hits = _search_demo_policies(complaint, category)
            timeline.append(event('Policy Retrieval',
                                  'Local demo keyword search; not FAISS/semantic RAG',
                                  sources=[hit['source'] for hit in policy_hits],
                                  sections=[hit['section'] for hit in policy_hits]))

        if input_check.needs_history:
            memory_checked = True
            memories = _search_demo_history(user['customer_id'], complaint)
            timeline.append(event('Customer History Search',
                                  'Customer-scoped saved demo ticket search; not ChromaDB',
                                  result_count=len(memories)))

        if requested and not order:
            response = ('I could not verify that order for your signed-in account. '
                        'Please check the order ID or contact support through a verified channel.')
            status, reason = 'Escalated', 'Order could not be verified for this customer'
        elif input_check.needs_policy and not policy_hits:
            response = 'I could not find enough official policy information to answer confidently, so I have escalated this for review.'
            status, reason = 'Escalated', 'Required policy evidence was unavailable'
        elif input_check.needs_history:
            if memories:
                latest = memories[0]
                previous_response = re.sub(r'\bORD\d{1,12}\b', '[order]', latest['response'], flags=re.I)
                category_labels = {
                    'shipping': 'order or delivery',
                    'refund': 'refund or return',
                    'account': 'account',
                    'product': 'product',
                    'payment': 'payment',
                }
                previous_category = category_labels.get(latest['category'], 'support')
                response = (f'I found a previous support request about {previous_category} '
                            f'from {latest["created_at"][:10]}. '
                            f'Its saved response was: {previous_response}')
            else:
                response = ('I could not find a saved support interaction for this account, '
                            'so I cannot verify what happened last time.')
            status, reason = 'Resolved', None
        elif requested and order:
            current_status = order['status']
            tracking = order['tracking_id']
            tracking_text = f'Tracking number: {tracking}.' if tracking else 'No tracking number is available in the order record yet.'
            if current_status.casefold() == 'processing':
                response = (f'Order {requested} is currently Processing, so the record does not show '
                            f'that it has shipped. {tracking_text}')
            else:
                response = f'Order {requested} is currently {current_status}. {tracking_text}'
            if category == 'refund' or 'refund' in complaint.casefold():
                response += (' The order record does not support a delivery guarantee, and I cannot '
                             'approve a refund. A specialist must review refund eligibility.')
                status, reason = 'Escalated', 'Refund or delivery guarantee requires human review'
            else:
                status, reason = 'Resolved', None
        elif category == 'refund' and policy_hits:
            if re.search(r'\b(?:late|delayed|arriv\w+\s+late)\b', complaint, re.I):
                response = (
                    'A late delivery does not automatically qualify for a refund. The refund '
                    'policy says each request must be checked against the order and eligibility '
                    'rules; ShopEase cannot promise approval here. I can arrange human review '
                    'if you share the relevant order ID.'
                )
                status, reason = 'Resolved', None
            else:
                excerpts = [_policy_excerpt(hit) for hit in policy_hits]
                response = (
                    'The refund policy says: ' + ' '.join(excerpts)
                )
                status, reason = 'Resolved', None
        elif input_check.needs_policy and policy_hits:
            hit = policy_hits[0]
            policy_labels = {
                'shipping': 'shipping and delivery',
                'product': 'product support',
                'account': 'account',
            }
            policy_label = policy_labels.get(category, 'store')
            excerpt = _policy_excerpt(hit)
            response = (
                f'I found the {policy_label} guidance in the "{hit["section"]}" section. '
                f'{excerpt}'
            )
            status, reason = 'Resolved', None
        else:
            response = ('Could you share your order ID and briefly describe what went wrong? '
                        'I need those details before I can check or request a fix.')
            status, reason = 'Resolved', None
        attempts = 1

        timeline.append(event('Resolution Agent Drafting', response, attempt=attempts))
        semantic_validator = (
            DeepEvalFaithfulnessValidator(
                complaint,
                threshold=DEEPEVAL_THRESHOLD,
                model=DEEPEVAL_MODEL,
            )
            if DEEPEVAL_ENABLED and input_check.needs_policy and policy_hits
            else None
        )
        gate = validate_final(
            reply=response,
            requested_order_id=requested or '',
            order_access=order_access,
            needs_policy=input_check.needs_policy,
            policies=policy_hits,
            needs_history=input_check.needs_history,
            memory_checked=memory_checked,
            authenticated=True,
            errors=errors,
            category=category,
            urgency=urgency,
            complaint=complaint,
            has_order=bool(order),
            memories=memories,
            needs_human=status == 'Escalated',
            confidence=0.95,
            human_reason=reason or '',
            order=order,
            semantic_validator=semantic_validator,
            require_semantic_for_policy=DEEPEVAL_ENABLED and input_check.needs_policy,
            sentiment=sentiment,
        )
        timeline.append(faithfulness_timeline_event(
            enabled=DEEPEVAL_ENABLED,
            needs_policy=input_check.needs_policy,
            has_policy_evidence=bool(policy_hits),
            validator=semantic_validator,
            final_check=gate,
            attempt=attempts,
        ))
        timeline.append(event('Final Guardrail',
                              'Deterministic demo evidence and response checks',
                              allowed=gate.allowed, reasons=gate.reasons))
        if not gate.allowed:
            status, reason = 'Escalated', '; '.join(gate.reasons)
            if 'Required official policy evidence missing' in gate.reasons:
                response = (
                    'I could not find a matching official policy to answer this confidently, '
                    'so I have marked your request for human review.'
                )
            elif not (requested and not order):
                response = SAFE_RESPONSE

    review_reason = UserApprovalAgent.review_reason(
        status=status,
        complaint=complaint,
        category=category,
        urgency=urgency,
    )
    if review_reason:
        was_already_escalated = status == 'Escalated'
        status = 'Escalated'
        if not was_already_escalated:
            reason, response = review_reason, SAFE_RESPONSE

    timeline.append(event(
        'Human Review Routing',
        'Human review requested' if status == 'Escalated' else 'No human review needed',
        required=status == 'Escalated',
        status='Pending' if status == 'Escalated' else 'Not required',
    ))
    timeline.append(event('Log Agent & Escalation Decision',
                          f'Status: {status}; saved to SQLite'))
    return dict(order_id=requested, complaint=complaint, category=category, urgency=urgency,
                sentiment=sentiment, response=response, status=status, reason=reason,
                policy_sources=list(dict.fromkeys(hit['source'] for hit in policy_hits)),
                answer_attempts=attempts, timeline=timeline)

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'mode': 'demo' if DEMO_MODE else 'unconfigured-production',
        'support_pipeline': SUPPORT_PIPELINE_MODE,
        'deepeval_enabled': DEEPEVAL_ENABLED,
        'deepeval_status': (
            'enabled' if DEEPEVAL_ENABLED
            else 'missing_openai_api_key' if DEEPEVAL_REQUESTED
            else 'disabled'
        ),
        'deepeval_threshold': DEEPEVAL_THRESHOLD,
    }


@app.on_event('shutdown')
async def close_graphflow_model():
    if _GRAPHFLOW_MODEL is not None:
        await _GRAPHFLOW_MODEL.close()

@app.get('/me')
def me(user=Depends(identity)):
    return user

@app.get('/demo/customers')
def demo_customers(x_demo_api_key: str | None = Header(default=None)):
    """List local sample accounts for the demo sign-in screen only."""
    if not DEMO_MODE:
        raise HTTPException(503, 'Demo account listing is disabled')
    if x_demo_api_key != DEMO_KEY:
        raise HTTPException(401, 'Invalid demo API key')
    with db() as conn:
        rows = conn.execute(
            'SELECT customer_id, name, email FROM customers ORDER BY name, customer_id'
        ).fetchall()
    return [dict(row) for row in rows]

@app.get('/me/orders')
def my_orders(user=Depends(identity)):
    """List only order IDs/statuses belonging to the signed-in demo customer."""
    with db() as conn:
        rows = conn.execute(
            '''SELECT order_id, status, tracking_id FROM orders
               WHERE customer_id=? ORDER BY order_id''',
            (user['customer_id'],),
        ).fetchall()
    return [dict(row) for row in rows]

@app.post('/support/tickets', response_model=TicketOut)
async def create_ticket(request: ComplaintRequest, user=Depends(identity)):
    if not DEMO_MODE:
        raise HTTPException(503, 'Production authentication adapter is not configured')
    if SUPPORT_PIPELINE_MODE == 'graphflow':
        from database.ticket_repository import list_tickets as list_agent_tickets
        from services.knowledge_service import RequestIdentity

        pipeline = _graphflow_pipeline()
        result = await pipeline.handle(
            request.complaint,
            identity=RequestIdentity(customer_id=user['customer_id']),
            customer_name=user['name'],
            selected_order_id=request.order_id,
        )
        record = next(
            (ticket for ticket in list_agent_tickets(_graphflow_db_path())
             if ticket['ticket_id'] == result.ticket_id),
            None,
        )
        if record is None:
            raise RuntimeError('GraphFlow completed without a persisted ticket')
        ticket = _agent_ticket_out(record, result.timeline)
        return _include_customer_review_status(
            [ticket], _review_db_path()
        )[0]

    result = demo_process(request.complaint, request.order_id, user)
    ticket_id = 'TKT-' + uuid.uuid4().hex[:8].upper()
    created_at = datetime.now(timezone.utc).isoformat()
    record = {'ticket_id': ticket_id, 'created_at': created_at,
              'customer_id': user['customer_id'], 'customer_name': user['name'], **result}
    with db() as conn:
        conn.execute('''INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            ticket_id, created_at, user['customer_id'], user['name'], result['order_id'],
            result['complaint'], result['category'], result['urgency'], result['sentiment'],
            result['response'], result['status'], result['reason'],
            json.dumps(result['policy_sources']), result['answer_attempts'], json.dumps(result['timeline'])
        ))
    if result['status'] == 'Escalated':
        review_reason = result.get('reason') or 'Escalated support request requires human review'
        register_pending_review(DB_PATH, ticket_id, review_reason)
    return _include_customer_review_status([record], DB_PATH)[0]

@app.get('/support/tickets', response_model=list[TicketOut])
def list_tickets(user=Depends(identity)):
    if SUPPORT_PIPELINE_MODE == 'graphflow':
        from database.ticket_repository import list_tickets as list_agent_tickets

        _graphflow_pipeline()

        records = list_agent_tickets(_graphflow_db_path())
        customer_tickets = [
            _agent_ticket_out(ticket)
            for ticket in records
            if ticket['customer_id'] == user['customer_id']
        ]
        return _include_customer_review_status(
            customer_tickets, _review_db_path()
        )
    with db() as conn:
        rows = conn.execute('SELECT * FROM tickets WHERE customer_id=? ORDER BY created_at DESC LIMIT 200', (user['customer_id'],)).fetchall()
    return _include_customer_review_status(
        [decode(row) for row in rows], DB_PATH
    )

@app.get('/support/tickets/{ticket_id}', response_model=TicketOut)
def get_ticket(ticket_id: str, user=Depends(identity)):
    if SUPPORT_PIPELINE_MODE == 'graphflow':
        from database.ticket_repository import list_tickets as list_agent_tickets

        _graphflow_pipeline()
        record = next(
            (ticket for ticket in list_agent_tickets(_graphflow_db_path())
             if ticket['ticket_id'] == ticket_id
             and ticket['customer_id'] == user['customer_id']),
            None,
        )
        if not record:
            raise HTTPException(404, 'Ticket not found')
        return _include_customer_review_status(
            [_agent_ticket_out(record)], _review_db_path()
        )[0]
    with db() as conn:
        row = conn.execute('SELECT * FROM tickets WHERE ticket_id=? AND customer_id=?', (ticket_id, user['customer_id'])).fetchone()
    if not row:
        raise HTTPException(404, 'Ticket not found')
    return _include_customer_review_status(
        [decode(row)], DB_PATH
    )[0]

@app.get('/support/tickets.csv')
def export_tickets(user=Depends(identity)):
    if SUPPORT_PIPELINE_MODE == 'graphflow':
        from database.ticket_repository import list_tickets as list_agent_tickets

        _graphflow_pipeline()
        rows = [
            {
                'ticket_id': ticket['ticket_id'],
                'created_at': ticket['timestamp'],
                'customer_name': ticket.get('customer_name') or '',
                'order_id': ticket.get('order_id'),
                'complaint': ticket['complaint'],
                'category': ticket['category'],
                'urgency': ticket['urgency'],
                'sentiment': ticket['sentiment'],
                'policy_sources': json.dumps(ticket['kb_sources']),
                'response': ticket['response'],
                'status': ticket['status'],
                'reason': ticket.get('escalation_reason'),
            }
            for ticket in list_agent_tickets(_graphflow_db_path())
            if ticket['customer_id'] == user['customer_id']
        ]
    else:
        with db() as conn:
            rows = conn.execute(
                '''SELECT ticket_id, created_at, customer_name, order_id, complaint,
                          category, urgency, sentiment, policy_sources, response,
                          status, reason
                   FROM tickets WHERE customer_id=? ORDER BY created_at DESC''',
                (user['customer_id'],),
            ).fetchall()
    output = io.StringIO()
    fields = [
        'ticket_id', 'timestamp', 'customer_name', 'order_id', 'complaint',
        'category', 'urgency', 'sentiment', 'kb_sources', 'response',
        'status', 'human_review_status', 'human_reviewed_at',
        'escalation_reason',
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    reviews = {
        review['ticket_id']: review
        for review in list_reviews(_review_db_path())
    }
    for row in rows:
        record = dict(row)
        review = reviews.get(record['ticket_id'])
        writer.writerow({
            'ticket_id': record['ticket_id'],
            'timestamp': record['created_at'],
            'customer_name': record['customer_name'],
            'order_id': record['order_id'],
            'complaint': record['complaint'],
            'category': record['category'],
            'urgency': record['urgency'],
            'sentiment': record['sentiment'],
            'kb_sources': ', '.join(json.loads(record['policy_sources'])),
            'response': record['response'],
            'status': record['status'],
            'human_review_status': review['status'] if review else '',
            'human_reviewed_at': review['reviewed_at'] if review else '',
            'escalation_reason': record['reason'],
        })
    return StreamingResponse(iter([output.getvalue()]), media_type='text/csv', headers={'Content-Disposition':'attachment; filename="shopease_tickets.csv"'})


@app.get('/admin/dashboard')
def admin_dashboard(_staff=Depends(human_review_staff)):
    tickets = _admin_tickets()
    reviews = list_reviews(_review_db_path())
    return {
        'total_tickets': len(tickets),
        'resolved_tickets': sum(ticket['status'] == 'Resolved' for ticket in tickets),
        'escalated_tickets': sum(ticket['status'] == 'Escalated' for ticket in tickets),
        'pending_reviews': sum(review['status'] == 'Pending' for review in reviews),
        'high_urgency': sum(ticket['urgency'] == 'High' for ticket in tickets),
        'recent_tickets': tickets[:10],
    }


@app.get('/admin/tickets')
def admin_tickets(_staff=Depends(human_review_staff)):
    return _admin_tickets()


@app.get('/admin/reviews')
def admin_reviews(status: Literal['Pending', 'Approved', 'Rejected'] | None = None,
                  _staff=Depends(human_review_staff)):
    tickets = {ticket['ticket_id']: ticket for ticket in _admin_tickets()}
    result = []
    for review in list_reviews(_review_db_path()):
        if status and review['status'] != status:
            continue
        ticket = tickets.get(review['ticket_id'])
        if ticket:
            result.append({**ticket, **review})
    return result


@app.post('/admin/reviews/{ticket_id}')
def decide_human_review(ticket_id: str, decision: HumanReviewDecision,
                        _staff=Depends(human_review_staff)):
    try:
        result = record_review_decision(
            _review_db_path(),
            ticket_id,
            decision=decision.decision,
            reviewer=decision.reviewer,
            note=decision.note,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return result


@app.get('/admin/audit')
def admin_audit(_staff=Depends(human_review_staff)):
    tickets = {ticket['ticket_id']: ticket for ticket in _admin_tickets()}
    return [
        {
            **item,
            'customer_name': tickets.get(item['ticket_id'], {}).get('customer_name', ''),
            'category': tickets.get(item['ticket_id'], {}).get('category', ''),
            'urgency': tickets.get(item['ticket_id'], {}).get('urgency', ''),
        }
        for item in list_review_audit(_review_db_path())
    ]


@app.get('/knowledge/policies')
def policies(user=Depends(identity)):
    return {'documents': [], 'message': 'Connect the existing policy repository and FAISS/BM25 reindexing service. Editing is disabled until authenticated administrator access is implemented.'}
