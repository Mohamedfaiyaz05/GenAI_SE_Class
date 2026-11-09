"""Starter repository for support-ticket persistence."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import uuid
from database.connection import connect

@dataclass(frozen=True)
class NewTicket:
    complaint: str
    category: str
    urgency: Literal['Low', 'Medium', 'High']
    sentiment: str
    response: str
    status: Literal['Resolved', 'Escalated']
    customer_id: str | None = None
    customer_name: str | None = None
    order_id: str | None = None
    kb_sources: list[str] = field(default_factory=list)
    escalation_reason: str | None = None
    evidence_status: Literal['FULL','PARTIAL','NONE','CONFLICTING'] | None = None
    answer_attempts: int = 0


def save_ticket(ticket: NewTicket, db_path: Path | None = None) -> str:
    if not ticket.complaint.strip() or not ticket.response.strip() or not ticket.category.strip() or not ticket.sentiment.strip():
        raise ValueError('Complaint, response, category, and sentiment are required')
    if ticket.status not in ('Resolved', 'Escalated') or ticket.urgency not in ('Low','Medium','High'):
        raise ValueError('Invalid ticket status or urgency')
    if ticket.status == 'Escalated' and not (ticket.escalation_reason or '').strip():
        raise ValueError('Escalated tickets require an escalation reason')
    if ticket.answer_attempts < 0:
        raise ValueError('answer_attempts cannot be negative')
    ticket_id = 'TKT-' + uuid.uuid4().hex[:12].upper()
    with connect(db_path) as conn:
        conn.execute('''INSERT INTO tickets
            (ticket_id, customer_id, customer_name, order_id, complaint, category,
            urgency, sentiment, kb_sources, response, status, escalation_reason,
            evidence_status, answer_attempts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (ticket_id, ticket.customer_id, ticket.customer_name, ticket.order_id,
             ticket.complaint, ticket.category, ticket.urgency, ticket.sentiment,
             json.dumps(ticket.kb_sources), ticket.response, ticket.status,
             ticket.escalation_reason, ticket.evidence_status, ticket.answer_attempts))
    return ticket_id


def list_tickets(db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute('SELECT * FROM tickets ORDER BY timestamp DESC, ticket_id DESC').fetchall()
    result = []
    for row in rows:
        record = dict(row)
        record['kb_sources'] = json.loads(record['kb_sources'])
        result.append(record)
    return result
