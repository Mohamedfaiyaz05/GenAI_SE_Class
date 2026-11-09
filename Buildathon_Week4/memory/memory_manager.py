"""Deterministic security and evidence checks before memory persistence."""
from __future__ import annotations
from dataclasses import dataclass
from agents.memory_agent import MemoryProposal
from memory.chroma_memory import CustomerMemory


@dataclass(frozen=True)
class VerifiedTicket:
    customer_id: str       # MUST originate from authenticated server session
    ticket_id: str         # MUST originate from trusted ticket database
    complaint: str
    verified_summary: str # MUST be independently verified by application
    order_id: str = ''      # MUST be validated against customer ownership


class MemoryManager:
    def __init__(self, store: CustomerMemory):
        self.store = store

    def save_proposal(self, ticket: VerifiedTicket, proposal: MemoryProposal) -> dict:
        if not ticket.customer_id or not ticket.ticket_id:
            return {'saved': False, 'reason': 'Missing trusted customer or ticket ID'}
        if proposal.action == 'SKIP':
            return {'saved': False, 'reason': 'Agent chose SKIP'}
        if not ticket.verified_summary.strip():
            return {'saved': False, 'reason': 'No verified ticket evidence'}
        if not proposal.summary.strip():
            return {'saved': False, 'reason': 'Empty summary'}
        if proposal.order_id and proposal.order_id != ticket.order_id:
            return {'saved': False, 'reason': 'Unverified order ID'}
        # Store trusted summary, NOT the agent's free-form claim. The proposal
        # controls classification; trusted evidence controls factual content.
        text = ticket.verified_summary.strip()
        if ticket.order_id and ticket.order_id not in text:
            text = f'Order {ticket.order_id}: {text}'
        memory_id = self.store.upsert(
            customer_id=ticket.customer_id, ticket_id=ticket.ticket_id,
            memory_type=proposal.memory_type, order_id=ticket.order_id,
            text=text,
        )
        return {'saved': True, 'memory_id': memory_id, 'stored_text': text}
