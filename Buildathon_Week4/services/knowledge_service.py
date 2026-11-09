"""Deterministic retrieval: policy hybrid RAG + authorized SQLite + scoped Chroma."""
from dataclasses import dataclass
from pathlib import Path
from database.order_repository import AuthenticatedCustomer, get_order

@dataclass(frozen=True)
class RequestIdentity:
    # Must be set by the trusted application/authentication layer, never the LLM.
    customer_id: str | None = None

class KnowledgeService:
    def __init__(self, retriever, memory=None, db_path: Path | None = None):
        self.retriever, self.memory, self.db_path = retriever, memory, db_path

    def gather(self, *, question: str, order_id: str = '',
               identity: RequestIdentity | None = None,
               needs_policy: bool = False, needs_history: bool = False,
               policy_category: str | list[str] | None = None):
        if not question.strip(): raise ValueError('Question is required')
        customer_id = identity.customer_id.strip() if identity and identity.customer_id else ''
        if needs_history and not customer_id:
            raise PermissionError('A verified customer is required for history search')
        if needs_history and self.memory is None:
            raise RuntimeError('Customer history search is not configured')
        policy_hits = (
            self.retriever.search(question, top_k=5, category=policy_category)
            if needs_policy else []
        )
        policies = [{'content': hit['content'], 'source': hit.get('source', hit.get('file_name', hit.get('category','policy'))),
                     'chunk_id': hit.get('chunk_id', '')} for hit in policy_hits]
        order = None
        if order_id and customer_id:
            order = get_order(order_id, AuthenticatedCustomer(customer_id), self.db_path)
        memories = []
        if needs_history and customer_id and self.memory is not None:
            memories = self.memory.recall(customer_id=customer_id, question=question, limit=3)
        return {'policies': policies, 'order': order, 'memories': memories,
                'requested_order_id': order_id, 'authenticated': bool(customer_id),
                'order_access': ('not_requested' if not order_id else
                                 'authentication_required' if not customer_id else
                                 'found' if order else 'not_found_or_not_owned')}
