"""Offline integration test: no OpenAI API calls. Run python -m tests.run_step5_demo"""
from __future__ import annotations
import hashlib
import tempfile
from agents.memory_agent import MemoryProposal
from memory.chroma_memory import CustomerMemory
from memory.memory_manager import MemoryManager, VerifiedTicket


class DemoEmbedder:
    """Deterministic test-only vectors; not suitable for real semantic search."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.lower().encode()).digest()
            vectors.append([float(x) / 255 for x in digest[:16]])
        return vectors


def main():
    with tempfile.TemporaryDirectory() as directory:
        clients = []
        try:
            store = CustomerMemory(directory=directory, embedder=DemoEmbedder())
            clients.append(store.client)
            manager = MemoryManager(store)
            ticket = VerifiedTicket('C001', 'TKT101',
                                    'My headphones arrived damaged',
                                    'Customer reported damaged headphones and requested replacement for ORD101.',
                                    'ORD101')
            proposal = MemoryProposal(action='STORE', memory_type='complaint_history',
                                      summary='Customer reported damaged headphones.', order_id='ORD101')
            first = manager.save_proposal(ticket, proposal)
            assert first['saved'], first
            assert store.collection.count() == 1
            again = manager.save_proposal(ticket, proposal)
            assert again['saved'] and store.collection.count() == 1, 'Upsert must prevent duplicates'
            bad = MemoryProposal(action='STORE', memory_type='complaint_history',
                                 summary='Unsupported', order_id='ORD999')
            assert not manager.save_proposal(ticket, bad)['saved']
            assert store.recall(customer_id='C002', question='headphones', max_distance=2.0) == []
            assert store.delete_ticket(customer_id='C002', ticket_id='TKT101') == 0
            reopened = CustomerMemory(directory=directory, embedder=DemoEmbedder())
            clients.append(reopened.client)
            assert reopened.collection.count() == 1
            found = reopened.recall(customer_id='C001', question=ticket.verified_summary,
                                    max_distance=2.0)
            assert found and found[0]['metadata']['customer_id'] == 'C001'
            assert reopened.delete_ticket(customer_id='C001', ticket_id='TKT101') == 1
            assert reopened.collection.count() == 0
            print('PASS: persistence, customer isolation, deduplication, validation, deletion')
            print('NOTE: DemoEmbedder tests storage mechanics, not semantic relevance or live AutoGen.')
        finally:
            for client in clients:
                client.close()


if __name__ == '__main__':
    main()
