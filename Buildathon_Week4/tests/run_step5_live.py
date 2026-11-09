"""Live AutoGen + OpenAI + Chroma integration. Requires OPENAI_API_KEY."""
from __future__ import annotations
import asyncio
import os
from dotenv import load_dotenv
from agents.memory_agent import MemoryAgent
from memory.chroma_memory import CustomerMemory, OpenAIEmbedder
from memory.memory_manager import MemoryManager, VerifiedTicket


async def main():
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY in .env before running live demo')
    ticket = VerifiedTicket(
        customer_id='C001', ticket_id='DEMO-TKT101',
        complaint='My headphones arrived damaged. Can I replace them?',
        verified_summary='Customer C001 reported damaged headphones for ORD101; replacement requested but not approved.',
        order_id='ORD101',
    )
    agent = MemoryAgent()
    try:
        proposal = await agent.extract(
            complaint=ticket.complaint,
            verified_ticket_summary=ticket.verified_summary,
            verified_order_id=ticket.order_id,
        )
        print('Agent proposal:', proposal.model_dump_json(indent=2))
        store = CustomerMemory(directory='data/chroma', embedder=OpenAIEmbedder())
        outcome = MemoryManager(store).save_proposal(ticket, proposal)
        print('Storage outcome:', outcome)
        hits = store.recall(customer_id='C001', question='My damaged headphones complaint',
                            max_distance=0.8)
        print('C001 memories:', hits)
        print('C002 memories:', store.recall(customer_id='C002',
              question='My damaged headphones complaint', max_distance=0.8))
    finally:
        await agent.close()


if __name__ == '__main__':
    asyncio.run(main())
