"""Live integration demonstration; calls OpenAI and writes SQLite tickets."""
import asyncio
import json
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from rag.hybrid_retriever import HybridRetriever
from memory.chroma_memory import CustomerMemory
from services.knowledge_service import KnowledgeService, RequestIdentity
from services.support_pipeline import SupportPipeline
from database.connection import initialize_database
from database.seed import seed_demo_data

async def main():
    load_dotenv()
    initialize_database()
    seed_demo_data()
    retriever = HybridRetriever()
    retriever.build()
    memory = CustomerMemory()
    model = OpenAIChatCompletionClient(model='gpt-4o-mini')
    try:
        service = KnowledgeService(retriever, memory)
        pipeline = SupportPipeline(model, service)
        result = await pipeline.handle('Where is my order ORD101? It seems delayed.',
            identity=RequestIdentity(customer_id='C001'), customer_name='Jane Doe')
        print(json.dumps(result.__dict__, indent=2, default=str))
        guest = await pipeline.handle('What is your general refund policy?', identity=RequestIdentity())
        print('\nGUEST TICKET:', guest.ticket_id, guest.status)
    finally:
        await model.close()

if __name__ == '__main__':
    asyncio.run(main())
