"""Persistent customer-scoped Chroma memory. Uses OpenAI embeddings in production."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Protocol
import chromadb
from openai import OpenAI


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str = 'text-embedding-3-small'):
        self.client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class CustomerMemory:
    def __init__(self, directory: str = 'data/chroma', embedder: Embedder | None = None):
        self.embedder = embedder or OpenAIEmbedder()
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(directory))
        self.collection = self.client.get_or_create_collection(
            name='shopease_customer_memories',
            metadata={'hnsw:space': 'cosine'},
        )

    @staticmethod
    def _id(customer_id: str, ticket_id: str, memory_type: str, order_id: str) -> str:
        # One current memory per (customer, ticket, type, order); updates overwrite.
        raw = '\x1f'.join([customer_id, ticket_id, memory_type, order_id])
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def upsert(self, *, customer_id: str, ticket_id: str, memory_type: str,
               text: str, order_id: str = '', status: str = 'historical') -> str:
        if not customer_id or not ticket_id or not text.strip():
            raise ValueError('customer_id, ticket_id and text are required')
        memory_id = self._id(customer_id, ticket_id, memory_type, order_id)
        vector = self.embedder.embed([text])[0]
        self.collection.upsert(
            ids=[memory_id], embeddings=[vector], documents=[text],
            metadatas=[{'customer_id': customer_id, 'ticket_id': ticket_id,
                        'memory_type': memory_type, 'order_id': order_id,
                        'status': status}],
        )
        return memory_id

    def recall(self, *, customer_id: str, question: str, limit: int = 3,
               max_distance: float = 0.65) -> list[dict]:
        if not customer_id or not question.strip() or limit < 1:
            return []
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed([question])[0]],
            n_results=min(limit, count),
            where={'customer_id': customer_id},
            include=['documents', 'metadatas', 'distances'],
        )
        hits = []
        for id_, document, metadata, distance in zip(
            result['ids'][0], result['documents'][0],
            result['metadatas'][0], result['distances'][0]
        ):
            if distance <= max_distance:
                hits.append({'id': id_, 'text': document, 'metadata': metadata,
                             'distance': float(distance)})
        return hits

    def delete_ticket(self, *, customer_id: str, ticket_id: str) -> int:
        if not customer_id or not ticket_id:
            raise ValueError('customer_id and ticket_id required')
        existing = self.collection.get(where={'$and': [
            {'customer_id': customer_id}, {'ticket_id': ticket_id}
        ]})
        if existing['ids']:
            self.collection.delete(ids=existing['ids'])
        return len(existing['ids'])
