"""ShopEase Step 4: persistent embedding cache + FAISS/BM25/RRF.

Run from repository root after `python -m rag.ingest`.
Requires: numpy, faiss-cpu, rank-bm25, openai.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
CACHE_SCHEMA = 1
INDEX_SCHEMA = 1


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(obj: object) -> str:
    return digest_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, obj: object) -> None:
    atomic_bytes(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode())


def load_chunks(path: Path) -> tuple[list[dict], str]:
    """Validate Step 3 schema and return active chunks + reproducible signature."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not isinstance(payload.get("chunks"), list):
        raise ValueError("Expected Step 3 policy_chunks.json schema_version=2")
    chunks = payload["chunks"]
    ids: set[str] = set()
    for c in chunks:
        for key in ("chunk_id", "source", "category", "section", "policy_version", "content"):
            if not isinstance(c.get(key), str) or not c[key].strip():
                raise ValueError(f"Invalid chunk field {key}: {c!r}")
        if c["chunk_id"] in ids:
            raise ValueError(f"Duplicate chunk_id: {c['chunk_id']}")
        ids.add(c["chunk_id"])
    if not chunks:
        raise ValueError("No active policy chunks")
    # Include content and metadata so policy version/section changes also refresh search metadata.
    return chunks, sha_json(chunks)


class EmbeddingCache:
    """SQLite durable cache; vector bytes are float32 and keyed by model + dimensions + text hash."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL
            )""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def key(model: str, dimensions: int, text: str) -> str:
        # Include the embedding input text, model and output dimensions.
        return sha_json({"schema": CACHE_SCHEMA, "provider": "openai", "model": model,
                         "dimensions": dimensions, "content_sha256": digest_bytes(text.encode("utf-8"))})

    def get(self, model: str, dimensions: int, text: str) -> np.ndarray | None:
        key = self.key(model, dimensions, text)
        with self._connect() as db:
            row = db.execute("SELECT vector FROM embeddings WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype="<f4").copy()
        if vector.size != dimensions or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
            return None
        return vector

    def put_many(self, model: str, dimensions: int, items: list[tuple[str, np.ndarray]]) -> None:
        records = []
        for content, vector in items:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (dimensions,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                raise ValueError("Embedding returned invalid dimension, nonfinite or zero vector")
            records.append((self.key(model, dimensions, content), model, dimensions,
                            digest_bytes(content.encode("utf-8")), vector.astype("<f4").tobytes()))
        with self._connect() as db:
            db.executemany("""INSERT OR REPLACE INTO embeddings
                (cache_key,model,dimensions,content_sha256,vector) VALUES (?,?,?,?,?)""", records)


@dataclass
class BuildStats:
    active_chunks: int
    unique_texts: int
    cache_hits: int
    embedded_new: int
    index_rebuilt: bool
    signature: str


class HybridRetriever:
    """Indexes active Step 3 chunks; never serves old versions after a successful rebuild."""

    def __init__(self, *, chunks_path: Path | None = None, index_dir: Path | None = None,
                 model: str = "text-embedding-3-small", dimensions: int = 1536,
                 batch_size: int = 64,
                 embedder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None):
        self.chunks_path = chunks_path or ROOT / "data" / "ingestion" / "policy_chunks.json"
        self.index_dir = index_dir or ROOT / "data" / "faiss"
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        if dimensions <= 0 or batch_size <= 0:
            raise ValueError("dimensions and batch_size must be positive")
        self.embedder = embedder or self._openai_embed
        self.cache = EmbeddingCache(self.index_dir / "embedding_cache.sqlite3")
        self.chunks: list[dict] = []
        self.index = None
        self.bm25 = None
        self.signature = ""

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        load_dotenv(ROOT / ".env")
        from openai import OpenAI
        response = OpenAI().embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise RuntimeError("Embedding response indexes are incomplete")
        return [item.embedding for item in ordered]

    def _index_paths(self) -> tuple[Path, Path]:
        return self.index_dir / "policies.faiss", self.index_dir / "manifest.json"

    def build(self) -> BuildStats:
        chunks, content_signature = load_chunks(self.chunks_path)
        signature = sha_json({"chunks": content_signature, "model": self.model,
                              "dimensions": self.dimensions, "index_schema": INDEX_SCHEMA})
        faiss_path, manifest_path = self._index_paths()
        manifest = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        # Validate index checksum before trusting persisted index.
        healthy = bool(manifest and manifest.get("signature") == signature and faiss_path.exists()
                       and manifest.get("faiss_sha256") == digest_bytes(faiss_path.read_bytes()))
        if healthy:
            try:
                index = faiss.read_index(str(faiss_path))
                healthy = index.ntotal == len(chunks) and index.d == self.dimensions
            except Exception:
                healthy = False
        if healthy:
            self.index = index
            self.chunks = chunks
            self.signature = signature
            self.bm25 = BM25Okapi([tokenize(c["content"]) or ["__empty__"] for c in chunks])
            return BuildStats(len(chunks), len({c["content"] for c in chunks}), 0, 0, False, signature)

        # Compute embeddings for unique text only. Persist each completed batch.
        unique_texts = list(dict.fromkeys(c["content"] for c in chunks))
        vectors: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for content in unique_texts:
            vector = self.cache.get(self.model, self.dimensions, content)
            if vector is None:
                missing.append(content)
            else:
                vectors[content] = vector
        hits = len(vectors)
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset:offset + self.batch_size]
            result = list(self.embedder(batch))
            if len(result) != len(batch):
                raise ValueError("Embedding API returned incorrect number of vectors")
            new_items = [(text, np.asarray(vec, dtype=np.float32)) for text, vec in zip(batch, result)]
            self.cache.put_many(self.model, self.dimensions, new_items)
            vectors.update(new_items)

        matrix = np.ascontiguousarray(np.stack([vectors[c["content"]] for c in chunks]).astype("float32"))
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(self.dimensions)
        index.add(matrix)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Write index first, manifest last. If interrupted, checksum forces repair on next run.
        tmp_index = self.index_dir / "policies.faiss.tmp"
        try:
            faiss.write_index(index, str(tmp_index))
            index_hash = digest_bytes(tmp_index.read_bytes())
            os.replace(tmp_index, faiss_path)
            atomic_json(manifest_path, {"schema_version": INDEX_SCHEMA, "signature": signature,
                                        "faiss_sha256": index_hash, "chunk_count": len(chunks),
                                        "model": self.model, "dimensions": self.dimensions})
        finally:
            if tmp_index.exists():
                tmp_index.unlink()
        self.index = index
        self.chunks = chunks
        self.signature = signature
        self.bm25 = BM25Okapi([tokenize(c["content"]) or ["__empty__"] for c in chunks])
        return BuildStats(len(chunks), len(unique_texts), hits, len(missing), True, signature)

    def search(self, query: str, *, top_k: int = 5, category: str | list[str] | None = None,
               candidate_k: int = 20, rrf_k: int = 60) -> list[dict]:
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Call build() before search()")
        if not query.strip() or top_k < 1 or candidate_k < 1 or rrf_k < 1:
            raise ValueError("Provide nonempty query and positive top_k/candidate_k/rrf_k")
        allowed = ({category.casefold()} if isinstance(category, str) else
                   {x.casefold() for x in category} if category else None)
        eligible = [i for i, c in enumerate(self.chunks)
                    if allowed is None or c["category"].casefold() in allowed]
        if not eligible:
            return []
        count = min(candidate_k, len(eligible))
        # Search full FAISS index and filter afterwards; for small policy corpora this is exact.
        query_vector = np.asarray(list(self.embedder([query]))[0], dtype=np.float32).reshape(1, -1)
        if query_vector.shape != (1, self.dimensions) or not np.all(np.isfinite(query_vector)) or np.linalg.norm(query_vector) == 0:
            raise ValueError("Invalid query embedding")
        faiss.normalize_L2(query_vector)
        scores, ids = self.index.search(query_vector, len(self.chunks))
        eligible_set = set(eligible)
        vector_rank = [int(i) for i in ids[0] if int(i) in eligible_set][:count]
        terms = tokenize(query)
        bm25_scores = self.bm25.get_scores(terms)
        # BM25 zero-score matches are not evidence of keyword relevance.
        keyword_rank = sorted((i for i in eligible if bm25_scores[i] > 0),
                              key=lambda i: (-float(bm25_scores[i]), self.chunks[i]["chunk_id"]))[:count]
        fused: dict[int, float] = {}
        for ranking in (vector_rank, keyword_rank):
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
        ordered = sorted(fused, key=lambda i: (-fused[i], self.chunks[i]["chunk_id"]))[:top_k]
        return [{**self.chunks[i], "rrf_score": round(fused[i], 8),
                 "bm25_score": round(float(bm25_scores[i]), 5),
                 "retrieval_methods": (["faiss"] if i in vector_rank else []) +
                                      (["bm25"] if i in keyword_rank else [])}
                for i in ordered]
