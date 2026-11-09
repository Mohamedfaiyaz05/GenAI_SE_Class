
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from rag.ingest import EMBEDDING_MODEL, KNOWLEDGE_FILE, MANIFEST_NAME

logger = logging.getLogger(__name__)
_INDEX_BUILD_LOCK = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "storage" / "faiss_index"
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "for", "here", "i", "in", "is", "it",
    "me", "of", "on", "our", "please", "the", "this", "to", "we",
    "what", "when", "where", "which", "who", "why", "with", "you",
}
_TERM_CANONICAL = {
    "cancelled": "cancel",
    "canceled": "cancel",
    "cancellation": "cancel",
    "cancellations": "cancel",
    "cancelling": "cancel",
    "refunds": "refund",
    "refunded": "refund",
    "refunding": "refund",
    "reimburse": "refund",
    "reimbursed": "refund",
    "reimbursement": "refund",
    "reimbursements": "refund",
}


def _normalized_terms(text: str) -> set[str]:
    terms = {
        _TERM_CANONICAL.get(term, term)
        for term in _WORD_PATTERN.findall(text.lower())
        if term not in _STOP_WORDS
    }
    if "money back" in text.lower() or "payment reversal" in text.lower():
        terms.add("refund")
    return terms


class MedicalKnowledgeRetriever:

    def __init__(self):

        load_dotenv(BASE_DIR / ".env")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing in .env")

        index_files = (
            FAISS_DIR / "index.faiss",
            FAISS_DIR / "index.pkl",
        )
        with _INDEX_BUILD_LOCK:
            if not self._index_is_current(index_files):
                logger.info(
                    "FAISS index is missing or stale; rebuilding from the handbook."
                )
                from rag.ingest import ingest_documents

                ingest_documents(faiss_dir=FAISS_DIR)

        if not self._index_is_current(index_files):
            raise FileNotFoundError(
                f"FAISS index at {FAISS_DIR} is incomplete or does not match "
                f"{KNOWLEDGE_FILE}. Rebuild with `python -m rag.ingest`."
            )

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )

        # Load only trusted, locally generated FAISS files.
        self.vector_db = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    @staticmethod
    def _index_is_current(index_files: tuple[Path, Path]) -> bool:
        if not KNOWLEDGE_FILE.is_file() or not all(
            path.is_file() for path in index_files
        ):
            return False

        manifest_path = index_files[0].parent / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_hash = hashlib.sha256(
                KNOWLEDGE_FILE.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(manifest, dict):
            return False

        return (
            manifest.get("version") == 1
            and manifest.get("source_file") == KNOWLEDGE_FILE.name
            and manifest.get("source_sha256") == source_hash
            and manifest.get("embedding_model") == EMBEDDING_MODEL
        )

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """Retrieve relevant chunks from the medical handbook."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")

        vector_count = int(self.vector_db.index.ntotal)
        candidate_count = min(vector_count, max(k * 4, 32))
        results = self.vector_db.similarity_search_with_score(
            query=question.strip(),
            k=candidate_count,
        )

        query_terms = _normalized_terms(question)
        ranked_results = []
        for document, score in results:
            content = document.page_content
            section_title = document.metadata.get("section_title", "")
            document_terms = _normalized_terms(
                f"{section_title} {content}"
            )
            matching_terms = len(query_terms & document_terms)
            ranked_results.append(
                (
                    -matching_terms,
                    float(score),
                    {
                        "content": document.page_content,
                        "source": document.metadata.get(
                            "source", "unknown"
                        ),
                        "chunk_id": document.metadata.get(
                            "chunk_id"
                        ),
                        "distance_score": float(score),
                    },
                )
            )

        ranked_results.sort(key=lambda result: (result[0], result[1]))
        return [result[2] for result in ranked_results[:k]]
