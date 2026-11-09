"""Tests for input validation and privacy-conscious fallback auditing."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rag.ingest as ingest
import rag.retriever as retriever_module
from rag.retriever import MedicalKnowledgeRetriever
from tools.csv_audit_tool import write_fallback_audit


@pytest.mark.parametrize(
    ("question", "expected_chunk_id"),
    [
        ("What is the refund policy here?", 3),
        ("Can I get my money back after cancelling?", 2),
    ],
)
def test_retrieval_prioritizes_policy_terms(question, expected_chunk_id):
    retriever = object.__new__(MedicalKnowledgeRetriever)
    retriever.vector_db = SimpleNamespace(
        index=SimpleNamespace(ntotal=4),
        similarity_search_with_score=lambda query, k: [
            (
                SimpleNamespace(
                    page_content="Clinical release rules for reports.",
                    metadata={"source": "handbook.md", "chunk_id": 1},
                ),
                0.01,
            ),
            (
                SimpleNamespace(
                    page_content=(
                        "Cancel at least 24 hours ahead to request a full "
                        "refund. Later cancellations are reviewed by staff."
                    ),
                    metadata={
                        "source": "handbook.md",
                        "section_title": "Refunds and payment reversals",
                        "chunk_id": 2,
                    },
                ),
                0.4,
            ),
            (
                SimpleNamespace(
                    page_content=(
                        "An approved refund is initiated within 7 business days."
                    ),
                    metadata={
                        "source": "handbook.md",
                        "section_title": "Refund policy",
                        "chunk_id": 3,
                    },
                ),
                0.5,
            ),
            (
                SimpleNamespace(
                    page_content="Fasting depends on the ordered blood test.",
                    metadata={"source": "handbook.md", "chunk_id": 4},
                ),
                0.03,
            ),
        ],
    )

    results = retriever.retrieve(question, k=3)

    assert results[0]["chunk_id"] == expected_chunk_id
    assert len(results) == 3


def test_retriever_builds_missing_index_automatically(tmp_path, monkeypatch):
    index_dir = tmp_path / "faiss_index"
    handbook = tmp_path / "handbook.md"
    handbook.write_text("# Handbook\nPolicy about refunds.", encoding="utf-8")
    monkeypatch.setattr(retriever_module, "FAISS_DIR", index_dir)
    monkeypatch.setattr(retriever_module, "KNOWLEDGE_FILE", handbook)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_ingest(faiss_dir):
        faiss_dir.mkdir(parents=True, exist_ok=True)
        (faiss_dir / "index.faiss").write_bytes(b"test index")
        (faiss_dir / "index.pkl").write_bytes(b"test metadata")
        (faiss_dir / ingest.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_file": handbook.name,
                    "source_sha256": hashlib.sha256(
                        handbook.read_text(encoding="utf-8").encode("utf-8")
                    ).hexdigest(),
                    "embedding_model": ingest.EMBEDDING_MODEL,
                    "chunk_count": 1,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ingest, "ingest_documents", fake_ingest)
    monkeypatch.setattr(
        retriever_module,
        "OpenAIEmbeddings",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        retriever_module.FAISS,
        "load_local",
        lambda **kwargs: "loaded index",
    )

    retriever = MedicalKnowledgeRetriever()

    assert retriever.vector_db == "loaded index"
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "index.pkl").is_file()


def test_retriever_rebuilds_index_when_handbook_changes(tmp_path, monkeypatch):
    handbook = tmp_path / "handbook.md"
    handbook.write_text("# Handbook\nUpdated policy.", encoding="utf-8")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"old index")
    (index_dir / "index.pkl").write_bytes(b"old metadata")
    (index_dir / ingest.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "source_file": handbook.name,
                "source_sha256": "stale-source-hash",
                "embedding_model": ingest.EMBEDDING_MODEL,
                "chunk_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retriever_module, "FAISS_DIR", index_dir)
    monkeypatch.setattr(retriever_module, "KNOWLEDGE_FILE", handbook)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    build_calls = []

    def fake_ingest(faiss_dir):
        build_calls.append(faiss_dir)
        source_hash = hashlib.sha256(
            handbook.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        (faiss_dir / ingest.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_file": handbook.name,
                    "source_sha256": source_hash,
                    "embedding_model": ingest.EMBEDDING_MODEL,
                    "chunk_count": 1,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ingest, "ingest_documents", fake_ingest)
    monkeypatch.setattr(
        retriever_module,
        "OpenAIEmbeddings",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        retriever_module.FAISS,
        "load_local",
        lambda **kwargs: "loaded index",
    )

    MedicalKnowledgeRetriever()

    assert build_calls == [index_dir]


def test_ingestion_saves_files_required_by_retriever(tmp_path, monkeypatch):
    handbook = tmp_path / "handbook.md"
    handbook.write_text(
        "# Handbook\n\n"
        "## Refund policy\n\n"
        + ("Refunds follow the cancellation timing policy. " * 30)
        + "\n\n## Report access\n\nReports appear after clinical release.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "faiss_index"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ingest, "OpenAIEmbeddings", lambda **kwargs: object())
    indexed_documents = []

    class FakeVectorStore:
        def save_local(self, path):
            output_dir = Path(path)
            (output_dir / "index.faiss").write_bytes(b"test index")
            (output_dir / "index.pkl").write_bytes(b"test metadata")

    monkeypatch.setattr(
        ingest.FAISS,
        "from_documents",
        lambda documents, embedding: (
            indexed_documents.extend(documents) or FakeVectorStore()
        ),
    )

    chunk_count = ingest.ingest_documents(
        knowledge_file=handbook,
        faiss_dir=index_dir,
    )

    assert chunk_count > 0
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "index.pkl").is_file()
    manifest = json.loads(
        (index_dir / ingest.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["source_sha256"] == hashlib.sha256(
        handbook.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    refund_chunks = [
        document
        for document in indexed_documents
        if document.metadata.get("section_title") == "Refund policy"
    ]
    assert len(refund_chunks) > 1
    assert all("Refund policy" in chunk.page_content for chunk in refund_chunks)


def test_retrieval_rejects_empty_query():
    retriever = object.__new__(MedicalKnowledgeRetriever)
    with pytest.raises(ValueError, match="Question must be a non-empty string"):
        retriever.retrieve("   ")


def test_fallback_audit_accepts_sanitized_question_summary(
    tmp_path,
    monkeypatch,
):
    audit_path = tmp_path / "fallback_audit.csv"
    monkeypatch.setattr("tools.csv_audit_tool.AUDIT_FILE", audit_path)

    write_fallback_audit(
        question="Medical information request (details omitted)",
        fallback_reason="Insufficient reliable evidence",
        source_urls=[],
        fallback_status="safe_response",
        monitoring_verdict="NOT_RUN",
        monitoring_reason="No web-supported answer was returned.",
    )

    with audit_path.open(newline="", encoding="utf-8") as audit_file:
        rows = list(csv.DictReader(audit_file))
    assert len(rows) == 1
    assert rows[0]["question"] == "Medical information request (details omitted)"
    assert rows[0]["fallback_status"] == "safe_response"
    assert rows[0]["timestamp_utc"]
