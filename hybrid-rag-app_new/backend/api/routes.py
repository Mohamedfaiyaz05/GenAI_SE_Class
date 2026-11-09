import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from generation.generator import RAGPipeline
from ingestion.loader import PDFLoader
from ingestion.chunker import DocumentChunker
from core.config import settings

router = APIRouter()
pipeline = RAGPipeline()
logger = logging.getLogger(__name__)
vector_sync_error: Optional[str] = None
graph_sync_error: Optional[str] = None

KB_DIR = Path(settings.KNOWLEDGE_BASE_DIR)
KB_DIR.mkdir(parents=True, exist_ok=True)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    session_id: Optional[str] = "default_session"
    k: int = Field(default=4, ge=1, le=20)


def _ensure_initial_vector_index() -> bool:
    """Build the initial vector index on first chat if PDFs exist but FAISS is absent."""
    if _vector_index_is_ready():
        return False
    if not list(KB_DIR.glob("*.pdf")):
        return False

    logger.info("FAISS is not ready; indexing PDFs before the first chat request.")
    _sync_vector_index()
    try:
        _sync_graph_index()
    except Exception:
        logger.exception(
            "Neo4j synchronization failed after vector indexing; continuing with "
            "semantic retrieval where available."
        )
    return True


def _retry_graph_sync_if_needed() -> None:
    if graph_sync_error and list(KB_DIR.glob("*.pdf")):
        try:
            _sync_graph_index()
        except Exception:
            logger.exception("Retrying Neo4j synchronization before chat failed.")


def _vector_index_is_ready() -> bool:
    vector_store = pipeline.vector_manager.vector_store
    index_path = pipeline.vector_manager.index_path
    return bool(
        vector_store
        and vector_store.index_to_docstore_id
        and (index_path / "index.faiss").is_file()
        and (index_path / "index.pkl").is_file()
        and pipeline.vector_manager.state_file_path.is_file()
    )


def initialize_semantic_index() -> None:
    """Synchronize semantic and graph indexes on backend startup."""
    global vector_sync_error, graph_sync_error
    if not list(KB_DIR.glob("*.pdf")):
        logger.info("No knowledge-base PDFs found at startup; index synchronization is deferred.")
        return

    try:
        processed_files = _sync_vector_index()
        if not _vector_index_is_ready():
            raise RuntimeError(
                "Semantic indexing finished without a non-empty persisted FAISS index."
            )
        logger.info(
            "Semantic index ready at startup: %d chunk(s), %d PDF(s) newly synchronized.",
            len(pipeline.vector_manager.vector_store.index_to_docstore_id),
            processed_files,
        )
    except Exception as exc:
        vector_sync_error = str(exc)
        logger.exception("Semantic index synchronization failed during backend startup.")

    try:
        processed_files = _sync_graph_index()
        graph_sync_error = None
        logger.info(
            "Knowledge graph ready at startup; %d PDF(s) newly synchronized.",
            processed_files,
        )
    except Exception as exc:
        graph_sync_error = str(exc)
        logger.exception("Knowledge graph synchronization failed during backend startup.")


@router.post("/chat/stream")
def stream_chat_endpoint(request: ChatRequest):
    """
    Streams the RAG pipeline response token by token using SSE formatting,
    maintaining session memory and sending full transparency metadata at completion.
    """
    def event_generator():
        try:
            if not _vector_index_is_ready() and list(KB_DIR.glob("*.pdf")):
                yield f"data: {json.dumps({'type': 'status', 'content': 'Building the document indexes before answering. This may take a few minutes.'})}\n\n"
                _ensure_initial_vector_index()
            _retry_graph_sync_if_needed()
            for event in pipeline.stream_run(
                query=request.query,
                session_id=request.session_id or "default_session",
                k=request.k,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Streaming chat request failed.")
            yield f"data: {json.dumps({'type': 'error', 'content': 'The request could not be completed. Check backend logs for details.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Standard non-streaming chat endpoint."""
    try:
        _ensure_initial_vector_index()
        _retry_graph_sync_if_needed()
        return pipeline.run(
            query=request.query,
            session_id=request.session_id or "default_session",
            k=request.k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
def ingest_documents_endpoint(file: UploadFile = File(...)):
    """
    Save an uploaded PDF, then synchronize FAISS and Neo4j from the knowledge base.
    """
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a file with a .pdf extension.")

    try:
        file_path = KB_DIR / filename
        with file_path.open("wb") as buffer:
            while content := file.file.read(1024 * 1024):
                buffer.write(content)

        vector_files = _sync_vector_index()
        try:
            graph_files = _sync_graph_index()
            graph_error = None
        except Exception as exc:
            graph_files = 0
            graph_error = str(exc)
            logger.exception(
                "Graph synchronization failed after semantic index persistence."
            )
        return {
            "status": "success",
            "filename": filename,
            "vector_files_processed": vector_files,
            "graph_files_processed": graph_files,
            "graph_error": graph_error,
            "message": f"Indexed {filename}; semantic vectors are persisted.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Ingestion failed for {filename}. Check backend logs for the underlying "
                f"error. Details: {e}"
            ),
        ) from e
    finally:
        file.file.close()


def _sync_vector_index() -> int:
    global vector_sync_error
    loader = PDFLoader(str(KB_DIR))
    chunker = DocumentChunker()
    try:
        processed_files = pipeline.vector_manager.sync_knowledge_base(loader, chunker)
        if list(KB_DIR.glob("*.pdf")) and not _vector_index_is_ready():
            raise RuntimeError(
                "PDF synchronization returned without a non-empty persisted FAISS index."
            )
        vector_sync_error = None
        return processed_files
    except Exception as exc:
        vector_sync_error = str(exc)
        raise


def _sync_graph_index() -> int:
    global graph_sync_error
    loader = PDFLoader(str(KB_DIR))
    try:
        processed_files = pipeline.graph_manager.sync_knowledge_base_graph(loader)
        if list(KB_DIR.glob("*.pdf")):
            counts = pipeline.graph_manager.query_graph(
                "MATCH (entity:Entity) RETURN count(entity) AS count"
            )
            if not counts or counts[0]["count"] == 0:
                raise RuntimeError(
                    "Graph synchronization produced no entity nodes in Neo4j."
                )
        graph_sync_error = None
        return processed_files
    except Exception as exc:
        graph_sync_error = str(exc)
        raise


@router.post("/ingest/sync")
def sync_knowledge_base_endpoint():
    """Index all PDFs currently in the configured knowledge-base folder."""
    try:
        if not list(KB_DIR.glob("*.pdf")):
            raise HTTPException(
                status_code=400,
                detail=f"No PDF files found in {KB_DIR}. Add PDFs before indexing.",
            )
        vector_files = _sync_vector_index()
        try:
            graph_files = _sync_graph_index()
            graph_error = None
        except Exception as exc:
            graph_files = 0
            graph_error = str(exc)
            logger.exception(
                "Graph synchronization failed after semantic index persistence."
            )
        return {
            "status": "success",
            "vector_files_processed": vector_files,
            "graph_files_processed": graph_files,
            "graph_error": graph_error,
            "message": "Semantic vectors are synchronized and persisted.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge-base synchronization failed: {e}",
        ) from e


@router.get("/stats")
def get_kb_stats():
    """Return document and index status so the UI can explain retrieval readiness."""
    try:
        files = sorted(KB_DIR.glob("*.pdf"))
        vector_manager = pipeline.vector_manager
        graph_counts = {"nodes": None, "relationships": None}
        graph_status = "unavailable"
        try:
            counts = pipeline.graph_manager.query_graph(
                """
                MATCH (n)
                WITH count(n) AS nodes
                OPTIONAL MATCH ()-[r]->()
                RETURN nodes, count(r) AS relationships
                """
            )
            if counts:
                graph_counts = {
                    "nodes": counts[0]["nodes"],
                    "relationships": counts[0]["relationships"],
                }
                graph_status = (
                    "ready"
                    if counts[0]["nodes"] > 0 or not files
                    else "empty"
                )
        except Exception as graph_error:
            graph_status = "error"
            graph_counts["error"] = str(graph_error)

        return {
            "total_documents": len(files),
            "documents": [file.name for file in files],
            "vector_index_ready": _vector_index_is_ready(),
            "vector_index_error": vector_sync_error,
            "vector_chunks": (
                len(vector_manager.vector_store.index_to_docstore_id)
                if vector_manager.vector_store is not None
                else 0
            ),
            "graph_status": graph_status,
            "graph_sync_error": graph_sync_error,
            "graph": graph_counts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load knowledge-base status: {e}") from e

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "Hybrid RAG Backend"}