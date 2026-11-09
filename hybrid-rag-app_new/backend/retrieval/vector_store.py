"""Create, persist, and search the FAISS vector index."""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from retrieval.embeddings import get_embedding_model
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages FAISS vector index creation, persistence, similarity search, 
    and incremental file synchronization based on content hashing.
    """
    def __init__(self, index_path: str = None):
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.state_file_path = self.index_path / "ingested_files_state.json"
        self.embeddings = get_embedding_model()
        self.vector_store: Optional[FAISS] = None
        self.ingested_state: Dict[str, str] = {} # {filename: file_hash}
        
        self.load_index_and_state()

    def compute_file_hash(self, file_path: Path) -> str:
        """Computes an MD5 hash of a file's content to detect changes."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def load_index_and_state(self) -> None:
        """Loads the FAISS index and the ingested files state from disk if they exist."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Load state tracking file
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r") as f:
                    state = json.load(f)
                if not isinstance(state, dict) or not all(
                    isinstance(filename, str) and isinstance(file_hash, str)
                    for filename, file_hash in state.items()
                ):
                    raise ValueError("Ingestion state must map filenames to hashes.")
                self.ingested_state = state
                logger.info(f"Loaded ingestion state with {len(self.ingested_state)} tracked file(s).")
            except (OSError, json.JSONDecodeError, ValueError):
                logger.exception("Failed to load ingestion state file %s.", self.state_file_path)
                self.ingested_state = {}

        # Load FAISS index
        if (self.index_path / "index.faiss").is_file() and (self.index_path / "index.pkl").is_file():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_path), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Successfully loaded persisted FAISS index from {self.index_path}")
            except Exception:
                logger.exception("Failed to load FAISS index from %s.", self.index_path)
                self.vector_store = None
        else:
            logger.info(f"No existing FAISS index found at {self.index_path}. A new one will be created.")
            self.vector_store = None

    def save_state(self) -> None:
        """Persists the ingestion state dictionary to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        temporary_state_path = self.state_file_path.with_suffix(".json.tmp")
        try:
            with temporary_state_path.open("w", encoding="utf-8") as f:
                json.dump(self.ingested_state, f, indent=4)
            temporary_state_path.replace(self.state_file_path)
        except Exception:
            temporary_state_path.unlink(missing_ok=True)
            logger.exception("Failed to save ingestion state to %s.", self.state_file_path)
            raise
        logger.info("Successfully updated ingestion state file.")

    def save_index(self) -> None:
        """Persists the current FAISS index to local disk."""
        if self.vector_store:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_path))
            self.save_state()
            logger.info(f"Successfully saved FAISS index and state to {self.index_path}")

    def sync_knowledge_base(self, pdf_loader, chunker) -> int:
        """
        Synchronizes the persisted FAISS index with the complete PDF knowledge base.
        Rebuilding on any source change prevents stale chunks for edited or removed PDFs.
        """
        pdf_files = pdf_loader.discover_pdfs()
        if not pdf_files:
            logger.info("No PDF files found in knowledge base during sync.")
            self.vector_store = None
            self.ingested_state = {}
            for filename in ("index.faiss", "index.pkl", self.state_file_path.name):
                (self.index_path / filename).unlink(missing_ok=True)
            return 0

        current_files_map = {}
        rebuild_required = (
            self.vector_store is None
            or not self.vector_store.index_to_docstore_id
        )

        for file_path in pdf_files:
            filename = file_path.name
            file_hash = self.compute_file_hash(file_path)
            current_files_map[filename] = file_hash

        changed_files = {
            filename
            for filename, file_hash in current_files_map.items()
            if self.ingested_state.get(filename) != file_hash
        }
        removed_files = set(self.ingested_state) - set(current_files_map)
        rebuild_required = rebuild_required or bool(changed_files or removed_files)
        if not rebuild_required:
            logger.info("Knowledge base is up-to-date; loaded FAISS index remains unchanged.")
            return 0

        logger.info(
            "Rebuilding the semantic index; changed=%s removed=%s",
            sorted(changed_files),
            sorted(removed_files),
        )

        new_raw_docs = []
        for file_path in pdf_files:
            docs = pdf_loader.load_pdf(file_path)
            if not docs:
                raise ValueError(
                    f"PDF loader returned no searchable content for {file_path.name}."
                )
            new_raw_docs.extend(docs)

        if not new_raw_docs:
            raise ValueError("PDF files exist, but the loader extracted no searchable text.")

        new_chunks = chunker.chunk_documents(new_raw_docs)
        if not new_chunks:
            raise ValueError("PDF files were loaded, but chunking produced no searchable text.")

        logger.info(
            "Embedding %d semantic chunks from %d PDF(s).",
            len(new_chunks),
            len(pdf_files),
        )
        documents = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in new_chunks
        ]
        updated_vector_store = FAISS.from_documents(documents, self.embeddings)
        previous_vector_store = self.vector_store
        previous_ingested_state = self.ingested_state
        self.vector_store = updated_vector_store
        self.ingested_state = current_files_map
        try:
            self.save_index()
        except Exception:
            self.vector_store = previous_vector_store
            self.ingested_state = previous_ingested_state
            raise

        logger.info(
            "Successfully synchronized and persisted %d semantic chunks from %d PDF(s).",
            len(updated_vector_store.index_to_docstore_id),
            len(pdf_files),
        )
        return len(pdf_files)

    def similarity_search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """
        Performs a semantic similarity search against the FAISS index,
        returning top-k chunks with their text, metadata, and distance scores.
        """
        if not self.vector_store:
            logger.warning("FAISS vector store is empty or not initialized.")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error during FAISS similarity search: {e}")
            return []