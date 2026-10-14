import os
import json
import difflib
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "based", "can", "could", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "tell", "that", "the", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "would",
}
_QUERY_TERM_ALIASES = {
    "tamilnadu": ("tamil", "nadu"),
}


class VectorStoreManager:
    """
    Manages embedding generation using OpenAI and local persistence/retrieval 
    using FAISS vector store with built-in idempotency and incremental updates.
    """
    def __init__(
        self, 
        index_dir: str = "vector_index", 
        embedding_model: str = "text-embedding-3-small",
        excluded_files: Optional[Set[str]] = None,
    ):
        self.index_dir = Path(index_dir)
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self._loaded_index_signature = None
        self.registry_file = self.index_dir / "processed_files.json"
        configured_exclusions = (
            excluded_files
            if excluded_files is not None
            else os.getenv(
                "EXCLUDED_PDF_FILES",
                "spotify_web_app_architecture.pdf",
            ).split(",")
        )
        self.excluded_files = {
            Path(filename.strip()).name.casefold()
            for filename in configured_exclusions
            if filename.strip()
        }

    def _is_excluded(self, filename: str) -> bool:
        return Path(filename).name.casefold() in self.excluded_files

    def _index_signature(self):
        index_file = self.index_dir / "index.faiss"
        try:
            stat = index_file.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _load_registry(self) -> set:
        """Load the set of already processed PDF filenames from local disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return {
                        filename
                        for filename in json.load(f)
                        if not self._is_excluded(filename)
                    }
            except Exception:
                return set()
        return set()

    def _save_registry(self, processed_files: set):
        """Save the updated set of processed PDF filenames to local disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(list(processed_files), f, indent=2)

    def load_vector_store(self) -> FAISS:
        """Load an existing FAISS vector store index from local disk."""
        index_file = self.index_dir / "index.faiss"
        if not self.index_dir.exists() or not index_file.exists():
            raise FileNotFoundError(
                f"No local FAISS index found at '{self.index_dir}'. "
                "Please run process_new_documents first."
            )
        
        print(f"Loading existing FAISS index from local disk ('{self.index_dir}')...")
        self.vector_store = FAISS.load_local(
            str(self.index_dir), 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        self._loaded_index_signature = self._index_signature()
        return self.vector_store

    def process_new_documents(
        self,
        extractor,
        data_dir: str = "data",
        filenames: Optional[Set[str]] = None,
    ) -> FAISS:
        """
        Idempotent incremental loader: Checks for newly added PDFs in the data folder,
        processes only those new files, generates embeddings, and updates the FAISS index.
        """
        discovered_pdfs = extractor.discover_pdfs()
        excluded_pdfs = [
            pdf for pdf in discovered_pdfs if self._is_excluded(pdf.name)
        ]
        if excluded_pdfs:
            print(
                "[Document Filter] Excluding configured PDF(s): "
                f"{[pdf.name for pdf in excluded_pdfs]}"
            )
        all_pdfs = [
            pdf for pdf in discovered_pdfs if not self._is_excluded(pdf.name)
        ]
        if filenames is not None:
            all_pdfs = [pdf for pdf in all_pdfs if pdf.name in filenames]
        processed_files = self._load_registry()

        # Identify only PDFs that haven't been embedded yet
        new_pdfs = [pdf for pdf in all_pdfs if pdf.name not in processed_files]

        if not new_pdfs:
            print("[Idempotency Check] No new PDFs found in 'data/' folder. Skipping embedding generation.")
            # Load existing store if not already loaded in memory
            if not self.vector_store:
                self.load_vector_store()
            return self.vector_store

        print(f"[Idempotency Check] Found {len(new_pdfs)} new PDF(s) to process: {[p.name for p in new_pdfs]}")

        # Extract chunks only from the new PDFs
        new_chunks = []
        for pdf_path in new_pdfs:
            print(f"Extracting chunks from new document: {pdf_path.name}...")
            chunks = extractor.process_pdf(pdf_path)
            new_chunks.extend(chunks)

        if not new_chunks:
            print("No valid text/visual chunks found in the new PDFs.")
            return self.load_vector_store() if (self.index_dir / "index.faiss").exists() else None

        # Convert chunks into LangChain Documents
        documents = []
        for chunk in new_chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "doc_name": chunk["doc_name"],
                    "page_num": chunk["page_num"],
                    "section": chunk["section"],
                    "content_type": chunk["content_type"]
                }
            )
            documents.append(doc)

        print(f"Generating embeddings for {len(documents)} new chunks via OpenAI API...")

        # Check if FAISS index already exists locally
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not self.vector_store:
            self.load_vector_store()

        if self.vector_store:
            # Incrementally add new documents to existing FAISS index
            print("Merging new embeddings into existing FAISS index...")
            self.vector_store.add_documents(documents)
        else:
            # Create a brand new FAISS index if none exists
            print("Creating brand new FAISS index from documents...")
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)

        # Persist updated index locally
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_dir))
        self._loaded_index_signature = self._index_signature()

        # Update and save the processed files registry
        for pdf in new_pdfs:
            processed_files.add(pdf.name)
        self._save_registry(processed_files)

        print(f"FAISS index successfully updated and persisted at {self.index_dir.resolve()}")
        print(f"Updated processed files ledger: {list(processed_files)}")
        return self.vector_store

    def rebuild_index(
        self,
        extractor,
        data_dir: str = "data",
        reprocess_visuals: bool = False,
    ) -> FAISS:
        """Build a fresh index and swap it in only after successful ingestion."""
        self.index_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.index_dir.name}.staging-",
                dir=str(self.index_dir.parent),
            )
        )
        backup_dir = staging_dir.with_name(f"{staging_dir.name}.backup")
        staging_manager = VectorStoreManager(
            index_dir=str(staging_dir),
            excluded_files=self.excluded_files,
        )
        staging_manager.embedding_model = self.embedding_model
        discovered_pdfs = extractor.discover_pdfs()
        excluded_pdfs = [
            pdf for pdf in discovered_pdfs if self._is_excluded(pdf.name)
        ]
        if excluded_pdfs:
            print(
                "[Document Filter] Excluding configured PDF(s): "
                f"{[pdf.name for pdf in excluded_pdfs]}"
            )
        source_pdfs = [
            pdf for pdf in discovered_pdfs if not self._is_excluded(pdf.name)
        ]
        active_pdf_names = {pdf.name for pdf in source_pdfs}
        preserved_visual_chunks = {}

        if (
            not reprocess_visuals
            and active_pdf_names
            and (self.index_dir / "index.faiss").exists()
        ):
            existing_store = self.vector_store or self.load_vector_store()
            for doc_id in existing_store.index_to_docstore_id.values():
                doc = existing_store.docstore.search(doc_id)
                if not isinstance(doc, Document):
                    continue
                source_name = doc.metadata.get("doc_name", "")
                if (
                    source_name in active_pdf_names
                    and doc.metadata.get("content_type") == "visual_chart"
                    and "vision extraction failed" not in doc.page_content.casefold()
                ):
                    preserved_visual_chunks.setdefault(source_name, []).append({
                        "doc_name": source_name,
                        "page_num": doc.metadata.get("page_num", 0),
                        "section": doc.metadata.get("section", "Visual/Chart"),
                        "content_type": "visual_chart",
                        "content": doc.page_content,
                    })

        class FilteredExtractor:
            def discover_pdfs(inner_self):
                return source_pdfs

            def process_pdf(inner_self, pdf_path):
                chunks = extractor.process_pdf(
                    pdf_path,
                    include_visuals=reprocess_visuals,
                )
                chunks.extend(preserved_visual_chunks.get(pdf_path.name, []))
                return chunks

        try:
            rebuilt_store = staging_manager.process_new_documents(
                FilteredExtractor(),
                data_dir=data_dir,
            )
            if rebuilt_store is None:
                raise RuntimeError(
                    "No PDF content was extracted; the existing index was left unchanged."
                )

            if self.index_dir.exists():
                self.index_dir.rename(backup_dir)
            try:
                staging_dir.rename(self.index_dir)
            except OSError:
                if backup_dir.exists() and not self.index_dir.exists():
                    backup_dir.rename(self.index_dir)
                raise

            self.vector_store = staging_manager.vector_store
            self.registry_file = self.index_dir / "processed_files.json"
            self._loaded_index_signature = self._index_signature()
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            print(f"FAISS index rebuilt successfully at {self.index_dir.resolve()}")
            return self.vector_store
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Combine semantic retrieval with exact keyword matches from the local index."""
        if (
            not self.vector_store
            or getattr(self, "_loaded_index_signature", None)
            != self._index_signature()
        ):
            self.load_vector_store()
        
        print(f"Executing hybrid search for query: '{query}' (Top {k} results)...")
        semantic_results = [
            doc
            for doc in self.vector_store.similarity_search(query, k=k)
            if not self._is_excluded(doc.metadata.get("doc_name", ""))
        ]

        documents = []
        term_document_frequency = {}
        for doc_id in self.vector_store.index_to_docstore_id.values():
            doc = self.vector_store.docstore.search(doc_id)
            if not isinstance(doc, Document):
                continue
            if self._is_excluded(doc.metadata.get("doc_name", "")):
                continue

            content_terms = set(
                re.findall(r"\b[\w'-]+\b", doc.page_content.casefold())
            )
            documents.append((doc, content_terms))
            for term in content_terms:
                term_document_frequency[term] = (
                    term_document_frequency.get(term, 0) + 1
                )

        vocabulary = set(term_document_frequency)
        query_terms = set()
        for term in re.findall(r"\b[\w'-]+\b", query.casefold()):
            if term in _QUERY_STOP_WORDS or len(term) <= 1:
                continue
            aliases = _QUERY_TERM_ALIASES.get(term)
            if aliases:
                query_terms.update(aliases)
                continue
            if term in vocabulary:
                query_terms.add(term)
                continue

            correction = difflib.get_close_matches(
                term,
                vocabulary,
                n=1,
                cutoff=0.82,
            )
            query_terms.add(correction[0] if correction else term)

        if not query_terms:
            return semantic_results[:k]

        min_matches = max(1, math.ceil(len(query_terms) * 0.6))
        keyword_results = []
        document_count = max(len(documents), 1)
        for doc, content_terms in documents:
            matched_terms = query_terms & content_terms
            if len(matched_terms) < min_matches:
                continue
            inverse_document_frequency = sum(
                math.log(
                    1
                    + (document_count - term_document_frequency[term] + 0.5)
                    / (term_document_frequency[term] + 0.5)
                )
                for term in matched_terms
            )
            keyword_results.append(
                (
                    inverse_document_frequency,
                    len(matched_terms),
                    doc.metadata.get("doc_name", ""),
                    doc,
                )
            )

        keyword_results.sort(
            key=lambda match: (
                -match[0],
                -match[1],
                match[2],
                match[3].metadata.get("page_num", 0),
            )
        )

        combined_results = []
        seen = set()
        for doc in (
            [match[3] for match in keyword_results]
            + semantic_results
        ):
            key = (
                doc.page_content,
                doc.metadata.get("doc_name"),
                doc.metadata.get("page_num"),
                doc.metadata.get("section"),
            )
            if key not in seen:
                seen.add(key)
                combined_results.append(doc)
            if len(combined_results) == k:
                break

        return combined_results