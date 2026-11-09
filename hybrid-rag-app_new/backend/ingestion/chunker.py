"""Split extracted document text into overlapping page-aware chunks."""
from typing import List, Dict, Any
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Splits loaded PDF documents and image descriptions into optimized, 
    overlapping text chunks while preserving source metadata.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain's recursive text splitter for semantic coherence
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def generate_chunk_id(self, source: str, page: int, chunk_index: int) -> str:
        """Generates a stable, unique identifier for a chunk based on its source and location."""
        raw_id = f"{source}_p{page}_c{chunk_index}"
        return hashlib.md5(raw_id.encode()).hexdigest()

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of raw document dicts (from PDFLoader) and returns 
        a flattened list of structured chunks with metadata and unique IDs.
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if not text.strip():
                continue
                
            # Split the text into smaller chunks
            text_chunks = self.text_splitter.split_text(text)
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                source = metadata.get("source", "unknown_source")
                page = metadata.get("page", 0)
                
                # Create a stable chunk ID for traceability
                chunk_id = self.generate_chunk_id(source, page, chunk_idx)
                
                # Clone metadata and enrich with chunk-specific info
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx
                chunk_metadata["chunk_id"] = chunk_id
                
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
        logger.info(f"Successfully chunked {len(documents)} raw source documents into {len(all_chunks)} fine-grained chunks.")
        return all_chunks