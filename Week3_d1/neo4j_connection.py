from __future__ import annotations

import fitz  # PyMuPDF
import json
import logging
import os
import re
import certifi
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlsplit
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase, TrustCustomCAs
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedIngestionPipeline:
    def __init__(self, base_dir: str = "rag_knowledge_base"):
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent
        
        possible_dirs = [
            root_dir / base_dir,
            Path(base_dir),
            current_dir / base_dir
        ]
        self.base_dir = next((d for d in possible_dirs if d.exists()), root_dir / base_dir)
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        
        # Initialize Neo4j driver with certificate verification pattern
        self.driver = self._create_neo4j_driver(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        
        self.vector_db_path = root_dir / settings.VECTOR_DB_PATH if not Path(settings.VECTOR_DB_PATH).is_absolute() else Path(settings.VECTOR_DB_PATH)

    def _create_neo4j_driver(self, uri: str, username: str, password: str):
        parsed_uri = urlsplit(uri)
        if parsed_uri.scheme == "neo4j+s":
            trusted_uri = uri.replace("neo4j+s://", "neo4j://", 1)
            driver = GraphDatabase.driver(
                trusted_uri,
                auth=(username, password),
                encrypted=True,
                trusted_certificates=TrustCustomCAs(certifi.where()),
            )
        else:
            driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
            )
        driver.verify_connectivity()
        return driver

    def _sanitize_label(self, label: str) -> str:
        """Sanitize entity type to be a valid Neo4j label."""
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
        if not cleaned or not cleaned[0].isalpha():
            cleaned = "Entity_" + cleaned
        return cleaned

    def _sanitize_rel_type(self, rel: str) -> str:
        """Sanitize relationship type to be a valid uppercase Neo4j relationship type."""
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', rel.strip()).upper()
        return cleaned if cleaned else "RELATED_TO"

    def _extract_graph_data(self, text: str) -> Dict[str, Any]:
        prompt = f"""
        Extract technical entities (services, databases, components, protocols, teams) and their relationships from the text below.
        Return valid JSON strictly adhering to this schema:
        {{
            "entities": [{{"name": "string", "type": "string"}}],
            "relations": [{{"source": "string", "target": "string", "relation": "string"}}]
        }}
        Text: {text}
        """
        try:
            response = self.llm.invoke(prompt)
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"Failed to extract graph data: {e}")
            return {"entities": [], "relations": []}

    def run(self):
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at: {self.base_dir}")
            
        pdf_files = list(self.base_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found inside directory: {self.base_dir}")

        logger.info(f"Found PDF files to ingest: {[f.name for f in pdf_files]}")
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        with self.driver.session() as session:
            # 1. Clear all existing nodes and relationships in Neo4j before fresh ingestion
            logger.info("Clearing existing Neo4j knowledge graph database...")
            session.run("MATCH (n) DETACH DELETE n")

            for pdf_path in pdf_files:
                logger.info(f"Processing document: {pdf_path.name}")
                doc = fitz.open(pdf_path)
                
                # Create Document Node in Neo4j
                session.run(
                    """
                    MERGE (d:Document {url: $url})
                    SET d.title = $title
                    """,
                    url=f"file://{pdf_path.name}", title=pdf_path.stem
                )

                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if not text.strip():
                        continue
                    
                    chunks = splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        token_count = len(chunk.split())
                        chunk_id = f"{pdf_path.stem}_p{page_num+1}_c{i}"
                        
                        all_chunks.append({
                            "content": chunk,
                            "metadata": {
                                "source_file": pdf_path.name,
                                "page": page_num + 1,
                                "chunk_id": chunk_id,
                                "chunk_size": len(chunk),
                                "token_count": token_count
                            }
                        })

                    # Extract graph entities & relations via OpenAI for graph building
                    graph_data = self._extract_graph_data(text[:4000])
                    
                    for ent in graph_data.get("entities", []):
                        name = ent.get("name")
                        raw_type = ent.get("type", "Entity")
                        if not name:
                            continue
                        label = self._sanitize_label(raw_type)
                        session.run(
                            f"""
                            MERGE (e:`{label}` {{name: $name}})
                            MERGE (d:Document {{url: $url}})
                            MERGE (d)-[:MENTIONS]->(e)
                            """,
                            name=name, url=f"file://{pdf_path.name}"
                        )

                    for rel in graph_data.get("relations", []):
                        source = rel.get("source")
                        target = rel.get("target")
                        raw_rel = rel.get("relation", "RELATED_TO")
                        if not source or not target:
                            continue
                        rel_type = self._sanitize_rel_type(raw_rel)
                        session.run(
                            f"""
                            MERGE (s {{name: $source}})
                            MERGE (t {{name: $target}})
                            MERGE (s)-[:`{rel_type}`]->(t)
                            """,
                            source=source, target=target
                        )

        # 2. Persist Vector Store (Load existing if present and append vectors, otherwise create new)
        texts = [c["content"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]

        if self.vector_db_path.exists() and any(self.vector_db_path.iterdir()):
            logger.info(f"Loading existing vector store from {self.vector_db_path} and appending new vectors...")
            vector_store = FAISS.load_local(
                str(self.vector_db_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            vector_store.add_texts(texts=texts, metadatas=metadatas)
        else:
            logger.info("Creating a new FAISS vector store index...")
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )

        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.vector_db_path))
        logger.info("Combined Vector DB persistence and Neo4j Knowledge Graph ingestion completed successfully.")

    def close(self):
        if self.driver:
            self.driver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

if __name__ == "__main__":
    pipeline = CombinedIngestionPipeline()
    pipeline.run()