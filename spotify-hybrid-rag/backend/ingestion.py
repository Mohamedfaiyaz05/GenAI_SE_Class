from __future__ import annotations

import json
import logging
import os
import re
import certifi
from pathlib import Path
from typing import List, Dict, Any
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase

if __package__:
    from .config import settings
else:
    from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_EXTRACTION_BATCH_CHARS = 18_000


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
        
        # Keep the URI scheme intact; neo4j+s enables Aura TLS and routing.
        self.driver = self._create_neo4j_driver(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        
        self.vector_db_path = root_dir / settings.VECTOR_DB_PATH if not Path(settings.VECTOR_DB_PATH).is_absolute() else Path(settings.VECTOR_DB_PATH)

    def _create_neo4j_driver(self, uri: str, username: str, password: str):
        if not uri:
            raise ValueError("NEO4J_URI is empty; set it in the project .env file.")
        if not username or not password:
            raise ValueError(
                "Neo4j credentials are missing; set NEO4J_USERNAME and "
                "NEO4J_PASSWORD in the project .env file."
            )

        ca_bundle = (
            os.getenv("NEO4J_CA_BUNDLE")
            or os.getenv("SSL_CERT_FILE")
            or certifi.where()
        )

        print('os.getenv("NEO4J_CA_BUNDLE")===================================>', os.getenv("NEO4J_CA_BUNDLE"))
        print('os.getenv("SSL_CERT_FILE")===================================>', os.getenv("SSL_CERT_FILE"))

        if not Path(ca_bundle).is_file():
            raise FileNotFoundError(
                f"Neo4j CA bundle does not exist: {ca_bundle}"
            )
        os.environ["SSL_CERT_FILE"] = ca_bundle

        driver = GraphDatabase.driver(uri, auth=(username, password))

        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            logger.exception(
                "Neo4j connectivity check failed. The URI scheme was preserved "
                "and certificate verification remains enabled."
            )
            raise
        return driver

    def _sanitize_rel_type(self, rel: str) -> str:
        """Sanitize relationship type to be a valid uppercase Neo4j relationship type."""
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', rel.strip()).upper()
        return cleaned if cleaned else "RELATED_TO"

    def _extract_graph_data(self, text: str, document_names: List[str]) -> Dict[str, Any]:
        prompt = f"""
        Extract the important entities and relationships from all source documents in this batch.
        Use the exact document filenames listed here in each item's source_documents field:
        {json.dumps(document_names, ensure_ascii=False)}

        Return valid JSON strictly adhering to this schema:
        {{
            "entities": [{{
                "name": "string",
                "type": "string",
                "source_documents": ["exact filename"]
            }}],
            "relations": [{{
                "source": "string",
                "target": "string",
                "relation": "string",
                "source_documents": ["exact filename"]
            }}]
        }}

        Only include facts supported by the text. Do not invent entities or relationships.
        Text from source documents:
        {text}
        """
        try:
            response = self.llm.invoke(prompt)
            if not isinstance(response.content, str):
                raise ValueError("The model returned non-text content instead of JSON.")
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            graph_data = json.loads(clean_content)
            if not isinstance(graph_data, dict):
                raise ValueError("The model response must be a JSON object.")
            if not isinstance(graph_data.get("entities", []), list):
                raise ValueError("'entities' must be a list.")
            if not isinstance(graph_data.get("relations", []), list):
                raise ValueError("'relations' must be a list.")
            return graph_data
        except Exception as e:
            logger.exception("Failed to extract graph data from an aggregated PDF batch.")
            raise RuntimeError("Graph extraction failed; ingestion stopped.") from e

    @staticmethod
    def _entity_key(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip().casefold()

    def _extract_all_graph_data(
        self, documents: List[Dict[str, str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        graph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=GRAPH_EXTRACTION_BATCH_CHARS - 500,
            chunk_overlap=300,
        )
        batches: List[Dict[str, Any]] = []

        for document in documents:
            for part in graph_splitter.split_text(document["text"]):
                section = f"SOURCE DOCUMENT: {document['name']}\n{part}"
                if not batches or (
                    len(batches[-1]["text"]) + len(section) + 2 > GRAPH_EXTRACTION_BATCH_CHARS
                ):
                    batches.append({"text": section, "documents": [document["name"]]})
                else:
                    batches[-1]["text"] += f"\n\n{section}"
                    if document["name"] not in batches[-1]["documents"]:
                        batches[-1]["documents"].append(document["name"])

        entities: Dict[str, Dict[str, Any]] = {}
        relations: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        known_names = {doc["name"].casefold(): doc["name"] for doc in documents}

        for batch_number, batch in enumerate(batches, start=1):
            logger.info(
                "Extracting graph data from batch %d of %d (%d source document(s))",
                batch_number, len(batches), len(batch["documents"])
            )
            graph_data = self._extract_graph_data(batch["text"], batch["documents"])

            def source_documents(item: Dict[str, Any]) -> set[str]:
                raw_sources = item.get("source_documents", [])
                if isinstance(raw_sources, str):
                    raw_sources = [raw_sources]
                if not isinstance(raw_sources, list):
                    return set()
                matched = {
                    known_names[source.casefold()]
                    for source in raw_sources
                    if isinstance(source, str)
                    and source.casefold() in known_names
                    and known_names[source.casefold()] in batch["documents"]
                }
                if not matched and len(batch["documents"]) == 1:
                    matched.add(batch["documents"][0])
                return matched

            for item in graph_data.get("entities", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                name = re.sub(r"\s+", " ", name).strip()
                key = self._entity_key(name)
                entity_type = item.get("type", "Entity")
                if not isinstance(entity_type, str) or not entity_type.strip():
                    entity_type = "Entity"
                record = entities.setdefault(
                    key,
                    {
                        "key": key,
                        "name": name,
                        "type": entity_type.strip(),
                        "types": set(),
                        "source_documents": set(),
                    },
                )
                record["types"].add(entity_type.strip())
                record["source_documents"].update(source_documents(item))

            for item in graph_data.get("relations", []):
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                target = item.get("target")
                relation = item.get("relation", "RELATED_TO")
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (source, target, relation)
                ):
                    continue

                source = re.sub(r"\s+", " ", source).strip()
                target = re.sub(r"\s+", " ", target).strip()
                relation = relation.strip()
                source_key = self._entity_key(source)
                target_key = self._entity_key(target)
                for key, name in ((source_key, source), (target_key, target)):
                    if key not in entities:
                        entities[key] = {
                            "key": key,
                            "name": name,
                            "type": "Entity",
                            "types": {"Entity"},
                            "source_documents": set(),
                        }
                relation_key = (
                    source_key,
                    self._sanitize_rel_type(relation),
                    target_key,
                )
                record = relations.setdefault(
                    relation_key,
                    {
                        "source_key": source_key,
                        "target_key": target_key,
                        "type": relation_key[1],
                        "source_documents": set(),
                    },
                )
                relation_sources = source_documents(item)
                record["source_documents"].update(relation_sources)
                entities[source_key]["source_documents"].update(relation_sources)
                entities[target_key]["source_documents"].update(relation_sources)

        return {
            "entities": [
                {
                    **entity,
                    "types": sorted(entity["types"]),
                    "source_documents": sorted(entity["source_documents"]),
                }
                for entity in entities.values()
            ],
            "relations": [
                {**relation, "source_documents": sorted(relation["source_documents"])}
                for relation in relations.values()
            ],
        }

    @staticmethod
    def _replace_knowledge_graph(
        tx,
        documents: List[Dict[str, str]],
        graph_data: Dict[str, List[Dict[str, Any]]],
    ):
        tx.run("MATCH (n) DETACH DELETE n").consume()
        tx.run(
            """
            UNWIND $documents AS document
            CREATE (:Document {url: document.url, title: document.name})
            """,
            documents=documents,
        ).consume()
        tx.run(
            """
            UNWIND $entities AS entity
            MERGE (e:Entity {key: entity.key})
            SET e.name = entity.name,
                e.type = entity.type,
                e.types = entity.types
            """,
            entities=graph_data["entities"],
        ).consume()
        tx.run(
            """
            UNWIND $entities AS entity
            UNWIND entity.source_documents AS source_name
            MATCH (d:Document {title: source_name})
            MATCH (e:Entity {key: entity.key})
            MERGE (d)-[:MENTIONS]->(e)
            """,
            entities=graph_data["entities"],
        ).consume()
        relation_types = {relation["type"] for relation in graph_data["relations"]}
        for relation_type in relation_types:
            typed_relations = [
                relation
                for relation in graph_data["relations"]
                if relation["type"] == relation_type
            ]
            tx.run(
                f"""
                UNWIND $relations AS relation
                MATCH (source:Entity {{key: relation.source_key}})
                MATCH (target:Entity {{key: relation.target_key}})
                MERGE (source)-[:`{relation_type}`]->(target)
                """,
                relations=typed_relations,
            ).consume()

    def run(self):
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at: {self.base_dir}")
            
        pdf_files = list(self.base_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found inside directory: {self.base_dir}")

        logger.info(f"Found PDF files to ingest: {[f.name for f in pdf_files]}")
        all_chunks = []
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for pdf_path in pdf_files:
            logger.info("Reading document: %s", pdf_path.name)
            with pymupdf.open(pdf_path) as doc:
                page_texts = [page.get_text() for page in doc]
            document_text = "\n\n".join(text.strip() for text in page_texts if text.strip())
            if not document_text:
                logger.warning("No extractable text in %s; skipping it.", pdf_path.name)
                continue

            documents.append({
                "name": pdf_path.name,
                "url": pdf_path.resolve().as_uri(),
                "text": document_text,
            })

            for page_num, text in enumerate(page_texts):
                if not text.strip():
                    continue
                for chunk_index, chunk in enumerate(splitter.split_text(text)):
                    all_chunks.append({
                        "content": chunk,
                        "metadata": {
                            "source_file": pdf_path.name,
                            "page": page_num + 1,
                            "chunk_id": f"{pdf_path.stem}_p{page_num + 1}_c{chunk_index}",
                            "chunk_size": len(chunk),
                            "token_count": len(chunk.split()),
                        },
                    })

        texts = [c["content"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        if not texts:
            raise ValueError(
                f"No extractable text found in the PDF files under {self.base_dir}."
            )

        graph_data = self._extract_all_graph_data(documents)
        if not graph_data["entities"]:
            raise ValueError(
                "No graph entities were extracted; the existing Neo4j graph was left unchanged."
            )

        logger.info("Building a fresh FAISS vector index from the current PDF collection...")
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.vector_db_path))

        graph_documents = [{"name": doc["name"], "url": doc["url"]} for doc in documents]
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(
                """
                CREATE CONSTRAINT entity_key_unique IF NOT EXISTS
                FOR (entity:Entity) REQUIRE entity.key IS UNIQUE
                """
            ).consume()
            session.execute_write(
                self._replace_knowledge_graph,
                graph_documents,
                graph_data,
            )

        logger.info(
            "Ingestion completed: %d document(s), %d unique entity node(s), "
            "%d unique relationship(s), and %d text chunk(s).",
            len(documents), len(graph_data["entities"]), len(graph_data["relations"]), len(texts)
        )

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