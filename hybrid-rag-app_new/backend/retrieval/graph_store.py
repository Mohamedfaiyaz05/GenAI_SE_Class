import os
import json
import logging
import re
import hashlib
import certifi
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
import pymupdf
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized chunk sizing to ensure deep, granular extraction without hitting token limits
GRAPH_CHUNK_SIZE = 8_000
GRAPH_CHUNK_OVERLAP = 500
GRAPH_EXTRACTION_PROMPT_VERSION = "domain-neutral-v2"

class Neo4jGraphManager:
    """
    Manages Neo4j connections, TLS verification, MD5 incremental file state tracking,
    empty-database fallback override, zero-drop chunking, custom ontological JSON graph extraction, 
    and atomic transactional graph persistence.
    """
    def __init__(self, uri: str = None, username: str = None, password: str = None, state_dir: str = None):
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.database = getattr(settings, "NEO4J_DATABASE", "neo4j")
        
        project_root = Path(__file__).resolve().parent.parent.parent
        self.state_dir = Path(state_dir or (project_root / "faiss_index"))
        self.state_file_path = self.state_dir / "graph_ingested_state.json"
        self.ingested_state: Dict[str, str] = {}
        
        # Configure TLS certificate bundle
        ca_bundle = os.getenv("NEO4J_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or certifi.where()
        if Path(ca_bundle).is_file():
            os.environ["SSL_CERT_FILE"] = ca_bundle

        # Initialize LLM with gpt-4o-mini for direct structured JSON extraction
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize official Neo4j driver
        self.driver = self._create_neo4j_driver(self.uri, self.username, self.password)
        self.load_state()

    def _create_neo4j_driver(self, uri: str, username: str, password: str):
        if not uri:
            raise ValueError("NEO4J_URI is empty; set it in the project .env file.")
        if not username or not password:
            raise ValueError("Neo4j credentials are missing; set NEO4J_USER and NEO4J_PASSWORD in the project .env file[cite: 1].")

        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            logger.exception("Neo4j connectivity check failed.")
            raise
        return driver

    def compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        hasher.update(GRAPH_EXTRACTION_PROMPT_VERSION.encode("utf-8"))
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def load_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r") as f:
                    self.ingested_state = json.load(f)
                logger.info(f"Loaded graph ingestion state with {len(self.ingested_state)} tracked file(s).")
            except Exception as e:
                logger.error(f"Failed to load graph ingestion state file: {e}")
                self.ingested_state = {}

    def save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary_state_path = self.state_file_path.with_suffix(".json.tmp")
        try:
            with temporary_state_path.open("w", encoding="utf-8") as f:
                json.dump(self.ingested_state, f, indent=4)
            temporary_state_path.replace(self.state_file_path)
        except Exception:
            temporary_state_path.unlink(missing_ok=True)
            logger.exception("Failed to save graph ingestion state to %s.", self.state_file_path)
            raise
        logger.info("Successfully updated graph ingestion state file.")

    def _sanitize_rel_type(self, rel: str) -> str:
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', rel.strip()).upper()
        return cleaned if cleaned else "RELATED_TO"

    @staticmethod
    def _entity_key(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip().casefold()

    def _extract_graph_data(self, text: str, document_names: List[str]) -> Dict[str, Any]:
        prompt = f"""
        Extract a useful, source-grounded knowledge graph from the provided document text,
        whatever
        its subject or document type. do not assume a particular industry, system, technology
        stack, or document format.

        Extract named people, organizations, places, concepts, products, services, processes,
        events, technologies, data stores, components, and other meaningful domain-specific
        entities. Prefer specific canonical names and include concise source-grounded descriptions that retain
        important qualifications stated by the source. Avoid ordinary words, duplicates, and
        unsupported inferences.

        Assign each entity one specific, concise type appropriate to the source (for example
        PERSON, ORGANIZATION, SERVICE, TECHNOLOGY, DATA_STORE, PROCESS, EVENT, or CONCEPT).
        Use another specific type when the examples do not fit; do not force every document
        into a fixed ontology.

        Extract directed relationships only when the text supports them. Use short uppercase
        underscore verbs (for example USES, PART_OF, LOCATED_IN, PRODUCES, SUPPORTS, or
        CAUSED_BY), preserve source-to-target direction, and include a short evidence phrase.
        Do not infer a connection from entities merely appearing together.

        Use only exact document filenames listed here in each item's source_documents field:
        {json.dumps(document_names, ensure_ascii=False)}

        Return valid JSON only, with no markdown, following this schema:
        {{
            "entities": [
                {{
                    "name": "Canonical entity name",
                    "type": "Specific category",
                    "description": "Concise source-grounded description",
                    "source_documents": ["exact filename"]
                }}
            ],
            "relations": [
                {{
                    "source": "Source entity name",
                    "target": "Target entity name",
                    "relation": "UPPERCASE_VERB",
                    "evidence": "Short supporting phrase from the text",
                    "source_documents": ["exact filename"]
                }}
            ]
        }}

        Guidelines:
        1. Include important facts and relationships, but do not extract every noun or fragment.
        2. Keep entity names consistent and use only information explicitly supported by text.
        3. If a source cannot be confidently assigned, use an empty source_documents list.

        Text from source documents:
        {text}
        """
        try:
            response = self.llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            clean_content = content.replace("```json", "").replace("```", "").strip()
            graph_data = json.loads(clean_content)
            if not isinstance(graph_data, dict) or not isinstance(graph_data.get("entities", []), list) or not isinstance(graph_data.get("relations", []), list):
                raise ValueError("The model response must be a JSON object with 'entities' and 'relations' lists.")
            return graph_data
        except Exception as e:
            logger.exception("Failed to extract graph data from text batch.")
            raise RuntimeError("Graph extraction failed.") from e

    def _extract_all_graph_data(self, documents: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Zero-drop chunking & batching strategy:
        Slices every document into overlapping chunks using RecursiveCharacterTextSplitter,
        ensuring 100% text coverage without dropping any paragraphs or sections.
        """
        graph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=GRAPH_CHUNK_SIZE,
            chunk_overlap=GRAPH_CHUNK_OVERLAP,
        )
        batches: List[Dict[str, Any]] = []

        for document in documents:
            doc_chunks = graph_splitter.split_text(document["text"])
            total_chunks = len(doc_chunks)
            logger.info(f"📄 Document '{document['name']}' split into {total_chunks} chunk(s) for graph extraction.")
            
            for idx, chunk_text in enumerate(doc_chunks, start=1):
                section = f"SOURCE DOCUMENT: {document['name']} (Part {idx}/{total_chunks})\n{chunk_text}"
                batches.append({
                    "text": section,
                    "documents": [document["name"]]
                })

        entities: Dict[str, Dict[str, Any]] = {}
        relations: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        known_names = {doc["name"].casefold(): doc["name"] for doc in documents}

        for batch_number, batch in enumerate(batches, start=1):
            logger.info("Extracting graph data from chunk %d of %d", batch_number, len(batches))
            graph_data = self._extract_graph_data(batch["text"], batch["documents"])

            def source_documents(item: Dict[str, Any]) -> set[str]:
                raw_sources = item.get("source_documents", [])
                if isinstance(raw_sources, str):
                    raw_sources = [raw_sources]
                if not isinstance(raw_sources, list):
                    return set()
                return {
                    known_names[source.casefold()]
                    for source in raw_sources
                    if isinstance(source, str) and source.casefold() in known_names and known_names[source.casefold()] in batch["documents"]
                } or set(batch["documents"])

            for item in graph_data.get("entities", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                name = re.sub(r"\s+", " ", name).strip()
                key = self._entity_key(name)
                entity_type = item.get("type", "Concept")
                description = item.get("description", "")
                
                record = entities.setdefault(key, {
                    "key": key,
                    "name": name,
                    "type": entity_type.strip(),
                    "types": set(),
                    "description": description.strip(),
                    "source_documents": set(),
                })
                normalized_type = entity_type.strip()
                record["types"].add(normalized_type)
                if record["type"] == "Concept" or len(normalized_type) > len(record["type"]):
                    record["type"] = normalized_type
                if description and len(description.strip()) > len(record["description"]):
                    record["description"] = description.strip()
                record["source_documents"].update(source_documents(item))

            for item in graph_data.get("relations", []):
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                target = item.get("target")
                relation = item.get("relation", "RELATED_TO")
                evidence = item.get("evidence", "")
                
                if not all(isinstance(v, str) and v.strip() for v in (source, target, relation)):
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
                            "type": "Concept",
                            "types": {"Concept"},
                            "description": "",
                            "source_documents": set(),
                        }

                relation_key = (source_key, self._sanitize_rel_type(relation), target_key)
                record = relations.setdefault(relation_key, {
                    "source_key": source_key,
                    "target_key": target_key,
                    "type": relation_key[1],
                    "evidence": [],
                    "source_documents": set(),
                })
                rel_sources = source_documents(item)
                evidence_text = evidence.strip()
                if evidence_text and evidence_text not in record["evidence"]:
                    record["evidence"].append(evidence_text)
                record["source_documents"].update(rel_sources)
                entities[source_key]["source_documents"].update(rel_sources)
                entities[target_key]["source_documents"].update(rel_sources)

        return {
            "entities": [{**e, "types": sorted(e["types"]), "source_documents": sorted(e["source_documents"])} for e in entities.values()],
            "relations": [{**r, "source_documents": sorted(r["source_documents"])} for r in relations.values()],
        }

    @staticmethod
    def _write_knowledge_graph(tx, documents: List[Dict[str, str]], graph_data: Dict[str, List[Dict[str, Any]]]):
        tx.run(
            """
            UNWIND $documents AS document
            MERGE (d:Document {url: document.url})
            SET d.title = document.name
            """,
            documents=documents,
        ).consume()
        tx.run(
            """
            UNWIND $entities AS entity
            MERGE (e:Entity {key: entity.key})
            SET e.name = entity.name,
                e.type = entity.type,
                e.types = entity.types,
                e.description = entity.description
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
        
        for relation_type in {r["type"] for r in graph_data["relations"]}:
            typed_rels = [r for r in graph_data["relations"] if r["type"] == relation_type]
            tx.run(
                f"""
                UNWIND $relations AS relation
                MATCH (source:Entity {{key: relation.source_key}})
                MATCH (target:Entity {{key: relation.target_key}})
                MERGE (source)-[r:`{relation_type}`]->(target)
                SET r.evidence = relation.evidence,
                    r.source_documents = relation.source_documents
                """,
                relations=typed_rels,
            ).consume()

    def get_graph_snapshot(
        self,
        limit: int = 100,
        node_skip: int = 0,
        relationship_skip: int = 0,
    ) -> Dict[str, Any]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")
        if node_skip < 0 or relationship_skip < 0:
            raise ValueError("skip values must be non-negative.")

        with self.driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()
            relationship_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()
            nodes = session.run(
                """
                MATCH (n)
                RETURN toString(id(n)) AS id, labels(n) AS labels,
                       properties(n) AS properties
                ORDER BY id(n)
                SKIP $skip LIMIT $limit
                """,
                skip=node_skip,
                limit=limit,
            )
            relationships = session.run(
                """
                MATCH (source)-[r]->(target)
                RETURN toString(id(r)) AS id, type(r) AS type,
                       properties(r) AS properties,
                       {id: toString(id(source)), labels: labels(source),
                        properties: properties(source)} AS source,
                       {id: toString(id(target)), labels: labels(target),
                        properties: properties(target)} AS target
                ORDER BY id(r)
                SKIP $skip LIMIT $limit
                """,
                skip=relationship_skip,
                limit=limit,
            )
            return {
                "totals": {
                    "nodes": node_count["count"] if node_count else 0,
                    "relationships": relationship_count["count"] if relationship_count else 0,
                },
                "nodes": [record.data() for record in nodes],
                "relationships": [record.data() for record in relationships],
            }

    def sync_knowledge_base_graph(self, pdf_loader) -> int:
        pdf_files = pdf_loader.discover_pdfs() if hasattr(pdf_loader, "discover_pdfs") else list(Path(pdf_loader.base_dir).glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in knowledge base for graph sync.")
            return 0

        # Check if Neo4j database is empty
        neo4j_is_empty = False
        try:
            with self.driver.session(database=self.database) as session:
                res = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()
                if not res or res["count"] == 0:
                    neo4j_is_empty = True
        except Exception:
            logger.exception("Could not determine whether the Neo4j entity graph is empty.")
            raise

        if neo4j_is_empty:
            logger.info("⚠️ Neo4j database is empty. Bypassing incremental checks and processing all files.")

        current_files_map = {}
        files_to_process = []

        for file_path in pdf_files:
            filename = file_path.name
            file_hash = self.compute_file_hash(file_path)
            current_files_map[filename] = file_hash

            if neo4j_is_empty:
                logger.info(f"Forcing processing of file due to empty Neo4j database: {filename}")
                files_to_process.append(file_path)
            elif filename not in self.ingested_state:
                logger.info(f"Detected NEW document for graph: {filename}")
                files_to_process.append(file_path)
            elif self.ingested_state[filename] != file_hash:
                logger.info(f"Detected MODIFIED document for graph: {filename}")
                files_to_process.append(file_path)
            else:
                logger.info(f"Graph document unchanged (skipping LLM extraction): {filename}")

        if not files_to_process:
            logger.info("✨ Knowledge base graph is fully up-to-date. No new or modified documents to process.")
            return 0

        documents_to_process = []
        for file_path in files_to_process:
            with pymupdf.open(file_path) as doc:
                page_texts = [page.get_text() for page in doc]
            document_text = "\n\n".join(text.strip() for text in page_texts if text.strip())
            if not document_text:
                logger.warning(f"No extractable text in {file_path.name}; skipping.")
                continue
            documents_to_process.append({
                "name": file_path.name,
                "url": file_path.resolve().as_uri(),
                "text": document_text,
            })

        if not documents_to_process:
            raise ValueError(
                "Graph sync found PDFs, but none contained extractable text."
            )

        logger.info(f"Extracting custom ontological graph data with zero-drop chunking for {len(documents_to_process)} file(s)...")
        graph_data = self._extract_all_graph_data(documents_to_process)
        if not graph_data["entities"]:
            raise ValueError(
                "Graph extraction returned no entities; refusing to report a successful sync."
            )

        graph_documents = [{"name": doc["name"], "url": doc["url"]} for doc in documents_to_process]

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                CREATE CONSTRAINT entity_key_unique IF NOT EXISTS
                FOR (entity:Entity) REQUIRE entity.key IS UNIQUE
                """
            ).consume()
            session.execute_write(
                self._write_knowledge_graph,
                graph_documents,
                graph_data,
            )

        # FETCH AND LOG EXACT DATABASE COUNTS AFTER INGESTION
        with self.driver.session(database=self.database) as session:
            node_count_res = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()
            rel_count_res = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
            total_nodes = node_count_res["count"] if node_count_res else 0
            total_rels = rel_count_res["count"] if rel_count_res else 0
            
            logger.info(f"📊 [Neo4j Status] Total Entity Nodes in DB: {total_nodes} | Total Relationship Edges in DB: {total_rels}")
            print(f"\n📊 [Neo4j Status] Total Entity Nodes in DB: {total_nodes}")
            print(f"📊 [Neo4j Status] Total Relationship Edges in DB: {total_rels}\n")

        for filename, file_hash in current_files_map.items():
            if any(f.name == filename for f in files_to_process) or filename not in self.ingested_state:
                self.ingested_state[filename] = file_hash

        self.save_state()
        logger.info(f"Successfully synchronized graph knowledge base. Processed {len(files_to_process)} file(s).")
        return len(files_to_process)

    def query_graph(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception:
            logger.exception("Error executing Cypher query.")
            raise

    def close(self):
        if self.driver:
            self.driver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass