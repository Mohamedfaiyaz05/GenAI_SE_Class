from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from neo4j_connection import create_neo4j_driver

# Load credentials from your .env file
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

INPUT_FILE = Path("extracted_graph_data.json")


def sanitize_label(label: str) -> str:
    """Sanitize entity type to be a valid Neo4j label."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "Entity_" + cleaned
    return cleaned


def sanitize_rel_type(rel: str) -> str:
    """Sanitize relationship type to be a valid uppercase Neo4j relationship type."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', rel.strip()).upper()
    return cleaned if cleaned else "RELATED_TO"


def ingest_graph_data(input_path: Path) -> None:
    """Connect to Neo4j and ingest extracted entities, documents, and relationship triples."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found. Please ensure Phase 2 has generated it.")
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", NEO4J_URI),
            ("NEO4J_USERNAME", NEO4J_USERNAME),
            ("NEO4J_PASSWORD", NEO4J_PASSWORD),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Neo4j configuration in the environment: "
            + ", ".join(missing)
        )
    with open(input_path, "r", encoding="utf-8") as f:
        graph_corpus = json.load(f)

    print("Connecting to Neo4j using certificate verification.")
    driver = create_neo4j_driver(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
    )

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            print(f"Starting ingestion of {len(graph_corpus)} document records into Neo4j...")

            for idx, item in enumerate(graph_corpus):
                url = item.get("url", "")
                title = item.get("title", "Untitled Document")
                entities = item.get("entities", [])
                relations = item.get("relations", [])

                print(f"[{idx + 1}/{len(graph_corpus)}] Ingesting: {title}")

                # 1. Create Document Node
                session.run(
                    """
                    MERGE (d:Document {url: $url})
                    SET d.title = $title
                    """,
                    url=url, title=title
                )

                # 2. Create Entity Nodes and Link to Document
                for ent in entities:
                    name = ent.get("name")
                    raw_type = ent.get("type", "Entity")
                    if not name:
                        continue
                    
                    label = sanitize_label(raw_type)
                    query = f"""
                    MERGE (e:`{label}` {{name: $name}})
                    MERGE (d:Document {{url: $url}})
                    MERGE (d)-[:MENTIONS]->(e)
                    """
                    session.run(query, name=name, url=url)

                # 3. Create Inter-Entity Relationships
                for rel in relations:
                    source = rel.get("source")
                    target = rel.get("target")
                    raw_rel = rel.get("relation", "RELATED_TO")

                    if not source or not target:
                        continue

                    rel_type = sanitize_rel_type(raw_rel)
                    query = f"""
                    MERGE (s {{name: $source}})
                    MERGE (t {{name: $target}})
                    MERGE (s)-[:`{rel_type}`]->(t)
                    """
                    session.run(query, source=source, target=target)
    finally:
        driver.close()
    print("\nPhase 3: Neo4j Knowledge Graph ingestion completed successfully!")


if __name__ == "__main__":
    ingest_graph_data(INPUT_FILE)