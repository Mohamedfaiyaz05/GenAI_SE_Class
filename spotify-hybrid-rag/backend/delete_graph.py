from pathlib import Path
import os

import certifi
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Assumes this script is saved in the project's backend/ folder.
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE") or "neo4j"

if not uri or not username or not password:
    raise RuntimeError("Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env.")

# Keep TLS verification enabled for neo4j+s:// connections.
ca_bundle = os.getenv("NEO4J_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or certifi.where()
if not Path(ca_bundle).is_file():
    raise FileNotFoundError(f"CA certificate bundle not found: {ca_bundle}")
os.environ["SSL_CERT_FILE"] = ca_bundle


def delete_all_graph_data(tx):
    summary = tx.run("MATCH (n) DETACH DELETE n").consume()
    counters = summary.counters
    return counters.nodes_deleted, counters.relationships_deleted


confirmation = input(
    f"This will delete all graph data in database '{database}'. "
    "Type DELETE to continue: "
)
if confirmation != "DELETE":
    print("Cancelled; no data was deleted.")
    raise SystemExit(0)

driver = GraphDatabase.driver(uri, auth=(username, password))
try:
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        nodes_deleted, relationships_deleted = session.execute_write(
            delete_all_graph_data
        )

    print(f"Deleted {nodes_deleted} node(s) and {relationships_deleted} relationship(s).")
finally:
    driver.close()