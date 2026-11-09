import os
import sys
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger("neo4j").setLevel(logging.DEBUG)

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

print("URI scheme:", uri.split("://")[0] if uri else "MISSING")
print("Username exists:", bool(username))
print("Password exists:", bool(password))


driver = GraphDatabase.driver(
    uri,
    auth=(username, password),
)

try:
    driver.verify_connectivity()
    print("Connection successful!")
except Exception as e:
    print("Error type:", type(e).__name__)
    print("Error:", repr(e))
finally:
    driver.close()