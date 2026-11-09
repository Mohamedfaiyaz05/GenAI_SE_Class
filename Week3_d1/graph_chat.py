from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv
from neo4j.exceptions import Neo4jError
from openai import OpenAIError


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from graph_qa import (  # noqa: E402
    answer_from_graph,
    create_openai_client,
    extract_search_terms,
    retrieve_graph_evidence,
)
from neo4j_connection import create_neo4j_driver  # noqa: E402


load_dotenv(PROJECT_DIR / ".env")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

st.set_page_config(
    page_title="Tamil Nadu Scheme Graph Assistant",
    page_icon="🔗",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(207, 226, 193, .5), transparent 34%),
            linear-gradient(145deg, #f6f8f1 0%, #eaf2e7 56%, #f8f7ef 100%);
    }
    .hero {
        background: linear-gradient(115deg, #0c3025, #15543d 70%, #34784f);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 22px;
        box-shadow: 0 14px 32px rgba(16,59,45,.14);
        color: #f8f4e6;
        margin-bottom: 1rem;
        padding: 1.5rem 1.8rem;
    }
    .hero-kicker {
        color: #ddce91;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: Georgia, serif;
        font-size: clamp(1.8rem, 4vw, 2.7rem);
        font-weight: 700;
        line-height: 1.13;
        margin: .35rem 0;
    }
    .hero-copy { color: #d8e6d7; max-width: 850px; }
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.8);
        border: 1px solid #e3eadd;
        border-radius: 14px;
    }
    .metric-card {
        background: rgba(255,255,255,.82);
        border: 1px solid #e3eadd;
        border-radius: 12px;
        padding: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_driver():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Neo4j settings are missing from .env: " + ", ".join(missing)
        )
    return create_neo4j_driver(uri, username, password)


def get_graph_counts(driver) -> tuple[int, int]:
    with driver.session(database=DATABASE) as session:
        record = session.run(
            """
            MATCH (node)
            OPTIONAL MATCH (node)-[relationship]->()
            RETURN count(DISTINCT node) AS nodes,
                   count(relationship) AS relationships
            """
        ).single()
    return record["nodes"], record["relationships"]


def process_question(question: str, driver, openai_client):
    terms, warning = extract_search_terms(question, openai_client)
    records = retrieve_graph_evidence(driver, DATABASE, terms)
    answer = answer_from_graph(question, records, openai_client)
    return answer, records, terms, warning


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Week 3 · Neo4j Knowledge Graph</div>
        <div class="hero-title">Ask the scheme graph.</div>
        <div class="hero-copy">
            Explore scheme, department, beneficiary, benefit, and relationship facts
            extracted into your Neo4j graph. Answers are grounded in matching graph
            nodes and edges—not a general web search.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Graph connection")
    try:
        driver = get_driver()
        node_count, relationship_count = get_graph_counts(driver)
        st.success("Connected securely to Neo4j Aura.")
        first, second = st.columns(2)
        with first:
            st.metric("Nodes", f"{node_count:,}")
        with second:
            st.metric("Relationships", f"{relationship_count:,}")
        if node_count == 0:
            st.info(
                "The graph is empty. Run `python .\\ingest_kg.py` after "
                "checking that the graph extraction output contains entities."
            )
    except (Neo4jError, RuntimeError, ValueError, OSError) as error:
        driver = None
        st.error(f"Could not connect to Neo4j: {error}")
        st.caption(
            "Check the Aura instance status, .env URI/credentials, and network "
            "access. This assistant only reads the graph."
        )

    st.divider()
    st.caption("Questions you can try")
    st.markdown(
        """
        - Which schemes benefit farmers?
        - Which department implements Training to Farmers?
        - What benefits are connected to seed production schemes?
        - Which officers are related to farmer applications?
        """
    )
    st.caption(
        "Tamil questions are supported when the OpenAI key is configured; "
        "the app translates the search concepts to English graph terms."
    )

if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = [
        {
            "role": "assistant",
            "content": (
                "Vanakkam! Ask me about the entities and relationships present "
                "in your Neo4j knowledge graph."
            ),
            "records": [],
            "terms": [],
        }
    ]

for message in st.session_state.graph_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("warning"):
            st.warning(message["warning"])
        if message["role"] == "assistant" and message.get("terms"):
            st.caption("Graph search terms: " + ", ".join(message["terms"]))
        if message["role"] == "assistant" and message.get("records"):
            with st.expander(f"Evidence from graph ({len(message['records'])} matches)"):
                for record in message["records"]:
                    node = record.get("node") or {}
                    node_name = node.get("name") or node.get("title") or str(node)
                    st.markdown(
                        f"**{', '.join(record.get('labels') or ['Node'])}:** "
                        f"{node_name}"
                    )
                    for connection in record.get("connections") or []:
                        if connection:
                            arrow = "→" if connection["direction"] == "outgoing" else "←"
                            st.write(
                                f"{arrow} {connection['type']} "
                                f"{connection['neighbor']}"
                            )
                    for source in record.get("sources") or []:
                        if source:
                            title = source.get("title") or "Source document"
                            url = source.get("url")
                            if url:
                                st.markdown(f"Source: [{title}]({url})")
                            else:
                                st.caption(f"Source: {title}")

question = st.chat_input(
    "Ask a question about schemes, departments, or graph relationships…"
)
if question:
    st.session_state.graph_messages.append(
        {
            "role": "user",
            "content": question,
            "records": [],
            "terms": [],
        }
    )
    warning = None
    if driver is None:
        answer = (
            "I can't query the graph until the Neo4j connection succeeds. "
            "Check the connection message in the sidebar."
        )
        records, terms = [], []
    else:
        client = create_openai_client()
        with st.spinner("Searching graph evidence and preparing an answer…"):
            try:
                answer, records, terms, warning = process_question(
                    question,
                    driver,
                    client,
                )
            except (Neo4jError, OpenAIError, RuntimeError, ValueError) as error:
                answer = f"I couldn't complete the graph question: {error}"
                records, terms = [], []
    st.session_state.graph_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "records": records,
            "terms": terms,
            "warning": warning,
        }
    )
    st.rerun()
