from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver
from openai import OpenAI
from openai import OpenAIError


load_dotenv()

STOP_WORDS = {
    "a", "about", "all", "and", "are", "based", "be", "benefit", "benefits",
    "can", "could", "did", "do", "does", "for", "from", "get", "give", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "please", "scheme",
    "schemes", "tell", "that", "the", "their", "this", "to", "what", "when",
    "where", "which", "who", "why", "with",
}

GRAPH_EVIDENCE_QUERY = """
MATCH (n)
WHERE coalesce(toString(n.name), toString(n.title), '') <> ''
  AND any(
    term IN $terms
    WHERE toLower(coalesce(toString(n.name), toString(n.title), ''))
      CONTAINS toLower(term)
  )
WITH DISTINCT n,
     size([
       term IN $terms
       WHERE toLower(coalesce(toString(n.name), toString(n.title), ''))
         CONTAINS toLower(term)
     ]) AS match_count
ORDER BY match_count DESC
LIMIT 20
OPTIONAL MATCH (n)-[r]-(neighbor)
WITH n, match_count,
     collect(DISTINCT CASE
       WHEN r IS NULL THEN null
       ELSE {
         direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
         type: type(r),
         neighbor: coalesce(neighbor.name, neighbor.title, labels(neighbor)[0])
       }
     END) AS connections
OPTIONAL MATCH (source:Document)-[:MENTIONS]->(n)
RETURN labels(n) AS labels,
       properties(n) AS node,
       match_count,
       connections,
       collect(DISTINCT CASE
         WHEN source IS NULL THEN null
         ELSE {title: source.title, url: source.url}
       END) AS sources
ORDER BY match_count DESC
LIMIT 20
"""


def create_openai_client() -> OpenAI | None:
    """Return a client when a key is configured; graph search still works without it."""
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


def fallback_search_terms(question: str) -> list[str]:
    """Produce conservative literal search terms if the language model is unavailable."""
    candidates = re.findall(r"[^\W_]+", question.casefold(), flags=re.UNICODE)
    terms = []
    for term in candidates:
        if len(term) < 3 or term in STOP_WORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:8]


def extract_search_terms(
    question: str,
    client: OpenAI | None,
) -> tuple[list[str], str | None]:
    """Extract graph-search terms, translating Tamil concepts to English when possible."""
    if client is None:
        terms = fallback_search_terms(question)
        if not terms:
            raise ValueError("Enter a question with searchable words.")
        return terms, (
            "OPENAI_API_KEY is not configured; searching with literal question words. "
            "Tamil questions may retrieve fewer English-language graph entries."
        )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert a user question into up to 8 concise search terms for "
                        "a Neo4j graph about Tamil Nadu government schemes. Return only "
                        'JSON: {"terms":["..."]}. Prefer English terms because graph '
                        "node names and relationships are stored in English. Translate "
                        "Tamil concepts to their English equivalents, preserve proper "
                        "names and scheme acronyms, and exclude question words."
                    ),
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        raw_terms = payload.get("terms", [])
        if not isinstance(raw_terms, list):
            raise ValueError("The keyword extractor returned an invalid terms field.")
        terms = []
        for item in raw_terms:
            if isinstance(item, str):
                term = item.strip()
                if len(term) >= 2 and term.casefold() not in {
                    existing.casefold() for existing in terms
                }:
                    terms.append(term)
        if terms:
            return terms[:8], None
    except (OpenAIError, json.JSONDecodeError, ValueError) as error:
        terms = fallback_search_terms(question)
        if not terms:
            raise RuntimeError(
                f"Could not prepare graph search terms: {error}"
            ) from error
        return terms, (
            "AI keyword extraction was unavailable; using literal question words "
            f"instead. Details: {error}"
        )

    terms = fallback_search_terms(question)
    if not terms:
        raise ValueError("No searchable terms could be extracted from the question.")
    return terms, "AI keyword extraction returned no terms; using literal question words."


def retrieve_graph_evidence(
    driver: Driver,
    database: str,
    terms: list[str],
) -> list[dict[str, Any]]:
    """Run a fixed, read-only Cypher query; user text is passed only as parameters."""
    with driver.session(database=database) as session:
        result = session.run(GRAPH_EVIDENCE_QUERY, terms=terms)
        return [record.data() for record in result]


def format_graph_evidence(records: list[dict[str, Any]]) -> str:
    lines = []
    for record in records:
        labels = ", ".join(record.get("labels") or ["Node"])
        properties = record.get("node") or {}
        name = properties.get("name") or properties.get("title") or str(properties)
        lines.append(f"{labels}: {name}")

        for connection in record.get("connections") or []:
            if connection:
                lines.append(
                    f"  {connection['direction']} {connection['type']} "
                    f"{connection['neighbor']}"
                )

        for source in record.get("sources") or []:
            if source and (source.get("title") or source.get("url")):
                lines.append(
                    f"  Source document: {source.get('title') or 'Untitled'} "
                    f"({source.get('url') or 'URL not recorded'})"
                )
    return "\n".join(lines)


def answer_from_graph(
    question: str,
    records: list[dict[str, Any]],
    client: OpenAI | None,
) -> str:
    """Summarize only graph evidence; use a transparent fallback without an API key."""
    evidence = format_graph_evidence(records)
    if not evidence:
        return (
            "I couldn't find matching nodes or relationships in the current knowledge "
            "graph. The answer may be missing because the source information has not "
            "been extracted into the graph or the search terms do not match its names."
        )

    if client is None:
        return "I found these matching graph facts:\n\n" + evidence

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the supplied Neo4j graph "
                    "evidence. Explain relationships in plain language. Do not invent "
                    "facts or claim absent values. Mention when the graph has no "
                    "evidence for part of the question. Include the source document "
                    "title and URL when provided. Respond in the user's language."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nGraph evidence:\n{evidence}"
                ),
            },
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("The answer model returned an empty response.")
    return answer.strip()
