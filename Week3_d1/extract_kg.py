from __future__ import annotations

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = Path("preprocessed_corpus.json")
OUTPUT_FILE = Path("extracted_graph_data.json")

# Strict extraction prompt template with escaped curly braces for JSON schema
EXTRACTION_PROMPT = """You are an expert knowledge graph engineer specializing in government schemes and public policy data. Extract all key entities (such as Schemes, Departments, Benefits, Districts, Officers, and Eligibility criteria) and the relationships between them from the following text chunk.

Return the result STRICTLY as a valid JSON object matching this schema:
{{
  "entities": [
    {{"name": "Entity Name", "type": "ENTITY_TYPE"}}
  ],
  "relations": [
    {{"source": "Entity Source Name", "relation": "RELATION_TYPE", "target": "Entity Target Name"}}
  ]
}}

Text Chunk:
{text}
"""


def extract_graph_from_text(text: str) -> dict:
    """Send preprocessed text chunk to LLM and retrieve structured graph triples."""
    prompt = EXTRACTION_PROMPT.format(text=text)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise knowledge graph builder. Always output strictly valid JSON matching the requested schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as error:
        print(f"[ERROR] LLM extraction failed: {error}")
        return {"entities": [], "relations": []}


def process_corpus_for_kg(input_path: Path, output_path: Path) -> None:
    """Read preprocessed corpus, extract entities/relations via LLM, and save structured graph output."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found. Please ensure Phase 1 crawler has generated it.")

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    all_graph_data = []
    print(f"Starting Phase 2: LLM Entity & Relation extraction for {len(corpus)} documents...")

    for idx, item in enumerate(corpus):
        title = item.get("title", "Untitled Document")
        url = item.get("url", "")
        cleaned_text = item.get("cleaned_text", "")
        
        if not cleaned_text.strip():
            continue
            
        print(f"[{idx + 1}/{len(corpus)}] Processing LLM Extraction: {title} ({url})")
        graph_data = extract_graph_from_text(cleaned_text)
        
        all_graph_data.append({
            "url": url,
            "title": title,
            "entities": graph_data.get("entities", []),
            "relations": graph_data.get("relations", [])
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_graph_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nPhase 2 Complete! Extracted graph data successfully saved to: {output_path.resolve()}")


if __name__ == "__main__":
    process_corpus_for_kg(INPUT_FILE, OUTPUT_FILE)