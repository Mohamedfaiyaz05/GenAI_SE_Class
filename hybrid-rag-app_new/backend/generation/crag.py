"""Evaluate retrieved evidence and trigger corrective retrieval when needed."""
import json
import logging
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRAGGrade(BaseModel):
    grade: Literal["correct", "ambiguous", "incorrect"] = Field(
        description="Relevance classification of the chunk against the user query."
    )
    reason: str = Field(description="Short explanation for the assigned grade.")

class CorrectiveRAGEvaluator:
    """
    Evaluates retrieved semantic vector chunks against the user query using an LLM grader
    with Pydantic structured output, filtering out irrelevant or noisy information.
    CRAG evaluation is strictly applied only to semantic vector results.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        base_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.llm = base_llm.with_structured_output(CRAGGrade)

    def evaluate_chunk(self, query: str, chunk_text: str) -> str:
        """
        Grades whether a specific semantic document chunk is 'correct' (relevant), 
        'ambiguous', or 'incorrect' (irrelevant) regarding the user query.
        """
        prompt = f"""
        You are a precise evaluation engine for a Corrective RAG (CRAG) system.
        Your task is to evaluate whether the provided document chunk contains information relevant and useful for answering the user query.

        User Query: "{query}"

        Document Chunk:
        "{chunk_text}"

        Classify the relevance of this chunk into exactly one of these three categories:
        - "correct": The chunk directly contains facts, data, or context that helps answer the query.
        - "ambiguous": The chunk is partially related or mentions related terminology but lacks definitive answers.
        - "incorrect": The chunk is completely irrelevant, off-topic, or noise.
        """
        try:
            result = self.llm.invoke(prompt)
            return result.grade if result and result.grade in ["correct", "ambiguous", "incorrect"] else "incorrect"
        except Exception as e:
            logger.error(f"Error during CRAG chunk evaluation: {e}")
            # Default to correct on failure to avoid dropping valid context during network hiccups
            return "correct"

    def evaluate_and_filter_context(self, query: str, hybrid_search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes hybrid search results, applies CRAG evaluation strictly to semantic vector chunks,
        filters out incorrect chunks, and checks semantic context sufficiency.
        """
        logger.info(f"Running CRAG evaluation strictly on semantic vector results for query: '{query}'")
        
        vector_results = hybrid_search_results.get("vector_results", [])
        graph_results = hybrid_search_results.get("graph_results", [])
        
        filtered_vector_results = []
        evaluation_log = []

        for idx, chunk in enumerate(vector_results, start=1):
            chunk_text = chunk.get("text", "")
            grade = self.evaluate_chunk(query, chunk_text)
            
            evaluation_log.append({
                "chunk_index": idx,
                "grade": grade,
                "source": chunk.get("metadata", {}).get("source", "Unknown")
            })

            if grade in ["correct", "ambiguous"]:
                chunk["crag_grade"] = grade
                filtered_vector_results.append(chunk)
            else:
                logger.info(f"❌ CRAG Filtered out semantic Chunk {idx} (Grade: incorrect)")

        # Graph evidence is also valid context when the vector index is empty or has
        # no relevant chunks; do not discard useful graph-only retrieval.
        is_context_sufficient = bool(filtered_vector_results or graph_results)
        if not is_context_sufficient:
            logger.warning(
                "No relevant semantic chunks or graph results were retrieved; context is insufficient."
            )
        elif not filtered_vector_results and graph_results:
            logger.info("Proceeding with graph-only evidence because no semantic chunks were available.")

        # Re-assemble refined context text
        refined_text_parts = []
        
        refined_text_parts.append("=== REFINED SEMANTIC CHUNKS (CRAG Verified) ===")
        if filtered_vector_results:
            for i, chunk in enumerate(filtered_vector_results, start=1):
                metadata = chunk.get("metadata", {})
                source = metadata.get("source") or metadata.get("filename") or "Unknown Document"
                page = metadata.get("page") if metadata.get("page") is not None else metadata.get("page_number", "Unknown")
                score = chunk.get("score", 0.0)
                grade = chunk.get("crag_grade", "correct")
                
                refined_text_parts.append(
                    f"[Verified Chunk {i}] | Source: {source} | Page: {page} | Grade: {grade} | Score: {score:.4f}\n{chunk['text']}"
                )
        else:
            refined_text_parts.append("No semantic chunks passed CRAG verification.")

        refined_text_parts.append("\n=== KNOWLEDGE GRAPH RELATIONSHIPS (Neo4j) ===")
        if graph_results:
            for i, g in enumerate(graph_results, start=1):
                source = g.get("source", "Unknown")
                source_type = g.get("source_type", "Entity")
                rel = g.get("relation")
                target = g.get("target")
                target_type = g.get("target_type", "Entity")
                desc = g.get("description", "")
                
                if rel and target:
                    refined_text_parts.append(f"[Graph Rel {i}] {source} ({source_type}) -[{rel}]-> {target} ({target_type})")
                elif source:
                    refined_text_parts.append(f"[Graph Entity {i}] {source} ({source_type}): {desc}")
        else:
            refined_text_parts.append("No direct graph relationships found.")

        refined_context_text = "\n\n".join(refined_text_parts)

        return {
            "query": query,
            "original_vector_count": len(vector_results),
            "filtered_vector_count": len(filtered_vector_results),
            "is_context_sufficient": is_context_sufficient,
            "evaluation_log": evaluation_log,
            "vector_results": filtered_vector_results,
            "graph_results": graph_results,
            "refined_context": refined_context_text
        }