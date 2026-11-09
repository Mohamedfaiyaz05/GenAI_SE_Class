"""Corrective retrieval-augmented generation logic."""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from config import settings

logger = logging.getLogger(__name__)

class CorrectiveRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=settings.OPENAI_API_KEY)

    def evaluate_and_correct(self, query: str, vector_chunks: List[Dict[str, Any]], graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the retrieved context using CRAG methodology without hardcoded truncation limits.
        Returns the evaluation score ('correct', 'ambiguous', 'incorrect') 
        and the corrected/refined context for generation.
        """
        context_text = "\n\n".join([chunk["content"] for chunk in vector_chunks])
        graph_evidence_lines = [f"{rel['source']} -[{rel['relation']}]-> {rel['target']}" for rel in graph_data.get("relationships", [])]
        graph_context_text = "\n".join(graph_evidence_lines)
        
        combined_raw_context = f"--- Vector Chunks ---\n{context_text}\n\n--- Knowledge Graph Relations ---\n{graph_context_text}"

        if not vector_chunks and not graph_data.get("nodes"):
            return {
                "score": "incorrect",
                "refined_context": "No local architectural context was retrieved for this query.",
                "action_taken": "Direct fallback due to empty retrieval."
            }

        prompt = f"""
        You are a Corrective RAG (CRAG) evaluation agent. Analyze the user query and the retrieved knowledge base context.
        Grade the relevance and sufficiency of the context as one of three strict categories:
        - "correct": The context fully answers or provides direct, highly relevant information for the query.
        - "ambiguous": The context is partially related or contains tangential architectural details, requiring cautious synthesis.
        - "incorrect": The context is completely irrelevant, off-topic, or fails to address the query.

        Return valid JSON strictly adhering to this schema:
        {{
            "score": "correct" | "ambiguous" | "incorrect",
            "reasoning": "string"
        }}
        
        Query: {query}
        Retrieved Context: {combined_raw_context}
        """
        
        try:
            response = self.llm.invoke(prompt)
            clean = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            score = data.get("score", "correct")
            if score not in ["correct", "ambiguous", "incorrect"]:
                score = "correct"
        except Exception as e:
            logger.error(f"CRAG evaluation failed: {e}")
            score = "correct"
            data = {"reasoning": "Fallback due to evaluation error exception."}

        logger.info(f"CRAG Evaluation Result -> Score: {score} | Reasoning: {data.get('reasoning')}")

        if score == "correct":
            refined_context = combined_raw_context
            action_taken = "Retrieved context accepted without modification."
        elif score == "ambiguous":
            refined_context = f"[Note: Retrieved context is partially ambiguous. Synthesize carefully with general architecture knowledge.]\n\n{combined_raw_context}"
            action_taken = "Context marked ambiguous; applied cautious synthesis wrapper."
        else:
            refined_context = "Local knowledge base retrieval was graded incorrect or off-topic for this specific prompt. Answer using general distributed systems and Spotify architecture best practices while noting the local retrieval mismatch."
            action_taken = "Retrieved context discarded; triggered corrective architectural fallback."

        return {
            "score": score,
            "refined_context": refined_context,
            "action_taken": action_taken
        }