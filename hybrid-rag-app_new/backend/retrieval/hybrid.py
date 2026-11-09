"""Combine retrieval results with Reciprocal Rank Fusion and merge context."""
import logging
from typing import List, Dict, Any, Optional
from retrieval.vector_store import VectorStoreManager
from retrieval.graph_store import Neo4jGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Orchestrates hybrid retrieval by querying FAISS for semantic context chunks 
    (with explicit source file and page number tracking) and Neo4j for structured 
    entity-relationship paths, merging them into a unified context window.
    """
    def __init__(self, vector_manager: Optional[VectorStoreManager] = None, graph_manager: Optional[Neo4jGraphManager] = None):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.graph_manager = graph_manager or Neo4jGraphManager()

    def search(self, query: str, k: int = 8) -> Dict[str, Any]:
        """
        Performs parallel vector search and graph relationship lookup, 
        extracting granular page/source provenance and structuring the final context.
        """
        logger.info(f"Executing hybrid search for query: '{query}'")
        
        # 1. Semantic Vector Search (FAISS)
        vector_results = []
        try:
            vector_results = self.vector_manager.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error executing FAISS vector search in hybrid engine: {e}")

        # 2. Graph Relational Lookup (Neo4j with Source Document Tracing)
        graph_results = []
        try:
            keywords = [word.strip() for word in query.split() if len(word.strip()) > 3]
            if not keywords:
                keywords = [query.strip()]
            
            cypher = """
            UNWIND $keywords AS kw
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(kw) OR toLower(e.description) CONTAINS toLower(kw)
            OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
            OPTIONAL MATCH (e)-[r]->(target:Entity)
            RETURN e.name AS source, e.type AS source_type, e.description AS description, 
                   type(r) AS relation, target.name AS target, target.type AS target_type,
                   collect(DISTINCT d.title) AS documents
            LIMIT 15
            """
            raw_graph_data = self.graph_manager.query_graph(cypher, {"keywords": keywords})
            
            seen = set()
            for record in raw_graph_data:
                item_str = str(record)
                if item_str not in seen:
                    seen.add(item_str)
                    graph_results.append(record)
        except Exception as e:
            logger.error(f"Error executing Neo4j graph search in hybrid engine: {e}")

        # 3. Assemble and format unified context package with explicit page numbers and provenance
        combined_text_parts = []
        
        combined_text_parts.append("=== SEMANTIC DOCUMENT CHUNKS (FAISS) ===")
        if vector_results:
            for i, chunk in enumerate(vector_results, start=1):
                metadata = chunk.get("metadata", {})
                source = metadata.get("source") or metadata.get("filename") or "Unknown Document"
                
                # Robust page number extraction across common keys
                page = metadata.get("page") if metadata.get("page") is not None else metadata.get("page_number")
                if page is not None:
                    page_str = f"Page {int(page) + 1}" if isinstance(page, int) and page == 0 else f"Page {page}"
                else:
                    page_str = "Page Unknown"
                    
                score = chunk.get("score", 0.0)
                
                combined_text_parts.append(
                    f"[Chunk {i}] | Source: {source} | {page_str} | Score: {score:.4f}\n{chunk['text']}"
                )
        else:
            combined_text_parts.append("No semantic chunks found.")

        combined_text_parts.append("\n=== KNOWLEDGE GRAPH RELATIONSHIPS (Neo4j) ===")
        if graph_results:
            for i, g in enumerate(graph_results, start=1):
                source = g.get("source", "Unknown")
                source_type = g.get("source_type", "Entity")
                rel = g.get("relation")
                target = g.get("target")
                target_type = g.get("target_type", "Entity")
                desc = g.get("description", "")
                docs = g.get("documents", [])
                doc_str = f"" if docs and any(docs) else ""
                
                if rel and target:
                    combined_text_parts.append(f"[Graph Rel {i}] {source} ({source_type}) -[{rel}]-> {target} ({target_type}){doc_str}")
                elif source:
                    combined_text_parts.append(f"[Graph Entity {i}] {source} ({source_type}): {desc}{doc_str}")
        else:
            combined_text_parts.append("No direct graph relationships found.")

        final_context_text = "\n\n".join(combined_text_parts)

        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_context": final_context_text
        }