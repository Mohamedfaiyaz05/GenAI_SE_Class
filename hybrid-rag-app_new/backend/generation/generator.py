import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from core.config import settings
from retrieval.vector_store import VectorStoreManager
from retrieval.graph_store import Neo4jGraphManager
from retrieval.hybrid import HybridSearchEngine
from generation.crag import CorrectiveRAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """Manages session-based conversation histories in memory for multi-turn chat continuity."""
    def __init__(self, max_history_turns: int = 5):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_turns = max_history_turns

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        history = self.sessions[session_id]
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            self.sessions[session_id] = history[-max_messages:]

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

class RAGPipeline:
    """Unified End-to-End Hybrid RAG Pipeline."""
    def __init__(
        self,
        index_path: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_history_turns: int = 5
    ):
        self.vector_manager = VectorStoreManager(index_path=index_path)
        self.graph_manager = Neo4jGraphManager()
        self.hybrid_engine = HybridSearchEngine(
            vector_manager=self.vector_manager,
            graph_manager=self.graph_manager
        )
        self.crag_evaluator = CorrectiveRAGEvaluator(model_name=model_name)
        self.memory_manager = ConversationMemoryManager(max_history_turns=max_history_turns)
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def _build_messages(self, query: str, refined_context: str, chat_history: List[Dict[str, str]]) -> List:
        system_prompt = (
            "You are an expert AI assistant and Knowledge Synthesizer. "
            "Your task is to provide a comprehensive, accurate, and professional answer to the user query "
            "strictly using the provided verified semantic document chunks, knowledge graph relationships, "
            "and conversation history.\n\n"
            "Rules:\n"
            "1. Base your answer primarily on the provided verified context. Do not invent unverified facts.\n"
            "2. Explicitly cite the source document and page number for every factual claim.\n"
            "3. Maintain conversational continuity using the chat history where relevant.\n"
            "4. Maintain a clear, objective, and structured tone."
        )

        messages = [SystemMessage(content=system_prompt)]
        for turn in chat_history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        user_prompt = f"""
        User Query: {query}

        Verified Knowledge Base Context:
        {refined_context}

        Provide your comprehensive answer below:
        """
        messages.append(HumanMessage(content=user_prompt))
        return messages

    def _extract_sources(self, crag_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        sources_cited = []
        seen_sources = set()
        for chunk in crag_output.get("vector_results", []):
            meta = chunk.get("metadata", {})
            src = meta.get("source") or meta.get("filename")
            page = meta.get("page") if meta.get("page") is not None else meta.get("page_number")
            
            source_key = f"{src}_{page}"
            if src and source_key not in seen_sources:
                seen_sources.add(source_key)
                sources_cited.append({"source": src, "page": page})
        return sources_cited

    def run(self, query: str, session_id: str = "default_session", k: int = 4) -> Dict[str, Any]:
        """Run the same retrieval and answer flow as streaming, returning one response."""
        hybrid_results = self.hybrid_engine.search(query, k=k)
        crag_output = self.crag_evaluator.evaluate_and_filter_context(query, hybrid_results)
        chat_history = self.memory_manager.get_history(session_id)

        if not crag_output.get("is_context_sufficient", False):
            answer = (
                "I couldn't find relevant content in the current indexes. Add PDFs to the "
                "knowledge-base folder and use the Index PDFs action, then try again."
            )
            self.memory_manager.add_turn(session_id, query, answer)
            return {
                "query": query,
                "session_id": session_id,
                "answer": answer,
                "sources_cited": [],
                "is_context_sufficient": False,
                "evaluation_log": crag_output.get("evaluation_log", []),
                "vector_results": [],
                "graph_results": [],
            }

        try:
            messages = self._build_messages(
                query,
                crag_output.get("refined_context", ""),
                chat_history,
            )
            response = self.llm.invoke(messages)
            answer = response.content if isinstance(response.content, str) else str(response.content)
            self.memory_manager.add_turn(session_id, query, answer)
            return {
                "query": query,
                "session_id": session_id,
                "answer": answer,
                "sources_cited": self._extract_sources(crag_output),
                "is_context_sufficient": True,
                "evaluation_log": crag_output.get("evaluation_log", []),
                "vector_results": crag_output.get("vector_results", []),
                "graph_results": crag_output.get("graph_results", []),
            }
        except Exception:
            logger.exception("Error during RAG answer generation.")
            raise

    def stream_run(self, query: str, session_id: str = "default_session", k: int = 4):
        """Executes the pipeline and yields streaming tokens, memory, and rich retrieval context."""
        logger.info(f"🚀 Running Streaming RAG Pipeline for query: '{query}' [Session: {session_id}]")

        hybrid_results = self.hybrid_engine.search(query, k=k)
        crag_output = self.crag_evaluator.evaluate_and_filter_context(query, hybrid_results)
        is_context_sufficient = crag_output.get("is_context_sufficient", True)
        refined_context = crag_output.get("refined_context", "")

        chat_history = self.memory_manager.get_history(session_id)

        if not is_context_sufficient:
            fallback_answer = (
                "I couldn't find relevant content in the current indexes. Add PDFs to the "
                "knowledge-base folder and use the Index PDFs action, then try again."
            )
            self.memory_manager.add_turn(session_id, query, fallback_answer)
            yield {"type": "token", "content": fallback_answer}
            yield {
                "type": "metadata",
                "sources_cited": [],
                "is_context_sufficient": False,
                "evaluation_log": crag_output.get("evaluation_log", []),
                "vector_results": [],
                "graph_results": []
            }
            return

        messages = self._build_messages(query, refined_context, chat_history)

        full_answer = ""
        try:
            for chunk in self.llm.stream(messages):
                token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if token:
                    full_answer += token
                    yield {"type": "token", "content": token}

            sources_cited = self._extract_sources(crag_output)
            self.memory_manager.add_turn(session_id, query, full_answer)

            yield {
                "type": "metadata",
                "sources_cited": sources_cited,
                "is_context_sufficient": True,
                "evaluation_log": crag_output.get("evaluation_log", []),
                "vector_results": crag_output.get("vector_results", []),
                "graph_results": crag_output.get("graph_results", [])
            }

        except Exception as e:
            logger.error(f"Error during streaming RAG generation: {e}")
            error_answer = "An error occurred while generating the response."
            yield {"type": "token", "content": error_answer}
            yield {
                "type": "metadata",
                "sources_cited": [],
                "is_context_sufficient": True,
                "evaluation_log": [],
                "vector_results": [],
                "graph_results": []
            }