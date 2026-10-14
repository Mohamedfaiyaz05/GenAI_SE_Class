import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class MemoryManager:
    """
    Manages multi-turn conversation history with local JSON persistence,
    session/thread tracking, and context window formatting.
    """
    def __init__(self, memory_dir: str = "chat_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _get_thread_file(self, session_id: str) -> Path:
        """Get the file path for a given session thread."""
        return self.memory_dir / f"{session_id}.json"

    def load_history(self, session_id: str) -> List[Dict[str, str]]:
        """Load conversation history for a given session ID."""
        thread_file = self._get_thread_file(session_id)
        if not thread_file.exists():
            return []
        try:
            with open(thread_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat history for session {session_id}: {str(e)}")
            return []

    def save_message(self, session_id: str, role: str, content: str):
        """Append a single user or AI message to the persisted JSON file."""
        history = self.load_history(session_id)
        history.append({"role": role, "content": content})
        
        thread_file = self._get_thread_file(session_id)
        with open(thread_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def get_langchain_messages(self, session_id: str, max_turns: int = 5) -> List[BaseMessage]:
        """
        Retrieves the last `max_turns` interactions as formal LangChain Message objects
        (HumanMessage / AIMessage) for context injection into LLM prompts.
        """
        history = self.load_history(session_id)
        recent_history = history[-(max_turns * 2):]  # Get last N Q&A pairs
        
        langchain_msgs = []
        for msg in recent_history:
            if msg["role"] == "user":
                langchain_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_msgs.append(AIMessage(content=msg["content"]))
        
        return langchain_msgs

    def clear_history(self, session_id: str):
        """Clear memory for a given session ID."""
        thread_file = self._get_thread_file(session_id)
        if thread_file.exists():
            thread_file.unlink()
            print(f"Memory cleared for session: {session_id}")