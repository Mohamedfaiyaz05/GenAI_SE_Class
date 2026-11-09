from pathlib import Path

backend_dir = Path(__file__).resolve().parent
import sys

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from generation.generator import RAGPipeline

def test_pipeline_with_memory():
    print("--- Testing End-to-End RAG Pipeline with Conversation Memory ---")
    
    project_root = backend_dir.parent
    test_index_path = project_root / "test_faiss_index"
    pipeline = RAGPipeline(index_path=str(test_index_path))
    session_id = "test_session_001"

    try:
        print("\n=== Turn 1 ===")
        query_1 = "What are the major components, and how are storage and caching handled?"
        print(f"🔍 User Query: '{query_1}'")
        response_1 = pipeline.run(query_1, session_id=session_id, k=3)
        print(f"🤖 Answer 1:\n{response_1['answer']}\n")

        print("=== Turn 2 (Follow-up) ===")
        query_2 = "Can you elaborate more on that?"
        print(f"🔍 User Query: '{query_2}'")
        response_2 = pipeline.run(query_2, session_id=session_id, k=3)
        print(f"🤖 Answer 2:\n{response_2['answer']}\n")

        turn_count = len(pipeline.memory_manager.get_history(session_id)) // 2
        print(f"✨ Total conversation turns stored: {turn_count}")
        print("✨ Memory integration test completed successfully!")
    finally:
        pipeline.graph_manager.close()

if __name__ == "__main__":
    test_pipeline_with_memory()