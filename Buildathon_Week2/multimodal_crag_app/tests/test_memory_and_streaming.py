from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor import PDFExtractor
from src.vector_store import VectorStoreManager
from src.crag import CRAGEngine

def test_memory_and_streaming():
    print("=== STARTING PHASE 4: MEMORY & STREAMING TEST ===")
    
    session_id = "test_user_session_101"
    
    # 1. Initialize Managers
    vector_manager = VectorStoreManager(index_dir="vector_index_test")
    extractor = PDFExtractor(data_dir="data")
    vector_manager.process_new_documents(extractor, data_dir="data")
    
    crag_engine = CRAGEngine(vector_store_manager=vector_manager)
    
    # --- Turn 1: Initial Question ---
    q1 = "What is the agriculture budget allocation for 2026-27?"
    print(f"\n[Turn 1 Question]: '{q1}'")
    print("[Streaming Response 1]: ", end="", flush=True)
    
    for token in crag_engine.run_stream(q1, session_id=session_id):
        print(token, end="", flush=True)
        time.sleep(0.01)  # Simulate typewriter streaming effect
    print("\n")

    # --- Turn 2: Contextual Follow-up Question ---
    q2 = "Can you summarize the number in one sentence based on what you just told me?"
    print(f"\n[Turn 2 Question (Memory Test)]: '{q2}'")
    print("[Streaming Response 2]: ", end="", flush=True)
    
    for token in crag_engine.run_stream(q2, session_id=session_id):
        print(token, end="", flush=True)
        time.sleep(0.01)
    print("\n")

    print("=== PHASE 4 TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_memory_and_streaming()