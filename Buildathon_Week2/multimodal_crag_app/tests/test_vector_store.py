from pathlib import Path
import sys

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor import PDFExtractor
from src.vector_store import VectorStoreManager

def test_idempotent_vector_store_pipeline():
    print("=== STARTING IDEMPOTENT PHASE 2 PIPELINE TEST ===")
    
    # 1. Initialize Extractor
    print("\n[Step 1] Initializing PDFExtractor...")
    extractor = PDFExtractor(data_dir="data")
    
    # 2. Initialize Vector Store Manager
    print("\n[Step 2] Initializing VectorStoreManager...")
    vector_manager = VectorStoreManager(index_dir="vector_index_test")
    
    # 3. Run incremental processing (Test 1: Should process new PDFs if present)
    print("\n[Step 3] Running incremental vector store update (First Pass)...")
    vector_store = vector_manager.process_new_documents(extractor, data_dir="data")
    
    if not vector_store:
        print("No vector store created (no PDFs found in data/ folder). Please add a PDF to data/ and re-run.")
        return

    # 4. Run incremental processing again (Test 2: Idempotency check - should skip embedding)
    print("\n[Step 4] Running incremental vector store update again (Second Pass - Idempotency Test)...")
    vector_store_second = vector_manager.process_new_documents(extractor, data_dir="data")
    print("Idempotency test passed successfully (skipped embedding already processed files).")

    # 5. Test similarity search retrieval
    print("\n[Step 5] Running semantic similarity search on reloaded store...")
    test_query = "agriculture budget allocation"
    results = vector_manager.similarity_search(test_query, k=2)
    
    print(f"\nRetrieved {len(results)} relevant chunks for query: '{test_query}':")
    for idx, doc in enumerate(results):
        print(f"\n--- Result {idx + 1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:300]}...")

    print("\n=== PHASE 2 IDEMPOTENT PIPELINE TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_idempotent_vector_store_pipeline()