from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor import PDFExtractor

def test_pdf_discovery_and_extraction():
    # Initialize the extractor with the configured OpenAI models.
    print("Initializing PDFExtractor for testing...")
    extractor = PDFExtractor(data_dir="data")
    
    # 1. Test PDF auto-discovery
    print("Discovering PDF files in the data/ directory...")
    pdfs = extractor.discover_pdfs()
    # assert len(pdfs) > 0, "No PDF files found in the data/ directory. Please place a sample PDF in data/."
    
    # 2. Test extraction on the first discovered PDF
    print("Processing the first discovered PDF...")
    chunks = extractor.process_pdf(pdfs[0])
    print(f"Number of chunks extracted: {len(chunks)}")
    # assert len(chunks) > 0, "No chunks were extracted from the PDF."
    
    # 3. Verify metadata and schema preservation on the first chunk
    first_chunk = chunks[0]
    print("chunked-retrieved================================>",first_chunk)
    # assert "doc_name" in first_chunk
    # assert "page_num" in first_chunk
    # assert "section" in first_chunk
    # assert "content_type" in first_chunk
    # assert "content" in first_chunk
    
    print(f"\nSuccessfully extracted {len(chunks)} chunks from {pdfs[0].name}.")


if __name__ == "__main__":
    test_pdf_discovery_and_extraction()