import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFExtractor:
    """
    Extracts text, tables, and visual chart information from local PDF documents.
    """
    def __init__(
        self, 
        data_dir: str = "data", 
        vision_model: str = "gpt-4o-mini",
        text_model: str = "gpt-4o-mini"
    ):
        self.data_dir = Path(data_dir)
        self.vision_model = vision_model
        self.text_model = text_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            add_start_index=True,
        )
        
        # OpenAI client handles both text and image requests.
        self.text_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def discover_pdfs(self) -> List[Path]:
        """Automatically discover all PDF files in the local data directory."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        return list(self.data_dir.glob("*.pdf"))

    def extract_visual_with_vision(
        self,
        image_bytes: bytes,
        image_mime_type: str = "image/jpeg",
    ) -> str:
        """
        Extracts visible captions and data from a rendered PDF page image.
        """
        try:
            import base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            expert_prompt = (
                "Transcribe the visible text on this rendered PDF page, including captions, "
                "titles, labels, dates, names, and chart/table values. Preserve names and "
                "numbers exactly as printed. Connect captions to the image or chart they "
                "describe. Do not identify people from their appearance or infer facts that "
                "are not explicitly written. Return concise, searchable text; use a Markdown "
                "table when the page contains tabular values."
            )

            response = self.text_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": expert_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_mime_type};base64,{base64_image}"
                                },
                            },
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.1  # Low temperature for high factual precision
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError("Vision extraction failed for a PDF image.") from e

    def process_pdf(
        self,
        pdf_path: Path,
        include_visuals: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process a single PDF document page by page, extracting text, tables, and images."""
        doc = fitz.open(pdf_path)
        extracted_chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")

            # 1. Native text extraction with bounded, overlapping chunks
            if page_text.strip():
                page_documents = self.text_splitter.create_documents([page_text])
                for idx, page_document in enumerate(page_documents):
                    content = page_document.page_content.strip()
                    if len(content) > 20:
                        extracted_chunks.append({
                            "doc_name": pdf_path.name,
                            "page_num": page_num + 1,
                            "section": f"Page {page_num + 1} - Block {idx + 1}",
                            "content_type": "text",
                            "content": content
                        })

            # 2. Table Extraction using PyMuPDF
            tabs = page.find_tables()
            if tabs.tables:
                for t_idx, tab in enumerate(tabs.tables):
                    table_df = tab.extract()
                    table_markdown = "\n".join([" | ".join([str(cell or "") for cell in row]) for row in table_df])
                    extracted_chunks.append({
                        "doc_name": pdf_path.name,
                        "page_num": page_num + 1,
                        "section": f"Page {page_num + 1} - Table {t_idx + 1}",
                        "content_type": "table",
                        "content": table_markdown
                    })

            if include_visuals:
                # Render the whole page once so captions and their images stay together.
                has_substantial_image = any(
                    len(doc.extract_image(image_info[0]).get("image", b"")) >= 5000
                    for image_info in page.get_images(full=True)
                )
                if has_substantial_image:
                    page_image = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    ).tobytes("png")
                    vision_description = self.extract_visual_with_vision(
                        page_image,
                        image_mime_type="image/png",
                    )
                    if vision_description.strip():
                        extracted_chunks.append({
                            "doc_name": pdf_path.name,
                            "page_num": page_num + 1,
                            "section": f"Page {page_num + 1} - Rendered Page Visual",
                            "content_type": "visual_chart",
                            "content": vision_description.strip()
                        })

        return extracted_chunks

    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Discover and process all local PDFs."""
        pdf_paths = self.discover_pdfs()
        all_chunks = []
        for path in pdf_paths:
            chunks = self.process_pdf(path)
            all_chunks.extend(chunks)
        return all_chunks