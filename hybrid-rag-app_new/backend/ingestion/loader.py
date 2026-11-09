"""Discover PDF files and extract text with page metadata."""
import os
from pathlib import Path
from typing import List, Dict, Any
import base64
import fitz  # PyMuPDF
from openai import OpenAI
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """
    Advanced PDF loader that extracts text, tables, and embedded images/charts,
    and converts images/charts into detailed descriptive text using OpenAI gpt-4o-mini.
    """
    def __init__(self, knowledge_base_dir: str = None):
        self.knowledge_base_dir = Path(knowledge_base_dir or settings.KNOWLEDGE_BASE_DIR)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client for vision-based image-to-text conversion
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def discover_pdfs(self) -> List[Path]:
        """Scans the knowledge base folder and returns a list of PDF file paths."""
        if not self.knowledge_base_dir.exists():
            logger.warning(f"Knowledge base directory {self.knowledge_base_dir} does not exist.")
            return []
        
        pdf_files = list(self.knowledge_base_dir.glob("*.pdf"))
        logger.info(f"Discovered {len(pdf_files)} PDF(s) in {self.knowledge_base_dir}")
        return pdf_files

    def encode_image_to_base64(self, image_bytes: bytes, image_ext: str) -> str:
        """Encodes raw image bytes into a base64 data URL string."""
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = f"image/{image_ext}" if image_ext in ["png", "jpeg", "jpg", "webp"] else "image/jpeg"
        return f"data:{mime_type};base64,{encoded_str}"

    def convert_image_to_text(self, image_bytes: bytes, image_ext: str, filename: str, page_num: int) -> str:
        """
        Uses OpenAI gpt-4o-mini vision capabilities to transcribe, extract, and describe 
        charts, tables, and images found in the document into searchable text.
        """
        try:
            data_url = self.encode_image_to_base64(image_bytes, image_ext)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document parser and data extractor. Your task is to analyze embedded images, charts, diagrams, or tables extracted from professional PDFs, and convert them into comprehensive, precise, structured text. Describe all data points, axes, labels, text content, and key insights so it can be fully indexed by a RAG system."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please extract and describe all text, data, charts, or visual information from this image found in document '{filename}' on page {page_num}."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            description = response.choices[0].message.content
            return description
        except Exception as e:
            logger.exception(
                "Image extraction failed for %s page %s.",
                filename,
                page_num,
            )
            raise RuntimeError(
                f"Image extraction failed for {filename} page {page_num}."
            ) from e

    def load_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text, tables, and visual elements page-by-page, converting images/charts 
        to text via gpt-4o-mini so every asset becomes a rich, searchable document chunk.
        """
        documents = []
        doc = None
        try:
            doc = fitz.open(file_path)
            filename = file_path.name
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_idx = page_num + 1
                
                # 1. Extract standard text and tables
                text = page.get_text()
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_idx,
                            "file_path": str(file_path),
                            "content_type": "text_or_table"
                        }
                    })
                
                # 2. Extract embedded images, charts, and diagrams
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Convert image/chart to descriptive text using gpt-4o-mini
                        logger.info(f"Processing image {img_index + 1} from {filename} (Page {page_idx}) with gpt-4o-mini...")
                        image_description = self.convert_image_to_text(image_bytes, image_ext, filename, page_idx)
                        
                        if image_description.strip():
                            documents.append({
                                "text": f"[Extracted Image/Chart Analysis from Page {page_idx}]:\n{image_description}",
                                "metadata": {
                                    "source": filename,
                                    "page": page_idx,
                                    "file_path": str(file_path),
                                    "content_type": "image_or_chart_analysis",
                                    "image_index": img_index + 1
                                }
                            })
                    except Exception as img_err:
                        raise RuntimeError(
                            f"Failed to process image {img_index + 1} on page {page_idx} "
                            f"of {filename}."
                        ) from img_err
                        
            logger.info(f"Successfully loaded and processed {filename} ({len(documents)} total blocks including text and image transcriptions)")
        except Exception as e:
            logger.exception("Failed to load PDF %s.", file_path)
            raise RuntimeError(f"Failed to load PDF {file_path.name}.") from e
        finally:
            if doc is not None:
                doc.close()
            
        return documents

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Discovers all PDFs in the directory and loads their pages and image descriptions."""
        pdf_files = self.discover_pdfs()
        all_documents = []
        
        for file_path in pdf_files:
            print(f"\n--- Processing PDF: {file_path.name} ---")
            docs = self.load_pdf(file_path)
            all_documents.extend(docs)
            
        logger.info(f"Total extracted documents (text + analyzed images): {len(all_documents)}")
        return all_documents