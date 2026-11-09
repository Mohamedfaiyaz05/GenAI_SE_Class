from __future__ import annotations

from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit
import json
import time
import re
import os
import base64
import io
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration Profile ---
CONFIG = {
    "start_url": "https://www.tn.gov.in/scheme_list.php?dep_id=Mg==",
    "allowed_domain": "tn.gov.in",
    "output_file": Path("preprocessed_corpus.json"),
    "request_delay": 0.75,
    "min_content_length": 30,
    "vision_model": "gpt-4o-mini",
}


def create_session() -> requests.Session:
    """Create an HTTP session with automatic retries and custom headers."""
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "TN-Scheme-KG-Reader/1.0"})
    return session


def is_valid_url(url: str, allowed_base_domain: str) -> bool:
    """Validate that the URL uses HTTP/HTTPS and belongs to the target domain or subdomains."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return False
    netloc = parsed.netloc.lower()
    return netloc == allowed_base_domain or netloc.endswith(f".{allowed_base_domain}")


def normalize_text(text: str) -> str:
    """Normalize text whitespace without removing stop words or altering vocabulary."""
    return " ".join(text.split())


def extract_scheme_from_table(url: str, html: str) -> dict | None:
    """Parse the specific TN government scheme details table and validate substance."""
    soup = BeautifulSoup(html, "html.parser")
    detail_table = None

    for table in soup.find_all("table"):
        table_text = table.get_text(" ", strip=True).casefold()
        if "scheme title/name" in table_text or "scheme title" in table_text:
            detail_table = table
            break

    if detail_table is None:
        return None

    fields: dict[str, str] = {}
    for row in detail_table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 2:
            continue

        label = " ".join(cells[0].get_text(" ", strip=True).split())
        value = " ".join(
            " ".join(cell.get_text(" ", strip=True).split())
            for cell in cells[1:]
        )

        label = label.rstrip(":").strip()
        if label and value:
            fields[label] = value

    title = fields.get("Scheme Title/Name") or fields.get("Scheme Title")

    if not title or not fields:
        return None

    raw_text = " ".join(f"{k}: {v}" for k, v in fields.items())
    return {
        "url": url,
        "title": title,
        "raw_text": raw_text
    }


def process_pdf_bytes_with_vision(pdf_bytes: bytes, client: OpenAI, vision_model: str) -> list[dict]:
    """Process PDF bytes using pdfplumber for robust native text and borderless tables, plus GPT-4o-mini Vision for charts/images."""
    extracted_chunks = []

    # 1. Native text and borderless table extraction via pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_chunks.append({
                        "section": f"Page {page_num + 1} - Text",
                        "content": page_text.strip()
                    })

                tables = page.extract_tables()
                if tables:
                    for t_idx, table in enumerate(tables):
                        table_rows = []
                        for row in table:
                            cleaned_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
                            table_rows.append(" | ".join(cleaned_row))
                        
                        table_markdown = "\n".join(table_rows)
                        if table_markdown.strip():
                            extracted_chunks.append({
                                "section": f"Page {page_num + 1} - Table {t_idx + 1}",
                                "content": table_markdown.strip()
                            })
    except Exception as plumber_err:
        print(f"[WARNING] pdfplumber extraction failed: {plumber_err}")

    # 2. Visual charts and image transcription via PyMuPDF & GPT-4o-mini Vision
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            has_substantial_image = any(
                len(doc.extract_image(image_info[0]).get("image", b"")) >= 5000
                for image_info in page.get_images(full=True)
            )
            if has_substantial_image:
                page_image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
                base64_image = base64.b64encode(page_image).decode("utf-8")
                
                expert_prompt = (
                    "Transcribe visible text, captions, and chart/table values on this rendered PDF page. "
                    "Preserve names and numbers exactly as printed."
                )

                response = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": expert_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ]
                        }
                    ],
                    max_tokens=800,
                    temperature=0.1
                )
                vision_description = response.choices[0].message.content
                if vision_description and vision_description.strip():
                    extracted_chunks.append({
                        "section": f"Page {page_num + 1} - Visual/Chart",
                        "content": vision_description.strip()
                    })
    except Exception as vis_err:
        print(f"[WARNING] Vision extraction failed: {vis_err}")

    return extracted_chunks


def crawl_and_preprocess(config: dict) -> list[dict]:
    """Crawl TN government scheme listings, HTML details, and embedded PDFs into unified JSON records."""
    session = create_session()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_url = config["start_url"]
    allowed_domain = config["allowed_domain"]
    vision_model = config["vision_model"]

    department_ids = parse_qs(urlsplit(start_url).query).get("dep_id", [])
    original_department_id = department_ids[0] if department_ids else None

    queue = deque([start_url])
    visited_listings: set[str] = set()
    visited_details: set[str] = set()
    visited_pdfs: set[str] = set()
    extracted_corpus: list[dict] = []

    print(f"Starting unified crawl and multimodal extraction on: {start_url}")

    while queue:
        current_url = queue.popleft()
        if current_url in visited_listings:
            continue
        visited_listings.add(current_url)

        try:
            time.sleep(config["request_delay"])
            response = session.get(current_url, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            
            soup = BeautifulSoup(response.text, "html.parser")

            for anchor in soup.select("a[href]"):
                href = anchor.get("href", "").strip()
                if not href:
                    continue
                absolute_url = urljoin(current_url, href)
                
                if not is_valid_url(absolute_url, allowed_domain):
                    continue

                parsed_link = urlsplit(absolute_url)
                page_name = Path(parsed_link.path).name

                if page_name == "scheme_list.php":
                    dep_ids = parse_qs(parsed_link.query).get("dep_id", [])
                    if dep_ids == [original_department_id] and absolute_url not in visited_listings:
                        queue.append(absolute_url)

                elif page_name == "scheme_details.php":
                    if absolute_url not in visited_details:
                        visited_details.add(absolute_url)
                        try:
                            time.sleep(config["request_delay"])
                            detail_res = session.get(absolute_url, timeout=30)
                            detail_res.raise_for_status()
                            
                            detail_soup = BeautifulSoup(detail_res.text, "html.parser")

                            # 1. HTML Table Processing & Empty Page Check
                            scheme_data = extract_scheme_from_table(absolute_url, detail_res.text)
                            if scheme_data is not None:
                                cleaned_text = normalize_text(scheme_data["raw_text"])
                                if len(cleaned_text.strip()) >= config["min_content_length"]:
                                    extracted_corpus.append({
                                        "url": absolute_url,
                                        "title": scheme_data["title"],
                                        "cleaned_text": cleaned_text
                                    })
                                    print(f"[PROCESSED & STORED HTML] {scheme_data['title']} ({absolute_url})")

                            # 2. Embedded PDF Scanning & Aggregating Full PDF Info into Unified cleaned_text
                            for sub_anchor in detail_soup.select("a[href]"):
                                sub_href = sub_anchor.get("href", "").strip()
                                if not sub_href:
                                    continue
                                pdf_url = urljoin(absolute_url, sub_href)

                                is_pdf_link = ".pdf" in pdf_url.lower()

                                if is_valid_url(pdf_url, allowed_domain) and is_pdf_link:
                                    if pdf_url not in visited_pdfs:
                                        visited_pdfs.add(pdf_url)
                                        try:
                                            time.sleep(config["request_delay"])
                                            pdf_res = session.get(pdf_url, timeout=30)
                                            pdf_res.raise_for_status()

                                            pdf_chunks = process_pdf_bytes_with_vision(pdf_res.content, openai_client, vision_model)
                                            
                                            if pdf_chunks:
                                                combined_pdf_raw = "\n".join([f"{chunk['section']}: {chunk['content']}" for chunk in pdf_chunks])
                                                cleaned_pdf_text = normalize_text(combined_pdf_raw)

                                                if len(cleaned_pdf_text.strip()) >= config["min_content_length"]:
                                                    pdf_filename = Path(urlsplit(pdf_url).path).name or "scheme_document.pdf"
                                                    extracted_corpus.append({
                                                        "url": pdf_url,
                                                        "title": f"PDF: {pdf_filename}",
                                                        "cleaned_text": cleaned_pdf_text
                                                    })
                                                    print(f"[PROCESSED & STORED PDF] {pdf_filename} -> Unified cleaned_text ({pdf_url})")
                                        except Exception as pdf_inner_err:
                                            print(f"[ERROR] Failed PDF vision processing {pdf_url}: {pdf_inner_err}")

                        except Exception as detail_err:
                            print(f"[ERROR] Failed detail page {absolute_url}: {detail_err}")

        except Exception as error:
            print(f"[ERROR] Failed to process listing {current_url}: {error}")

    return extracted_corpus


def save_corpus(corpus: list[dict], output_path: Path) -> None:
    """Save the unified multimodal preprocessed corpus as a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
    print(f"\nUnified multimodal corpus successfully saved to: {output_path.resolve()} (Total items: {len(corpus)})")


if __name__ == "__main__":
    corpus_data = crawl_and_preprocess(CONFIG)
    if corpus_data:
        save_corpus(corpus_data, CONFIG["output_file"])
    else:
        print("No valid data extracted. Check your configuration selectors and URLs.")