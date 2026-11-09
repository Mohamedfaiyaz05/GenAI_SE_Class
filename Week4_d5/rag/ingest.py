
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge_base" / "medical_support_handbook.md"
FAISS_DIR = BASE_DIR / "storage" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
MANIFEST_NAME = "index_manifest.json"


def _section_documents(content: str, source_name: str) -> list[Document]:
    """Keep Markdown section titles attached to the text they describe."""
    sections: list[tuple[str, list[str]]] = []
    title = "Handbook"
    lines: list[str] = []

    for line in content.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if any(part.strip() for part in lines):
                sections.append((title, lines))
            title = line.lstrip("#").strip()
            lines = [line]
        else:
            lines.append(line)

    if any(part.strip() for part in lines):
        sections.append((title, lines))

    return [
        Document(
            page_content="\n".join(section_lines).strip(),
            metadata={"source": source_name, "section_title": section_title},
        )
        for section_title, section_lines in sections
    ]


def ingest_documents(
    knowledge_file: Path = KNOWLEDGE_FILE,
    faiss_dir: Path = FAISS_DIR,
) -> int:
    """Read the medical handbook and build the FAISS index."""

    load_dotenv(BASE_DIR / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing in .env")

    if not knowledge_file.is_file():
        raise FileNotFoundError(
            f"Knowledge document not found: {knowledge_file}"
        )

    content = knowledge_file.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(f"Knowledge document is empty: {knowledge_file}")

    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(
        _section_documents(content, knowledge_file.name)
    )
    if not chunks:
        raise ValueError(f"No text chunks were created from {knowledge_file}")

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["source_sha256"] = source_hash
        section_title = chunk.metadata.get("section_title", "")
        if section_title and not chunk.page_content.startswith(section_title):
            chunk.page_content = f"{section_title}\n{chunk.page_content}"

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    faiss_dir.mkdir(parents=True, exist_ok=True)

    vector_db.save_local(str(faiss_dir))

    required_files = (faiss_dir / "index.faiss", faiss_dir / "index.pkl")
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(
            "FAISS save completed without creating expected file(s): "
            + ", ".join(missing_files)
            + f" in {faiss_dir}"
        )

    manifest = {
        "version": 1,
        "source_file": knowledge_file.name,
        "source_sha256": source_hash,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
    }
    manifest_path = faiss_dir / MANIFEST_NAME
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    return len(chunks)


if __name__ == "__main__":
    chunk_count = ingest_documents()
    print(
        f"Indexed {chunk_count} handbook chunk(s) into "
        f"{FAISS_DIR}"
    )
