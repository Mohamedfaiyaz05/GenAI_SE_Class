
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = {
    "shipping_policy.md": "Shipping",
    "refund_policy.md": "Refund",
    "account_and_login.md": "Account",
    "product_faq.md": "Product",
}


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    source: str
    category: str
    section: str
    policy_id: str
    policy_version: str
    effective_date: str
    content: str


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_field(text: str, names: tuple[str, ...]) -> str:
    for name in names:
        pattern = rf"(?im)^\s*\*\*{re.escape(name)}:\*\*\s*(.+?)\s*$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def split_sections(text: str) -> list[tuple[str, str]]:
    """Keep heading context with each section; ignore metadata-only preamble."""
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    buffer: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{2,3}\s+", line):
            if buffer and any(x.strip() for x in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = re.sub(r"^#{2,3}\s+", "", line).strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer and any(x.strip() for x in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def ingest_policy(path: Path, category: str, chunk_size: int, overlap: int) -> list[PolicyChunk]:
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if len(raw.strip()) < 100:
        raise ValueError(f"Policy is empty or too short: {path.name}")
    policy_id = extract_field(raw, ("Policy ID",))
    version = extract_field(raw, ("Version", "Policy version"))
    effective = extract_field(raw, ("Effective date", "Effective Date"))
    if not policy_id or not version or not effective:
        raise ValueError(f"Missing Policy ID, Version or Effective date in {path.name}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    result: list[PolicyChunk] = []
    for section, body in split_sections(raw):
        # Do not index the title/metadata preamble as an independent chunk.
        if section == "Overview":
            continue
        for index, piece in enumerate(splitter.split_text(body)):
            if not piece.strip():
                continue
            digest = sha256(f"{path.name}|{version}|{section}|{index}|{piece}")[:20]
            result.append(PolicyChunk(
                chunk_id=f"policy_{digest}", source=path.name, category=category,
                section=section, policy_id=policy_id, policy_version=version,
                effective_date=effective, content=piece.strip(),
            ))
    if not result:
        raise ValueError(f"No indexable sections (## headings) in {path.name}")
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == serialized:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def build_index(knowledge_dir: Path, output_dir: Path, chunk_size: int = 1000, overlap: int = 150) -> dict:
    """Incrementally rebuild chunks; unchanged files retain their saved chunks.

    Note: embedding skipping is implemented in Step 4 using this registry;
    this Step 3 function never calls an embedding API.
    """
    if chunk_size < 100 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be >=100 and 0 <= overlap < chunk_size")
    missing = [name for name in POLICY_FILES if not (knowledge_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required policies: {', '.join(missing)}")

    destination = output_dir / "policy_chunks.json"
    previous = None
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
    compatible = bool(previous and previous.get("schema_version") == 2
                      and previous.get("chunk_size") == chunk_size
                      and previous.get("chunk_overlap") == overlap)
    old_files = previous.get("files", {}) if compatible else {}
    old_chunks = {}
    if compatible:
        for chunk in previous.get("chunks", []):
            old_chunks.setdefault(chunk["source"], []).append(chunk)

    chunks = []
    signatures = {}
    files = {}
    reused = []
    changed = []
    for filename, category in POLICY_FILES.items():
        path = knowledge_dir / filename
        raw_bytes = path.read_bytes()
        signature = hashlib.sha256(raw_bytes).hexdigest()
        size = len(raw_bytes)
        signatures[filename] = signature
        files[filename] = {"sha256": signature, "size_bytes": size}
        old = old_files.get(filename, {})
        if (compatible and old.get("size_bytes") == size
                and old.get("sha256") == signature
                and old_chunks.get(filename)):
            chunks.extend(old_chunks[filename])
            reused.append(filename)
        else:
            chunks.extend(asdict(c) for c in ingest_policy(path, category, chunk_size, overlap))
            changed.append(filename)

    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs detected")
    payload = {
        "schema_version": 2,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "files": files,
        "source_signatures": signatures,
        "chunks": chunks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, payload)
    return {
        "documents": len(POLICY_FILES),
        "chunks": len(chunks),
        "reused_documents": reused,
        "processed_documents": changed,
        "by_category": {category: sum(c["category"] == category for c in chunks)
                        for category in POLICY_FILES.values()},
        "output": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ShopEase policies into local JSON chunks")
    parser.add_argument("--knowledge-dir", type=Path, default=ROOT / "knowledge_base")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "ingestion")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()
    summary = build_index(args.knowledge_dir, args.output_dir, args.chunk_size, args.overlap)
    print("\nShopEase policy ingestion complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
