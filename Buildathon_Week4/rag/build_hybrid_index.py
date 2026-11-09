"""CLI entry point: python -m rag.build_hybrid_index"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from rag.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=1536)
    args = parser.parse_args()
    retriever = HybridRetriever(chunks_path=args.chunks, index_dir=args.index_dir,
                                model=args.model, dimensions=args.dimensions)
    stats = retriever.build()
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
