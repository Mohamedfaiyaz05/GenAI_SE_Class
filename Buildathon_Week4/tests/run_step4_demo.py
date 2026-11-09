"""Run: python -m tests.run_step4_demo (requires OPENAI_API_KEY)."""
from __future__ import annotations
from dataclasses import asdict
import json
from rag.hybrid_retriever import HybridRetriever


def main() -> None:
    retriever = HybridRetriever()
    print("\nSTEP 4: FIRST BUILD / REOPEN\n")
    print(json.dumps(asdict(retriever.build()), indent=2))
    print("\nSTEP 4: SECOND BUILD (should use persisted FAISS)\n")
    print(json.dumps(asdict(retriever.build()), indent=2))
    cases = [
        ("My order is delayed, when should I contact support?", ["Shipping"]),
        ("How many days do I have to return my shoes?", ["Refund"]),
        ("I forgot my password. How do I reset it?", ["Account"]),
        ("My blender arrived broken. Can I replace it?", ["Product", "Refund"]),
    ]
    for query, categories in cases:
        print("\n" + "=" * 75)
        print("INPUT:", query, "\nCATEGORIES:", categories)
        results = retriever.search(query, category=categories, top_k=3)
        print("OUTPUT: retrieved", len(results), "policy chunks")
        for n, r in enumerate(results, 1):
            print(f"  {n}. [{r['category']}] {r['source']} | {r['section']} | "
                  f"RRF={r['rrf_score']:.5f} | via {', '.join(r['retrieval_methods'])}")
            print("     ", r["content"][:250].replace("\n", " "))


if __name__ == "__main__":
    main()
