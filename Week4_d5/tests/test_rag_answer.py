"""Interactive smoke test for handbook retrieval and answer generation."""

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.support_agent import run_support_agent
from rag.retriever import MedicalKnowledgeRetriever


def main() -> None:
    load_dotenv()
    question = "What is the refund policy here?"

    retriever = MedicalKnowledgeRetriever()
    chunks = retriever.retrieve(question, k=3)

    if not chunks:
        print("No matching handbook chunks were retrieved.")
        return

    evidence = "\n\n".join(
        chunk["content"]
        for chunk in chunks
        if isinstance(chunk.get("content"), str)
        and chunk["content"].strip()
    )
    if not evidence:
        print("The retrieved chunks did not contain readable text.")
        return

    print(f"Question: {question}\n")
    print("Retrieved handbook sections:")
    for chunk in chunks:
        print(
            f"- chunk {chunk.get('chunk_id')} "
            f"(distance {chunk.get('distance_score', 0):.4f})"
        )

    answer = run_support_agent(question=question, evidence=evidence)
    print("\nFinal answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
