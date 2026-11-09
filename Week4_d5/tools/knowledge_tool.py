
import json

from crewai.tools import tool
from rag.retriever import MedicalKnowledgeRetriever


# Initialize the retriever once
retriever = MedicalKnowledgeRetriever()


@tool("medical_knowledge_search")
def medical_knowledge_search(question: str) -> str:
    """
    Search the internal MediCare Plus medical handbook
    using FAISS semantic retrieval.

    Use this tool for questions about appointments,
    refunds, blood tests, laboratory reports,
    and clinic policies.

    Returns relevant document chunks and source metadata.
    Retrieved chunks are not verified answers.
    """

    # Retrieve relevant chunks from FAISS
    chunks = retriever.retrieve(
        question=question,
        k=3
    )

    # Prepare structured output
    result = {
        "question": question,
        "retrieved_count": len(chunks),
        "chunks": chunks
    }

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )
