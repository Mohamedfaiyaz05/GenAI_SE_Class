
from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str


# Create data following the expected structure
state: GraphState = {
    "question": "What is Python?",
    "documents": [
        Document(
            page_content="Python is a high-level programming language."
        ),
        Document(
            page_content="Python is used in AI and data engineering."
        )
    ],
    "generation": ""
}


# Read the question
print(state["question"])

# Read the retrieved documents
for document in state["documents"]:
    print(document.page_content)

# Update the answer
state["generation"] = (
    "Python is a high-level programming language "
    "used in AI and data engineering."
)

print(state["generation"])


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

grader = llm.with_structured_output(GradeDocument)

result = grader.invoke("""
Question: What is Python used for?

Document: Python is widely used in artificial intelligence
and data engineering.

Is this document relevant? Return yes or no.
""")

print(result.binary_score)