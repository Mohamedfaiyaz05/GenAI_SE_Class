import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

load_dotenv()

app = FastAPI(title="Production CRAG API", version="1.0.0")

DB_PATH = "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Pydantic Schemas ---
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question for the CRAG system")

class DocumentMetadata(BaseModel):
    source: str
    page: int

class SourceDocument(BaseModel):
    content: str
    metadata: DocumentMetadata

class QueryResponse(BaseModel):
    generation: str
    documents: List[SourceDocument]

class GradeDocument(BaseModel):
    binary_score: str = Field(description="Binary score 'yes' or 'no' for relevance")


# --- Helper: Build CRAG Graph ---
def get_retriever():
    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    return None

def build_crag_graph(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocument)
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader evaluating whether a retrieved document is relevant to a user question. Grade 'yes' or 'no'."),
        ("human", "Retrieved document:\n\n{document}\n\nUser Question: {question}")
    ])
    document_grader = grader_prompt | structured_llm_grader

    rag_system_prompt = (
        "You are an expert technical Q&A assistant.\n"
        "Answer strictly based on the provided context. If context lacks info, state your fallback message.\n\n"
        "CONTEXT:\n{context}"
    )
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("human", "{question}")
    ])
    rag_chain = rag_prompt | llm

    # State Definition
    class GraphState(TypedDict if 'TypedDict' in globals() else dict):
        question: str
        documents: List[Document]
        generation: str

    workflow = StateGraph(GraphState)

    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "question": state["question"]}

    def grade_documents(state):
        filtered_docs = []
        for d in state["documents"]:
            score = document_grader.invoke({"question": state["question"], "document": d.page_content})
            if score.binary_score.lower() == "yes":
                filtered_docs.append(d)
        return {"documents": filtered_docs, "question": state["question"]}

    def generate(state):
        context_text = "\n\n".join([doc.page_content for doc in state["documents"]])
        response = rag_chain.invoke({"context": context_text, "question": state["question"]})
        return {"generation": response.content, "documents": state["documents"]}

    def fallback_no_context(state):
        return {
            "generation": "I am sorry, but the provided document does not contain enough information to answer this question.",
            "documents": []
        }

    def decide_to_generate(state):
        return "generate" if state["documents"] else "no_context"

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback_no_context", fallback_no_context)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "no_context": "fallback_no_context"})
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback_no_context", END)

    return workflow.compile()


# --- API Endpoints ---
@app.get("/health")
def health_check():
    has_keys = bool(os.getenv("OPENAI_API_KEY"))
    index_exists = os.path.exists(DB_PATH)
    return {"status": "healthy", "openai_key_configured": has_keys, "vector_index_exists": index_exists}

@app.post("/upload", summary="Upload and Index PDF Document")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        raw_docs = loader.load()
        for d in raw_docs:
            d.metadata["source"] = file.filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = splitter.split_documents(raw_docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(DB_PATH)
        os.remove(temp_path)

        return {"message": "PDF successfully processed and indexed in FAISS.", "filename": file.filename, "chunks_indexed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, summary="Execute CRAG Pipeline")
def query_crag(request: QueryRequest):
    retriever = get_retriever()
    if not retriever:
        raise HTTPException(status_code=400, detail="No vector index found. Please upload a PDF first.")
    
    app_graph = build_crag_graph(retriever)
    result = app_graph.invoke({"question": request.question})

    formatted_docs = []
    for doc in result.get("documents", []):
        formatted_docs.append(SourceDocument(
            content=doc.page_content,
            metadata=DocumentMetadata(
                source=doc.metadata.get("source", "Uploaded PDF"),
                page=doc.metadata.get("page", 0) + 1
            )
        ))

    return QueryResponse(
        generation=result.get("generation", ""),
        documents=formatted_docs
    )

@app.delete("/index", summary="Clear Local FAISS Index")
def clear_index():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        return {"message": "Local vector index cleared successfully."}
    return {"message": "No index found to clear."}