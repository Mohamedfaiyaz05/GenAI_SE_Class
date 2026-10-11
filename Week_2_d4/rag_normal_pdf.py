import os
import tempfile
import shutil
import sys
import streamlit as st
from typing import List, TypedDict, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

# ==========================================
# 1. ENVIRONMENT & LANGSMITH SETUP
# ==========================================
load_dotenv()

PROJECT_NAME = "Production-CRAG-FAISS-App"
DB_PATH = "faiss_index"

def check_environment() -> bool:
    """Check required environment variables for LangSmith and OpenAI without printing secret values."""
    required = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
    }

    missing = [name for name, value in required.items() if not value]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}. Please set them in your .env file or terminal.")
        return False

    tracing_value = required["LANGSMITH_TRACING"].strip().lower()
    if tracing_value not in {"true", "1", "yes", "on"}:
        st.error('LANGSMITH_TRACING is set, but not to a true value. Set it to "true".')
        return False

    return True

# Run preflight check
is_env_valid = check_environment()


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str


class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# ==========================================
# 3. CRAG WORKFLOW BUILDER
# ==========================================
def build_crag_graph(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Document Grader
    structured_llm_grader = llm.with_structured_output(GradeDocument)
    grader_system = (
        "You are a grader evaluating whether a retrieved document is relevant to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'.\n"
        "Otherwise, grade it as 'no'."
    )
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", grader_system),
        ("human", "Retrieved document:\n\n{document}\n\nUser Question: {question}")
    ])
    document_grader = grader_prompt | structured_llm_grader

    # Strict Generation Chain
    rag_system_prompt = (
        "You are an expert technical Q&A assistant.\n"
        "Your task is to provide comprehensive, well-structured answers based strictly and only on the provided context.\n\n"
        "RULES:\n"
        "1. DO NOT use external knowledge or make assumptions outside the provided context. Every claim must be grounded in the text.\n"
        "2. Match the depth of your response to the user's request (provide a detailed explanation if requested, or a brief overview if concise is preferred).\n"
        "3. Synthesize and combine information across multiple retrieved context chunks to build a complete and coherent answer.\n"
        "4. If the context completely lacks the information needed to address the question, state:\n"
        "   'I am sorry, but the provided document does not contain enough information to answer this question.'\n"
        "   (Note: If partial relevant information is available, explain what is known rather than rejecting the question outright).\n"
        "5. Keep answers factual, clear, and professional.\n\n"
        "CONTEXT:\n{context}"
        )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("human", "{question}")
    ])
    rag_chain = rag_prompt | llm

    # --- NODES ---
    def retrieve(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = retriever.invoke(question)
        return {"documents": docs, "question": question}

    def grade_documents(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = state["documents"]
        filtered_docs = []
        
        for d in docs:
            score = document_grader.invoke({"question": question, "document": d.page_content})
            if score.binary_score.lower() == "yes":
                filtered_docs.append(d)
                
        return {"documents": filtered_docs, "question": question}

    def generate(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = state["documents"]
        context_text = "\n\n".join([doc.page_content for doc in docs])
        response = rag_chain.invoke({"context": context_text, "question": question})
        return {"generation": response.content, "documents": docs}

    def fallback_no_context(state: GraphState) -> Dict[str, Any]:
        return {
            "generation": "I am sorry, but the provided document does not contain enough information to answer this question.",
            "documents": []
        }

    # --- CONDITIONAL ROUTER ---
    def decide_to_generate(state: GraphState) -> str:
        """Route to generator if relevant docs exist; otherwise route to fallback."""
        if not state["documents"]:
            return "no_context"
        return "generate"

    # --- GRAPH ASSEMBLY ---
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback_no_context", fallback_no_context)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "no_context": "fallback_no_context"
        }
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback_no_context", END)

    return workflow.compile()


# ==========================================
# 4. STREAMLIT FRONTEND & PERSISTENCE
# ==========================================
st.set_page_config(page_title="Production CRAG with LangGraph & FAISS", layout="wide")
st.title("📄 Production Corrective RAG (CRAG)")

if not is_env_valid:
    st.warning("⚠️ Please configure your environment variables (OpenAI & LangSmith keys, Tracing, and Project name) before querying.")

# Initialize Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Automatically load the local FAISS index on application startup if it exists
if "retriever" not in st.session_state and os.path.exists(DB_PATH):
    try:
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        st.session_state.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 8}
        )
    except Exception as e:
        st.error(f"Error loading local vector store: {e}")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # Process PDF if uploaded and no retriever is active in session
    if uploaded_file is not None:
        print("enterd inside uploaded_file is not None")
        if "retriever" not in st.session_state:
            print("enterd inside retriever is not in session_state")
            with st.spinner("Processing PDF & Indexing in FAISS..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_pdf_path = tmp_file.name

                loader = PyPDFLoader(temp_pdf_path)
                raw_docs = loader.load()

                for d in raw_docs:
                    d.metadata["source"] = uploaded_file.name

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=250
                )
                chunks = text_splitter.split_documents(raw_docs)

                # Create vector store from chunks and save locally
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(DB_PATH)
                
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 10}
                )
                os.remove(temp_pdf_path)
                st.success("PDF processed, indexed, and saved locally!")

    # Utility button to clear the local database index if needed
    if os.path.exists(DB_PATH):
        if st.button("🗑️ Clear Local Index"):
            shutil.rmtree(DB_PATH)
            if "retriever" in st.session_state:
                del st.session_state.retriever
            st.success("Local index deleted! Please refresh or upload a new file.")
            st.rerun()

if "retriever" in st.session_state:
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Executing CRAG Pipeline..."):
            app = build_crag_graph(st.session_state.retriever)
            result = app.invoke({"question": question})

            st.subheader("💡 Answer:")
            st.write(result["generation"])

            st.subheader("📌 Retrieved Source Passages & Page Numbers:")
            retrieved_docs = result.get("documents", [])
            
            if retrieved_docs:
                for idx, doc in enumerate(retrieved_docs):
                    page = doc.metadata.get("page", 0) + 1  # 1-indexed page display
                    source = doc.metadata.get("source", "Uploaded PDF")
                    
                    with st.expander(f"Source {idx+1}: {source} — Page {page}"):
                        st.write(doc.page_content)
            else:
                st.info("No matching source pages passed relevance evaluation.")
else:
    st.info("Please upload a PDF file in the sidebar to begin (or ensure your local index exists).")