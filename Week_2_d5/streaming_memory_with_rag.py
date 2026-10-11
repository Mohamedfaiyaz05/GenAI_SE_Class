import os
import tempfile
import shutil
import sys
import streamlit as st
from typing import List, TypedDict, Dict, Any, Annotated
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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

is_env_valid = check_environment()


# ==========================================
# 2. STATE DEFINITION WITH CHAT MEMORY
# ==========================================
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: List[Document]
    generation: str


class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# ==========================================
# 3. CRAG WORKFLOW BUILDER WITH CHECKPOINTER
# ==========================================
def build_crag_graph(retriever, checkpointer: MemorySaver):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)

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
        messages = state["messages"]
        # Extract the latest human message as the current question
        question = messages[-1].content if messages else ""
        docs = retriever.invoke(question)
        return {"documents": docs}

    def grade_documents(state: GraphState) -> Dict[str, Any]:
        messages = state["messages"]
        question = messages[-1].content if messages else ""
        docs = state["documents"]
        filtered_docs = []
        
        for d in docs:
            score = document_grader.invoke({"question": question, "document": d.page_content})
            if score.binary_score.lower() == "yes":
                filtered_docs.append(d)
                
        return {"documents": filtered_docs}

    def generate(state: GraphState) -> Dict[str, Any]:
        messages = state["messages"]
        question = messages[-1].content if messages else ""
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

    # Compile graph with memory checkpointer attached
    return workflow.compile(checkpointer=checkpointer)


# ==========================================
# 4. STREAMLIT FRONTEND & SESSION SETUP
# ==========================================
st.set_page_config(page_title="Production CRAG Chatbot with Memory", layout="wide")
st.title("📄 Production CRAG Chatbot — Multi-PDF & Memory Engine")

if not is_env_valid:
    st.warning("⚠️ Please configure your environment variables (OpenAI & LangSmith keys, Tracing, and Project name) before querying.")

# Initialize Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Session Memory Saver and Thread ID for conversation persistence
if "memory_saver" not in st.session_state:
    st.session_state.memory_saver = MemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_user_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Automatically load local FAISS index on startup if it exists
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

# ==========================================
# 5. SIDEBAR: DOCUMENT INGESTION
# ==========================================
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Index All PDFs"):
            with st.spinner("Processing documents & building unified FAISS index..."):
                all_chunks = []
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=250
                )
                
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_pdf_path = tmp_file.name

                    try:
                        loader = PyPDFLoader(temp_pdf_path)
                        raw_docs = loader.load()

                        for d in raw_docs:
                            d.metadata["source"] = uploaded_file.name

                        chunks = text_splitter.split_documents(raw_docs)
                        all_chunks.extend(chunks)
                    finally:
                        if os.path.exists(temp_pdf_path):
                            os.remove(temp_pdf_path)

                if all_chunks:
                    vectorstore = FAISS.from_documents(all_chunks, embeddings)
                    vectorstore.save_local(DB_PATH)
                    
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="similarity", search_kwargs={"k": 10}
                    )
                    st.success(f"Successfully processed {len(uploaded_files)} PDF(s) and indexed {len(all_chunks)} chunks!")
                else:
                    st.warning("No text could be extracted from the uploaded PDFs.")

    if os.path.exists(DB_PATH):
        if st.button("🗑️ Clear Local Index & Memory"):
            shutil.rmtree(DB_PATH)
            if "retriever" in st.session_state:
                del st.session_state.retriever
            st.session_state.messages = []
            st.session_state.memory_saver = MemorySaver()
            st.success("Local index and chat memory cleared!")
            st.rerun()

# ==========================================
# 6. CHAT INTERFACE & STREAMING
# ==========================================
if "retriever" in st.session_state:
    # Render existing conversation history bubbles
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "documents" in message and message["documents"]:
                with st.expander("📌 Retrieved Source Passages & Page Numbers"):
                    for idx, doc in enumerate(message["documents"]):
                        page = doc.metadata.get("page", 0) + 1
                        source = doc.metadata.get("source", "Uploaded PDF")
                        st.markdown(f"**Source {idx+1}: {source} — Page {page}**")
                        st.text(doc.page_content)

    # Chat input box
    if question := st.chat_input("Ask a question across your documents..."):
        # Append & display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate & stream assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Compile graph with persistent memory
            app = build_crag_graph(st.session_state.retriever, st.session_state.memory_saver)
            
            inputs = {"messages": [HumanMessage(content=question)]}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            def stream_response_generator():
                # Stream token deltas from the graph generation node
                for chunk, metadata in app.stream(inputs, config=config, stream_mode="messages"):
                    if metadata.get("langgraph_node") == "generate" and chunk.content:
                        yield chunk.content

            try:
                # Real-time token streaming into UI
                full_generation = message_placeholder.write_stream(stream_response_generator())
            except Exception as e:
                # Fallback if streaming encounters any edge case
                result = app.invoke(inputs, config=config)
                full_generation = result.get("generation", "Error generating response.")
                message_placeholder.markdown(full_generation)

            # Retrieve final state documents for source citations
            current_state = app.get_state(config=config)
            retrieved_docs = current_state.values.get("documents", []) if current_state.values else []

            if retrieved_docs:
                with st.expander("📌 Retrieved Source Passages & Page Numbers"):
                    for idx, doc in enumerate(retrieved_docs):
                        page = doc.metadata.get("page", 0) + 1
                        source = doc.metadata.get("source", "Uploaded PDF")
                        st.markdown(f"**Source {idx+1}: {source} — Page {page}**")
                        st.text(doc.page_content)

            # Save assistant response to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_generation,
                "documents": retrieved_docs
            })
else:
    st.info("Please upload one or more PDF files in the sidebar and click **Process & Index All PDFs** to begin chatting.")