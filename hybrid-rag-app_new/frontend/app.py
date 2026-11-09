"""Main Streamlit application."""
import html
import streamlit as st
import uuid
from api_client import (
    get_stats_api,
    stream_chat_api,
    sync_knowledge_base_api,
    upload_document_api,
)

st.set_page_config(
    page_title="Hybrid RAG Knowledge Assistant",
    page_icon="✨",
    layout="wide"
)

# Custom Modern & Colorful CSS Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #2a2a40 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #d1d5db !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: #e5e7eb !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        color: #f9fafb !important;
    }
    .active-document {
        color: #f3f4f6;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 6px;
        padding: 8px 10px;
        margin: 5px 0;
        overflow-wrap: anywhere;
        font-size: 0.85rem;
    }
    .chunk-card {
        background: #ffffff;
        border-left: 5px solid #6366f1;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #1f2937;
    }
    .graph-card {
        background: #ffffff;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #1f2937;
    }
    .badge-correct {
        background-color: #d1fae5;
        color: #065f46;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-ambiguous {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize Message History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📂 Knowledge Control")
    st.markdown("Upload documents & inspect repository status.")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file:
        if st.button("🚀 Ingest & Index Document", use_container_width=True):
            with st.spinner("Processing embeddings & extracting graph..."):
                res = upload_document_api(uploaded_file)
                if res.get("status") == "success":
                    st.success(res.get("message"))
                    if res.get("graph_error"):
                        st.warning(
                            "Semantic vectors were persisted, but graph synchronization "
                            f"failed: {res['graph_error']}"
                        )
                    st.rerun()
                else:
                    st.error(f"Error: {res.get('message')}")

    st.markdown("---")
    st.subheader("📊 System Stats")
    stats = get_stats_api()
    if stats.get("status") == "error":
        st.error(f"Backend status is unavailable: {stats.get('message')}")
    else:
        st.metric("PDF documents", stats.get("total_documents", 0))
        if stats.get("total_documents", 0) > 0 and (
            not stats.get("vector_index_ready")
            or stats.get("vector_chunks", 0) == 0
            or stats.get("graph_status") != "ready"
        ):
            st.warning(
                "Documents are present, but one or more search indexes are not ready. "
                "Index the knowledge base before chatting."
            )
            if stats.get("vector_index_error"):
                st.error(f"Semantic indexing failed: {stats['vector_index_error']}")
            if st.button("🔄 Index / Sync PDFs", use_container_width=True):
                with st.spinner("Building semantic and graph indexes. This may take a while..."):
                    sync_result = sync_knowledge_base_api()
                if sync_result.get("status") == "success":
                    st.success(sync_result.get("message", "Index synchronization completed."))
                    if sync_result.get("graph_error"):
                        st.warning(
                            "Semantic vectors were persisted, but graph synchronization "
                            f"failed: {sync_result['graph_error']}"
                        )
                    st.rerun()
                else:
                    st.error(f"Indexing failed: {sync_result.get('message')}")

        if stats.get("vector_index_ready"):
            st.caption(f"Semantic chunks indexed: {stats.get('vector_chunks', 0)}")
        graph_stats = stats.get("graph", {})
        if stats.get("graph_status") == "ready":
            st.caption(
                f"Graph: {graph_stats.get('nodes', 0)} nodes, "
                f"{graph_stats.get('relationships', 0)} relationships"
            )
        elif stats.get("graph_status") == "error":
            st.warning(f"Knowledge graph status could not be checked: {graph_stats.get('error')}")
        elif stats.get("graph_status") == "empty":
            st.warning("Neo4j is reachable, but no knowledge-graph entities are loaded.")
        if stats.get("graph_sync_error"):
            st.error(f"Knowledge-graph ingestion failed: {stats['graph_sync_error']}")
    
    if stats.get("documents"):
        with st.expander("📚 Active Documents"):
            for doc in stats["documents"]:
                st.markdown(
                    f"<div class='active-document'>📄 {html.escape(doc)}</div>",
                    unsafe_allow_html=True,
                )
                
    st.markdown("---")
    if st.button("🧹 Clear Chat Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Main Header
st.title("⚡ Hybrid RAG Knowledge Assistant")
st.caption("Powered by FAISS, Neo4j, Corrective RAG (CRAG), & Multi-Turn Session Memory")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("role") == "assistant" and (message.get("sources") or message.get("vector_results") or message.get("graph_results")):
            with st.expander("🔍 Transparency & Retrieval Inspector"):
                tab1, tab2, tab3 = st.tabs(["📚 Cited Sources", "📄 Verified Vector Chunks", "🕸️ Knowledge Graph Relations"])
                
                with tab1:
                    sources = message.get("sources", [])
                    if sources:
                        for src in sources:
                            st.markdown(f"- **Document:** `{src.get('source')}` | **Page:** `{src.get('page')}`")
                    else:
                        st.info("No specific sources cited.")
                        
                with tab2:
                    v_chunks = message.get("vector_results", [])
                    if v_chunks:
                        for idx, chunk in enumerate(v_chunks, 1):
                            meta = chunk.get("metadata", {})
                            grade = chunk.get("crag_grade", "correct")
                            score = chunk.get("score", 0.0)
                            badge_class = "badge-correct" if grade == "correct" else "badge-ambiguous"
                            
                            st.markdown(f"""
                            <div class="chunk-card">
                                <b>Chunk {idx}</b> | Score: <code>{score:.4f}</code> | Grade: <span class="{badge_class}">{grade.upper()}</span><br>
                                <small><b>Source:</b> {meta.get('source') or meta.get('filename')} (Page {meta.get('page') or meta.get('page_number')})</small>
                                <p style="margin-top: 8px; font-size: 13px;">{chunk.get('text')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No semantic chunks used for this response.")
                        
                with tab3:
                    g_results = message.get("graph_results", [])
                    if g_results:
                        for idx, g in enumerate(g_results, 1):
                            source = g.get("source", "Unknown")
                            rel = g.get("relation")
                            target = g.get("target")
                            desc = g.get("description", "")
                            
                            if rel and target:
                                content = f"<b>{source}</b> &nbsp;&nbsp;➔&nbsp;&nbsp; <code>[{rel}]</code> &nbsp;&nbsp;➔&nbsp;&nbsp; <b>{target}</b>"
                            else:
                                content = f"<b>{source}</b>: {desc}"
                                
                            st.markdown(f"""
                            <div class="graph-card">
                                <b>Graph Entity/Relation {idx}</b><br>
                                <p style="margin-top: 5px; font-size: 13px;">{content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No direct knowledge graph entities matched.")

# User Input Handling
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""
        sources_cited = []
        vector_results = []
        graph_results = []
        stream_error = None
        
        for event in stream_chat_api(prompt, session_id=st.session_state.session_id):
            if event.get("type") == "token":
                full_response += event.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            elif event.get("type") == "metadata":
                sources_cited = event.get("sources_cited", [])
                vector_results = event.get("vector_results", [])
                graph_results = event.get("graph_results", [])
            elif event.get("type") == "error":
                stream_error = event.get("content", "The chat request failed.")
            elif event.get("type") == "status":
                status_placeholder.info(event.get("content", "Preparing document indexes..."))

        status_placeholder.empty()
        if stream_error:
            message_placeholder.error(stream_error)
        else:
            message_placeholder.markdown(full_response)
        
        if sources_cited or vector_results or graph_results:
            with st.expander("🔍 Transparency & Retrieval Inspector"):
                tab1, tab2, tab3 = st.tabs(["📚 Cited Sources", "📄 Verified Vector Chunks", "🕸️ Knowledge Graph Relations"])
                
                with tab1:
                    if sources_cited:
                        for src in sources_cited:
                            st.markdown(f"- **Document:** `{src.get('source')}` | **Page:** `{src.get('page')}`")
                    else:
                        st.info("No specific sources cited.")
                        
                with tab2:
                    if vector_results:
                        for idx, chunk in enumerate(vector_results, 1):
                            meta = chunk.get("metadata", {})
                            grade = chunk.get("crag_grade", "correct")
                            score = chunk.get("score", 0.0)
                            badge_class = "badge-correct" if grade == "correct" else "badge-ambiguous"
                            
                            st.markdown(f"""
                            <div class="chunk-card">
                                <b>Chunk {idx}</b> | Score: <code>{score:.4f}</code> | Grade: <span class="{badge_class}">{grade.upper()}</span><br>
                                <small><b>Source:</b> {meta.get('source') or meta.get('filename')} (Page {meta.get('page') or meta.get('page_number')})</small>
                                <p style="margin-top: 8px; font-size: 13px;">{chunk.get('text')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No semantic chunks used.")
                        
                with tab3:
                    if graph_results:
                        for idx, g in enumerate(graph_results, 1):
                            source = g.get("source", "Unknown")
                            rel = g.get("relation")
                            target = g.get("target")
                            desc = g.get("description", "")
                            
                            if rel and target:
                                content = f"<b>{source}</b> &nbsp;&nbsp;➔&nbsp;&nbsp; <code>[{rel}]</code> &nbsp;&nbsp;➔&nbsp;&nbsp; <b>{target}</b>"
                            else:
                                content = f"<b>{source}</b>: {desc}"
                                
                            st.markdown(f"""
                            <div class="graph-card">
                                <b>Graph Entity/Relation {idx}</b><br>
                                <p style="margin-top: 5px; font-size: 13px;">{content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No knowledge graph entities matched.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": stream_error or full_response,
            "sources": sources_cited,
            "vector_results": vector_results,
            "graph_results": graph_results
        })