import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Production CRAG Client", layout="wide")
st.title("📄 Production Corrective RAG (CRAG) Client")

# Initialize a persistent HTTPX client in session state for connection pooling
if "http_client" not in st.session_state:
    st.session_state.http_client = httpx.Client(base_url=API_URL, timeout=60.0)

client: httpx.Client = st.session_state.http_client

# Preflight check to verify backend connectivity
try:
    health_res = client.get("/health")
    health_data = health_res.json()
    backend_ready = health_data.get("vector_index_exists", False)
except httpx.ConnectError:
    st.error("⚠️ Could not connect to FastAPI backend. Please ensure it is running (`uvicorn backend:app --reload`).")
    st.stop()

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Index PDF"):
            with st.spinner("Streaming file to FastAPI backend..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = client.post("/upload", files=files)
                if res.status_code == 200:
                    st.success("PDF processed and indexed successfully!")
                    st.rerun()
                else:
                    st.error(f"Error: {res.json().get('detail')}")

    if st.button("🗑️ Clear Local Index"):
        res = client.delete("/index")
        if res.status_code == 200:
            st.success("Index cleared!")
            st.rerun()

# --- Main Chat & Query Interface ---
if backend_ready or health_data.get("openai_key_configured"):
    question = st.text_input("Ask a question about your document:")
    
    if question:
        with st.spinner("Executing CRAG Pipeline via FastAPI..."):
            res = client.post("/query", json={"question": question})
            
            if res.status_code == 200:
                data = res.json()
                st.subheader("💡 Answer:")
                st.write(data["generation"])

                st.subheader("📌 Retrieved Source Passages:")
                docs = data.get("documents", [])
                if docs:
                    for idx, doc in enumerate(docs):
                        with st.expander(f"Source {idx+1}: {doc['metadata']['source']} — Page {doc['metadata']['page']}"):
                            st.write(doc["content"])
                else:
                    st.info("No matching source pages passed relevance evaluation.")
            else:
                st.error(f"API Error: {res.json().get('detail')}")
else:
    st.info("Please upload a PDF file in the sidebar to begin.")