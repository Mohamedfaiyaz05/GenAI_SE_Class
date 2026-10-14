from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from datetime import datetime, timezone
import os
import shutil
import json
import uuid

from src.extractor import PDFExtractor
from src.vector_store import VectorStoreManager
from src.crag import CRAGEngine

app = FastAPI(
    title="Vivasaya Thozan AI API",
    description="Production-grade FastAPI backend for Multimodal Corrective RAG Agricultural Assistant",
    version="1.0.0"
)

# Enable CORS for frontend and Streamlit/React UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core pipeline engines and directories
DATA_DIR = Path("data")
INDEX_DIR = Path("vector_index")
FEEDBACK_DIR = Path("feedback")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

extractor = PDFExtractor(data_dir=str(DATA_DIR))
vector_manager = VectorStoreManager(index_dir=str(INDEX_DIR))

# Startup initialization: idempotent check of data folder
try:
    vector_manager.process_new_documents(extractor, data_dir=str(DATA_DIR))
    print("[Startup] Vector store initialized successfully.")
except Exception as e:
    print(f"[Startup Warning] Vector store initialization warning: {e}")

crag_engine = CRAGEngine(vector_store_manager=vector_manager)


@app.get("/health")
def health_check():
    """Health check endpoint to verify API status."""
    return {"status": "healthy", "service": "Vivasaya Thozan AI Backend"}


@app.post("/api/chat/stream")
def chat_stream(question: str = Form(...), session_id: str = Form("default_session")):
    """
    Streams responses token by token using Server-Sent Events (SSE) 
    backed by the CRAG workflow and persistent chat memory.
    """
    def event_generator():
        try:
            for token in crag_engine.run_stream(question, session_id=session_id):
                # Send token payload formatted for SSE
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/feedback")
def save_response_feedback(
    response_id: str = Form(...),
    rating: str = Form(...),
):
    """Persist a response rating without storing conversation or session data."""
    if len(response_id) != 32 or any(
        character not in "0123456789abcdef" for character in response_id
    ):
        raise HTTPException(status_code=400, detail="Invalid response ID.")
    if rating not in {"like", "dislike"}:
        raise HTTPException(status_code=400, detail="Rating must be like or dislike.")

    feedback = {
        "response_id": response_id,
        "rating": rating,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = FEEDBACK_DIR / f".{response_id}.{uuid.uuid4().hex}.tmp"
    destination = FEEDBACK_DIR / f"{response_id}.json"
    try:
        temporary_path.write_text(json.dumps(feedback), encoding="utf-8")
        os.replace(temporary_path, destination)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Could not save response feedback.",
        ) from exc

    return {"status": "saved", "rating": rating}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Saves a selected PDF for review. Indexing is an explicit separate action.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = Path(file.filename).name
    if filename != file.filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid PDF filename.")

    file_path = DATA_DIR / filename
    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"A file named '{filename}' already exists. Rename the new PDF "
                "before uploading so the existing document and index are not overwritten."
            ),
        )

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "uploaded",
            "filename": filename,
            "message": "PDF uploaded. Create chunks to add it to the searchable knowledge base.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")


@app.post("/api/process-upload")
def process_uploaded_pdf(filename: str = Form(...)):
    """Extract chunks and index one previously uploaded PDF."""
    safe_filename = Path(filename).name
    if (
        not filename
        or safe_filename != filename
        or safe_filename in {".", ".."}
        or not safe_filename.lower().endswith(".pdf")
    ):
        raise HTTPException(status_code=400, detail="Invalid PDF filename.")

    file_path = DATA_DIR / safe_filename
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Uploaded PDF '{safe_filename}' was not found on the server.",
        )

    try:
        vector_store = vector_manager.process_new_documents(
            extractor,
            data_dir=str(DATA_DIR),
            filenames={safe_filename},
        )
        if vector_store is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No searchable chunks could be extracted from '{safe_filename}'. "
                    "Check that the PDF contains readable text or supported page images."
                ),
            )

        processed_files = vector_manager._load_registry()
        if safe_filename not in processed_files:
            raise HTTPException(
                status_code=422,
                detail=f"'{safe_filename}' was not added to the vector store.",
            )

        return {
            "status": "success",
            "filename": safe_filename,
            "message": f"Chunks created and '{safe_filename}' added to the vector store.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create chunks for '{safe_filename}': {str(e)}",
        ) from e


@app.get("/api/stats")
def get_knowledge_base_stats():
    """
    Returns live statistics matching your UI sidebar metrics:
    - Knowledge Base count (processed PDFs)
    - Total Vector DB Chunks stored in FAISS
    """
    processed_registry = vector_manager._load_registry()
    
    total_chunks = 0
    if vector_manager.vector_store and hasattr(vector_manager.vector_store, "index"):
        total_chunks = vector_manager.vector_store.index.ntotal
        
    return {
        "knowledge_base_count": len(processed_registry),
        "vector_db_chunks": total_chunks,
        "processed_files": list(processed_registry)
    }


@app.post("/api/clear-memory")
def clear_memory(session_id: str = Form("default_session")):
    """Clears chat conversation history for a given user session."""
    crag_engine.memory_manager.clear_history(session_id)
    return {"status": "success", "message": f"Memory cleared for session {session_id}"}