"""HTTP and server-sent-event client for the backend API."""
import os
import requests
import json
from typing import Any, Dict, Iterator

BACKEND_URL = os.getenv(
    "BACKEND_API_URL",
    "http://localhost:8000/api/v1",
).rstrip("/")

def stream_chat_api(query: str, session_id: str = "default_session") -> Iterator[Dict[str, Any]]:
    """
    Consumes the Server-Sent Events (SSE) streaming endpoint from FastAPI.
    """
    url = f"{BACKEND_URL}/chat/stream"
    payload = {"query": query, "session_id": session_id}
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 900)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    yield json.loads(decoded[6:])
    except requests.RequestException as e:
        yield {"type": "error", "content": f"Could not reach the RAG backend: {e}"}
    except json.JSONDecodeError as e:
        yield {"type": "error", "content": f"The backend returned invalid streaming data: {e}"}

def upload_document_api(uploaded_file) -> Dict[str, Any]:
    """Uploads a PDF file to the backend for ingestion."""
    url = f"{BACKEND_URL}/ingest"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    try:
        response = requests.post(url, files=files, timeout=(10, 600))
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except (ValueError, AttributeError):
            detail = str(e)
        return {"status": "error", "message": detail}
    except (requests.RequestException, ValueError) as e:
        return {"status": "error", "message": str(e)}

def sync_knowledge_base_api() -> Dict[str, Any]:
    """Index all PDF files already present in the configured knowledge-base folder."""
    url = f"{BACKEND_URL}/ingest/sync"
    try:
        response = requests.post(url, timeout=(10, 600))
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except (ValueError, AttributeError):
            detail = str(e)
        return {"status": "error", "message": detail}
    except (requests.RequestException, ValueError) as e:
        return {"status": "error", "message": str(e)}


def get_stats_api() -> Dict[str, Any]:
    """Fetches knowledge base stats from the backend."""
    url = f"{BACKEND_URL}/stats"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except (ValueError, AttributeError):
            detail = str(e)
        return {"status": "error", "message": detail}
    except (requests.RequestException, ValueError) as e:
        return {"status": "error", "message": str(e)}