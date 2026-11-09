# Hybrid RAG App

FastAPI and Streamlit application combining FAISS semantic search, Neo4j graph
retrieval, corrective RAG, and session-based chat history.

## Project layout

- `rag_knowledge_base/` — add PDF files here for indexing.
- `backend/` — API, PDF ingestion, retrieval, graph, and generation modules.
- `frontend/` — Streamlit interface and backend API client.
- `faiss_index/` — generated semantic index and ingestion state.
- `tests/` — focused tests for retrieval and graph behavior.

## Configuration

Create a project-root `.env` file with `OPENAI_API_KEY`, `NEO4J_URI`,
`NEO4J_USERNAME`, and `NEO4J_PASSWORD`. The application resolves the default
knowledge-base and index paths relative to the project directory, regardless of
the shell's current directory. Override them with `KNOWLEDGE_BASE_DIR` and
`FAISS_INDEX_PATH` if needed.

## Run locally

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

Start the backend from `backend/`:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend in a separate terminal from `frontend/`:

```powershell
streamlit run app.py
```

## Run with Docker Compose

Install and start Docker Desktop, then open PowerShell in the project root. If
you do not already have a `.env`, copy `.env.example` to `.env`; otherwise keep
your existing file. Set `OPENAI_API_KEY`, and set `NEO4J_URI`,
`NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` to the credentials and
connection details from your Neo4j Aura instance. Use the Aura connection URI
starting with `neo4j+s://` (or `neo4j+ssc://` if Aura specifically provides that
scheme). The backend no longer defaults to a local Neo4j address, so an
explicit cloud URI is required. Do not commit `.env`.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose config
docker compose up --build -d
docker compose ps
```

Compose runs only the backend and frontend; Neo4j is hosted by Aura and is not
started as a local container. The backend connects to Aura using the configured
URI and credentials. Ensure the Aura instance is running and reachable from
your network. Open the Streamlit UI at <http://localhost:8501> and FastAPI docs
at <http://localhost:8000/docs>. The UI waits for the API health check, and the
frontend uses `http://backend:8000/api/v1` to reach the API over Compose's
internal network.

Place PDFs in the project-root `rag_knowledge_base/` folder before starting, or
upload them in the UI. FAISS data/state is stored in a named Docker volume and
survives container rebuilds. Graph data remains in your Aura database. To
inspect startup and ingestion errors, use:

```powershell
docker compose logs -f backend frontend
```

To stop the services without deleting persisted indexes or graph data:

```powershell
docker compose down
```

To remove persisted data as well, explicitly remove the named volumes:

```powershell
docker compose down --volumes
```

At backend startup, PDFs are synchronized into FAISS and Neo4j before the API
accepts requests. On every restart, the application verifies the PDF hashes
against the loaded vector and graph state and reprocesses files when the index
or graph is empty or files are new or changed. The frontend reports backend,
vector-index, and graph status. Use **Index / Sync PDFs** in the sidebar or
`POST /api/v1/ingest/sync` to synchronize manually. A graph synchronization
failure does not undo a successfully persisted semantic index and is surfaced
in the UI and backend logs. Chat streaming and ingestion failures are displayed
in the UI rather than silently appearing as empty responses.

## Tests

From the project root, run:

```powershell
python -m pytest -q tests
```
