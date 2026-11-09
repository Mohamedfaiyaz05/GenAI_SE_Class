from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import initialize_semantic_index, router as api_router

@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_semantic_index()
    yield

app = FastAPI(
    title="Hybrid RAG Backend API",
    version="1.0.0",
    description="Production-grade Hybrid RAG API with FAISS, Neo4j, CRAG, and Session Memory.",
    lifespan=lifespan,
)

# Enable CORS for frontend integration (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to the Hybrid RAG API. Visit /docs for Swagger documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)