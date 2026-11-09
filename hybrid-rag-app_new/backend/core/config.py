"""Validated backend settings."""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Resolve default data paths from the project root, independent of the shell's
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# Fallback to local .env if root isn't found
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv()


def project_path_from_env(name: str, default_name: str) -> str:
    configured = Path(os.getenv(name, str(PROJECT_ROOT / default_name)))
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return str(configured.resolve())


class Settings(BaseSettings):
    """
    Centralized configuration class using Pydantic for validation and type safety.
    Reads credentials and paths directly from environment variables.
    """
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    FAISS_INDEX_PATH: str = project_path_from_env("FAISS_INDEX_PATH", "faiss_index")
    BM25_INDEX_PATH: str = project_path_from_env("BM25_INDEX_PATH", "bm25_index")
    BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:8000")
    KNOWLEDGE_BASE_DIR: str = project_path_from_env("KNOWLEDGE_BASE_DIR", "rag_knowledge_base")

    class Config:
        case_sensitive = True
        extra = "ignore"

# Instantiate a global settings object to be imported across backend modules
settings = Settings()