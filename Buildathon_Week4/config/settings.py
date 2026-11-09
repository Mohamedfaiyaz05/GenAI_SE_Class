# """Central application settings and local storage paths."""

# import os
# from pathlib import Path

# from dotenv import load_dotenv

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# load_dotenv(PROJECT_ROOT / ".env")

# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DATA_DIR = PROJECT_ROOT / "data"
# CHROMA_DIR = DATA_DIR / "chroma"
# FAISS_DIR = DATA_DIR / "faiss"
# CACHE_DIR = DATA_DIR / "cache"
# LOG_DIR = PROJECT_ROOT / "logs"
# KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

# MAX_AGENT_RETRIES = 2
# INITIAL_RETRIEVAL_K = 5
# EXPANDED_RETRIEVAL_K = 10
# RRF_K = 60


from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

@dataclass(frozen=True)
class Settings:
    database_path: Path


def get_settings() -> Settings:
    raw = os.getenv('SHOPEASE_DB_PATH', 'data/shopease.db')
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return Settings(database_path=path.resolve())
