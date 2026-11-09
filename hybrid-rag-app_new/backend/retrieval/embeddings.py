"""Initialize the configured text embedding model."""
from langchain_openai import OpenAIEmbeddings
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAI embeddings model configured 
    with the API key from central settings. Uses text-embedding-3-small 
    for optimal cost, speed, and retrieval precision.
    """
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is missing in core settings.")
        raise ValueError("OpenAI API key is required for generating embeddings.")
        
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY
        )
        logger.info("Successfully initialized OpenAIEmbeddings (text-embedding-3-small).")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise e