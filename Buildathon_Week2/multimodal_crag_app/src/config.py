import os
from dotenv import load_dotenv

# Load environment variables from .env file into os.environ
load_dotenv()

# Export validated keys/settings for use across your modules
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")