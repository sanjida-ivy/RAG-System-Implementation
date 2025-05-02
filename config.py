# File: config.py
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")
# Pinecone serverless region (defaults to PINECONE_ENVIRONMENT if not set)
PINECONE_SERVERLESS_REGION = os.getenv(
    "PINECONE_SERVERLESS_REGION",
    PINECONE_ENVIRONMENT
)


# Validate required variables
required = {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
    "PINECONE_INDEX": PINECONE_INDEX,
}
missing = [name for name, val in required.items() if not val]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")