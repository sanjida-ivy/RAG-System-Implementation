# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Required
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "test")  # Changed from quickstart
