# pinecone_setup.py
import pinecone
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX
import sys

# pinecone_setup.py
# pinecone_setup.py
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    EMBEDDING_DIMENSION = 768  # Matches multi-qa-mpnet-base-dot-v1
    
    if PINECONE_INDEX in pc.list_indexes().names():
        pc.delete_index(PINECONE_INDEX)  # Force delete old index
    
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIMENSION,  # 768
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
    
    return pc.Index(PINECONE_INDEX)

def create_new_index(pc):
    """Helper function to create a new index with consistent settings"""
    
    pc.create_index(
    name=PINECONE_INDEX,
    dimension=768,  # Updated to match multi-qa-mpnet-base-dot-v1
    metric="cosine",
    spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
    print(f"Created new Pinecone index '{PINECONE_INDEX}' with dimension 768")

if __name__ == "__main__":
    try:
        index = initialize_pinecone()
        print(f"Pinecone index '{PINECONE_INDEX}' is ready")
    except Exception as e:
        print(f"Failed to set up Pinecone: {e}", file=sys.stderr)
        sys.exit(1)