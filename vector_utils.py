# File: vector_utils.py

from config import EMBEDDING_MODEL, PINECONE_NAMESPACE
from pinecone_setup import init_pinecone
from sentence_transformers import SentenceTransformer

# Clean env values
_model_name = EMBEDDING_MODEL.split('#', 1)[0].strip()
_namespace = PINECONE_NAMESPACE.split('#', 1)[0].strip()


# Init Pinecone index
_index = init_pinecone()

# Load local embedding model
model = SentenceTransformer(_model_name)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text using SentenceTransformer"""
    
    return model.encode(texts, convert_to_numpy=True).tolist()

   
def upsert_vectors(data: list[dict]):
    vectors = []
    texts = [d["text"] for d in data]
    embeddings = embed_texts(texts)

    for d, emb in zip(data, embeddings):
        vectors.append({
            "id": d["id"],
            "values": emb,
            "metadata": {"text": d["text"]},
        })

    print(f"📥 Upserting {len(vectors)} vectors into Pinecone...")
    _index.upsert(vectors=vectors, namespace=_namespace)

    # Test: Immediately query back what we just inserted
    test_query = _index.query(vector=embeddings[0], top_k=1, namespace=_namespace, include_metadata=True)
    print("🧪 Immediate test query result:", test_query)

