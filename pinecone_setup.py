# File: pinecone_setup.py

from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX,
    PINECONE_SERVERLESS_REGION
)

# Strip any trailing comments/whitespace from your env-loaded values
_env = PINECONE_ENVIRONMENT.split('#', 1)[0].strip()
_srv = PINECONE_SERVERLESS_REGION.split('#', 1)[0].strip()
_idx = PINECONE_INDEX.split('#', 1)[0].strip()

_client: Pinecone | None = None
_index = None

def init_pinecone():
    global _client, _index
    if _client is None:
        # 1) Control-plane client
        _client = Pinecone(
            api_key=PINECONE_API_KEY,
            environment=_env,
        )

        # 2) Create index if missing
        if _idx not in _client.list_indexes().names():
            _client.create_index(
                name=_idx,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=_srv,
                )
            )

        # 3) Grab its host and spin up data-plane client
        cfg = _client.describe_index(name=_idx)
        _index = _client.Index(host=cfg["host"])

    return _index

def get_pinecone_client() -> Pinecone:
    """Returns the control-plane Pinecone client (for embeddings)."""
    if _client is None:
        init_pinecone()
    return _client
