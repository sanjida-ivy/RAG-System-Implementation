import os
import uuid
from typing import List

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from .vector_utils import upsert_vectors
from pinecone_setup import init_pinecone

def process_pdf(path: str,
                chunk_size: int = 1000,
                chunk_overlap: int = 200) -> int:
    """
    1) Starts Pinecone (if needed).
    2) Reads the PDF at `path`.
    3) Splits its text into overlapping chunks.
    4) Embeds & upserts them.
    Returns the number of chunks indexed.
    """
    # ensure Pinecone is up
    init_pinecone()

    # read raw text
    reader = PdfReader(path)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    print("Full text extracted:")
    print(full_text)

    # split into chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks: List[str] = splitter.split_text(full_text)

    # prepare payload for Pinecone
    to_upsert = []
    for chunk in chunks:
        to_upsert.append({
            "id": uuid.uuid4().hex,
            "text": chunk
        })

    # embed & push
    upsert_vectors(to_upsert)
    print(f"✅ Indexed {len(chunks)} chunks.")
    return len(chunks)
