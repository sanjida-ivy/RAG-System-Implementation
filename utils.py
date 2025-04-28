import subprocess
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import PINECONE_API_KEY, OLLAMA_MODEL, PINECONE_ENVIRONMENT, PINECONE_INDEX
from pdf_processing import extract_text_from_pdf
import streamlit as st
import tempfile

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check/create index
    if PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,  # Must match SentenceTransformer
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
        )
    index = pc.Index(PINECONE_INDEX)

except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    raise

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # This model outputs 768-dimension embeddings
print(model.get_sentence_embedding_dimension())

def ollama_embed(text: str):
    try:
        cleaned_text = text.encode('utf-8', errors='ignore').decode('utf-8')[:10000]
        
        # Check if Ollama is running and model exists
        subprocess.run(["ollama", "list"], check=True, capture_output=True, timeout=10)
        
        # Try embedding with Ollama
        result = subprocess.run(
            ["ollama", "run", "llama3", "embed", cleaned_text],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            return [float(i) for i in result.stdout.strip().split(",")]
        else:
            raise Exception(result.stderr)
    
    except Exception as e:
        print(f"Ollama embedding failed, using SentenceTransformer: {str(e)[:200]}")
        return model.encode(text).tolist()  # Fallback to SentenceTransformer
    

def retrieve_docs(query: str):
    query_embedding = ollama_embed(query)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def summarize_docs(docs: list):
    context = "\n".join(docs)
    prompt = f"Summarize the following documents:\n{context}"
    return query_ollama(prompt)

def answer_question(query: str, context: str):
    prompt = f"Context: {context}\n\nAnswer the query: {query}"
    return query_ollama(prompt)

def query_ollama(prompt: str):
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt.encode(),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        return result.stdout
    except Exception as e:
        return f"Error querying Ollama: {e}"


def process_pdf(pdf_file_path):
    try:
        text = extract_text_from_pdf(pdf_file_path)
        embedding = ollama_embed(text)
        
        # Verify embedding dimension matches index
        index_info = pc.describe_index(PINECONE_INDEX)
        if len(embedding) != index_info.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} doesn't match index dimension {index_info.dimension}")
            
        index.upsert([(pdf_file_path, embedding, {"text": text})])
        return text, embedding
    except Exception as e:
        raise Exception(f"Failed to process PDF: {e}")