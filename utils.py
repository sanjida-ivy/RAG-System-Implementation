import time
from config import OLLAMA_HOST, OLLAMA_MODEL
from langchain_ollama.llms import OllamaLLM
from agents.common.vector_utils import embed_texts
from pinecone_setup import init_pinecone
from config import PINECONE_NAMESPACE
import streamlit as st




llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_HOST)
_idx = init_pinecone()

RETRIEVAL_SCORE_THRESHOLD = 0.0  # Lowered to allow any match


def retrieve_docs(query: str, top_k: int = 3) -> list[str]:
    start = time.time()
    emb = embed_texts([query])[0]
    res = _idx.query(vector=emb, top_k=top_k, include_metadata=True)
    duration = time.time() - start
    print(f"⏱️ Pinecone query took {duration:.2f}s")

    print("🔍 Pinecone matches:")
    for m in res["matches"]:
        print(f"- Score: {m['score']:.4f} | Preview: {m['metadata']['text'][:100].replace('\n',' ')}")

    return [m["metadata"]["text"] for m in res["matches"]]



def answer_query(query: str) -> str:
    query_clean = query.strip()

    # 🧠 Build conversation context from history
    history = st.session_state.get("history", [])
    past_turns = ""
    for turn in history[-6:]:  # last 3 exchanges (user+assistant)
        role = "User" if turn["role"] == "user" else "Assistant"
        past_turns += f"{role}: {turn['content']}\n"

    # ✅ If PDF has been uploaded
    if "pdf_raw_text" in st.session_state:
        print("📄 Using full PDF content with Ollama and history")

        full_text = st.session_state["pdf_raw_text"]
        prompt = f"""
You are a helpful assistant.

The user uploaded a PDF. Answer the question using only the content of the PDF and the conversation history.

PDF Content:
\"\"\"{full_text[:4000]}\"\"\"

Conversation History:
{past_turns}

Current Question: {query_clean}

Answer:"""
        return llm.invoke(prompt.strip())

    # ❌ No PDF uploaded
    print("💬 No PDF — using Ollama with chat history")
    prompt = f"""
You are a helpful assistant.

Use the conversation history to answer the user's current question.

Conversation History:
{past_turns}

Current Question: {query_clean}

Answer:"""
    return llm.invoke(prompt.strip())

