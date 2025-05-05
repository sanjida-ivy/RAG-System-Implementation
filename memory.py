# memory.py

from typing import Dict
from config import OLLAMA_HOST, OLLAMA_MODEL
from langchain_ollama.llms import OllamaLLM

# In-memory key-value store per user
memory_store: Dict[str, Dict[str, str]] = {}

def remember(user_id: str, key: str, value: str):
    if user_id not in memory_store:
        memory_store[user_id] = {}
    memory_store[user_id][key] = value

def recall(user_id: str) -> str:
    user_memory = memory_store.get(user_id, {})
    if not user_memory:
        return "None"
    return "\n".join(f"{k}: {v}" for k, v in user_memory.items())

def clear_memory(user_id: str):
    if user_id in memory_store:
        del memory_store[user_id]

def auto_learn_facts(text: str, user_id: str = "default"):
    """
    Uses the LLM to extract 'key: value' facts from a text and stores them as memory.
    Typically called on assistant responses or user statements.
    """
    llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_HOST)

    prompt = f"""
Extract useful long-term facts from the following text. These should help personalize future answers.
Return them in the format: key: value — one per line.
Ignore anything irrelevant or vague.

Text:
\"\"\"{text}\"\"\"

Extracted Facts:
"""
    output = llm.invoke(prompt).strip()

    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            remember(user_id, key.strip(), value.strip())
