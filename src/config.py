# src/config.py
"""
Central configuration — reads from .env file or uses safe defaults.
All other modules import from here so settings stay in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (won't overwrite already-set env vars)
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DOCS_PATH   = Path(os.getenv("DOCS_PATH",   str(BASE_DIR / "docs")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db")))

# ── Ollama / Model settings ──────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL",        "gemma3:4b")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "nomic-embed-text")

# ── ChromaDB ────────────────────────────────────────────────────────
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "rag_documents")

# ── Text splitting ───────────────────────────────────────────────────
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",   "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP","100"))

# ── Retriever ────────────────────────────────────────────────────────
RETRIEVER_K      = int(os.getenv("RETRIEVER_K",      "4"))
RETRIEVER_FETCH_K= int(os.getenv("RETRIEVER_FETCH_K","8"))

# ── LLM generation ───────────────────────────────────────────────────
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE","0.1"))
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS",  "1024"))

# ── Streamlit UI ─────────────────────────────────────────────────────
APP_TITLE        = os.getenv("APP_TITLE", "RAG Chatbot")
APP_ICON         = os.getenv("APP_ICON",  "📄")

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY \
using the context provided below from the user's uploaded documents.

Rules:
1. If the answer is clearly in the context, answer concisely and accurately.
2. If the answer is NOT in the context, say exactly:
   "I don't have enough information in the provided documents to answer that."
3. Never fabricate facts or use outside knowledge.
4. When relevant, mention which document the information comes from.

Context:
{context}"""
