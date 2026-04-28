# src/vector_store.py
"""
DAY 1 — Steps 4 & 5: Embed chunks and persist them in ChromaDB.

ChromaDB stores vectors on disk so you never re-embed the same
documents twice. The collection is append-safe: calling
`add_documents()` on an existing collection simply adds new chunks.

Embedding model: nomic-embed-text (via Ollama, runs locally, FREE)
"""

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from src.logger import logger


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _get_embeddings() -> OllamaEmbeddings:
    """Return a configured OllamaEmbeddings instance."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def embed_and_store(chunks: List[Document]) -> Chroma:
    """
    Embed `chunks` and persist them into ChromaDB.

    If the collection already exists the new chunks are APPENDED —
    existing vectors are not touched.  This lets you add new documents
    to a running chatbot without re-indexing everything.

    Args:
        chunks: Chunked LangChain Documents (from text_splitter.py).

    Returns:
        Loaded Chroma vectorstore instance.
    """
    if not chunks:
        logger.warning("embed_and_store received empty chunk list.")
        return load_vector_store()

    embeddings = _get_embeddings()
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    if vector_store_exists():
        logger.info(f"ChromaDB exists — appending {len(chunks)} new chunk(s).")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings,
        )
        vectorstore.add_documents(chunks)
    else:
        logger.info(f"Creating new ChromaDB collection with {len(chunks)} chunk(s).")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_PATH),
        )

    logger.info(f"Vector store saved to '{CHROMA_PATH}'.")
    return vectorstore


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB collection from disk.

    Raises:
        FileNotFoundError: if the collection has not been created yet.
    """
    if not vector_store_exists():
        raise FileNotFoundError(
            f"No ChromaDB found at '{CHROMA_PATH}'. "
            "Run `python ingest.py` first, or upload documents in the UI."
        )

    embeddings = _get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
    )
    count = vectorstore._collection.count()
    logger.info(f"Loaded ChromaDB with {count} stored chunk(s).")
    return vectorstore


def vector_store_exists() -> bool:
    """Return True if a ChromaDB collection exists on disk."""
    chroma_sqlite = CHROMA_PATH / "chroma.sqlite3"
    return chroma_sqlite.exists()


def delete_vector_store() -> None:
    """Delete the entire ChromaDB (useful for a full re-index)."""
    import shutil
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        logger.info(f"Deleted vector store at '{CHROMA_PATH}'.")
    else:
        logger.warning("No vector store found to delete.")
