# src/retriever.py
"""
DAY 2 — Step 1: Build a retriever from the ChromaDB vector store.

Search strategy: MMR (Max Marginal Relevance)
  MMR balances relevance AND diversity.  Instead of returning the 4
  most similar chunks (which might all say the same thing), it fetches
  8 candidates and picks 4 that together cover the query best.

  fetch_k=8  — how many raw candidates to pull from the DB
  k=4        — how many to keep after MMR re-ranking
"""

from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import RETRIEVER_K, RETRIEVER_FETCH_K
from src.logger import logger


def build_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    """
    Wrap a Chroma vectorstore as an MMR retriever.

    Args:
        vectorstore: Loaded Chroma instance from vector_store.py.

    Returns:
        LangChain VectorStoreRetriever ready to plug into a chain.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
        },
    )
    logger.info(
        f"Retriever ready — MMR (k={RETRIEVER_K}, fetch_k={RETRIEVER_FETCH_K})."
    )
    return retriever
