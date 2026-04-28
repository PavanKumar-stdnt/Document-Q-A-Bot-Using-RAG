# src/text_splitter.py
"""
DAY 1 — Step 3: Split documents into overlapping chunks.

Why overlap?
  Sentences that cross chunk boundaries would otherwise be cut off,
  losing context. A 100-token overlap ensures continuity.

Why RecursiveCharacterTextSplitter?
  It tries to split on paragraph → sentence → word → character,
  preserving natural reading units as much as possible.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import logger


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Each output chunk inherits the metadata of its parent document
    (source, page, file_type) plus a new `chunk_index` field so you
    can trace exactly where in the file a chunk came from.

    Args:
        docs: Raw documents from document_loader.py

    Returns:
        List of chunked Documents ready for embedding.
    """
    if not docs:
        logger.warning("split_documents received an empty list — nothing to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # natural split hierarchy
    )

    chunks = splitter.split_documents(docs)

    # Tag each chunk with its index within the same source document
    source_counters: dict = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[src]

    logger.info(
        f"Split {len(docs)} document(s) into {len(chunks)} chunks "
        f"(size≈{CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )
    return chunks
