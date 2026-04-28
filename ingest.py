#!/usr/bin/env python3
# ingest.py
"""
DAY 1 — CLI entry point for document ingestion.

Usage:
    python ingest.py                  # indexes all files in ./docs
    python ingest.py --reset          # deletes existing DB first, then re-indexes
    python ingest.py --docs path/to/  # specify a custom docs folder

What it does:
    1. Load documents from ./docs (PDFs, TXTs, MDs)
    2. Split them into overlapping chunks
    3. Embed each chunk with nomic-embed-text (via Ollama, free)
    4. Persist vectors to ChromaDB on disk
"""

import argparse
import sys
from pathlib import Path

from src.document_loader import load_from_directory
from src.text_splitter import split_documents
from src.vector_store import embed_and_store, delete_vector_store, vector_store_exists
from src.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Path to folder containing documents (default: ./docs)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector store before indexing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Optional: wipe existing DB
    if args.reset:
        logger.warning("--reset flag detected. Deleting existing vector store…")
        delete_vector_store()

    # Resolve docs path
    docs_path = args.docs or Path("./docs")
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logger.info(f"Created '{docs_path}'. Add your PDF/TXT/MD files and re-run.")
        sys.exit(0)

    # ── Pipeline ────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1/3 — Loading documents…")
    docs = load_from_directory(docs_path)
    if not docs:
        logger.error("No documents loaded. Aborting.")
        sys.exit(1)

    logger.info("STEP 2/3 — Splitting into chunks…")
    chunks = split_documents(docs)

    logger.info("STEP 3/3 — Embedding and storing in ChromaDB…")
    vectorstore = embed_and_store(chunks)

    # Summary
    count = vectorstore._collection.count()
    logger.info("=" * 50)
    logger.info(f"Ingestion complete!")
    logger.info(f"  Documents loaded : {len(docs)}")
    logger.info(f"  Chunks stored    : {count}")
    logger.info(f"  Next step        : streamlit run app.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
