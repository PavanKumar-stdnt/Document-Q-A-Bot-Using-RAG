# src/document_loader.py
"""
DAY 1 — Step 1 & 2: Load documents from disk or uploaded bytes.

Supports: PDF, TXT, Markdown (.md)
Each loaded page is a LangChain Document with metadata:
  - source   : original file name
  - page      : page number (PDFs only)
  - file_type : "pdf" | "txt" | "md"
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

from src.config import DOCS_PATH
from src.logger import logger


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


# ────────────────────────────────────────────────────────────────────
# Load from the ./docs folder (used by ingest.py CLI)
# ────────────────────────────────────────────────────────────────────

def load_from_directory(directory: Path = DOCS_PATH) -> List[Document]:
    """
    Walk `directory` and load every supported file.
    Returns a flat list of LangChain Documents.
    """
    if not directory.exists():
        logger.warning(f"Docs directory '{directory}' does not exist. Creating it.")
        directory.mkdir(parents=True, exist_ok=True)
        return []

    all_docs: List[Document] = []
    files_found = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files_found:
        logger.warning(f"No supported files found in '{directory}'. "
                       f"Add PDFs, TXTs, or MDs and retry.")
        return []

    for file_path in files_found:
        docs = _load_single_file(file_path, source_name=file_path.name)
        all_docs.extend(docs)

    logger.info(f"Loaded {len(all_docs)} pages/sections from {len(files_found)} files.")
    return all_docs


# ────────────────────────────────────────────────────────────────────
# Load from Streamlit UploadedFile objects (used by app.py)
# ────────────────────────────────────────────────────────────────────

def load_from_uploaded_files(uploaded_files) -> List[Document]:
    """
    Accept a list of Streamlit UploadedFile objects.
    Writes each to a temp file, loads it, then cleans up.
    Returns a flat list of LangChain Documents.
    """
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            continue

        # Write to a temp file so loaders can read from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        try:
            docs = _load_single_file(tmp_path, source_name=uploaded_file.name)
            all_docs.extend(docs)
        finally:
            tmp_path.unlink(missing_ok=True)   # always clean up

    logger.info(f"Loaded {len(all_docs)} pages/sections from "
                f"{len(uploaded_files)} uploaded file(s).")
    return all_docs


# ────────────────────────────────────────────────────────────────────
# Internal helper
# ────────────────────────────────────────────────────────────────────

def _load_single_file(file_path: Path, source_name: str) -> List[Document]:
    """Load one file and tag every Document with clean metadata."""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()

        # Normalise metadata so downstream code can rely on it
        for doc in docs:
            doc.metadata["source"]    = source_name
            doc.metadata["file_type"] = suffix.lstrip(".")

        logger.debug(f"  Loaded '{source_name}' → {len(docs)} section(s).")
        return docs

    except Exception as e:
        logger.error(f"Failed to load '{source_name}': {e}")
        return []
