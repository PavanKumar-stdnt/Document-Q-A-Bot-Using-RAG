#!/usr/bin/env bash
# scripts/reset.sh
# Wipe the vector store and re-index everything from scratch.

set -e
source rag-env/bin/activate 2>/dev/null || true

echo "[INFO] Deleting existing ChromaDB..."
python -c "from src.vector_store import delete_vector_store; delete_vector_store()"

echo "[INFO] Re-indexing documents from ./docs ..."
python ingest.py

echo "[DONE] Reset complete. Run: streamlit run app.py"
