# tests/test_pipeline.py
"""
Unit tests for the RAG pipeline.

Run with:
    pytest tests/ -v
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


# ── Document loader ──────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_from_empty_directory(self, tmp_path):
        """Empty folder should return empty list, not raise."""
        from src.document_loader import load_from_directory
        result = load_from_directory(tmp_path)
        assert result == []

    def test_load_txt_file(self, tmp_path):
        """A .txt file should load as a Document."""
        from src.document_loader import load_from_directory
        txt = tmp_path / "sample.txt"
        txt.write_text("Hello, this is a test document.")
        docs = load_from_directory(tmp_path)
        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "sample.txt"

    def test_unsupported_extension_ignored(self, tmp_path):
        """A .csv file should be silently ignored."""
        from src.document_loader import load_from_directory
        (tmp_path / "data.csv").write_text("a,b,c")
        docs = load_from_directory(tmp_path)
        assert docs == []


# ── Text splitter ────────────────────────────────────────────────────

class TestTextSplitter:
    def test_splits_long_text(self):
        """Long text should be split into multiple chunks."""
        from src.text_splitter import split_documents
        long_text = "This is a sentence. " * 200   # ~600 words
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})
        chunks = split_documents([doc])
        assert len(chunks) > 1

    def test_chunks_inherit_metadata(self):
        """Every chunk should carry the source metadata."""
        from src.text_splitter import split_documents
        doc = Document(page_content="Word " * 300, metadata={"source": "my_file.pdf"})
        chunks = split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "my_file.pdf"

    def test_empty_input_returns_empty(self):
        from src.text_splitter import split_documents
        assert split_documents([]) == []


# ── Vector store ─────────────────────────────────────────────────────

class TestVectorStore:
    def test_vector_store_not_exists_on_fresh_path(self, tmp_path, monkeypatch):
        """vector_store_exists() should return False for a new directory."""
        import src.vector_store as vs_module
        monkeypatch.setattr(vs_module, "CHROMA_PATH", tmp_path / "chroma")
        from src.vector_store import vector_store_exists
        assert not vector_store_exists()


# ── Chain (mocked) ───────────────────────────────────────────────────

class TestChain:
    def test_ask_empty_question(self):
        """Empty question should return a safe fallback, not crash."""
        from src.chain import ask
        mock_chain = MagicMock()
        result = ask(mock_chain, "   ")
        assert "Please enter a question" in result["answer"]
        mock_chain.invoke.assert_not_called()

    def test_ask_returns_structured_result(self):
        """ask() should always return answer + sources + chunks keys."""
        from src.chain import ask
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "Test answer.",
            "source_documents": [
                Document(page_content="chunk 1", metadata={"source": "doc.pdf"})
            ],
        }
        result = ask(mock_chain, "What is in the doc?")
        assert "answer"  in result
        assert "sources" in result
        assert "chunks"  in result
        assert result["answer"] == "Test answer."
        assert "doc.pdf" in result["sources"]
