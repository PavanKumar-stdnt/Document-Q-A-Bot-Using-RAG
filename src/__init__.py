# src/__init__.py
"""
RAG Chatbot — source package.

Module map:
  config          → all settings (read from .env)
  logger          → centralised loguru logger
  document_loader → load PDFs / TXTs from disk or upload
  text_splitter   → chunk documents with overlap
  vector_store    → embed chunks and persist to ChromaDB
  retriever       → MMR retriever from ChromaDB
  chain           → ConversationalRetrievalChain + ask()
  evaluator       → simple eval loop + RAGAS metrics
"""
