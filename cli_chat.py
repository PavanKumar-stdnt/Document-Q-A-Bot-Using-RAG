#!/usr/bin/env python3
# cli_chat.py
"""
DAY 2 — Terminal chatbot for testing the RAG chain before building the UI.

Usage:
    python cli_chat.py

Commands inside the chat:
    exit / quit   — stop the session
    /reset        — clear conversation history
    /sources      — toggle source citation display
    /help         — show commands
"""

import sys
from src.vector_store import load_vector_store
from src.retriever import build_retriever
from src.chain import build_chain, ask
from src.logger import logger


HELP_TEXT = """
Commands:
  /reset    — clear conversation memory
  /sources  — toggle source display on/off
  /help     — show this message
  exit      — quit
"""


def main():
    print("\n" + "=" * 55)
    print("  RAG CHATBOT  |  Powered by Gemma3:4b + ChromaDB (FREE)")
    print("=" * 55)
    print("  Type your question and press Enter.")
    print("  Type /help for commands, or 'exit' to quit.\n")

    # ── Load pipeline ────────────────────────────────────────────────
    try:
        logger.info("Loading vector store…")
        vectorstore = load_vector_store()
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)

    retriever = build_retriever(vectorstore)
    chain = build_chain(retriever)
    show_sources = True

    print("Ready! Ask anything about your documents.\n")

    # ── REPL loop ────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break
        elif user_input == "/help":
            print(HELP_TEXT)
            continue
        elif user_input == "/sources":
            show_sources = not show_sources
            state = "ON" if show_sources else "OFF"
            print(f"[Source display: {state}]\n")
            continue
        elif user_input == "/reset":
            retriever = build_retriever(vectorstore)
            chain = build_chain(retriever)
            print("[Conversation memory cleared.]\n")
            continue

        # Ask the chain
        result = ask(chain, user_input)

        print(f"\nBot: {result['answer']}")

        if show_sources and result["sources"]:
            print(f"     📚 Sources: {', '.join(result['sources'])}")

        print()


if __name__ == "__main__":
    main()
