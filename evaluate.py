#!/usr/bin/env python3
# evaluate.py
"""
DAY 3 — Run evaluation against the RAG chain.

Usage:
    python evaluate.py                  # uses built-in sample questions
    python evaluate.py --ragas          # full RAGAS metrics (needs ground truths)

Edit SAMPLE_QUESTIONS and GROUND_TRUTHS below to match your documents.
"""

import argparse
from src.vector_store import load_vector_store
from src.retriever import build_retriever
from src.chain import build_chain
from src.evaluator import run_evaluation, run_ragas_evaluation
from src.logger import logger


# ── Edit these to match YOUR documents ──────────────────────────────

SAMPLE_QUESTIONS = [
    "What is the main topic of the uploaded document?",
    "Summarise the key points from the document.",
    "What are the important rules or policies mentioned?",
]

# Only needed for RAGAS — provide your expected correct answers here
GROUND_TRUTHS = [
    "The document covers ...",   # replace with actual expected answers
    "The key points are ...",
    "The important rules are ...",
]

# ────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG chain quality.")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run full RAGAS metrics (requires ragas + datasets installed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading pipeline for evaluation…")
    vectorstore = load_vector_store()
    retriever   = build_retriever(vectorstore)
    chain       = build_chain(retriever)

    if args.ragas:
        logger.info("Running full RAGAS evaluation…")
        scores = run_ragas_evaluation(chain, SAMPLE_QUESTIONS, GROUND_TRUTHS)
        print("\nRAGAS Scores:")
        for metric, score in scores.items():
            bar = "█" * int(score * 20)
            print(f"  {metric:<22} {score:.3f}  {bar}")
    else:
        run_evaluation(chain, SAMPLE_QUESTIONS)


if __name__ == "__main__":
    main()
