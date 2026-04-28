# src/evaluator.py
"""
DAY 3 — Step 4: Evaluate RAG quality using RAGAS metrics (all free).

RAGAS metrics explained:
  - faithfulness      : Does the answer stick to the retrieved context?
                        (catches hallucinations — higher is better)
  - answer_relevancy  : How well does the answer address the question?
                        (catches off-topic answers — higher is better)
  - context_recall    : Did we retrieve the chunks that were actually needed?
                        (requires ground-truth answers — higher is better)
  - context_precision : Are all retrieved chunks actually relevant?
                        (catches noisy retrieval — higher is better)

All metrics score 0–1.  Anything above 0.7 is considered good.

Usage:
  from src.evaluator import run_evaluation
  results = run_evaluation(chain, test_questions)
"""

from typing import List, Dict, Any

from src.chain import ask
from src.logger import logger


def run_evaluation(
    chain,
    test_questions: List[str],
    ground_truths: List[str] = None,
) -> Dict[str, Any]:
    """
    Run a simple evaluation loop and print per-question stats.

    For full RAGAS metrics (faithfulness, answer_relevancy, etc.)
    install ragas and call `run_ragas_evaluation()` below.

    Args:
        chain:           Built ConversationalRetrievalChain.
        test_questions:  List of questions to evaluate.
        ground_truths:   Optional list of correct answers (one per question).

    Returns:
        Dictionary with questions, answers, and retrieved sources.
    """
    results = []
    for i, question in enumerate(test_questions):
        logger.info(f"Evaluating question {i+1}/{len(test_questions)}: {question!r}")
        output = ask(chain, question)
        result = {
            "question":      question,
            "answer":        output["answer"],
            "sources":       output["sources"],
            "num_chunks":    len(output["chunks"]),
        }
        if ground_truths and i < len(ground_truths):
            result["ground_truth"] = ground_truths[i]
        results.append(result)

    _print_evaluation_table(results)
    return {"results": results}


def run_ragas_evaluation(
    chain,
    test_questions: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Full RAGAS evaluation with faithfulness + answer_relevancy metrics.
    Requires: pip install ragas datasets

    Args:
        chain:           Built ConversationalRetrievalChain.
        test_questions:  Questions to evaluate.
        ground_truths:   Correct reference answers (one per question).

    Returns:
        Dict of metric_name → average score.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        logger.error("RAGAS not installed. Run: pip install ragas datasets")
        return {}

    # Collect answers and contexts
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for question, truth in zip(test_questions, ground_truths):
        output = ask(chain, question)
        data["question"].append(question)
        data["answer"].append(output["answer"])
        data["contexts"].append([c.page_content for c in output["chunks"]])
        data["ground_truth"].append(truth)

    dataset = Dataset.from_dict(data)

    logger.info("Running RAGAS metrics (faithfulness + answer_relevancy)…")
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    logger.info(f"RAGAS scores: {dict(scores)}")
    return dict(scores)


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────

def _print_evaluation_table(results: List[Dict]) -> None:
    """Pretty-print evaluation results to the console."""
    print("\n" + "=" * 70)
    print(f"{'EVALUATION RESULTS':^70}")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n[Q{i}] {r['question']}")
        print(f"  Answer   : {r['answer'][:120]}{'…' if len(r['answer'])>120 else ''}")
        print(f"  Sources  : {', '.join(r['sources']) or 'none'}")
        print(f"  Chunks   : {r['num_chunks']}")
        if "ground_truth" in r:
            print(f"  Expected : {r['ground_truth'][:120]}")
    print("=" * 70 + "\n")
