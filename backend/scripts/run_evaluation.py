"""
scripts/run_evaluation.py

Lightweight academic evaluation for the RAG system's retrieval quality.
Uses the manually-edited test cases in eval/eval_dataset.json (see that
file's "_instructions" for how to fill it in with real, verified answers).

Computes, for each question:
  - Hit@K       : was ANY expected location retrieved in the top K?
  - Precision@K : fraction of the top K retrieved chunks that match an
                  expected (filename, location) pair
  - Recall@K    : fraction of the expected locations that were retrieved
                  somewhere in the top K

Also supports a simple ablation mode so you can compare retrieval
strategies for your report:
    --mode hybrid       (default: semantic + BM25 + cross-encoder rerank)
    --mode semantic     (embeddings only)
    --mode bm25         (BM25 only)
    --mode no-rerank    (semantic + BM25, no cross-encoder)

Usage (run from the backend/ directory):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --mode bm25
    python scripts/run_evaluation.py --k 3
    python scripts/run_evaluation.py --compare-modes   # runs all 4 modes and prints a comparison table

This script does NOT judge answer correctness automatically -- it prints
each generated answer next to your 'expected_key_facts' so you can mark it
correct/incorrect by eye and note that in your report. Automatically
grading free-text answer correctness is out of scope for a lightweight,
explainable final-year project; retrieval metrics are computed exactly,
while answer-quality is manually verified (this distinction is worth
stating explicitly in your report).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src import retriever
from src import vector_store
from src import rag


EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "eval_dataset.json"


def load_test_cases():
    with open(EVAL_FILE, "r") as f:
        data = json.load(f)
    return data.get("test_cases", [])


def resolve_doc_id(filename: str):
    docs = vector_store.list_active_documents()
    for doc in docs:
        if doc["filename"] == filename:
            return doc["doc_id"]
    return None


def apply_mode(mode: str):
    """Temporarily reconfigure retrieval weights/toggles for ablation runs."""
    original = {
        "SEMANTIC_WEIGHT": config.SEMANTIC_WEIGHT,
        "KEYWORD_WEIGHT": config.KEYWORD_WEIGHT,
        "ENABLE_CROSS_ENCODER_RERANK": config.ENABLE_CROSS_ENCODER_RERANK,
    }
    if mode == "semantic":
        config.SEMANTIC_WEIGHT, config.KEYWORD_WEIGHT = 1.0, 0.0
        config.ENABLE_CROSS_ENCODER_RERANK = False
    elif mode == "bm25":
        config.SEMANTIC_WEIGHT, config.KEYWORD_WEIGHT = 0.0, 1.0
        config.ENABLE_CROSS_ENCODER_RERANK = False
    elif mode == "no-rerank":
        config.ENABLE_CROSS_ENCODER_RERANK = False
    elif mode == "hybrid":
        pass  # defaults (already set in config.py)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return original


def restore_mode(original: dict):
    config.SEMANTIC_WEIGHT = original["SEMANTIC_WEIGHT"]
    config.KEYWORD_WEIGHT = original["KEYWORD_WEIGHT"]
    config.ENABLE_CROSS_ENCODER_RERANK = original["ENABLE_CROSS_ENCODER_RERANK"]


def evaluate_retrieval(test_cases, k: int, show_answers: bool):
    hits, precisions, recalls = [], [], []
    skipped = []

    for case in test_cases:
        doc_id = resolve_doc_id(case["expected_filename"])
        if not doc_id:
            skipped.append(case["id"])
            continue

        retrieved = retriever.retrieve(case["question"], doc_ids=[doc_id], top_k=k)
        retrieved_locations = {r.get("location_label") for r in retrieved}
        expected_locations = set(case.get("expected_locations", []))

        matched = retrieved_locations & expected_locations
        hit = 1 if matched else 0
        precision = len(matched) / k if k else 0
        recall = len(matched) / len(expected_locations) if expected_locations else 0

        hits.append(hit)
        precisions.append(precision)
        recalls.append(recall)

        print(f"[{case['id']}] \"{case['question']}\"")
        print(f"    expected: {sorted(expected_locations)}  |  retrieved: {sorted(retrieved_locations)}")
        print(f"    Hit@{k}: {hit}  Precision@{k}: {precision:.2f}  Recall@{k}: {recall:.2f}")

        if show_answers:
            try:
                answer_result = rag.answer_question(case["question"], doc_ids=[doc_id])
                print(f"    Generated answer: {answer_result['answer'][:300]}")
                print(f"    Key facts to verify manually: {case.get('expected_key_facts', [])}")
                print(f"    Confidence: {answer_result['confidence']}")
            except rag.RAGError as exc:
                print(f"    (Could not generate answer: {exc})")
        print()

    if skipped:
        print(f"Skipped {len(skipped)} test case(s) - expected file not currently indexed: {skipped}")
        print("   Upload/process that file through the app first, then re-run.\n")

    n = len(hits)
    if n == 0:
        print("No evaluable test cases found. Edit eval/eval_dataset.json and index the referenced files.")
        return None

    summary = {
        "n_cases": n,
        f"Hit@{k}": sum(hits) / n,
        f"Precision@{k}": sum(precisions) / n,
        f"Recall@{k}": sum(recalls) / n,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("--k", type=int, default=config.TOP_K_FINAL, help="Top-K to evaluate")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "bm25", "no-rerank"], default="hybrid")
    parser.add_argument("--answers", action="store_true", help="Also generate and print LLM answers")
    parser.add_argument("--compare-modes", action="store_true", help="Run all modes and print a comparison table")
    args = parser.parse_args()

    test_cases = load_test_cases()
    if not test_cases:
        print("No test cases found in eval/eval_dataset.json.")
        return

    if args.compare_modes:
        print("=" * 70)
        print(f"ABLATION COMPARISON (K={args.k})")
        print("=" * 70)
        results = {}
        for mode in ["semantic", "bm25", "no-rerank", "hybrid"]:
            print(f"\n--- Mode: {mode} ---")
            original = apply_mode(mode)
            try:
                summary = evaluate_retrieval(test_cases, args.k, show_answers=False)
            finally:
                restore_mode(original)
            if summary:
                results[mode] = summary

        print("\n" + "=" * 70)
        print("SUMMARY TABLE (paste into your report)")
        print("=" * 70)
        header = f"{'Mode':<12} {'Hit@K':>8} {'Precision@K':>13} {'Recall@K':>10}"
        print(header)
        print("-" * len(header))
        for mode, summary in results.items():
            print(f"{mode:<12} {summary[f'Hit@{args.k}']:>8.2f} "
                  f"{summary[f'Precision@{args.k}']:>13.2f} {summary[f'Recall@{args.k}']:>10.2f}")
        return

    print("=" * 70)
    print(f"RETRIEVAL EVALUATION (mode={args.mode}, K={args.k})")
    print("=" * 70)
    original = apply_mode(args.mode)
    try:
        summary = evaluate_retrieval(test_cases, args.k, show_answers=args.answers)
    finally:
        restore_mode(original)

    if summary:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for key, value in summary.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
