"""Collect generated LexiAssist answers and FULL supporting passages for human review.

Run from repository root after installing Ollama and pulling llama3:
  python phase2/evaluation/collect_answer_faithfulness.py --limit 2
  python phase2/evaluation/collect_answer_faithfulness.py --limit 10

This writes actual model answers and leaves ALL assessment fields empty.
If interrupted, rerun the same command; previously collected rows are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "phase2" / "shared"))

from test_retrieval import check_same_corpus, import_retriever
from generator import generate_answer

OUTPUT = EVAL / "results" / "answer_faithfulness" / "answers_for_review.csv"
COLUMNS = ["query_id", "question", "variant", "llm_model", "answer",
           "sources_json", "answer_relevance_0_1_2", "factual_support_0_1_2",
           "citation_support_0_1_2", "jurisdiction_correct_0_1",
           "unsupported_claims_or_notes", "reviewer", "review_date"]


def questions(path, limit):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        data = list(csv.DictReader(f))
    return data[:limit] if limit else data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--variant", choices=["lightweight", "pretrained"], default="pretrained")
    ap.add_argument("--questions", type=Path, default=EVAL / "benchmark_questions.csv")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    if args.limit < 1 or args.limit > 20:
        ap.error("--limit must be 1–20 for this pilot")

    check_same_corpus()
    data = questions(args.questions, args.limit)
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != COLUMNS:
                raise ValueError("Existing CSV columns differ. Use another --output path.")
            existing = list(reader)
    complete = {(r["query_id"], r["variant"]) for r in existing}
    needed = [q for q in data if (q["query_id"], args.variant) not in complete]
    if not needed:
        print(f"All {len(data)} selected questions are already collected in {path}")
        return

    print(f"Loading {args.variant} retriever. Ollama llama3 must be running locally.")
    retriever = import_retriever(args.variant)()
    with path.open("a" if path.exists() else "w", encoding="utf-8-sig" if not path.exists() else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not existing:
            writer.writeheader()
        for n, q in enumerate(needed, start=1):
            print(f"[{n}/{len(needed)}] Generating answer for {q['query_id']}: {q['query']}", flush=True)
            retrieved = retriever.retrieve(q["query"], sub_queries=[])[:5]
            if not retrieved:
                answer = "No passages retrieved; no answer generated."
            else:
                try:
                    response = generate_answer(q["query"], retrieved)
                except Exception as exc:
                    print(f"Generation failed: {type(exc).__name__}: {exc}")
                    print("Previously written rows are saved. Start Ollama and rerun this command.")
                    raise SystemExit(1) from exc
                answer = response["answer"].strip()
                disclaimer = "Disclaimer: This is general legal information, not legal advice."
                if disclaimer.lower() not in answer.lower():
                    answer += "\n\n" + disclaimer
            sources = [
                {"index": i, "citation": r["chunk"].citation,
                 "jurisdiction": r["chunk"].jurisdiction,
                 "url": r["chunk"].url, "passage": r["chunk"].text}
                for i, r in enumerate(retrieved, 1)
            ]
            writer.writerow({"query_id": q["query_id"], "question": q["query"],
                             "variant": args.variant, "llm_model": "llama3",
                             "answer": answer, "sources_json": json.dumps(sources, ensure_ascii=False)})
            f.flush()
    print(f"Saved generated answers to {path}")
    print("Now REVIEW each answer against its source passages and fill the blank assessment columns.")


if __name__ == "__main__":
    main()
