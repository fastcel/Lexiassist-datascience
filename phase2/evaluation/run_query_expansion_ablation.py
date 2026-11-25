"""Compare LexiAssist full retrieval with query expansion OFF and ON.

Run from repository root with the EXISTING 1,000-document indexes:
  python phase2/evaluation/run_query_expansion_ablation.py --limit 3 --output phase2/evaluation/results/query_expansion/pilot_3
  python phase2/evaluation/run_query_expansion_ablation.py

Saves ranks, query expansions, timings, and a BLANK relevance-judgment pool.
No accuracy or nDCG is reported until source-based relevance review.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "phase2" / "shared"))

from query_expander import expand_query
from test_retrieval import check_same_corpus, import_retriever

K = 5
VARIANTS = ("lightweight", "pretrained")
DEFAULT_OUTPUT = EVAL / "results" / "query_expansion"


def read_questions(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))
    if not questions or not {"query_id", "query", "domain"}.issubset(questions[0]):
        raise ValueError("Question CSV must contain query_id, query, domain")
    if len({q["query_id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs are not unique")
    return questions[:limit] if limit else questions


def save_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=EVAL / "benchmark_questions.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    if len(set(args.variants)) != len(args.variants):
        ap.error("Do not list a variant twice")

    output = args.output.resolve()
    if output == (EVAL / "results").resolve():
        ap.error("Use a NEW output folder, not phase2/evaluation/results")
    existing = [name for name in ("rankings.csv", "judgments_TO_FILL.csv", "overlap_summary.csv")
                if (output / name).exists()]
    if existing:
        ap.error(f"Output files already exist in {output}: {existing}. Choose a new --output folder so prior work is not lost.")

    check_same_corpus()
    questions = read_questions(args.questions, args.limit)
    retrievers = {}
    for variant in args.variants:
        print(f"Loading {variant} index once...")
        retrievers[variant] = import_retriever(variant)()

    ranks: list[dict] = []
    pool: dict[tuple[str, str], dict] = {}
    expansions: list[dict] = []
    overlap_rows: list[dict] = []
    timing_rows: list[dict] = []

    for number, q in enumerate(questions, start=1):
        qid, query = q["query_id"], q["query"]
        variants = expand_query(query, n=3)
        expansions.append({"query_id": qid, "question": query,
                           "expansion_count": len(variants),
                           "expanded_queries": " || ".join(variants)})
        print(f"[{number}/{len(questions)}] {qid}: {query}")
        for variant in args.variants:
            sets: dict[str, set[str]] = {}
            times: dict[str, float] = {}
            for enabled, suffix in ((False, "off"), (True, "on")):
                start = time.perf_counter()
                found = retrievers[variant].retrieve(
                    query, sub_queries=(variants if enabled else []))[:K]
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                method = f"{variant}_{suffix}"
                times[suffix] = latency_ms
                keys = set()
                for rank, item in enumerate(found, start=1):
                    c = item["chunk"]
                    key = f"{c.doc_id}::chunk_{c.chunk_index}"
                    keys.add(key)
                    ranks.append({"query_id": qid, "method": method, "rank": rank,
                                  "doc_id": c.doc_id, "chunk_index": c.chunk_index,
                                  "chunk_key": key, "citation": c.citation,
                                  "jurisdiction": c.jurisdiction, "url": c.url,
                                  "latency_ms": latency_ms})
                    pool[(qid, key)] = {
                        "query_id": qid, "query": query, "domain": q.get("domain", ""),
                        "chunk_key": key, "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index, "citation": c.citation,
                        "url": c.url, "jurisdiction": c.jurisdiction,
                        "passage_text": c.text, "relevance_grade": ""
                    }
                sets[suffix] = keys
                timing_rows.append({"query_id": qid, "method": method,
                                    "latency_ms": latency_ms, "retrieved": len(found)})
                print(f"  {method}: {len(found)} passages, {latency_ms:.2f} ms")
            shared = len(sets["off"] & sets["on"])
            overlap_rows.append({"query_id": qid, "variant": variant,
                                 "common_top5": shared,
                                 "new_in_top5_with_expansion": len(sets["on"] - sets["off"]),
                                 "top5_jaccard": round(shared / len(sets["off"] | sets["on"]), 4)
                                 if sets["off"] | sets["on"] else 1.0,
                                 "off_latency_ms": times["off"],
                                 "on_latency_ms": times["on"]})

    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "rankings.csv",
             ["query_id", "method", "rank", "doc_id", "chunk_index", "chunk_key",
              "citation", "jurisdiction", "url", "latency_ms"], ranks)
    save_csv(output / "judgments_TO_FILL.csv",
             ["query_id", "query", "domain", "chunk_key", "doc_id", "chunk_index",
              "citation", "url", "jurisdiction", "passage_text", "relevance_grade"],
             sorted(pool.values(), key=lambda x: (x["query_id"], x["chunk_key"])))
    save_csv(output / "query_expansions.csv",
             ["query_id", "question", "expansion_count", "expanded_queries"], expansions)
    save_csv(output / "overlap_summary.csv",
             ["query_id", "variant", "common_top5", "new_in_top5_with_expansion",
              "top5_jaccard", "off_latency_ms", "on_latency_ms"], overlap_rows)
    save_csv(output / "timings.csv", ["query_id", "method", "latency_ms", "retrieved"], timing_rows)

    methods = sorted({r["method"] for r in timing_rows})
    timing_summary = [
        {"method": m, "questions": len([r for r in timing_rows if r["method"] == m]),
         "mean_latency_ms": round(statistics.mean(
             r["latency_ms"] for r in timing_rows if r["method"] == m), 2)}
        for m in methods
    ]
    save_csv(output / "timing_summary.csv",
             ["method", "questions", "mean_latency_ms"], timing_summary)
    (output / "run_metadata.json").write_text(json.dumps({
        "question_count": len(questions), "variants": args.variants,
        "retrieval_methods": methods, "top_k": K,
        "queries_with_expansion": sum(x["expansion_count"] > 0 for x in expansions),
        "unique_question_passage_pairs": len(pool),
        "note": "No retrieval-quality scores calculated. Candidate passages require source-based judgments.",
    }, indent=2), encoding="utf-8")
    print(f"\nSaved results in {output}")
    print(f"Unique question–passage pairs to review: {len(pool)}")
    print("Review ALL judgments_TO_FILL.csv passages, save as judgments_reviewed.csv,")
    print("then use existing phase2/evaluation/score_results.py for comparable retrieval metrics.")


if __name__ == "__main__":
    main()
