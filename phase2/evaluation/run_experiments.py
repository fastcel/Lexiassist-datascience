"""LexiAssist Phase 2: fair retrieval comparisons on one shared chunk corpus.

Reuses original TokenlessLabs retriever methods and the two indexes already built.
Only retrieval is evaluated here: no Ollama, legal advice, or fabricated scores.
Run from repository root:
    python phase2/evaluation/run_experiments.py --limit 3
    python phase2/evaluation/run_experiments.py
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "phase2" / "shared"))
from test_retrieval import check_same_corpus, import_retriever  

METHODS = (
    "bm25", "tfidf", "minilm", "lightweight_rrf", "pretrained_rrf",
    "lightweight_full", "pretrained_full",
)
METHOD_VARIANT = {
    "bm25": "lightweight", "tfidf": "lightweight", "minilm": "pretrained",
    "lightweight_rrf": "lightweight", "pretrained_rrf": "pretrained",
    "lightweight_full": "lightweight", "pretrained_full": "pretrained",
}
K = 5


def read_questions(path, limit=None):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"query_id", "query", "domain"}.issubset(rows[0]):
        raise ValueError("Question file must contain query_id, query, domain")
    if any(not r["query_id"].strip() or not r["query"].strip() for r in rows):
        raise ValueError("Each question needs a nonblank query_id and query")
    ids = [r["query_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("query_id values must be unique")
    return rows[:limit] if limit else rows


def as_results(retriever, indexed):
    """Same original Jaccard deduper, applied to every baseline/ablation."""
    candidates = [
        {"chunk": retriever.chunks[int(i)], "score": float(s), "rank": rank}
        for rank, (i, s) in enumerate(indexed, 1)
    ]
    return retriever._dedupe(candidates)[:K]


def execute(method, retriever, question):
    if method == "bm25":
        ids = retriever._bm25_search([question])[0]
        return as_results(retriever, [(i, len(ids) - r) for r, i in enumerate(ids)])
    if method == "tfidf":
        ids = retriever._tfidf_search([question])[0]
        return as_results(retriever, [(i, len(ids) - r) for r, i in enumerate(ids)])
    if method == "minilm":
        ids = retriever._dense([question])[0]
        return as_results(retriever, [(i, len(ids) - r) for r, i in enumerate(ids)])
    if method == "lightweight_rrf":
        a = retriever._tfidf_search([question]); b = retriever._bm25_search([question])
        return as_results(retriever, retriever._rrf(a, b))
    if method == "pretrained_rrf":
        a = retriever._dense([question]); b = retriever._sparse([question])
        return as_results(retriever, retriever._rrf(a, b))
    if method in ("lightweight_full", "pretrained_full"):
        return retriever.retrieve(question)[:K]  
    raise ValueError(method)


def write_csv(path, columns, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        out.writeheader()
        out.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "benchmark_questions.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="Number of questions to test first (e.g. 3). Omit for all.")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--output", type=Path, default=EVAL_DIR / "results")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >=1")
    if len(set(args.methods)) != len(args.methods):
        ap.error("Do not repeat method names")

    check_same_corpus()
    questions = read_questions(args.questions, args.limit)
    needed = sorted(set(METHOD_VARIANT[m] for m in args.methods))
    retrievers = {}
    load_times = {}
    for variant in needed:
        print(f"Loading {variant} retriever once...")
        t = time.perf_counter()
        retrievers[variant] = import_retriever(variant)()
        load_times[variant] = time.perf_counter() - t

    results, pool = [], {}
    times = defaultdict(list)
    for ix, q in enumerate(questions, 1):
        print(f"[{ix}/{len(questions)}] {q['query_id']}: {q['query']}")
        for method in args.methods:
            retriever = retrievers[METHOD_VARIANT[method]]
            t = time.perf_counter()
            found = execute(method, retriever, q["query"])
            latency_ms = round((time.perf_counter() - t) * 1000, 2)
            times[method].append(latency_ms)
            for rank, item in enumerate(found[:K], 1):
                c = item["chunk"]
                chunk_key = f"{c.doc_id}::chunk_{c.chunk_index}"
                results.append(dict(query_id=q["query_id"], method=method,
                    rank=rank, doc_id=c.doc_id, chunk_index=c.chunk_index,
                    chunk_key=chunk_key, citation=c.citation, jurisdiction=c.jurisdiction,
                    url=c.url, latency_ms=latency_ms))
                pool[(q["query_id"], chunk_key)] = dict(
                    query_id=q["query_id"], query=q["query"], domain=q.get("domain", ""),
                    chunk_key=chunk_key, doc_id=c.doc_id, chunk_index=c.chunk_index,
                    citation=c.citation, url=c.url, jurisdiction=c.jurisdiction,
                    passage_text=c.text, relevance_grade="")
            print(f"  {method}: {len(found[:K])} passages, {latency_ms:.1f} ms")

    args.output.mkdir(parents=True, exist_ok=True)
    rankings_path = args.output / "rankings.csv"
    judgments_path = args.output / "judgments_TO_FILL.csv"
    write_csv(rankings_path, ["query_id", "method", "rank", "doc_id", "chunk_index",
        "chunk_key", "citation", "jurisdiction", "url", "latency_ms"], results)
    write_csv(judgments_path, ["query_id", "query", "domain", "chunk_key", "doc_id",
        "chunk_index", "citation", "url", "jurisdiction", "passage_text", "relevance_grade"],
        sorted(pool.values(), key=lambda r: (r["query_id"], r["doc_id"], int(r["chunk_index"]))))
    summary = {"questions": len(questions), "methods": args.methods, "top_k": K,
               "load_seconds": load_times,
               "mean_retrieval_ms": {m: round(sum(t)/len(t), 2) for m, t in times.items()},
               "warning": "Latency and candidates only. No relevance labels verified yet."}
    (args.output / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved:", rankings_path)
    print("Saved:", judgments_path)
    print("Next: manually fill relevance_grade (0/1/2) for EVERY row before scoring.")
    print("No accuracy or retrieval-quality score is reported until the labels are reviewed.")


if __name__ == "__main__":
    main()
