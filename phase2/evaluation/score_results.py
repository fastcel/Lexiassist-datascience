"""Compute retrieval scores ONLY from manually reviewed passage judgments.

The relevance pool is candidate-limited, so pooled_recall@5 and nDCG@5
cannot be interpreted as exhaustive corpus-wide recall and nDCG.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent
K = 5


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def score(rankings, judgments):
    if not rankings or not judgments:
        raise ValueError("No rankings or judgments found; run the retrieval experiment first")
    qrels = {}
    for row in judgments:
        key = (row["query_id"], row["chunk_key"])
        if key in qrels:
            raise ValueError(f"Duplicate judgment: {key}")
        grade = row["relevance_grade"].strip()
        if grade not in {"0", "1", "2"}:
            raise ValueError(f"Unreviewed/invalid grade for {key}: {grade!r}. Fill ALL rows with 0, 1, or 2.")
        qrels[key] = int(grade)
    by_query, by_combo = defaultdict(list), defaultdict(list)
    for (qid, _), grade in qrels.items():
        by_query[qid].append(grade)
    for row in rankings:
        key = (row["query_id"], row["chunk_key"])
        if key not in qrels:
            raise ValueError(f"Missing judgment for result: {key}")
        by_combo[(row["method"], row["query_id"])].append(row)

    detail = []
    for (method, qid), rows in sorted(by_combo.items()):
        ranks = [int(r["rank"]) for r in rows]
        if len(rows) > K or sorted(ranks) != list(range(1, len(rows)+1)):
            raise ValueError(f"Unexpected ranks in {(method, qid)}: {ranks}")
        rows.sort(key=lambda r: int(r["rank"]))
        grades = [qrels[(qid, r["chunk_key"])] for r in rows]
        binary = [int(g >= 1) for g in grades]
        rr = next((1/(i+1) for i,v in enumerate(binary) if v), 0.0)
        positives_in_pool = sum(g >= 1 for g in by_query[qid])
        ideal = sorted(by_query[qid], reverse=True)[:K]
        dcg = sum((2**g-1) / math.log2(i+2) for i,g in enumerate(grades))
        idcg = sum((2**g-1) / math.log2(i+2) for i,g in enumerate(ideal))
        detail.append(dict(method=method, query_id=qid, retrieved=len(rows),
            hit_at_5=int(any(binary)), precision_at_5=round(sum(binary)/K, 4),
            pooled_recall_at_5=round(sum(binary)/positives_in_pool,4) if positives_in_pool else "",
            reciprocal_rank=round(rr, 4), ndcg_at_5=round(dcg/idcg, 4) if idcg else "",
            latency_ms=round(float(rows[0]["latency_ms"]), 2) if rows else 0))
    per_method = defaultdict(list)
    for d in detail:
        per_method[d["method"]].append(d)
    summaries=[]
    for method, items in sorted(per_method.items()):
        def avg(field):
            vals=[float(r[field]) for r in items if r[field] != ""]
            return round(statistics.mean(vals), 4) if vals else ""
        summaries.append(dict(method=method, questions=len(items), hit_at_5=avg("hit_at_5"),
            precision_at_5=avg("precision_at_5"), pooled_recall_at_5=avg("pooled_recall_at_5"),
            mrr_at_5=avg("reciprocal_rank"), ndcg_at_5=avg("ndcg_at_5"),
            mean_latency_ms=avg("latency_ms")))
    return detail, summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DIR / "results")
    ap.add_argument("--judgments", type=Path, default=None,
                    help="Path to reviewed copy of judgments_TO_FILL.csv")
    args = ap.parse_args()
    rankings = read(args.results / "rankings.csv")
    judgments_path = args.judgments or args.results / "judgments_TO_FILL.csv"
    judgments = read(judgments_path)
    detail, summaries = score(rankings, judgments)
    write(args.results / "per_question_metrics.csv", list(detail[0]), detail)
    write(args.results / "comparison_metrics.csv", list(summaries[0]), summaries)
    print("\nRetrieval evaluation complete (manually graded):")
    for row in summaries:
        print(f" {row['method']:<20} Hit@5={row['hit_at_5']} "
              f"P@5={row['precision_at_5']} MRR@5={row['mrr_at_5']} "
              f"nDCG@5={row['ndcg_at_5']} latency={row['mean_latency_ms']}ms")
    print("\nNote: pooled recall/nDCG depend on the judged candidate pool, not all 38,223 chunks.")


if __name__ == "__main__":
    main()
