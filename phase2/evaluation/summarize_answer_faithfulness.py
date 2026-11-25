"""Summarize independently reviewed answer-quality ratings (never grades answers)."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

EVAL = Path(__file__).resolve().parent
DEFAULT = EVAL / "results" / "answer_faithfulness" / "answers_for_review.csv"
KEYS = {"answer_relevance_0_1_2": {"0", "1", "2"},
        "factual_support_0_1_2": {"0", "1", "2"},
        "citation_support_0_1_2": {"0", "1", "2"},
        "jurisdiction_correct_0_1": {"0", "1"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT)
    args = ap.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        ap.error("No answer rows to summarize")
    for i, row in enumerate(rows, start=2):
        for k, options in KEYS.items():
            if row.get(k, "").strip() not in options:
                ap.error(f"Row {i}, {k}: expected one of {sorted(options)}. Review every answer first.")
    print(f"Reviewed answers: {len(rows)}")
    for key, allowed in KEYS.items():
        nums = [int(r[key]) for r in rows]
        print(f"{key}: mean={statistics.mean(nums):.3f} (max={max(map(int, allowed))}), "
              f"full-score={sum(n == max(map(int, allowed)) for n in nums)}/{len(nums)}")
    print("Report the grading rubric, sample size, human review method, and representative failures.")


if __name__ == "__main__":
    main()
