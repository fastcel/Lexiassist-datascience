# LexiAssist- Phase 2 retrieval evaluation (pilot)
> **Final submission note:** This document describes the initial pilot evaluation workflow. The completed 100-question benchmark, final development and held-out test experiments, results, and limitations are documented in the [main README](../../README.md).

This evaluation **reuses both original LexiAssist retrieval implementations** and the two index sets you already generated from the same 1,000-document Phase 1 CSV (38,223 chunks). It does **not** need Ollama or rerun indexing.

## 1. Run a short pilot
From the repository root, in VS Code PowerShell:

```powershell
.\.venv\Scripts\python.exe phase2/evaluation/run_experiments.py --limit 3
```

This loads the two retrievers **once** and runs the **same three questions** for seven configs:

- `bm25`: original BM25 sparse search (baseline)
- `tfidf`: original TF-IDF cosine similarity (baseline)
- `minilm`: original MiniLM + FAISS semantic search (baseline)
- `lightweight_rrf`: TF-IDF + BM25 + RRF; no reranker
- `pretrained_rrf`: MiniLM + BM25 + RRF; no reranker
- `lightweight_full`: original lightweight RRF + BM25 re-scoring
- `pretrained_full`: original pretrained RRF + cross-encoder reranking

The baseline and no-reranker configurations use the original retrievers' internal search/fusion/dedupe methods. Each method's returned results are truncated at K=5. The first pass uses **no query expansion**, so it isolates retrieval methods; expansion is a later experiment.

**Important:** The original deduper is unchanged, and the full lightweight pipeline returns up to 8 before truncation to 5. We retain this behavior for reuse. Timing measurements are approximate local wall-clock times without loading time, and methods run sequentially (order/caches can influence latency).

## 2. Human-label candidate passages

Open `phase2/evaluation/results/judgments_TO_FILL.csv` in Excel (or a CSV editor).

Each row is a UNIQUE *(question, legal passage)* pair pooled from all seven configurations. Inspect the question and **complete passage**; verify legal relevance against the source if necessary. Enter an integer in **`relevance_grade`** for every row:

- `0` = irrelevant or misleading for the specific question
- `1` = partially useful / provides context
- `2` = directly relevant, materially answers the question

Don't enter grades based only on common keywords. Don't invent relevant sources. If a question has no answer in the sampled corpus, grade its unrelated results 0 and document that limitation.

**Avoid accidental data loss:** save the reviewed CSV as `judgments_reviewed.csv` in `phase2/evaluation/results/`. A CSV editor must preserve IDs and column order. Do not modify `chunk_key` or `query_id`.

## 3. Calculate honest metrics

After every pooled row has a reviewed grade:

```powershell
.\.venv\Scripts\python.exe phase2/evaluation/score_results.py --judgments phase2/evaluation/results/judgments_reviewed.csv
```

You'll get:

- `comparison_metrics.csv` — method-level averages
- `per_question_metrics.csv` — question-level results

Metrics: Hit Rate@5, Precision@5 (always dividing by 5, even if fewer than 5 results), MRR@5, nDCG@5, and *pooled* Recall@5. Since we judge only passages that any tested method retrieved, Recall@5 measures recall **within the reviewed candidate pool** and is not an estimate of exhaustive 38,223-chunk recall. Similarly, pooled nDCG's ideal ranking is based on available judgments.

This is **retrieval evaluation only**, not proof of factual answer faithfulness or citation accuracy. That requires a separate LLM-answer review.

## 4. Expand the benchmark

`benchmark_questions.csv` contains **20 proposed pilot questions**, with the original project's ten questions at the top. These questions are **not labeled**, and their answerability in your 1,000-document sample has **not been verified**. Review and adjust them, then expand to roughly **100 carefully sourced legal questions** to align with your proposal. Avoid creating a benchmark only from the system's retrieved passages; source-grounded labels reduce selection bias.

Once ready, rerun without `--limit`:

```powershell
.\.venv\Scripts\python.exe phase2/evaluation/run_experiments.py
```

Running experiments overwrites `results/judgments_TO_FILL.csv` and `results/rankings.csv`; **back up completed judgments first**. You may run a separate output directory with `--output`.

## 5. Repository etiquette

Commit the scripts, benchmark question file, and reviewed annotations/experiment summaries after checking for proprietary content and large payloads. The candidate-pool CSV includes text passages from the corpus and may be large; avoid committing it blindly. Generated indexes under `phase2/artifacts/` are already ignored by Git. Add a short narrative of results and limitations to your README.

The original `phase2/evaluation/evaluate.py` remains untouched for provenance, but its keyword-matching labels are not suitable as ground truth for final reported metrics.
