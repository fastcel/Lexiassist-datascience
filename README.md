# LexiAssist- Australian Legal Research RAG (Data Science Course Project)

**Status:** Phase 1 and Phase 2 completed; experimental evaluation performed on a sampled corpus.  
**Website:** Local Flask application (`website/server.py`); no public deployment URL.  
**Project Report:** [Read the Final Project Report](LexiAssistFinalReport.pdf)  
**Disclaimer:** For educational research only. Retrieved materials and generated explanations may be incomplete, outdated or inaccurate; not legal advice.

## 1. Problem and approach

Australian law is spread across Commonwealth, state and territory sources. LexiAssist helps users search legal text and inspect source-backed answers while retaining the ability to view original excerpts. It is a **retrieval-augmented generation (RAG)** project, not a model-training project. The user-facing demo runs locally.

The system contains two retrieval pipelines:

| Pipeline | Retrieval and reranking |
|---|---|
| Lightweight | TF-IDF + BM25, reciprocal rank fusion (RRF), optional rule-based legal query expansion, BM25 reranking and deduplication |
| Pretrained | `all-MiniLM-L6-v2` embeddings with FAISS, BM25, RRF, optional query expansion, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking and deduplication |

The shared preprocessing layer uses **sentence-aware chunks with a 250-word target and two-sentence overlap**; the sampled dataset of **1,000 documents** produces **38,223 chunks**. The Flask website presents source excerpts or, when Ollama is available, locally generated answers from **Llama 3**. Prompt instructions and source citation format are in `phase2/shared/generator.py`; query rewriting is in `phase2/shared/query_expander.py`.

**Deployed website vs research configurations:** The current website uses each pipeline's normal retrieval path **without expansion** (`sub_queries=[]`). The best *experimental* pretrained setting, **RRF 30 + expansion ON**, was evaluated in a separate frozen research script; this research selection has **not** been automatically applied to the website. The source-excerpts mode works without Ollama; LLM answering and text simplification require Ollama and `llama3`.

## 2. Dataset and Phase 1

Data: [Open Australian Legal Corpus](https://huggingface.co/datasets/isaacus/open-australian-legal-corpus). Phase 1 samples **1,000 documents** (seed 42), audits document types/jurisdictions and missing metadata, conservatively cleans text and produces retrieval-ready CSV records. The sample is **not comprehensive or guaranteed current** Australian law.

Phase 1 notebooks with saved outputs:

- [`01_data_acquisition_and_initial_audit.ipynb`](notebooks/01_data_acquisition_and_initial_audit.ipynb)
- [`02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb)
- [`03_exploratory_data_analysis.ipynb`](notebooks/03_exploratory_data_analysis.ipynb)
- [`04_text_preprocessing.ipynb`](notebooks/04_text_preprocessing.ipynb)

Complete dataset downloads are in the [Phase 1 GitHub Release](https://github.com/fastcel/Lexiassist-datascience/releases/tag/v1.0-phase1), including `australian_legal_corpus_preprocessed.csv`. `data/australian_legal_corpus_readable.xlsx` is only an inspection preview, not the complete text dataset. The complete CSV is intentionally excluded from Git because of its size. See the `.gitignore` file.

## 3. Phase 2 experiments and results

The retrieval study compares lexical, semantic and fused/reranked configurations using **Hit@5, Precision@5, MRR@5, nDCG@5 and retrieval latency**. A 100-question *source-linked* benchmark was assembled as 40 development and 60 held-out test questions; the development questions were used to tune configurations **before** testing. Source-link and relevance assessments are **model-based/provisional**, **not independent human or legal-expert verification**. The judged passage pool is not an exhaustive sample of all relevant passages in the 38,223-chunk corpus. The numbers below describe an exploratory course study, not established legal accuracy.

### Locked final 60-question test

| Metric | TF-IDF lightweight | Pretrained RRF 30 + expansion ON |
|---|---:|---:|
| Hit@5 | 0.9333 | **0.9500** |
| Precision@5 | 0.4333 | **0.4933** |
| MRR@5 | 0.7050 | **0.7908** |
| nDCG@5 | 0.6564 | **0.7502** |
| Mean retrieval latency | **390.14 ms** | 1,597.90 ms |

**Interpretation:** The preselected pretrained pipeline ranks relevant evidence better on this judged test pool, but is approximately **4.1 times slower** than TF-IDF. The final configuration was selected by development-set nDCG@5, not by inspecting test results. Earlier 20-question, query-expansion ON/OFF, and 40-question tuning comparisons are included in the evaluation results folders.

### Generated answer quality (10-question pilot)

Comparing locally generated Llama 3 answers to the five retrieved passages for each question gave provisional mean scores: **answer relevance 1.20/2**, **factual support 0.90/2**, **citation support 0.70/2**, **jurisdiction correctness 0.70/1**. Common problems include incomplete responses, unsupported generalizations, and incorrect citation markers. These passage-based ratings do **not** establish whether every legal claim reflects current law. See `phase2/evaluation/results/answer_faithfulness/answers_scored_provisional.csv` and [`notebooks/05_phase2_experiments.ipynb`](notebooks/05_phase2_experiments.ipynb).

### Reproducible outputs and visualizations

- [`notebooks/05_phase2_experiments.ipynb`](notebooks/05_phase2_experiments.ipynb) — Phase 2 result tables and **saved plot outputs**.
- [`docs/figures/`](docs/figures/) — static plots of held-out scores, runtime and answer-quality screening.
- [`phase2/evaluation/results/option_b_dev_combined_40/`](phase2/evaluation/results/option_b_dev_combined_40/) — development scores for 19 configurations against one candidate pool.
- [`phase2/evaluation/results/option_b_test_final/`](phase2/evaluation/results/option_b_test_final/) — locked held-out retrieval outputs and provisional grades.
- [`phase2/evaluation/results/query_expansion/full_20/`](phase2/evaluation/results/query_expansion/full_20/) — query-expansion ablation.
- [`phase2/evaluation/option_b/preprocessing_sensitivity.csv`](phase2/evaluation/option_b/preprocessing_sensitivity.csv) — chunk count and processing time for 27 settings; **not** a relevance test.
- [`docs/screenshots/`](docs/screenshots/) — live interface screenshots.

### Limitations

- Model-based question/source and passage relevance grades are **provisional**; there was no independent expert adjudication.
- The small, geographically unbalanced sample omits many valid legal authorities and may include outdated material.
- Measured query latency excludes index/model loading and generated-answer time, and timings were not repeatedly sampled for uncertainty intervals.
- Generated answers sometimes misinterpret cases, miss the legal question or provide unsupported citations.
- The experimental best configuration and the currently deployed local website's default settings are different, as described above.

## 4. Local installation and website demo

> **No online deployment is required to run the project.** The website is tested locally; this repository does not claim a public live URL.

1. Clone the project and use the **final submission branch** (the same version as the default branch after merging):

   ```bash
   git clone https://github.com/fastcel/Lexiassist-datascience.git
   cd Lexiassist-datascience
   ```

2. Create a Python environment and install requirements (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   # Windows PowerShell: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   # macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
   ```

3. Download `australian_legal_corpus_preprocessed.csv` from the [Phase 1 release](https://github.com/fastcel/Lexiassist-datascience/releases/tag/v1.0-phase1) into `data/` (full-text CSV, not the `.xlsx` preview).

4. Build reproducible indexes for the **1,000-document run**. The `--limit 1000` flag is intentional: it gives the artifact folder names expected by the existing Flask server and experiment scripts:

   ```powershell
   .\.venv\Scripts\python.exe phase2/lightweight/Indexer.py --limit 1000
   .\.venv\Scripts\python.exe phase2/pretrained/Indexer.py --limit 1000
   .\.venv\Scripts\python.exe phase2/test_retrieval.py --variant both
   ```

   On macOS/Linux replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
   The indexes are large, generated locally and ignored by Git; no manual binary upload is necessary.

5. Start the web app:

   ```powershell
   .\.venv\Scripts\python.exe website/server.py
   ```

   Open **http://127.0.0.1:5000**. Select **Lightweight / Source excerpts** to demonstrate local retrieval without an LLM.

6. For **LLM answer** or **Simplify legal text**, install [Ollama](https://ollama.com/download), run `ollama pull llama3` and ensure Ollama is running. Then select the LLM answer mode in the website. No paid API key is required.

For more detailed UI troubleshooting, see [`website/README_WEBSITE.md`](website/README_WEBSITE.md). If inference is unavailable, the website identifies that and falls back to source excerpts.

## 5. Repository structure

```text
notebooks/                           Phase 1 notebooks + Phase 2 experiment notebook
phase2/lightweight/                 TF-IDF/BM25 indexer and retriever
phase2/pretrained/                  MiniLM/FAISS/BM25 indexer and retriever
phase2/shared/                     Chunker, corpus adapter, prompt, query expander
phase2/evaluation/                 Benchmarks, metrics, ablations and answer screening
phase2/evaluation/option_b/frozen/  Fixed development/test questions and selected configs
phase2/evaluation/results/         Measured results, model-based passage grades
phase2/artifacts/                  Generated indexes (ignored by Git)
website/                           Flask server and HTML frontend
notebooks/                          Phase 1/2 notebooks with output cells
docs/screenshots/                   Interface screenshots
docs/figures/                       Final evaluation charts
requirements.txt                    Dependencies for Phase 1 and 2
requirements_phase2.txt             Compatibility installer, including root requirements
.env.example                        Optional environment settings, no keys
.gitignore                          Prevents committing secrets, local indexes, local dataset
```

## 6. Notes on evaluation and reproducibility

This project **does not train an LLM**. It evaluates retrieval settings and manually inspectable source excerpts. The Phase 2 scripts reproduce indexing and retrieval experiments when provided with the sampled corpus. Test results in `phase2/evaluation/results/option_b_test_final/` are preserved and should **not** be used to reselect hyperparameters. The original seven-method study, the 20-question expansion ablation, and the development-only 19-way tuning run are retained for provenance. See `phase2/evaluation/README_EVALUATION.md` for grading rules and metrics.

The provided relevance CSVs and answer-quality scores contain **provisional model-generated labels**. The CLI scorer's printed phrase “manually graded” is a generic status line and should not be interpreted as proof of independent human grading. Generated answers should always be verified against authoritative legal sources.

## 7. Conclusion

LexiAssist demonstrates the effectiveness of different retrieval strategies for Australian legal question answering. The pretrained hybrid configuration achieved better retrieval performance than the lightweight TF-IDF baseline, although at a higher computational cost. The generated answer evaluation also revealed limitations in factual grounding and citation accuracy, highlighting the importance of reliable retrieval and source verification.

Overall, the project demonstrates the trade-off between retrieval quality, computational efficiency and answer reliability in legal RAG systems. Future work could include independent human evaluation, a larger legal corpus and improved answer grounding.

## 8. Submission

This course project is submitted **via GitHub**. The repository is public, making the source available without collaborator invitations. Phase 2 belongs on the default branch for easy assessment. Project screenshots and evaluation plots are provided under `docs/`.
