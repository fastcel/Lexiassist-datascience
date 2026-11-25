# LexiAssist Phase 2 — Flask Website (Original UI Reused)

This uses the original TokenlessLabs LexiAssist Flask/HTML design and existing Phase 2 retrievers. **No index rebuilding is required.**

## 1. File locations

Copy `website/server.py` and `website/gui.html` from this ZIP into your existing repository's `website/` folder (replace the old versions). Copy `requirements_phase2.txt` and `.env.example` to the repository root. The original Phase 1 notebooks and Phase 2 algorithms stay unchanged.

You should already have:

- `phase2/artifacts/lightweight_test_1000/` (saved TF-IDF + BM25 artifacts)
- `phase2/artifacts/pretrained_test_1000/` (saved MiniLM + FAISS + BM25 artifacts)
- `phase2/shared/{chunker.py,query_expander.py,generator.py}`
- `phase2/test_retrieval.py`

## 2. Start website from VS Code PowerShell (repository root)

```powershell
.\.venv\Scripts\python.exe -m pip install flask ollama
.\.venv\Scripts\python.exe website/server.py
```

Then open **http://127.0.0.1:5000** in a browser. Do not double-click `gui.html`, because the UI and API need to share the same web origin.

- **Retriever** dropdown: `Lightweight` uses original TF-IDF/BM25/RRF/rerank; `Pretrained` uses original MiniLM/FAISS/BM25/RRF/cross-encoder. The pretrained option may initially take more memory and time to load.
- **Answer** dropdown: `Source excerpts` works without Ollama and returns retrieved excerpts with citations. This is *not an LLM-generated answer*. `LLM answer` uses the existing `generator.py` and a locally running Ollama `llama3` model.
- **Simplify Text** needs Ollama.
- **Document Library** now displays actual indexed sample sources rather than hard-coded demonstration cards.

## 3. Enable actual LLM answers

Install and run **Ollama** separately. Download the `llama3` model with:

```powershell
ollama pull llama3
ollama list
```

Ollama should be running locally (typically as a Windows background app). Choose **LLM answer (requires Ollama)** on the website. If unavailable, the server clearly labels a fallback to source excerpts; it never silently presents excerpts as LLM output.

If your Ollama model has a different name, set the environment variable `LEXIASSIST_LLM_MODEL` and ensure the same name is used in `phase2/shared/generator.py` (the original generator currently uses the literal `llama3`).

## 4. Basic verification

1. Open `http://127.0.0.1:5000` and verify the status shows `38,223 chunks indexed`.
2. Choose **Lightweight** + **Source excerpts**.
3. Ask: `What is a de facto relationship under Australian law?`.
4. Confirm the app returns excerpt text and clickable source citations.
5. Choose **Pretrained** and repeat. Initial model loading may be slower.
6. For LLM Q&A/simplification, set up Ollama and switch mode.

## 5. Troubleshooting

- `ModuleNotFoundError: No module named 'flask'`: run the pip install command above with the **same** `.venv\Scripts\python.exe` interpreter.
- Missing index: verify both folders in `phase2/artifacts/` exist exactly as named above; they are local and ignored by Git.
- `Port 5000 already in use`: stop the previous server with Ctrl+C in its terminal.
- `Could not reach server`: navigate to the Flask URL, not the local HTML file.
- `Ollama generation unavailable`: start Ollama and run `ollama pull llama3`; the Source excerpts mode remains functional without it.
- **Deployment:** this package is first intended for **local demonstration**. A cloud deployment will need persistent index storage, model resources and access to an LLM backend; it is not automatically deployable to Streamlit Community Cloud.

## 6. Safety and reproducibility

This application searches a static 1,000-document sample, which can be incomplete or outdated. Outputs are for general educational legal research, not legal advice. The original models and retrieval source code are reused, not trained from scratch.

Do not commit: `.env`, `data/australian_legal_corpus_preprocessed.csv`, `phase2/artifacts/`. Ensure `.gitignore` excludes them. Keep your evaluation benchmark and transparently report AI-assisted draft relevance labels until a person verifies them.
