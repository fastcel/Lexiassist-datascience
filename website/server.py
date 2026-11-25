"""LexiAssist Phase 2 Flask server, adapted from original TokenlessLabs/LexiAssist.

Keeps the original retrievers, query expander, generator and HTML frontend.
Run from repository root: python website/server.py

The Source excerpts mode works without Ollama. For generated answers, install
Ollama, pull llama3, and choose 'LLM answer' in the website.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
import re

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "phase2" / "shared"))

from test_retrieval import check_same_corpus, import_retriever  # noqa: E402
from query_expander import expand_query  # noqa: E402

app = Flask(__name__)
VARIANTS = {"lightweight", "pretrained"}
DEFAULT_VARIANT = os.getenv("LEXIASSIST_VARIANT", "lightweight").strip().lower()
if DEFAULT_VARIANT not in VARIANTS:
    raise ValueError("LEXIASSIST_VARIANT must be lightweight or pretrained")

# Load each retriever only when requested; reuse it for subsequent queries.
_CACHE = {}
_CACHE_LOCK = threading.RLock()


def get_retriever(variant: str):
    if variant not in VARIANTS:
        raise ValueError("Invalid variant; choose 'lightweight' or 'pretrained'.")
    with _CACHE_LOCK:
        if variant not in _CACHE:
            # Validates that both complete indexes share the exact same dataset.
            check_same_corpus()
            _CACHE[variant] = import_retriever(variant)()
        return _CACHE[variant]


def citation_info(retrieved):
    citations = []
    for rank, row in enumerate(retrieved, 1):
        c = row["chunk"]
        citations.append({
            "index": rank,
            "citation": c.citation,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "jurisdiction": c.jurisdiction,
            "url": c.url,
            "rerank_score": round(float(row["score"]), 4),
            "excerpt": c.text[:350],
        })
    return citations


def source_only_answer(retrieved):
    if not retrieved:
        return "No matching passages were retrieved from the indexed sample."
    sections = [
        "Source-excerpts mode (not an LLM-generated answer). "
        "The passages below were retrieved from the sampled legal corpus. "
        "They may not fully answer the question; verify the citation, jurisdiction and current law."
    ]
    for rank, row in enumerate(retrieved, 1):
        c = row["chunk"]
        excerpt = " ".join(c.text.split())[:460]
        sections.append(f"[{rank}] {c.citation or 'Untitled source'} — "
                        f"{c.jurisdiction or 'unspecified jurisdiction'}\n{excerpt}…")
    sections.append("Disclaimer: This is general legal information, not legal advice.")
    return "\n\n".join(sections)


@app.get("/")
def serve_gui():
    return send_from_directory(str(Path(__file__).parent), "gui.html")


@app.get("/api/status")
def api_status():
    try:
        retriever = get_retriever(DEFAULT_VARIANT)
        return jsonify({
            "status": "ok",
            "chunks_indexed": len(retriever.chunks),
            "default_variant": DEFAULT_VARIANT,
            "available_variants": sorted(VARIANTS),
            "embed_model": "TF-IDF" if DEFAULT_VARIANT == "lightweight" else "all-MiniLM-L6-v2",
            "rerank_model": "BM25" if DEFAULT_VARIANT == "lightweight" else "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm": os.getenv("LEXIASSIST_LLM_MODEL", "llama3"),
            "note": "LLM mode requires a running Ollama service and downloaded model.",
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(exc)}), 503



@app.post("/api/ask")
def api_ask():
    data = request.get_json(silent=True) or {}

    query = str(data.get("query") or "").strip()

    # Normalize common legal spelling variations
    query = re.sub(
        r"\bdefacto\b",
        "de facto",
        query,
        flags=re.IGNORECASE
    )
    query = query.rstrip(" ?!.,")

    if not query:
        return jsonify({
            "error": "Please enter a legal research question."
        }), 400

    if len(query) > 600:
        return jsonify({"error": "Question is too long (maximum 600 characters)."}), 400

    variant = str(data.get("variant") or DEFAULT_VARIANT).lower()
    mode = str(data.get("mode") or "sources").lower()
    if variant not in VARIANTS or mode not in {"sources", "llm"}:
        return jsonify({"error": "Invalid retrieval variant or response mode."}), 400

    try:
        retriever = get_retriever(variant)
        sub_queries = []
        rows = retriever.retrieve(query, sub_queries=sub_queries)
        # The original lightweight retriever returns up to 8; the frontend
        # uses a common top-5 for both variants. This is not an evaluation.
        rows = rows[:5]
        answer = source_only_answer(rows)
        citations = citation_info(rows)
        used_mode = "sources"
        warning = None
        if mode == "llm" and rows:
            try:
                # Import lazily so a missing Ollama package does not stop the
                # website from working in source-excerpts mode.
                from generator import generate_answer
                
                response = generate_answer(query, rows)
                answer = response["answer"]

                disclaimer = "Disclaimer: This is general legal information, not legal advice."

                if disclaimer.lower() not in answer.lower():
                    answer = answer.rstrip() + "\n\n" + disclaimer

                used_mode = "llm"

            except Exception as exc:
                warning = (
                    "Ollama generation is unavailable; showing retrieved sources instead. "
                    "Install the ollama Python package, start Ollama and pull llama3. "
                    f"Technical detail: {type(exc).__name__}."
                )
                app.logger.warning("LLM generation unavailable: %s", exc)
        return jsonify({
            "answer": answer,
            "citations": citations,
            "sub_queries": sub_queries,
            "variant": variant,
            "response_mode": used_mode,
            "warning": warning,
            "query": query,
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Retrieval failed. Check the VS Code terminal for details."}), 500


@app.post("/api/simplify")
def api_simplify():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please paste legal text to simplify."}), 400
    if len(text) > 8000:
        return jsonify({"error": "Text too long (maximum 8,000 characters)."}), 400
    try:
        import ollama
        response = ollama.chat(
            model=os.getenv("LEXIASSIST_LLM_MODEL", "llama3"),
            messages=[
                {"role": "system", "content": "Simplify legal text into clear, plain English. Preserve legal meaning. Do not fabricate legal facts."},
                {"role": "user", "content": text},
            ],
        )
        return jsonify({"simplified": response.message.content})
    except Exception as exc:
        app.logger.warning("Simplify unavailable: %s", exc)
        return jsonify({"error": "Text simplification requires Ollama running with the llama3 model."}), 503


@app.get("/api/documents")
def api_documents():
    """Display actual indexed corpus documents, instead of hardcoded examples."""
    try:
        retriever = get_retriever(DEFAULT_VARIANT)
        seen = set()
        docs = []
        for c in retriever.chunks:
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            docs.append({"doc_id": c.doc_id, "citation": c.citation or "Untitled legal document",
                         "jurisdiction": c.jurisdiction, "url": c.url})
            if len(docs) >= 15:
                break
        return jsonify({"documents": docs, "count": len(docs), "note": "Sample of indexed documents"})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Unable to load document library."}), 503


if __name__ == "__main__":
    print("LexiAssist Phase 2 — http://127.0.0.1:5000")
    print("Source-excerpts mode works without Ollama; LLM mode needs Ollama + llama3.")
    app.run(host="127.0.0.1", port=5000, debug=False)
