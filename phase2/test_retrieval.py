r"""Smoke-test existing LexiAssist retrieval without starting Ollama or Flask.

Examples (run from repository root):
    .\.venv\Scripts\python.exe phase2/test_retrieval.py --variant lightweight
    .\.venv\Scripts\python.exe phase2/test_retrieval.py --variant pretrained
    .\.venv\Scripts\python.exe phase2/test_retrieval.py --variant both --expand
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2" / "shared"))
from query_expander import expand_query  # noqa: E402


ARTIFACTS = ROOT / "phase2" / "artifacts"


def import_retriever(variant):
    file_path = ROOT / "phase2" / variant / "retriever.py"
    spec = importlib.util.spec_from_file_location(f"lexiassist_{variant}_retriever", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load retriever from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Retriever


def check_same_corpus():
    paths = [ARTIFACTS / v / "manifest.json" for v in ["lightweight_test_1000", "pretrained_test_1000"]]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Both full-index manifest files are required to compare the variants.")
    a, b = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for field in ["source_sha256", "documents_indexed", "chunks_indexed", "chunk_size_words", "overlap_sentences"]:
        if a.get(field) != b.get(field):
            raise ValueError(f"Different {field}: lightweight={a.get(field)}, pretrained={b.get(field)}")
    print(f"Verified same CSV (SHA-256) and {a['chunks_indexed']:,} chunks for both implementations.\n")


def main():
    parser = argparse.ArgumentParser(description="Test Phase 2 retrieval; no website or LLM required")
    parser.add_argument("--variant", choices=["lightweight", "pretrained", "both"], default="lightweight")
    parser.add_argument("--query", default="What is a de facto relationship under Australian law?")
    parser.add_argument("--expand", action="store_true", help="Use original rule-based query expander (3 subqueries)")
    args = parser.parse_args()

    check_same_corpus()
    variants = ["lightweight", "pretrained"] if args.variant == "both" else [args.variant]
    subqueries = expand_query(args.query, n=3) if args.expand else []
    print(f"Question: {args.query}\n")
    if subqueries:
        print("Expanded queries:", subqueries, "\n")

    for variant in variants:
        print("=" * 65)
        print("Retrieval method:", variant)
        retriever_cls = import_retriever(variant)
        start = time.perf_counter()
        retriever = retriever_cls()
        load_seconds = time.perf_counter() - start
        start = time.perf_counter()
        results = retriever.retrieve(args.query, sub_queries=subqueries)
        search_seconds = time.perf_counter() - start
        print(f"Model/index loading: {load_seconds:.2f}s; retrieval only: {search_seconds:.2f}s")
        print(f"Retrieved {len(results)} passages:\n")
        for i, row in enumerate(results, 1):
            c = row["chunk"]
            print(f"{i}. Citation: {c.citation or '(not available)'}")
            print(f"   Document ID: {c.doc_id} | Chunk: {c.chunk_index}")
            print(f"   Jurisdiction: {c.jurisdiction or 'unspecified'}")
            print(f"   Score: {row['score']:.4f} | Source: {c.url or 'not available'}")
            print(f"   Excerpt: {' '.join(c.text.split())[:300]}\n")
    print("Index retrieval smoke test completed. Relevance and answer quality are NOT yet evaluated.")


if __name__ == "__main__":
    main()
