"""Lightweight LexiAssist indexer: TF-IDF + BM25.

Based closely on TokenlessLabs/LexiAssist branch models, Indexer.py.
Phase 2 changes: read Phase 1 CSV, locate shared chunker, and save indexes
in a separate output folder so the two variants cannot overwrite each other.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2" / "shared"))
from chunker import chunk_document, Chunk  # noqa: E402 (original chunker)
from corpus import DEFAULT_DATASET, load_phase1_corpus  # noqa: E402

# Original implementation settings.
CHUNK_SIZE = 250        # Original code counts words, not tokenizer tokens.
OVERLAP_SENTS = 2


def load_corpus(limit=None, dataset_path=DEFAULT_DATASET):
    return load_phase1_corpus(dataset_path, limit=limit)


def build_chunks(docs):
    print("Chunking documents...")
    chunks = []
    for i, doc in enumerate(docs):
        text = doc.get("text", "").strip()
        if not text:
            continue
        chunks.extend(
            chunk_document(
                text=text,
                doc_id=doc.get("id", str(i)),
                citation=doc.get("citation", ""),
                url=doc.get("url", ""),
                jurisdiction=doc.get("jurisdiction", ""),
                chunk_size=CHUNK_SIZE,
                overlap_sentences=OVERLAP_SENTS,
            )
        )
    print("Total chunks:", len(chunks))
    if not chunks:
        raise ValueError("No chunks created. Check the input dataset.")
    return chunks


def build_tfidf(chunks):
    print("Building TF-IDF vectorizer from corpus...")
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def build_bm25(chunks):
    print("Building BM25 index...")
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def save_artifacts(chunks, vectorizer, tfidf_matrix, bm25,
                   output_dir, source_csv, num_docs):
    print("Saving artifacts to disk...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    with (output_dir / "tfidf_vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    save_npz(output_dir / "tfidf_matrix.npz", tfidf_matrix)
    with (output_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    sha256 = hashlib.sha256(Path(source_csv).read_bytes()).hexdigest()
    manifest = {
        "variant": "lightweight",
        "source_filename": Path(source_csv).name,
        "source_sha256": sha256,
        "documents_indexed": num_docs,
        "chunks_indexed": len(chunks),
        "chunk_size_words": CHUNK_SIZE,
        "overlap_sentences": OVERLAP_SENTS,
        "tfidf_ngram_range": [1, 2],
        "tfidf_max_df": 0.85,
        "tfidf_min_df": 2,
        "retrieval": ["TF-IDF", "BM25"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved lightweight artifacts to:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Index the Phase 1 legal corpus (lightweight)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET,
                        help="Path to full Phase 1 preprocessed CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional small number of documents for testing (e.g. 10)")
    args = parser.parse_args()

    docs = load_corpus(limit=args.limit, dataset_path=args.data)
    chunks = build_chunks(docs)
    vectorizer, tfidf_matrix = build_tfidf(chunks)
    bm25 = build_bm25(chunks)

    folder_name = "lightweight" if args.limit is None else f"lightweight_test_{args.limit}"
    output_dir = ROOT / "phase2" / "artifacts" / folder_name
    save_artifacts(chunks, vectorizer, tfidf_matrix, bm25,
                   output_dir, args.data, len(docs))
    print("DONE: lightweight indexing finished.")


if __name__ == "__main__":
    main()
