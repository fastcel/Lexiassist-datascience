"""Pretrained LexiAssist indexer: MiniLM + FAISS + BM25.

Based closely on TokenlessLabs/LexiAssist branch main, Indexer.py.
Phase 2 changes: read Phase 1 CSV, locate shared chunker, and save indexes
in a separate output folder so the two variants cannot overwrite each other.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2" / "shared"))
from chunker import chunk_document, Chunk  
from corpus import DEFAULT_DATASET, load_phase1_corpus  

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 250          
OVERLAP_SENTS = 2


def load_corpus(limit=None, dataset_path=DEFAULT_DATASET):
    return load_phase1_corpus(dataset_path, limit=limit)


def build_chunks(docs):
    print("Chunking...")
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


def embed(chunks):
    print("Embedding using", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c.text for c in chunks]
    emb = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.array(emb, dtype="float32")


def build_faiss(emb):
    print("Building FAISS...")
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print("Vectors:", index.ntotal)
    return index


def build_bm25(chunks):
    print("Building BM25...")
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def save(index, chunks, bm25, output_dir, source_csv, num_docs):
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "legal.faiss"))
    with (output_dir / "chunks.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    with (output_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    sha256 = hashlib.sha256(Path(source_csv).read_bytes()).hexdigest()
    manifest = {
        "variant": "pretrained",
        "source_filename": Path(source_csv).name,
        "source_sha256": sha256,
        "documents_indexed": num_docs,
        "chunks_indexed": len(chunks),
        "chunk_size_words": CHUNK_SIZE,
        "overlap_sentences": OVERLAP_SENTS,
        "embedding_model": EMBED_MODEL,
        "index": "faiss.IndexFlatIP",
        "retrieval": ["MiniLM/FAISS", "BM25"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved pretrained artifacts to:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Index the Phase 1 legal corpus (pretrained)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET,
                        help="Path to full Phase 1 preprocessed CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional small number of documents for testing (e.g. 10)")
    args = parser.parse_args()

    docs = load_corpus(limit=args.limit, dataset_path=args.data)
    chunks = build_chunks(docs)
    emb = embed(chunks)
    index = build_faiss(emb)
    bm25 = build_bm25(chunks)

    folder_name = "pretrained" if args.limit is None else f"pretrained_test_{args.limit}"
    output_dir = ROOT / "phase2" / "artifacts" / folder_name
    save(index, chunks, bm25, output_dir, args.data, len(docs))
    print("DONE: pretrained indexing finished.")


if __name__ == "__main__":
    main()
