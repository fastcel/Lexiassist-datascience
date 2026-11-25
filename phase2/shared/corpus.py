"""Load the Phase 1 preprocessed corpus for both LexiAssist implementations.

This adapter preserves the original indexers' expected document dictionary keys:
  id, text, citation, url, jurisdiction

The real Phase 1 fields are:
  version_id, retrieval_text, citation, url, jurisdiction
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "australian_legal_corpus_preprocessed.csv"


def load_phase1_corpus(csv_path=DEFAULT_DATASET, limit=None):
    """Read Phase 1's FULL CSV (not the Excel preview).

    Args:
        csv_path: Full preprocessed CSV released with Phase 1.
        limit: Optional number of documents for a small smoke test.

    Returns:
        List[dict]: Original LexiAssist-compatible document dictionaries.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download australian_legal_corpus_preprocessed.csv from the Phase 1 "
            "GitHub Release (v1.0-phase1), and put it in the repository's data/ folder."
        )

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    print(f"Reading Phase 1 dataset: {csv_path.name}")
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", low_memory=False)

    expected = {"version_id", "retrieval_text", "citation", "url", "jurisdiction"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing Phase 1 CSV columns: {sorted(missing)}")

    if df["version_id"].isna().any() or df["version_id"].duplicated().any():
        raise ValueError("version_id values must be present and unique")
    if df["retrieval_text"].isna().any() or df["retrieval_text"].str.strip().eq("").any():
        raise ValueError("Every document must have nonempty retrieval_text")

    if limit is not None:
        df = df.head(limit)

    docs = []
    for row in df.to_dict(orient="records"):
        docs.append({
            "id": row["version_id"],
            "text": row["retrieval_text"],
            "citation": row["citation"] if pd.notna(row["citation"]) else "",
            "url": row["url"] if pd.notna(row["url"]) else "",
            "jurisdiction": row["jurisdiction"] if pd.notna(row["jurisdiction"]) else "",
        })

    print(f"Loaded {len(docs):,} Phase 1 legal documents.")
    return docs
