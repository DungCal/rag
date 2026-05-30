from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .pdf_rag import ChunkRecord


def save_index(index_dir: Path, index: faiss.Index, records: list[ChunkRecord]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    payload = [asdict(record) for record in records]
    (index_dir / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_dir: Path) -> tuple[faiss.Index, list[ChunkRecord]]:
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing FAISS index or metadata in {index_dir}")

    index = faiss.read_index(str(index_path))
    raw_records = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = [ChunkRecord(**item) for item in raw_records]
    return index, records


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No vectors were created from the input PDF")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index
