from __future__ import annotations

import faiss
import numpy as np

from .pdf_rag import ChunkRecord


def retrieve_results(
    index: faiss.Index,
    records: list[ChunkRecord],
    query_vector: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, object]]:
    limit = min(top_k, len(records))
    scores, ids = index.search(query_vector, limit)

    results: list[dict[str, object]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        record = records[idx]
        results.append(
            {
                "score": float(score),
                "source_file": record.source_file,
                "page_number": record.page_number,
                "chunk_id": record.chunk_id,
                "text": record.text,
            }
        )

    return results
