from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np


ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_INDEX_DIR = Path("storage")
DEFAULT_NAMESPACE = "default"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"


def _load_env_value(name: str, env_path: Path = ENV_FILE_PATH) -> str | None:
    if value := os.getenv(name):
        return value
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            resolved = value.strip().strip('"').strip("'")
            return resolved or None

    return None


def load_faiss_vectors_and_metadata(index_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing FAISS index or metadata in {index_dir.resolve()}")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError("metadata.json must contain a JSON array")
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Vector count ({index.ntotal}) does not match metadata count ({len(metadata)}) in {index_dir.resolve()}"
        )

    vectors = np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32")
    return vectors, metadata


def ensure_index(
    *,
    index_name: str,
    dimension: int,
    cloud: str,
    region: str,
    api_key: str,
) -> Any:
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise ImportError(
            "pinecone is required for FAISS-to-Pinecone sync. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    client = Pinecone(api_key=api_key)
    if not client.has_index(index_name):
        client.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
            deletion_protection="disabled",
        )

    for _ in range(60):
        description = client.describe_index(index_name)
        status = getattr(description, "status", None)
        ready = None
        if isinstance(status, dict):
            ready = status.get("ready")
        elif status is not None:
            ready = getattr(status, "ready", None)
        if ready is True:
            break
        time.sleep(2)

    return client.Index(name=index_name)


def build_vectors_payload(vectors: np.ndarray, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for idx, (vector, record) in enumerate(zip(vectors, metadata, strict=False)):
        chunk_id = record.get("chunk_id", idx)
        vector_id = str(chunk_id)
        normalized_metadata = {
            "chunk_id": chunk_id,
            "source_file": record.get("source_file"),
            "page_number": record.get("page_number"),
            "text": record.get("text", ""),
        }
        payload.append(
            {
                "id": vector_id,
                "values": vector.tolist(),
                "metadata": normalized_metadata,
            }
        )
    return payload


def upsert_batches(index: Any, vectors: list[dict[str, Any]], *, namespace: str, batch_size: int) -> int:
    total = 0
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total += len(batch)
    return total


def sync_faiss_to_pinecone(
    *,
    index_dir: Path,
    index_name: str,
    namespace: str,
    cloud: str,
    region: str,
    batch_size: int,
    api_key: str | None = None,
) -> int:
    resolved_api_key = api_key or _load_env_value("PINECONE_API_KEY")
    if not resolved_api_key:
        raise ValueError("Missing PINECONE_API_KEY in the environment or .env file")

    vectors, metadata = load_faiss_vectors_and_metadata(index_dir)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No vectors were reconstructed from the FAISS index")

    index = ensure_index(
        index_name=index_name,
        dimension=vectors.shape[1],
        cloud=cloud,
        region=region,
        api_key=resolved_api_key,
    )
    payload = build_vectors_payload(vectors, metadata)
    return upsert_batches(index, payload, namespace=namespace, batch_size=batch_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync local FAISS vectors and metadata into Pinecone")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory containing faiss.index and metadata.json")
    parser.add_argument("--pinecone-index-name", required=True, help="Pinecone index name")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Pinecone namespace")
    parser.add_argument("--cloud", default=DEFAULT_CLOUD, help="Pinecone serverless cloud")
    parser.add_argument("--region", default=DEFAULT_REGION, help="Pinecone serverless region")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of vectors per upsert batch")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inserted = sync_faiss_to_pinecone(
        index_dir=Path(args.index_dir),
        index_name=args.pinecone_index_name,
        namespace=args.namespace,
        cloud=args.cloud,
        region=args.region,
        batch_size=args.batch_size,
    )
    print(f"Upserted {inserted} vectors into Pinecone index '{args.pinecone_index_name}' namespace '{args.namespace}'")


if __name__ == "__main__":
    main()
