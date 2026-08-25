from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    HuggingFaceInferenceEmbeddingProvider,
)
from .index_store import build_faiss_index, save_index
from .pdf_rag import ChunkRecord


HEADING3_RE = re.compile(r"^###\s+(.+)$")


def load_index_records(chunks_dir: Path) -> list[dict]:
    index_jsonl = chunks_dir / "index.jsonl"
    if not index_jsonl.exists():
        raise FileNotFoundError(f"index.jsonl not found in {chunks_dir}")
    records = []
    with open(index_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_section_at_h3(section: dict, content: str) -> list[dict]:
    """Split a level-2 section into level-3 sub-chunks at ### headings."""
    lines = content.split("\n")
    parent_heading = section.get("heading")
    parent_chunk_file = section.get("chunk_file")

    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = HEADING3_RE.match(line)
        if m:
            starts.append((i, m.group(1).strip()))

    if not starts:
        return [{"content": content, "heading": parent_heading, "heading_level": section.get("heading_level", 2)}]

    children = []
    for idx, (start_i, h3) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body_lines = lines[start_i + 1:end_i]
        child_content = "\n".join(body_lines).strip()
        children.append({
            "content": child_content,
            "heading": h3,
            "heading_level": 3,
            "parent_heading": parent_heading,
            "parent_chunk_file": parent_chunk_file,
        })

    return children


def load_chunks_with_h3_split(chunks_dir: Path) -> list[dict]:
    """Load chunks from directory, split sections at ### headings."""
    records = load_index_records(chunks_dir)
    all_chunks = []

    for rec in records:
        chunk_file = rec.get("chunk_file")
        if not chunk_file:
            continue
        file_path = chunks_dir / chunk_file
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        if rec.get("chunk_type") == "section":
            sub_chunks = split_section_at_h3(rec, content)
        else:
            sub_chunks = [{"content": content, "heading": rec.get("heading"), "heading_level": rec.get("heading_level")}]

        for chunk in sub_chunks:
            heading = chunk.get("heading") or ""
            heading_level = chunk.get("heading_level")
            # Prepend heading to content for embedding context
            if heading_level and heading:
                prefix = "#" * heading_level
                chunk_text = f"{prefix} {heading}\n\n{chunk['content']}"
            else:
                chunk_text = chunk["content"]

            all_chunks.append({
                "text": chunk_text,
                "source_file": chunk.get("parent_chunk_file") or chunk_file,
                "page_number": (rec.get("pages") or [0])[0],
                "metadata": rec,
            })

    return all_chunks


def command_index_chunks(args: argparse.Namespace) -> None:
    chunks_dir = Path(args.chunks_dir).expanduser().resolve()
    if not chunks_dir.is_dir():
        raise NotADirectoryError(f"Chunks directory not found: {chunks_dir}")

    output_dir = Path(args.index_dir).expanduser().resolve()

    print(f"Loading chunks from {chunks_dir} ...")
    chunks = load_chunks_with_h3_split(chunks_dir)
    if not chunks:
        raise ValueError("No chunks found in the chunks directory")
    print(f"Loaded {len(chunks)} chunks (after ### split)")

    texts = [c["text"] for c in chunks]
    records = [
        ChunkRecord(
            chunk_id=i,
            source_file=c["source_file"],
            page_number=c["page_number"],
            text=c["text"],
        )
        for i, c in enumerate(chunks)
    ]

    print(f"Embedding {len(texts)} chunks with model={args.model_name}, provider={args.embedding_provider or 'default'} ...")
    embedding_provider = HuggingFaceInferenceEmbeddingProvider(
        model_name=args.model_name,
        provider=args.embedding_provider,
    )
    vectors = embedding_provider.embed_texts(texts)
    print(f"Embedding vectors shape: {vectors.shape}")

    index = build_faiss_index(vectors)
    save_index(output_dir, index, records)
    print(f"Saved FAISS index to {output_dir.resolve()}")
    print(f"  faiss.index: {output_dir / 'faiss.index'}")
    print(f"  metadata.json: {output_dir / 'metadata.json'}")
