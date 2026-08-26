#!/usr/bin/env python3
"""Token statistics for FAISS metadata chunks using BAAI/bge-reranker-v2-m3 tokenizer."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

Tokenizer = None


def _get_tokenizer(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global Tokenizer
    if Tokenizer is not None:
        return Tokenizer
    try:
        from transformers import AutoTokenizer

        Tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loaded tokenizer: {model_name}", file=sys.stderr)
        return Tokenizer
    except Exception as exc:
        print(f"Warning: failed to load tokenizer ({exc}), using word-count heuristic", file=sys.stderr)
        Tokenizer = None
        return None


def count_tokens(text: str, model_name: str = "BAAI/bge-reranker-v2-m3") -> int:
    tok = _get_tokenizer(model_name)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return max(1, int(len(text.split()) * 1.3))


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_stats(chunks: list[dict], model_name: str) -> list[dict]:
    rows = []
    for c in chunks:
        text = c.get("text", "")
        rows.append({
            "chunk_id": c["chunk_id"],
            "source_file": c.get("source_file", ""),
            "page_number": c.get("page_number", 0),
            "char_count": len(text),
            "token_count": count_tokens(text, model_name),
        })
    return rows


def group_by_file(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_file"]].append(r)
    return dict(sorted(groups.items()))


def compute_aggregate(rows: list[dict]) -> dict:
    tokens = [r["token_count"] for r in rows]
    chars = [r["char_count"] for r in rows]
    return {
        "total_chunks": len(rows),
        "token": {
            "min": min(tokens),
            "max": max(tokens),
            "mean": round(statistics.mean(tokens), 1),
            "median": round(statistics.median(tokens), 1),
            "stdev": round(statistics.stdev(tokens), 1) if len(tokens) > 1 else 0,
            "p25": round(sorted(tokens)[len(tokens) // 4], 1) if tokens else 0,
            "p75": round(sorted(tokens)[len(tokens) * 3 // 4], 1) if tokens else 0,
            "p90": round(sorted(tokens)[int(len(tokens) * 0.90)], 1) if tokens else 0,
            "p95": round(sorted(tokens)[int(len(tokens) * 0.95)], 1) if tokens else 0,
            "total": sum(tokens),
        },
        "char": {
            "min": min(chars),
            "max": max(chars),
            "mean": round(statistics.mean(chars), 1),
            "median": round(statistics.median(chars), 1),
            "total": sum(chars),
        },
    }


def print_table(rows: list[dict], grouped: dict[str, list[dict]], agg: dict):
    print(f"\n{'='*70}")
    print(f"CHUNK TOKEN STATISTICS  (tokenizer: BAAI/bge-reranker-v2-m3)")
    print(f"{'='*70}")
    print(f"Total chunks: {agg['total_chunks']}")
    print(f"\n--- Token counts ---")
    print(f"  Min:    {agg['token']['min']}")
    print(f"  Max:    {agg['token']['max']}")
    print(f"  Mean:   {agg['token']['mean']}")
    print(f"  Median: {agg['token']['median']}")
    print(f"  Stdev:  {agg['token']['stdev']}")
    print(f"  P25:    {agg['token']['p25']}")
    print(f"  P75:    {agg['token']['p75']}")
    print(f"  P90:    {agg['token']['p90']}")
    print(f"  P95:    {agg['token']['p95']}")
    print(f"  Total:  {agg['token']['total']}")
    print(f"\n--- Char counts ---")
    print(f"  Min:    {agg['char']['min']}")
    print(f"  Max:    {agg['char']['max']}")
    print(f"  Mean:   {agg['char']['mean']}")
    print(f"  Median: {agg['char']['median']}")
    print(f"  Total:  {agg['char']['total']}")

    print(f"\n--- By source_file ---")
    print(f"{'source_file':<50} {'chunks':>6} {'tok_min':>7} {'tok_max':>7} {'tok_avg':>8} {'tok_total':>9}")
    print("-" * 90)
    for src, grp in grouped.items():
        toks = [r["token_count"] for r in grp]
        print(f"{src:<50} {len(grp):>6} {min(toks):>7} {max(toks):>7} {statistics.mean(toks):>8.1f} {sum(toks):>9}")


def save_csv(rows: list[dict], grouped: dict[str, list[dict]], agg: dict, out_dir: Path, timestamp: str):
    # Per-chunk CSV
    chunks_csv = out_dir / f"chunk_token_stats_{timestamp}.csv"
    with chunks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "source_file", "page_number", "char_count", "token_count"])
        writer.writeheader()
        writer.writerows(rows)

    # Per-file summary CSV
    files_csv = out_dir / f"chunk_token_stats_by_file_{timestamp}.csv"
    with files_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "chunks", "tok_min", "tok_max", "tok_avg", "tok_median", "tok_total", "char_avg", "char_total"])
        writer.writeheader()
        for src, grp in grouped.items():
            toks = [r["token_count"] for r in grp]
            chars = [r["char_count"] for r in grp]
            writer.writerow({
                "source_file": src,
                "chunks": len(grp),
                "tok_min": min(toks),
                "tok_max": max(toks),
                "tok_avg": round(statistics.mean(toks), 1),
                "tok_median": round(statistics.median(toks), 1),
                "tok_total": sum(toks),
                "char_avg": round(statistics.mean(chars), 1),
                "char_total": sum(chars),
            })

    # Aggregate summary JSON
    summary_json = out_dir / f"chunk_token_stats_summary_{timestamp}.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    return chunks_csv, files_csv, summary_json


def main():
    parser = argparse.ArgumentParser(description="Token statistics for FAISS metadata chunks")
    parser.add_argument("--metadata-path", default="storage_hierarchical/metadata.json")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    meta_path = Path(args.metadata_path)
    if not meta_path.exists():
        print(f"Error: {meta_path} not found", file=sys.stderr)
        sys.exit(1)

    chunks = load_metadata(meta_path)
    print(f"Loaded {len(chunks)} chunks from {meta_path}", file=sys.stderr)

    rows = compute_stats(chunks, args.model)
    grouped = group_by_file(rows)
    agg = compute_aggregate(rows)

    print_table(rows, grouped, agg)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_csv, files_csv, summary_json = save_csv(rows, grouped, agg, out_dir, ts)
    print(f"\nSaved: {chunks_csv}", file=sys.stderr)
    print(f"Saved: {files_csv}", file=sys.stderr)
    print(f"Saved: {summary_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
