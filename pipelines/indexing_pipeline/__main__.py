from __future__ import annotations

import argparse

from . import indexing, scope
from .index_from_chunks import command_index_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Indexing pipeline commands for PDF ingestion, querying, and scope generation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a PDF into FAISS storage")
    index_parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    index_parser.add_argument("--index-dir", default=str(indexing.DEFAULT_INDEX_DIR), help="Directory for FAISS index and metadata")
    index_parser.add_argument("--model-name", default=indexing.DEFAULT_MODEL_NAME, help="Embedding model name")
    index_parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    index_parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters")
    index_parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    index_parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
    index_parser.set_defaults(func=indexing.command_index)

    index_chunks_parser = subparsers.add_parser("index-chunks", help="Index pre-chunked markdown files into FAISS")
    index_chunks_parser.add_argument("--chunks-dir", required=True, help="Directory containing index.jsonl and chunks/")
    index_chunks_parser.add_argument("--index-dir", default="storage_hierarchical", help="Output directory for FAISS index")
    index_chunks_parser.add_argument("--model-name", default=indexing.DEFAULT_MODEL_NAME, help="Embedding model name")
    index_chunks_parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    index_chunks_parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
    index_chunks_parser.set_defaults(func=command_index_chunks)

    query_parser = subparsers.add_parser("query", help="Query an existing FAISS index")
    query_parser.add_argument("--query", required=True, help="User query text")
    query_parser.add_argument("--index-dir", default=str(indexing.DEFAULT_INDEX_DIR), help="Directory containing FAISS index and metadata")
    query_parser.add_argument("--model-name", default=indexing.DEFAULT_MODEL_NAME, help="Embedding model name")
    query_parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    query_parser.add_argument("--generate-answer", action="store_true", help="Generate an answer from retrieved chunks")
    query_parser.add_argument("--llm-model", default=indexing.DEFAULT_LLM_MODEL, help="Hugging Face model used for answer generation")
    query_parser.add_argument("--off-topic-threshold", type=float, default=0.35, help="Minimum similarity score required to treat a query as document-relevant")
    query_parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
    query_parser.add_argument("--as-json", action="store_true", help="Print results as JSON")
    query_parser.set_defaults(func=indexing.command_query)

    scope_parser = subparsers.add_parser("scope", help="Generate a scope summary from indexed metadata")
    scope_parser.add_argument("--metadata-path", default=str(scope.DEFAULT_METADATA_PATH), help="Path to metadata.json")
    scope_parser.add_argument("--prompt-path", default=str(scope.DEFAULT_PROMPT_PATH), help="Path to prompt template")
    scope_parser.add_argument("--results-dir", default=str(scope.DEFAULT_RESULTS_DIR), help="Directory for generated output")
    scope_parser.add_argument("--output-prompt-path", default=str(scope.DEFAULT_OUTPUT_PROMPT_PATH), help="Path for the rendered prompt snapshot")
    scope_parser.add_argument("--sample-size", type=int, default=10, help="Number of random chunks to sample")
    scope_parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    scope_parser.add_argument("--model-name", default=scope.DEFAULT_LLM_MODEL, help="Hugging Face model used for generation")
    scope_parser.set_defaults(func=scope.main_from_args)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
