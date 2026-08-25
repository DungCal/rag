from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.indexing_pipeline.index_store import load_index
from pipelines.indexing_pipeline.pdf_rag import DEFAULT_MODEL_NAME, PDFRAG
from pipelines.indexing_pipeline.retriever import retrieve_results

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "mcp is required for the MCP server. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


mcp = FastMCP("retrieve_context")


@mcp.tool()
async def retrieve_context(
    query: str,
    top_k: int = 5,
    index_dir: str = "storage_hierarchical",
) -> list[dict[str, Any]]:
    """Retrieve the most relevant document chunks from the local FAISS vector index.

    Use this tool when the user's question is about the indexed document
    (the TYM tractor operator manual) and can be answered from its contents.
    Do not use this tool for greetings, off-topic questions, or questions that
    require external or up-to-date information.

    Args:
        query: The search query or user question.
        top_k: Number of most relevant chunks to return (default: 5).
        index_dir: Directory containing faiss.index and metadata.json (default: "storage_hierarchical").

    Returns:
        A list of retrieved chunks. Each chunk contains:
        - text: the chunk text
        - score: cosine-style similarity score
        - page_number: PDF page number
        - chunk_id: chunk identifier
        - source_file: source PDF path
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    index_path = Path(index_dir)
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path

    index, records = load_index(index_path)
    rag = PDFRAG(model_name=DEFAULT_MODEL_NAME)
    query_vector = rag.embed_query(query)
    results = retrieve_results(index, records, query_vector, top_k=top_k)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve context MCP server")
    parser.add_argument(
        "--index-dir",
        default="storage",
        help="Default directory containing faiss.index and metadata.json",
    )
    args = parser.parse_args()
    mcp.run()


if __name__ == "__main__":
    main()
