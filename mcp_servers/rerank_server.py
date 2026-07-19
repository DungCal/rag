from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rerank.hf_reranker import DEFAULT_RERANKER_MODEL_NAME, HuggingFaceHubReranker

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "mcp is required for the MCP server. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


mcp = FastMCP("rerank")


@mcp.tool()
async def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Rerank a list of retrieved document chunks by semantic relevance to the query.

    Use this tool when you already have candidate chunks from retrieve_context
    and want to improve their ordering before generating an answer.
    This is especially useful when the initial retrieval returns many chunks
    with varying relevance.

    Args:
        query: The original user query.
        results: Candidate chunks returned by retrieve_context.
        top_k: Number of top reranked chunks to return (default: 5).
        model_name: Hugging Face reranker model name (default: BAAI/bge-reranker-v2-m3).

    Returns:
        A list of reranked chunks. Each chunk contains the original metadata plus:
        - rerank_score: cross-encoder relevance score
        - rerank_rank: new rank after reranking
        - retrieval_rank: original rank from retrieve_context
        - retrieval_score: original retrieval score
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not results:
        return []

    reranker = HuggingFaceHubReranker(model_name=model_name)
    reranked = reranker.rerank(query, results)[:top_k]
    return reranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank MCP server")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_RERANKER_MODEL_NAME,
        help="Hugging Face reranker model name",
    )
    parser.parse_args()
    mcp.run()


if __name__ == "__main__":
    main()
