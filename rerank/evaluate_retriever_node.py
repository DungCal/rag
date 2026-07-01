from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.agent_pipeline.rerank.rerank_node import (
    DEFAULT_RERANK_INPUT_TOP_K,
    DEFAULT_RERANK_OUTPUT_TOP_K,
    RerankNode,
)
from pipelines.agent_pipeline.retriever.retriever_node import RetrieverNode
from pipelines.agent_pipeline.routers.routing_classification import PromptRouteDecision
from rerank.hf_reranker import DEFAULT_RERANKER_MODEL_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Pinecone retriever scores with Hugging Face reranker scores")
    parser.add_argument("--query", required=True, help="User query to send into the retriever node")
    parser.add_argument("--pinecone-index-name", required=True, help="Pinecone index name for retrieval")
    parser.add_argument("--pinecone-namespace", default="default", help="Pinecone namespace for retrieval")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3", help="Embedding model for retriever queries")
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_OUTPUT_TOP_K, help="Final number of reranked results to return")
    parser.add_argument("--reranker-model-name", default=DEFAULT_RERANKER_MODEL_NAME, help="Hugging Face reranker model name")
    parser.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranking")
    parser.add_argument("--reranker-max-length", type=int, default=4096, help="Max token length for reranking")
    parser.add_argument("--reranker-instruction", default=None, help="Reserved reranker option kept for CLI compatibility")
    parser.add_argument("--reranker-fp16", action="store_true", help="Reserved reranker option kept for CLI compatibility")
    parser.add_argument("--reranker-sigmoid", action="store_true", help="Convert reranker scores to 0-1 probabilities")
    parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
    parser.add_argument("--as-json", action="store_true", help="Print the comparison payload as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retriever = RetrieverNode(
        index_name=args.pinecone_index_name,
        namespace=args.pinecone_namespace,
        model_name=args.embedding_model_name,
        provider=args.embedding_provider,
        top_k=DEFAULT_RERANK_INPUT_TOP_K,
        use_fp16=args.use_fp16,
    )
    reranker = RerankNode(
        model_name=args.reranker_model_name,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        instruction=args.reranker_instruction,
        use_fp16=args.reranker_fp16,
        apply_sigmoid=args.reranker_sigmoid,
        input_top_k=DEFAULT_RERANK_INPUT_TOP_K,
        output_top_k=args.top_k,
    )
    decision = PromptRouteDecision(
        route="retrieval",
        label="related",
        message=args.query,
        raw_output="related",
        prompt="",
    )
    result = retriever.run(decision, args.query)
    if result is None:
        raise RuntimeError("Retriever node returned no results")
    rerank_result = reranker.run(result.query, result.results)

    payload = {
        "route": result.route,
        "query": result.query,
        "reranking_enabled": True,
        "retriever_results": result.results,
        "rerank_input_results": rerank_result.input_results,
        "rerank_results": rerank_result.results,
        "results": rerank_result.results,
    }

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"route={payload['route']}")
    print(f"query={payload['query']}")
    print(f"reranking_enabled={payload['reranking_enabled']}")
    print("-" * 80)
    print("retriever_results")
    print("-" * 80)
    for item in payload["retriever_results"]:
        print(
            "score={score:.4f} page={page} chunk={chunk}".format(
                score=item.get("score", 0.0),
                page=item.get("page_number"),
                chunk=item.get("chunk_id"),
            )
        )
        print(item.get("text", ""))
        print("-" * 80)
    print("rerank_results")
    print("-" * 80)
    for item in payload["rerank_results"]:
        print(
            "retrieval_rank={retrieval_rank} rerank_rank={rerank_rank} "
            "retrieval_score={retrieval_score:.4f} rerank_score={rerank_score:.4f} "
            "page={page} chunk={chunk}".format(
                retrieval_rank=item.get("retrieval_rank"),
                rerank_rank=item.get("rerank_rank"),
                retrieval_score=item.get("retrieval_score", item.get("score", 0.0)),
                rerank_score=item.get("rerank_score", 0.0),
                page=item.get("page_number"),
                chunk=item.get("chunk_id"),
            )
        )
        print(item.get("text", ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
