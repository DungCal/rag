from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "agent_pipeline.log"
logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)

from pipelines.indexing_pipeline.indexing import (
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL as DEFAULT_INDEXING_LLM_MODEL,
    DEFAULT_MODEL_NAME as DEFAULT_INDEXING_MODEL_NAME,
    command_index,
    command_query,
)
from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL
from pipelines.indexing_pipeline.scope import (
    DEFAULT_METADATA_PATH,
    DEFAULT_OUTPUT_PROMPT_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_LLM_MODEL as DEFAULT_SCOPE_LLM_MODEL,
    main_from_args as run_scope_generation,
)
from pipelines.agent_pipeline.routers.routing_classification import (
    DEFAULT_DOCUMENT_SCOPE,
    PromptQueryRouter,
)
from pipelines.agent_pipeline.routers.routing_response import (
    GreetingNode,
    OffTopicNode,
    load_default_scope,
)
from pipelines.agent_pipeline.rerank import (
    DEFAULT_RERANK_INPUT_TOP_K,
    DEFAULT_RERANK_OUTPUT_TOP_K,
    DEFAULT_RERANKER_MODEL_NAME,
)


DEFAULT_SCOPE_FILE = PROJECT_ROOT / "results" / "scope_result_20260606_193507.txt"
DEFAULT_PINECONE_NAMESPACE = "default"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def _load_scope(scope: str | None, scope_file: str | None) -> str:
    if scope and scope.strip():
        return scope.strip()
    if scope_file:
        scope_path = Path(scope_file)
        if not scope_path.exists():
            raise FileNotFoundError(f"Scope file not found: {scope_path.resolve()}")
        loaded_scope = scope_path.read_text(encoding="utf-8").strip()
        if not loaded_scope:
            raise ValueError(f"Scope file is empty: {scope_path.resolve()}")
        return loaded_scope
    if DEFAULT_SCOPE_FILE.exists():
        return load_default_scope(DEFAULT_SCOPE_FILE)
    return DEFAULT_DOCUMENT_SCOPE


class RoutedRAGPipeline:
    def __init__(
        self,
        *,
        prompt_path: str,
        scope: str,
        model_name: str,
        embedding_model_name: str,
        pinecone_index_name: str | None,
        pinecone_namespace: str,
        top_k: int,
        use_fp16: bool,
        enable_rerank: bool,
        reranker_model_name: str,
        reranker_batch_size: int,
        reranker_max_length: int,
        reranker_instruction: str | None,
        reranker_use_fp16: bool,
        reranker_apply_sigmoid: bool,
    ) -> None:
        try:
            from langchain_core.runnables import RunnableBranch, RunnableLambda
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required for the LangChain pipeline orchestration. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.prompt_path = prompt_path
        self.scope = scope
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_namespace = pinecone_namespace
        self.top_k = top_k
        self.use_fp16 = use_fp16
        self.enable_rerank = enable_rerank
        self.reranker_model_name = reranker_model_name
        self.reranker_batch_size = reranker_batch_size
        self.reranker_max_length = reranker_max_length
        self.reranker_instruction = reranker_instruction
        self.reranker_use_fp16 = reranker_use_fp16
        self.reranker_apply_sigmoid = reranker_apply_sigmoid

        logger.info(
            "Initializing RoutedRAGPipeline: llm_model={}, embedding_model={}, reranker_model={}, "
            "pinecone_index_name={}, pinecone_namespace={}, top_k={}, enable_rerank={}",
            self.model_name,
            self.embedding_model_name,
            self.reranker_model_name,
            self.pinecone_index_name,
            self.pinecone_namespace,
            self.top_k,
            self.enable_rerank,
        )

        self._router = PromptQueryRouter(
            prompt_path=self.prompt_path,
            scope=self.scope,
            model_name=self.model_name,
        )
        self._greeting_node = GreetingNode(scope=self.scope, model_name=self.model_name)
        self._off_topic_node = OffTopicNode(scope=self.scope, model_name=self.model_name)
        self._retriever = None
        self._reranker = None

        self._chain = (
            RunnableLambda(self._build_state)
            | RunnableBranch(
                (lambda state: state["decision"].route == "greeting", RunnableLambda(self._run_greeting)),
                (lambda state: state["decision"].route == "off_topic", RunnableLambda(self._run_off_topic)),
                RunnableLambda(self._run_retrieval),
            )
            | RunnableLambda(self._build_payload)
        )

    def invoke(self, query: str, *, include_prompt: bool = False) -> dict[str, Any]:
        return self._chain.invoke({
            "query": query,
            "include_prompt": include_prompt,
        })

    def _build_state(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = str(inputs["query"]).strip()
        if not query:
            raise ValueError("User query must not be empty")

        decision = self._router.route(query)
        logger.info(
            "Router decision: query={!r}, route={}, label={}, message={!r}, raw_output={!r}",
            query,
            decision.route,
            decision.label,
            decision.message,
            decision.raw_output,
        )
        return {
            "query": query,
            "include_prompt": bool(inputs.get("include_prompt", False)),
            "decision": decision,
            "node_result": None,
            "retriever_result": None,
            "rerank_result": None,
        }

    def _run_greeting(self, state: dict[str, Any]) -> dict[str, Any]:
        state["node_result"] = self._greeting_node.run(state["query"])
        return state

    def _run_off_topic(self, state: dict[str, Any]) -> dict[str, Any]:
        state["node_result"] = self._off_topic_node.run(state["query"])
        return state

    def _run_retrieval(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.pinecone_index_name:
            raise ValueError("--pinecone-index-name is required when the route is retrieval")

        if self._retriever is None:
            from pipelines.agent_pipeline.retriever.retriever_node import RetrieverNode

            self._retriever = RetrieverNode(
                index_name=self.pinecone_index_name,
                namespace=self.pinecone_namespace,
                model_name=self.embedding_model_name,
                top_k=DEFAULT_RERANK_INPUT_TOP_K if self.enable_rerank else self.top_k,
                use_fp16=self.use_fp16,
            )

        state["retriever_result"] = self._retriever.run(state["decision"], state["query"])
        if state["retriever_result"] is not None:
            logger.info(
                "Retriever returned {} result(s) for query={!r}",
                len(state["retriever_result"].results),
                state["retriever_result"].query,
            )
            for idx, item in enumerate(state["retriever_result"].results, start=1):
                logger.info(
                    "Retriever result #{}: score={:.4f} page={} chunk_id={} text_preview={!r}",
                    idx,
                    item.get("score", 0.0),
                    item.get("page_number"),
                    item.get("chunk_id"),
                    (item.get("text", "")[:120] + "...") if len(item.get("text", "")) > 120 else item.get("text", ""),
                )

        if self.enable_rerank and state["retriever_result"] is not None:
            if self._reranker is None:
                from pipelines.agent_pipeline.rerank.rerank_node import RerankNode

                self._reranker = RerankNode(
                    model_name=self.reranker_model_name,
                    batch_size=self.reranker_batch_size,
                    max_length=self.reranker_max_length,
                    instruction=self.reranker_instruction,
                    use_fp16=self.reranker_use_fp16,
                    apply_sigmoid=self.reranker_apply_sigmoid,
                    input_top_k=DEFAULT_RERANK_INPUT_TOP_K,
                    output_top_k=min(self.top_k, DEFAULT_RERANK_OUTPUT_TOP_K),
                )

            state["rerank_result"] = self._reranker.run(
                state["retriever_result"].query,
                state["retriever_result"].results,
            )
            if state["rerank_result"] is not None:
                logger.info(
                    "Reranker returned {} result(s)",
                    len(state["rerank_result"].results),
                )
                for idx, item in enumerate(state["rerank_result"].results, start=1):
                    logger.info(
                        "Reranker result #{}: rerank_score={:.4f} retrieval_score={:.4f} "
                        "retrieval_rank={} rerank_rank={} page={} chunk_id={} text_preview={!r}",
                        idx,
                        item.get("rerank_score", 0.0),
                        item.get("retrieval_score", item.get("score", 0.0)),
                        item.get("retrieval_rank"),
                        item.get("rerank_rank"),
                        item.get("page_number"),
                        item.get("chunk_id"),
                        (item.get("text", "")[:120] + "...") if len(item.get("text", "")) > 120 else item.get("text", ""),
                    )
        return state

    @staticmethod
    def _build_payload(state: dict[str, Any]) -> dict[str, Any]:
        decision = state["decision"]
        node_result = state["node_result"]
        retriever_result = state["retriever_result"]
        rerank_result = state["rerank_result"]

        payload: dict[str, Any] = {
            "route": decision.route,
            "label": decision.label,
            "message": decision.message,
            "raw_output": decision.raw_output,
        }
        if node_result is not None:
            payload["response"] = node_result.response
            payload["node_raw_output"] = node_result.raw_output
        if retriever_result is not None:
            payload["retrieval_query"] = retriever_result.query
            payload["retriever_results"] = retriever_result.results
            payload["results"] = retriever_result.results
        if rerank_result is not None:
            payload["reranking_enabled"] = True
            payload["rerank_input_results"] = rerank_result.input_results
            payload["rerank_results"] = rerank_result.results
            payload["results"] = rerank_result.results
        if state["include_prompt"]:
            payload["prompt"] = decision.prompt
            if node_result is not None:
                payload["node_prompt"] = node_result.prompt
        return payload


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", required=True, help="User query to classify")
    parser.add_argument(
        "--prompt-path",
        default=str(PROJECT_ROOT / "prompts" / "route_node_prompt.txt"),
        help="Path to the router prompt template",
    )
    parser.add_argument("--scope", default=None, help="Document scope text to inject into the router prompt")
    parser.add_argument("--scope-file", default=None, help="Path to a text file containing the document scope")
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL, help="Hugging Face model used for routing")
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="Embedding model used for Pinecone retrieval",
    )
    parser.add_argument("--pinecone-index-name", required=False, help="Pinecone index name for retrieval route")
    parser.add_argument(
        "--pinecone-namespace",
        default=DEFAULT_PINECONE_NAMESPACE,
        help="Pinecone namespace for retrieval route",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Final number of results to return")
    parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for the embedding model when supported")
    parser.add_argument("--enable-rerank", action="store_true", help="Rerank Pinecone matches with Hugging Face Hub")
    parser.add_argument(
        "--reranker-model-name",
        default=DEFAULT_RERANKER_MODEL_NAME,
        help="Hugging Face reranker model used after Pinecone retrieval",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranking")
    parser.add_argument("--reranker-max-length", type=int, default=4096, help="Max sequence length for reranking")
    parser.add_argument("--reranker-instruction", default=None, help="Reserved reranker option kept for CLI compatibility")
    parser.add_argument("--reranker-fp16", action="store_true", help="Reserved reranker option kept for CLI compatibility")
    parser.add_argument("--reranker-sigmoid", action="store_true", help="Convert reranker scores to 0-1 probabilities")
    parser.add_argument("--as-json", action="store_true", help="Print the pipeline result as JSON")
    parser.add_argument("--print-prompt", action="store_true", help="Include the rendered prompt in the output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connected agent pipeline CLI with indexing, query, scope, and routed retrieval commands"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the routed agent pipeline with Pinecone retrieval")
    _add_run_arguments(run_parser)
    run_parser.set_defaults(func=run_agent_pipeline)

    index_parser = subparsers.add_parser("index", help="Index a PDF into FAISS storage")
    index_parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    index_parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory for FAISS index and metadata",
    )
    index_parser.add_argument("--model-name", default=DEFAULT_INDEXING_MODEL_NAME, help="Embedding model name")
    index_parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters")
    index_parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    index_parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for faster inference on supported hardware")
    index_parser.set_defaults(func=command_index)

    query_parser = subparsers.add_parser("query", help="Query an existing FAISS index")
    query_parser.add_argument("--query", required=True, help="User query text")
    query_parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory containing FAISS index and metadata",
    )
    query_parser.add_argument("--model-name", default=DEFAULT_INDEXING_MODEL_NAME, help="Embedding model name")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    query_parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Generate an answer from retrieved chunks",
    )
    query_parser.add_argument(
        "--llm-model",
        default=DEFAULT_INDEXING_LLM_MODEL,
        help="Hugging Face model used for answer generation",
    )
    query_parser.add_argument(
        "--off-topic-threshold",
        type=float,
        default=0.35,
        help="Minimum similarity score required to treat a query as document-relevant",
    )
    query_parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for faster inference on supported hardware")
    query_parser.add_argument("--as-json", action="store_true", help="Print results as JSON")
    query_parser.set_defaults(func=command_query)

    scope_parser = subparsers.add_parser("scope", help="Generate a scope summary from indexed metadata")
    scope_parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Path to metadata.json",
    )
    scope_parser.add_argument(
        "--prompt-path",
        default=str(DEFAULT_PROMPT_PATH),
        help="Path to prompt template",
    )
    scope_parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for generated output",
    )
    scope_parser.add_argument(
        "--output-prompt-path",
        default=str(DEFAULT_OUTPUT_PROMPT_PATH),
        help="Path for the rendered prompt snapshot",
    )
    scope_parser.add_argument("--sample-size", type=int, default=10, help="Number of random chunks to sample")
    scope_parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    scope_parser.add_argument(
        "--model-name",
        default=DEFAULT_SCOPE_LLM_MODEL,
        help="Hugging Face model used for generation",
    )
    scope_parser.set_defaults(func=run_scope_generation)

    return parser


def run_agent_pipeline(args: argparse.Namespace) -> None:
    scope = _load_scope(args.scope, args.scope_file)
    pipeline = RoutedRAGPipeline(
        prompt_path=args.prompt_path,
        scope=scope,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        pinecone_index_name=args.pinecone_index_name,
        pinecone_namespace=args.pinecone_namespace,
        top_k=args.top_k,
        use_fp16=args.use_fp16,
        enable_rerank=args.enable_rerank,
        reranker_model_name=args.reranker_model_name,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_instruction=args.reranker_instruction,
        reranker_use_fp16=args.reranker_fp16,
        reranker_apply_sigmoid=args.reranker_sigmoid,
    )
    payload = pipeline.invoke(args.query, include_prompt=args.print_prompt)

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"route={payload['route']}")
    print(f"label={payload['label']}")
    print(f"raw_output={payload['raw_output']}")
    print(payload["message"])
    if "response" in payload:
        print("-" * 80)
        print(payload["response"])
    if "results" in payload:
        print("-" * 80)
        print(f"retrieval_query={payload['retrieval_query']}")
        if payload.get("reranking_enabled"):
            print("retriever_results")
            print("-" * 80)
            for item in payload.get("retriever_results", []):
                print(f"score={item['score']:.4f} page={item.get('page_number')} chunk={item.get('chunk_id')}")
                print(item.get("text", ""))
                print("-" * 80)
            print("rerank_results")
            print("-" * 80)
            for item in payload.get("rerank_results", []):
                print(
                    "retrieval_score={retrieval_score:.4f} rerank_score={rerank_score:.4f} "
                    "retrieval_rank={retrieval_rank} rerank_rank={rerank_rank} "
                    "page={page} chunk={chunk}".format(
                        retrieval_score=item.get("retrieval_score", item.get("score", 0.0)),
                        rerank_score=item.get("rerank_score", 0.0),
                        retrieval_rank=item.get("retrieval_rank"),
                        rerank_rank=item.get("rerank_rank"),
                        page=item.get("page_number"),
                        chunk=item.get("chunk_id"),
                    )
                )
                print(item.get("text", ""))
                print("-" * 80)
        else:
            for item in payload["results"]:
                print(f"score={item['score']:.4f} page={item.get('page_number')} chunk={item.get('chunk_id')}")
                print(item.get("text", ""))
                print("-" * 80)
    if args.print_prompt:
        print("-" * 80)
        print(payload["prompt"])
        if "node_prompt" in payload:
            print("-" * 80)
            print(payload["node_prompt"])


def _normalize_legacy_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["run"]
    if argv[0] in {"run", "index", "query", "scope", "-h", "--help"}:
        return argv
    if argv[0].startswith("-"):
        return ["run", *argv]
    return argv


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_normalize_legacy_argv(sys.argv[1:]))
    args.func(args)


if __name__ == "__main__":
    main()
