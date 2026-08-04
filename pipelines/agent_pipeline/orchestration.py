from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.agent_pipeline.shared.config import (
    ENABLE_JUDGE as DEFAULT_ENABLE_JUDGE,
    ENABLE_GENERATION as DEFAULT_ENABLE_GENERATION,
    JUDGE_TOP_K as DEFAULT_JUDGE_TOP_K,
    JUDGE_MIN_SCORE as DEFAULT_JUDGE_MIN_SCORE,
    ENABLE_CONVERSATION_COMPACTION as DEFAULT_ENABLE_CONVERSATION_COMPACTION,
    MAX_INPUT_TOKENS as DEFAULT_MAX_INPUT_TOKENS,
    CONTEXT_TOKEN_THRESHOLD_PCT as DEFAULT_CONTEXT_TOKEN_THRESHOLD_PCT,
    MIN_KEEP_RECENT_TURNS as DEFAULT_MIN_KEEP_RECENT_TURNS,
    COMPACTION_MAX_SUMMARY_TOKENS as DEFAULT_COMPACTION_MAX_SUMMARY_TOKENS,
)

from pipelines.indexing_pipeline.indexing import (
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL as DEFAULT_INDEXING_LLM_MODEL,
    DEFAULT_MODEL_NAME as DEFAULT_INDEXING_MODEL_NAME,
    command_index,
    command_query,
)
from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL, DEFAULT_HF_INFERENCE_PROVIDER
from pipelines.indexing_pipeline.scope import (
    DEFAULT_METADATA_PATH,
    DEFAULT_OUTPUT_PROMPT_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_LLM_MODEL as DEFAULT_SCOPE_LLM_MODEL,
    main_from_args as run_scope_generation,
)
from pipelines.agent_pipeline.traditional_rag.rerank.rerank_node import (
    DEFAULT_RERANK_INPUT_TOP_K,
    DEFAULT_RERANK_OUTPUT_TOP_K,
    DEFAULT_RERANKER_MODEL_NAME,
)
from pipelines.agent_pipeline.traditional_rag.retriever_judge.retriever_judge_node import (
    DEFAULT_JUDGE_MIN_SCORE,
    DEFAULT_JUDGE_PROMPT_PATH,
    DEFAULT_JUDGE_TOP_K,
)


DEFAULT_PRESIDIO_ANALYZER_URL = "http://localhost:5002/analyze"
DEFAULT_PRESIDIO_ANONYMIZER_URL = "http://localhost:5001/anonymize"
DEFAULT_SAFETY_NSFW_THRESHOLD = 0.95
DEFAULT_SAFETY_TOXIC_THRESHOLD = 0.5
DEFAULT_PINECONE_NAMESPACE = "default"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def is_agentic_rag_enabled(args: argparse.Namespace) -> bool:
    """Resolve whether to run agentic RAG.

    Priority:
      1. CLI flag --turn-on-agent-rag
      2. Environment variable TURN_ON_AGENT_RAG
      3. Default False
    """
    if hasattr(args, "turn_on_agent_rag") and args.turn_on_agent_rag:
        return True
    env_value = os.getenv("TURN_ON_AGENT_RAG", "").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


def run(args: argparse.Namespace) -> None:
    """Dispatch to the traditional or agentic RAG pipeline."""
    if args.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = args.langsmith_project

    if hasattr(args, "enable_langsmith") and args.enable_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    elif not os.environ.get("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    if is_agentic_rag_enabled(args):
        from pipelines.agent_pipeline.agent_rag.agent_graph import run_agentic_pipeline

        asyncio.run(run_agentic_pipeline(args))
    else:
        from pipelines.agent_pipeline.traditional_rag.pipeline import run_traditional_pipeline

        run_traditional_pipeline(args)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by the traditional and agentic RAG run commands."""
    parser.add_argument("--query", required=True, help="User query to classify")
    parser.add_argument(
        "--prompt-path",
        default=str(PROJECT_ROOT / "prompts" / "route_node_prompt.txt"),
        help="Path to the router prompt template",
    )
    parser.add_argument("--scope", default=None, help="Document scope text to inject into the router prompt")
    parser.add_argument("--scope-file", default=None, help="Path to a text file containing the document scope")
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL, help="Hugging Face model used for routing and generation")
    parser.add_argument(
        "--hf-inference-provider",
        default=None,
        help="Hugging Face Inference Provider for LLM calls (e.g. 'scaleway', 'together', 'fireworks-ai'). "
             "Defaults to the HF_INFERENCE_PROVIDER env var or 'scaleway'.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="Embedding model used for retrieval",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    parser.add_argument("--pinecone-index-name", required=False, help="Pinecone index name for traditional retrieval route")
    parser.add_argument(
        "--pinecone-namespace",
        default=DEFAULT_PINECONE_NAMESPACE,
        help="Pinecone namespace for traditional retrieval route",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Final number of results to return")
    parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
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
    parser.add_argument("--enable-judge", action="store_true", default=DEFAULT_ENABLE_JUDGE, help="Score reranked chunks with an LLM-as-a-judge")
    parser.add_argument(
        "--judge-prompt-path",
        default=str(DEFAULT_JUDGE_PROMPT_PATH),
        help="Path to the retriever judge prompt template",
    )
    parser.add_argument(
        "--judge-model-name",
        default=DEFAULT_LLM_MODEL,
        help="Hugging Face model used for the retriever judge",
    )
    parser.add_argument(
        "--judge-top-k",
        type=int,
        default=DEFAULT_JUDGE_TOP_K,
        help="Maximum number of judged chunks to keep for further processing",
    )
    parser.add_argument(
        "--judge-min-score",
        type=int,
        default=DEFAULT_JUDGE_MIN_SCORE,
        help="Minimum judge score (0-10) a chunk must have to be kept",
    )
    parser.add_argument("--enable-input-guard", action="store_true", help="Enable input PII + safety guard")
    parser.add_argument("--enable-output-guard", action="store_true", help="Enable output PII + safety guard")
    parser.add_argument(
        "--presidio-analyzer-url",
        default=DEFAULT_PRESIDIO_ANALYZER_URL,
        help="URL of the Presidio analyzer service",
    )
    parser.add_argument(
        "--presidio-anonymizer-url",
        default=DEFAULT_PRESIDIO_ANONYMIZER_URL,
        help="URL of the Presidio anonymizer service",
    )
    parser.add_argument(
        "--safety-nsfw-threshold",
        type=float,
        default=DEFAULT_SAFETY_NSFW_THRESHOLD,
        help="Threshold for the NSFWText safety validator",
    )
    parser.add_argument(
        "--safety-toxic-threshold",
        type=float,
        default=DEFAULT_SAFETY_TOXIC_THRESHOLD,
        help="Threshold for the ToxicLanguage safety validator",
    )
    parser.add_argument(
        "--refusal-messages-path",
        default=None,
        help="Path to the refusal messages file used by safety guards",
    )
    parser.add_argument("--as-json", action="store_true", help="Print the pipeline result as JSON")
    parser.add_argument("--print-prompt", action="store_true", help="Include the rendered prompt in the output")
    parser.add_argument("--enable-langsmith", action="store_true", help="Enable LangSmith tracing for this run")
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="Override the LANGCHAIN_PROJECT value read from .env",
    )
    parser.add_argument(
        "--turn-on-agent-rag",
        action="store_true",
        help="Enable the agentic RAG path (LangGraph ReAct + MCP tools). "
             "Can also be set via TURN_ON_AGENT_RAG env var.",
    )
    parser.add_argument(
        "--tavily-api-key",
        default=None,
        help="Tavily API key for the web_search MCP tool in agentic RAG mode",
    )
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        default=DEFAULT_ENABLE_GENERATION,
        help="Enable GenerationNode for final answer generation (agentic RAG)",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread ID for persistent agentic RAG (auto-generates UUID if omitted)",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=None,
        help="Path to SQLite checkpoint database for agentic RAG (default: logs/agent_checkpoints.sqlite)",
    )
    parser.add_argument(
        "--enable-conversation-compaction",
        action="store_true",
        default=DEFAULT_ENABLE_CONVERSATION_COMPACTION,
        help="Enable conversation summarization when agentic RAG context grows too long",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Max input tokens for the LLM (used to compute compaction threshold, default: 256000)",
    )
    parser.add_argument(
        "--context-token-threshold-pct",
        type=float,
        default=DEFAULT_CONTEXT_TOKEN_THRESHOLD_PCT,
        help="Percentage of max input tokens that triggers summarization (default: 0.30)",
    )
    parser.add_argument(
        "--min-keep-recent-turns",
        type=int,
        default=DEFAULT_MIN_KEEP_RECENT_TURNS,
        help="Minimum number of recent query/response pairs to keep unsummarized (default: 1)",
    )
    parser.add_argument(
        "--conversation-history-dir",
        default=str(PROJECT_ROOT / "logs" / "conversation_history"),
        help="Directory to save full conversation history before summarization",
    )
    parser.add_argument(
        "--compaction-max-summary-tokens",
        type=int,
        default=DEFAULT_COMPACTION_MAX_SUMMARY_TOKENS,
        help="Max output tokens for the compaction summary LLM call (default: 2048)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connected agent pipeline CLI with indexing, query, scope, and routed retrieval commands"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the routed agent pipeline")
    _add_run_arguments(run_parser)
    run_parser.set_defaults(func=run)

    index_parser = subparsers.add_parser("index", help="Index a PDF into FAISS storage")
    index_parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    index_parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory for FAISS index and metadata",
    )
    index_parser.add_argument("--model-name", default=DEFAULT_INDEXING_MODEL_NAME, help="Embedding model name")
    index_parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    index_parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters")
    index_parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    index_parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
    index_parser.set_defaults(func=command_index)

    query_parser = subparsers.add_parser("query", help="Query an existing FAISS index")
    query_parser.add_argument("--query", required=True, help="User query text")
    query_parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory containing FAISS index and metadata",
    )
    query_parser.add_argument("--model-name", default=DEFAULT_INDEXING_MODEL_NAME, help="Embedding model name")
    query_parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
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
    query_parser.add_argument("--use-fp16", action="store_true", help="Kept for CLI compatibility; ignored by inference providers")
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
