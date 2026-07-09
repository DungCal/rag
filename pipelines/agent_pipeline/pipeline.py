from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file into a dictionary."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV_VALUES = _parse_env_file(ENV_FILE_PATH)
for _key in ("LANGCHAIN_ENDPOINT", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
    if _ENV_VALUES.get(_key) and not os.environ.get(_key):
        os.environ[_key] = _ENV_VALUES[_key]

from loguru import logger

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

try:
    from langsmith import traceable

    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional for base pipeline
    _LANGSMITH_AVAILABLE = False

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


_trace_router = traceable(run_type="llm", name="router_classification")
_trace_greeting = traceable(run_type="llm", name="greeting_response")
_trace_off_topic = traceable(run_type="llm", name="off_topic_response")
_trace_judge = traceable(run_type="llm", name="retriever_judge")

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
    PromptRouteDecision,
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
from pipelines.agent_pipeline.retriever_judge import (
    DEFAULT_JUDGE_MIN_SCORE,
    DEFAULT_JUDGE_PROMPT_PATH,
    DEFAULT_JUDGE_TOP_K,
    RetrieverJudgeNode,
)
from commons.guardrails import PresidioPIIGuard, GuardrailsSafetyGuard
from pipelines.agent_pipeline.rejected_nodes import RejectedNode
from pipelines.agent_pipeline.safety_input_nodes import SafetyInputNode, SafetyInputResult
from pipelines.agent_pipeline.safety_output_nodes import SafetyOutputNode, SafetyOutputResult


DEFAULT_SCOPE_FILE = PROJECT_ROOT / "results" / "scope_result_20260606_193507.txt"
DEFAULT_PRESIDIO_ANALYZER_URL = "http://localhost:5002/analyze"
DEFAULT_PRESIDIO_ANONYMIZER_URL = "http://localhost:5001/anonymize"
DEFAULT_REFUSAL_MESSAGES_PATH = str(PROJECT_ROOT / "prompts" / "guardrails" / "refusal_messages.txt")
DEFAULT_SAFETY_NSFW_THRESHOLD = 0.95
DEFAULT_SAFETY_TOXIC_THRESHOLD = 0.5
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
        embedding_provider: str | None,
        use_fp16: bool,
        enable_rerank: bool,
        reranker_model_name: str,
        reranker_batch_size: int,
        reranker_max_length: int,
        reranker_instruction: str | None,
        reranker_use_fp16: bool,
        reranker_apply_sigmoid: bool,
        enable_judge: bool,
        judge_prompt_path: str,
        judge_model_name: str,
        judge_top_k: int,
        judge_min_score: int,
        enable_input_guard: bool = False,
        enable_output_guard: bool = False,
        presidio_analyzer_url: str = DEFAULT_PRESIDIO_ANALYZER_URL,
        presidio_anonymizer_url: str = DEFAULT_PRESIDIO_ANONYMIZER_URL,
        safety_nsfw_threshold: float = DEFAULT_SAFETY_NSFW_THRESHOLD,
        safety_toxic_threshold: float = DEFAULT_SAFETY_TOXIC_THRESHOLD,
        refusal_messages_path: str | None = None,
        enable_langsmith: bool = False,
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
        self.embedding_provider = embedding_provider
        self.use_fp16 = use_fp16
        self.enable_rerank = enable_rerank
        self.reranker_model_name = reranker_model_name
        self.reranker_batch_size = reranker_batch_size
        self.reranker_max_length = reranker_max_length
        self.reranker_instruction = reranker_instruction
        self.reranker_use_fp16 = reranker_use_fp16
        self.reranker_apply_sigmoid = reranker_apply_sigmoid
        self.enable_judge = enable_judge
        self.judge_prompt_path = judge_prompt_path
        self.judge_model_name = judge_model_name
        self.judge_top_k = judge_top_k
        self.judge_min_score = judge_min_score
        self.enable_input_guard = enable_input_guard
        self.enable_output_guard = enable_output_guard
        self.presidio_analyzer_url = presidio_analyzer_url
        self.presidio_anonymizer_url = presidio_anonymizer_url
        self.safety_nsfw_threshold = safety_nsfw_threshold
        self.safety_toxic_threshold = safety_toxic_threshold
        self.refusal_messages_path = refusal_messages_path
        self.enable_langsmith = enable_langsmith

        logger.info(
            "Initializing RoutedRAGPipeline: llm_model={}, embedding_model={}, embedding_provider={}, "
            "reranker_model={}, pinecone_index_name={}, pinecone_namespace={}, top_k={}, enable_rerank={}, "
            "enable_judge={}, judge_top_k={}, judge_min_score={}",
            self.model_name,
            self.embedding_model_name,
            self.embedding_provider,
            self.reranker_model_name,
            self.pinecone_index_name,
            self.pinecone_namespace,
            self.top_k,
            self.enable_rerank,
            self.enable_judge,
            self.judge_top_k,
            self.judge_min_score,
        )
        logger.info(
            "LangSmith tracing {}: available={}, endpoint={}, project={}",
            "enabled" if self.enable_langsmith else "disabled",
            _LANGSMITH_AVAILABLE,
            os.environ.get("LANGCHAIN_ENDPOINT", "default"),
            os.environ.get("LANGCHAIN_PROJECT", "default"),
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
        self._judge = None

        # Guardrail nodes
        if self.enable_input_guard or self.enable_output_guard:
            self._rejected_node = RejectedNode(refusal_messages_path=self.refusal_messages_path)
            pii_guard = PresidioPIIGuard(
                analyzer_url=self.presidio_analyzer_url,
                anonymizer_url=self.presidio_anonymizer_url,
            )
            safety_guard = GuardrailsSafetyGuard(
                nsfw_threshold=self.safety_nsfw_threshold,
                toxic_threshold=self.safety_toxic_threshold,
            )
            self._input_guard = (
                SafetyInputNode(
                    pii_guard=pii_guard,
                    safety_guard=safety_guard,
                    rejected_node=self._rejected_node,
                )
                if self.enable_input_guard
                else None
            )
            self._output_guard = (
                SafetyOutputNode(
                    pii_guard=pii_guard,
                    safety_guard=safety_guard,
                    rejected_node=self._rejected_node,
                )
                if self.enable_output_guard
                else None
            )
        else:
            self._rejected_node = None
            self._input_guard = None
            self._output_guard = None

        self._chain = (
            RunnableLambda(self._build_state)
            | RunnableBranch(
                (lambda state: state["decision"].route == "rejected", RunnableLambda(self._run_rejected)),
                (lambda state: state["decision"].route == "greeting", RunnableLambda(self._run_greeting)),
                (lambda state: state["decision"].route == "off_topic", RunnableLambda(self._run_off_topic)),
                RunnableLambda(self._run_retrieval),
            )
            | RunnableLambda(self._apply_output_guard)
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

        input_guard_result = None
        if self._input_guard is not None:
            input_guard_result = self._input_guard.run(query)
            if input_guard_result.rejected:
                logger.warning("Input guard rejected query; skipping router")
                return {
                    "query": query,
                    "include_prompt": bool(inputs.get("include_prompt", False)),
                    "decision": PromptRouteDecision(
                        route="rejected",
                        label="safety_violation",
                        message="Input rejected by safety guard.",
                        raw_output="",
                        prompt="",
                    ),
                    "node_result": input_guard_result.node_result,
                    "retriever_result": None,
                    "rerank_result": None,
                    "judge_result": None,
                    "input_guard_result": input_guard_result.to_dict(),
                }
            query = input_guard_result.query

        decision = self._classify_route(query)
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
            "judge_result": None,
            "input_guard_result": input_guard_result.to_dict() if input_guard_result else None,
        }

    @_trace_router
    def _classify_route(self, query: str) -> Any:
        """Classify the user query into greeting/related/off_topic."""
        return self._router.route(query)

    def _run_greeting(self, state: dict[str, Any]) -> dict[str, Any]:
        state["node_result"] = self._generate_greeting(state["query"])
        return state

    @_trace_greeting
    def _generate_greeting(self, query: str) -> Any:
        """Generate a greeting response using the LLM."""
        return self._greeting_node.run(query)

    def _run_off_topic(self, state: dict[str, Any]) -> dict[str, Any]:
        state["node_result"] = self._generate_off_topic(state["query"])
        return state

    @_trace_off_topic
    def _generate_off_topic(self, query: str) -> Any:
        """Generate an off-topic response using the LLM."""
        return self._off_topic_node.run(query)

    def _run_rejected(self, state: dict[str, Any]) -> dict[str, Any]:
        """No-op branch: the rejection node_result was already set by the input guard."""
        return state

    def _apply_output_guard(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run output safety + PII guard on any generated textual response."""
        if self._output_guard is None:
            return state

        # Already rejected inputs do not need their refusal message re-checked.
        if state["decision"].route == "rejected":
            return state

        node_result = state.get("node_result")
        if node_result is None or not getattr(node_result, "response", None):
            return state

        output_result = self._output_guard.run(node_result)
        state["node_result"] = output_result.node_result
        state["output_guard_result"] = output_result.to_dict()

        if output_result.safety_result and not output_result.safety_result.passed:
            logger.warning("Output guard rejected generated response")
            state["decision"] = PromptRouteDecision(
                route="rejected",
                label="safety_violation",
                message="Output rejected by safety guard.",
                raw_output="",
                prompt="",
            )

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
                provider=self.embedding_provider,
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

            judge_input = state["rerank_result"] or state["retriever_result"]
            if self.enable_judge and judge_input is not None:
                if self._judge is None:
                    self._judge = RetrieverJudgeNode(
                        prompt_path=self.judge_prompt_path,
                        model_name=self.judge_model_name,
                        top_k=self.judge_top_k,
                        min_score=self.judge_min_score,
                    )

                state["judge_result"] = self._judge_chunk_relevance(
                    judge_input.query,
                    judge_input.results,
                )
                if state["judge_result"] is not None:
                    logger.info(
                        "Judge returned {} scored result(s)",
                        len(state["judge_result"].results),
                    )
                    for idx, item in enumerate(state["judge_result"].results, start=1):
                        judge = item.get("judge", {})
                        logger.info(
                            "Judge result #{}: final_score={} score_band={} page={} chunk_id={} "
                            "text_preview={!r}",
                            idx,
                            judge.get("final_score"),
                            judge.get("score_band"),
                            item.get("page_number"),
                            item.get("chunk_id"),
                            (item.get("text", "")[:120] + "...") if len(item.get("text", "")) > 120 else item.get("text", ""),
                        )
        return state

    @_trace_judge
    def _judge_chunk_relevance(self, query: str, results: list[dict[str, Any]]) -> Any:
        """Score retrieved/reranked chunks for relevance using the LLM."""
        return self._judge.run(query, results)

    @staticmethod
    def _build_payload(state: dict[str, Any]) -> dict[str, Any]:
        decision = state["decision"]
        node_result = state["node_result"]
        retriever_result = state["retriever_result"]
        rerank_result = state["rerank_result"]
        judge_result = state["judge_result"]

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
        if judge_result is not None:
            payload["judge_enabled"] = True
            payload["judge_results"] = judge_result.results
            payload["results"] = judge_result.results
        if state.get("input_guard_result") is not None:
            payload["input_guard_result"] = state["input_guard_result"]
        if state.get("output_guard_result") is not None:
            payload["output_guard_result"] = state["output_guard_result"]
        if state["include_prompt"]:
            payload["prompt"] = decision.prompt
            if node_result is not None:
                payload["node_prompt"] = node_result.prompt
            if judge_result is not None and judge_result.results:
                payload["judge_prompt"] = judge_result.results[0].get("judge_prompt")
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
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Hugging Face Hub inference provider for embeddings (e.g. 'together', 'fireworks-ai'). "
             "Defaults to the Hugging Face Inference API.",
    )
    parser.add_argument("--pinecone-index-name", required=False, help="Pinecone index name for retrieval route")
    parser.add_argument(
        "--pinecone-namespace",
        default=DEFAULT_PINECONE_NAMESPACE,
        help="Pinecone namespace for retrieval route",
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
    parser.add_argument("--enable-judge", action="store_true", help="Score reranked chunks with an LLM-as-a-judge")
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


def run_agent_pipeline(args: argparse.Namespace) -> None:
    if args.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = args.langsmith_project

    if args.enable_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        logger.info(
            "LangSmith tracing active: endpoint={}, project={}",
            os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            os.environ.get("LANGCHAIN_PROJECT", "default"),
        )
    elif not os.environ.get("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    scope = _load_scope(args.scope, args.scope_file)
    pipeline = RoutedRAGPipeline(
        prompt_path=args.prompt_path,
        scope=scope,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        pinecone_index_name=args.pinecone_index_name,
        pinecone_namespace=args.pinecone_namespace,
        top_k=args.top_k,
        embedding_provider=args.embedding_provider,
        use_fp16=args.use_fp16,
        enable_rerank=args.enable_rerank,
        reranker_model_name=args.reranker_model_name,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_instruction=args.reranker_instruction,
        reranker_use_fp16=args.reranker_fp16,
        reranker_apply_sigmoid=args.reranker_sigmoid,
        enable_judge=args.enable_judge,
        judge_prompt_path=args.judge_prompt_path,
        judge_model_name=args.judge_model_name,
        judge_top_k=args.judge_top_k,
        judge_min_score=args.judge_min_score,
        enable_input_guard=args.enable_input_guard,
        enable_output_guard=args.enable_output_guard,
        presidio_analyzer_url=args.presidio_analyzer_url,
        presidio_anonymizer_url=args.presidio_anonymizer_url,
        safety_nsfw_threshold=args.safety_nsfw_threshold,
        safety_toxic_threshold=args.safety_toxic_threshold,
        refusal_messages_path=args.refusal_messages_path,
        enable_langsmith=args.enable_langsmith,
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
        if payload.get("judge_enabled"):
            print("judge_results")
            print("-" * 80)
            for item in payload.get("judge_results", []):
                judge = item.get("judge", {})
                print(
                    "judge_score={final_score} score_band={score_band} "
                    "page={page} chunk={chunk}".format(
                        final_score=judge.get("final_score"),
                        score_band=judge.get("score_band"),
                        page=item.get("page_number"),
                        chunk=item.get("chunk_id"),
                    )
                )
                print(item.get("text", ""))
                print("-" * 80)
        if not payload.get("reranking_enabled") and not payload.get("judge_enabled"):
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
        if "judge_prompt" in payload:
            print("-" * 80)
            print(payload["judge_prompt"])


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
