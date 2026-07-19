from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from pipelines.agent_pipeline.shared import (
    GenerationNode,
    PromptNodeResult,
    RetrieverJudgeNode,
    SafetyInputNode,
    SafetyInputResult,
    SafetyOutputNode,
)
from pipelines.agent_pipeline.shared.utils import extract_tool_results
from pipelines.agent_pipeline.traditional_rag.routers.routing_classification import (
    DEFAULT_DOCUMENT_SCOPE,
    PromptQueryRouter,
    PromptRouteDecision,
)
from pipelines.agent_pipeline.traditional_rag.routers.routing_response import (
    GreetingNode,
    OffTopicNode,
    load_default_scope,
)
from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL, _load_api_key_from_env_file


try:
    from langgraph.prebuilt import create_react_agent
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "langgraph is required for the agentic RAG agent. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "langchain-huggingface is required for the agentic RAG agent. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


DEFAULT_SCOPE_FILE = PROJECT_ROOT / "results" / "scope_result_20260606_193507.txt"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "route_node_prompt.txt"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

AGENT_SYSTEM_PROMPT = """You are a helpful assistant for a TYM tractor operator manual.

You have access to the following tools:
- retrieve_context: use this for questions about the tractor operator manual.
- rerank: use this when you have candidate chunks from retrieve_context and want to improve relevance ordering.
- web_search: use this only when the question needs current or external information not in the manual.

Follow these rules:
1. Use the minimum number of tools needed.
2. Prefer retrieve_context for manual-related questions.
3. Use web_search only when the manual cannot answer the question.
4. Synthesize a concise, accurate final answer from the tool results.
5. If no tool result is useful, say that you do not have enough information.
"""


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


class AgentRAGPipeline:
    """Agentic RAG pipeline using LangGraph ReAct + guarded MCP tools."""

    def __init__(
        self,
        *,
        prompt_path: str = str(DEFAULT_PROMPT_PATH),
        scope: str | None = None,
        scope_file: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        top_k: int = 5,
        tavily_api_key: str | None = None,
        enable_input_guard: bool = False,
        enable_output_guard: bool = False,
        enable_judge: bool = True,
        enable_generation: bool = True,
        judge_top_k: int = 3,
        judge_min_score: int = 5,
    ) -> None:
        self.prompt_path = prompt_path
        self.scope = _load_scope(scope, scope_file)
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.tavily_api_key = tavily_api_key
        self.enable_input_guard = enable_input_guard
        self.enable_output_guard = enable_output_guard
        self.enable_judge = enable_judge
        self.enable_generation = enable_generation
        self.judge_top_k = judge_top_k
        self.judge_min_score = judge_min_score

        self._router = PromptQueryRouter(
            prompt_path=self.prompt_path,
            scope=self.scope,
            model_name=self.model_name,
        )
        self._greeting_node = GreetingNode(scope=self.scope, model_name=self.model_name)
        self._off_topic_node = OffTopicNode(scope=self.scope, model_name=self.model_name)
        self._input_guard = SafetyInputNode() if self.enable_input_guard else None
        self._output_guard = SafetyOutputNode() if self.enable_output_guard else None

        resolved_api_key = os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        endpoint = HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=resolved_api_key,
            task="conversational",
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=False,
        )
        self._llm = ChatHuggingFace(llm=endpoint)

        # Initialize judge and generation nodes
        self._judge = (
            RetrieverJudgeNode(
                model_name=self.model_name,
                top_k=self.judge_top_k,
                min_score=self.judge_min_score,
            )
            if self.enable_judge
            else None
        )
        self._generation_node = GenerationNode(model_name=self.model_name) if self.enable_generation else None

    async def invoke(self, query: str, *, include_prompt: bool = False) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("User query must not be empty")

        input_guard_result = None
        if self._input_guard is not None:
            input_guard_result = self._input_guard.run(query)
            if input_guard_result.rejected:
                logger.warning("Agentic input guard rejected query")
                return {
                    "route": "rejected",
                    "label": "safety_violation",
                    "message": "Input rejected by safety guard.",
                    "response": input_guard_result.node_result.response if input_guard_result.node_result else None,
                    "input_guard_result": input_guard_result.to_dict(),
                }
            query = input_guard_result.query

        decision = self._router.route(query)
        logger.info(
            "Agentic router decision: query={!r}, route={}, label={}",
            query,
            decision.route,
            decision.label,
        )

        if decision.route == "greeting":
            node_result = self._greeting_node.run(query)
            return self._finalize(
                decision=decision,
                node_result=node_result,
                input_guard_result=input_guard_result,
                include_prompt=include_prompt,
            )

        if decision.route == "off_topic":
            node_result = self._off_topic_node.run(query)
            return self._finalize(
                decision=decision,
                node_result=node_result,
                input_guard_result=input_guard_result,
                include_prompt=include_prompt,
            )

        # related -> agentic RAG loop
        from pipelines.agent_pipeline.agent_rag.tool_client import AgentToolClient

        tool_client = AgentToolClient(
            tavily_api_key=self.tavily_api_key,
            enable_input_guard=self.enable_input_guard,
            enable_output_guard=self.enable_output_guard,
        )
        tools = await tool_client.connect()

        try:
            agent = create_react_agent(
                model=self._llm,
                tools=tools,
                prompt=AGENT_SYSTEM_PROMPT,
            )

            agent_input = {"messages": [("human", query)]}
            agent_output = await agent.ainvoke(agent_input)
            
            # Log tool calls made by the agent
            messages = agent_output.get("messages", [])
            for msg in messages:
                if hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
                    tool_calls = msg.additional_kwargs["tool_calls"]
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_args = tool_call.get("function", {}).get("arguments", "{}")
                        logger.info("Agent calling tool: {} with args={}", tool_name, tool_args)
            
            # Extract tool results from agent output
            tool_results = extract_tool_results(agent_output)
            chunks = tool_results.get("chunks", [])
            web_results = tool_results.get("web_results", [])
            
            # Judge chunks if enabled
            judged_chunks = chunks
            if self._judge is not None and chunks:
                logger.info("Judging {} chunks", len(chunks))
                judge_result = self._judge.run(query, chunks)
                judged_chunks = judge_result.results
                logger.info("Judged {} chunks, filtered to top {} (min_score={})", 
                           len(chunks), len(judged_chunks), self._judge.min_score)
                
                # Log top chunk details
                if judged_chunks:
                    top_chunk = judged_chunks[0]
                    top_chunk_id = top_chunk.get("chunk_id", "N/A")
                    top_score = top_chunk.get("judge", {}).get("final_score", "N/A")
                    top_page = top_chunk.get("page_number", "N/A")
                    logger.info("Top chunk: chunk_id={} final_score={} page={}", 
                               top_chunk_id, top_score, top_page)
            
            # Combine chunks and web results for generation
            combined_context = judged_chunks + web_results
            
            # Generate final answer using GenerationNode if enabled
            if self._generation_node is not None and combined_context:
                logger.info("Generating answer from {} context items", len(combined_context))
                node_result = self._generation_node.run(query, combined_context)
            else:
                # Fallback to agent's own answer if generation is disabled or no context
                messages = agent_output.get("messages", [])
                final_message = messages[-1] if messages else None
                final_answer = str(final_message.content).strip() if final_message else ""
                node_result = PromptNodeResult(
                    route="agentic_rag",
                    response=final_answer,
                    raw_output=final_answer,
                    prompt="",
                )
        finally:
            await tool_client.close()

        return self._finalize(
            decision=decision,
            node_result=node_result,
            input_guard_result=input_guard_result,
            include_prompt=include_prompt,
            agent_output=agent_output,
        )

    def _finalize(
        self,
        *,
        decision: PromptRouteDecision,
        node_result: PromptNodeResult,
        input_guard_result: SafetyInputResult | None,
        include_prompt: bool,
        agent_output: Any | None = None,
    ) -> dict[str, Any]:
        output_guard_result = None
        if self._output_guard is not None:
            output_result = self._output_guard.run(node_result)
            node_result = output_result.node_result
            output_guard_result = output_result.to_dict()
            if output_result.safety_result and not output_result.safety_result.passed:
                decision = PromptRouteDecision(
                    route="rejected",
                    label="safety_violation",
                    message="Output rejected by safety guard.",
                    raw_output="",
                    prompt="",
                )

        payload: dict[str, Any] = {
            "route": decision.route,
            "label": decision.label,
            "message": decision.message,
            "raw_output": decision.raw_output,
            "response": node_result.response,
            "node_raw_output": node_result.raw_output,
        }
        if input_guard_result is not None:
            payload["input_guard_result"] = input_guard_result.to_dict()
        if output_guard_result is not None:
            payload["output_guard_result"] = output_guard_result
        if agent_output is not None:
            payload["agent_output"] = agent_output
        if include_prompt:
            payload["prompt"] = decision.prompt
            payload["node_prompt"] = node_result.prompt
        return payload


async def run_agentic_pipeline(args: argparse.Namespace) -> None:
    """Run the agentic RAG pipeline."""
    pipeline = AgentRAGPipeline(
        prompt_path=args.prompt_path,
        scope=args.scope,
        scope_file=args.scope_file,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        top_k=args.top_k,
        tavily_api_key=args.tavily_api_key,
        enable_input_guard=args.enable_input_guard,
        enable_output_guard=args.enable_output_guard,
        enable_judge=getattr(args, 'enable_judge', True),
        enable_generation=getattr(args, 'enable_generation', True),
        judge_top_k=getattr(args, 'judge_top_k', 3),
        judge_min_score=getattr(args, 'judge_min_score', 5),
    )
    payload = await pipeline.invoke(args.query, include_prompt=args.print_prompt)

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    print(f"route={payload['route']}")
    print(f"label={payload['label']}")
    print(f"raw_output={payload['raw_output']}")
    print(payload["message"])
    if "response" in payload:
        print("-" * 80)
        print(payload["response"])
    if args.print_prompt:
        print("-" * 80)
        print(payload.get("prompt", ""))
        if payload.get("node_prompt"):
            print("-" * 80)
            print(payload["node_prompt"])
