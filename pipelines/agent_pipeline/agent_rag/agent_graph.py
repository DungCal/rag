from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from pipelines.agent_pipeline.shared import (
    GenerationNode,
    PromptNodeResult,
    RetrieverJudgeNode,
    SafetyInputNode,
    SafetyOutputNode,
)
from pipelines.agent_pipeline.shared.qa_logger import QALogger
from pipelines.agent_pipeline.shared.utils import extract_tool_results
from pipelines.agent_pipeline.agent_rag.conversation_compaction import (
    ConversationCompactionNode,
    append_turn_to_file,
    _replace_system_context_in_file,
    _get_next_turn_number,
    DEFAULT_HISTORY_DIR,
)
from pipelines.agent_pipeline.agent_rag.token_utils import count_messages_tokens, count_tokens
from pipelines.agent_pipeline.traditional_rag.routers.routing_classification import (
    DEFAULT_DOCUMENT_SCOPE,
    PromptQueryRouter,
)
from pipelines.agent_pipeline.traditional_rag.routers.routing_response import (
    GreetingNode,
    OffTopicNode,
    load_default_scope,
)
from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL, DEFAULT_HF_INFERENCE_PROVIDER, _load_api_key_from_env_file, _load_provider_from_env_file


try:
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "langgraph is required for the agentic RAG agent. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langgraph is required for the agentic RAG agent. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langgraph-checkpoint-sqlite is required for persistent conversation state. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from huggingface_hub import InferenceClient
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "huggingface_hub and langchain-core are required for the agentic RAG agent. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


class AgentState(TypedDict):
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: str | None
    label: str | None
    router_message: str | None
    router_raw_output: str | None
    router_prompt: str | None
    input_guard_result: dict | None
    output_guard_result: dict | None
    node_result: dict | None
    combined_context: list | None
    final_answer: str | None
    safety_passed: bool
    summarized: bool
    conversation_history_file: str | None
    system_context: str | None
    recent_turns: list[BaseMessage]


class InferenceClientChatAdapter(BaseChatModel):
    """Adapter to use huggingface_hub InferenceClient with LangChain.

    Supports native tool calling by converting LangChain tools to OpenAI-compatible
    tool schemas and parsing tool_calls from the InferenceClient response.
    """

    client: Any
    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.2
    tools: list[Any] | None = None
    provider: str | None = None

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "InferenceClientChatAdapter":
        """Bind tools to this model for tool calling."""
        return InferenceClientChatAdapter(
            client=self.client,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=tools,
            provider=self.provider,
        )

    def _convert_tool_to_openai_schema(self, tool: Any) -> dict[str, Any]:
        """Convert a LangChain tool to OpenAI-compatible tool schema."""
        from langchain_core.tools import BaseTool
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Expected BaseTool, got {type(tool).__name__}")

        args_schema = tool.args_schema
        properties = {}
        required = []
        if args_schema is not None:
            schema = args_schema.schema() if hasattr(args_schema, 'schema') else {}
            properties = schema.get("properties", {})
            required = schema.get("required", [])

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _parse_tool_calls(self, response: Any) -> list[Any]:
        """Parse tool_calls from InferenceClient response into LangChain ToolCall objects."""
        from langchain_core.messages import ToolCall
        message = response.choices[0].message
        if not message.tool_calls:
            return []

        tool_calls = []
        for tc in message.tool_calls:
            tool_calls.append(ToolCall(
                name=tc.function.name,
                args=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                id=tc.id,
                type="tool_call",
            ))
        return tool_calls

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convert LangChain messages to InferenceClient format
        hf_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict: dict[str, Any] = {"role": "assistant"}
                # Include tool_calls if present (from previous iterations)
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in tool_calls
                    ]
                else:
                    msg_dict["content"] = msg.content
                hf_messages.append(msg_dict)
            elif hasattr(msg, "role") and msg.role == "tool":
                hf_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            else:
                hf_messages.append({"role": "user", "content": msg.content})

        # Build tool schemas for the InferenceClient
        tools_schema = None
        if self.tools:
            tools_schema = [self._convert_tool_to_openai_schema(t) for t in self.tools]

        response = self.client.chat_completion(
            messages=hf_messages,
            tools=tools_schema,
            tool_choice="auto" if tools_schema else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        
        content = response.choices[0].message.content or ""
        tool_calls = self._parse_tool_calls(response)
        
        message = AIMessage(content=content)
        if tool_calls:
            message = AIMessage(content=content, tool_calls=tool_calls)
        
        return ChatResult(generations=[ChatGeneration(message=message)])


DEFAULT_SCOPE_FILE = PROJECT_ROOT / "results" / "scope_result_20260606_193507.txt"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "route_node_prompt.txt"
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


class AgentRAGPipeline:
    """Agentic RAG pipeline as a single checkpointed LangGraph StateGraph."""

    DEFAULT_CHECKPOINT_DB = PROJECT_ROOT / "logs" / "agent_checkpoints.sqlite"

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
        provider: str | None = None,
        thread_id: str | None = None,
        checkpoint_db: str | Path | None = None,
        enable_conversation_compaction: bool = False,
        max_input_tokens: int = 256_000,
        context_token_threshold_pct: float = 0.30,
        min_keep_recent_turns: int = 1,
        conversation_history_dir: str | Path | None = None,
        compaction_max_summary_tokens: int = 2048,
        enable_token_debug_log: bool = False,
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
        self.thread_id = thread_id
        self.checkpoint_db = Path(checkpoint_db) if checkpoint_db else self.DEFAULT_CHECKPOINT_DB
        self.enable_conversation_compaction = enable_conversation_compaction
        self.max_input_tokens = max_input_tokens
        self.context_token_threshold_pct = context_token_threshold_pct
        self.min_keep_recent_turns = min_keep_recent_turns
        self.conversation_history_dir = Path(conversation_history_dir) if conversation_history_dir else PROJECT_ROOT / "logs" / "conversation_history"
        self.compaction_max_summary_tokens = compaction_max_summary_tokens
        self.enable_token_debug_log = enable_token_debug_log

        resolved_provider = provider or os.getenv("HF_INFERENCE_PROVIDER") or _load_provider_from_env_file() or DEFAULT_HF_INFERENCE_PROVIDER

        self._router = PromptQueryRouter(
            prompt_path=self.prompt_path,
            scope=self.scope,
            model_name=self.model_name,
            provider=resolved_provider,
        )
        self._greeting_node = GreetingNode(scope=self.scope, model_name=self.model_name, provider=resolved_provider)
        self._off_topic_node = OffTopicNode(scope=self.scope, model_name=self.model_name, provider=resolved_provider)
        self._input_guard = SafetyInputNode() if self.enable_input_guard else None
        self._output_guard = SafetyOutputNode() if self.enable_output_guard else None

        self._compaction_node = None
        if self.enable_conversation_compaction:
            from pipelines.agent_pipeline.agent_rag.conversation_compaction import ConversationCompactionNode

            self._compaction_node = ConversationCompactionNode(
                model_name=self.model_name,
                provider=resolved_provider,
                max_input_tokens=self.max_input_tokens,
                threshold_pct=self.context_token_threshold_pct,
                min_keep_recent_turns=self.min_keep_recent_turns,
                history_dir=self.conversation_history_dir,
                max_summary_tokens=self.compaction_max_summary_tokens,
            )

        resolved_api_key = os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        inference_client = InferenceClient(
            model=self.model_name,
            token=resolved_api_key,
            provider=resolved_provider,
        )
        self._llm = InferenceClientChatAdapter(
            client=inference_client,
            model_name=self.model_name,
            max_tokens=1024,
            temperature=0.2,
            provider=resolved_provider,
        )

        self._judge = (
            RetrieverJudgeNode(
                model_name=self.model_name,
                top_k=self.judge_top_k,
                min_score=self.judge_min_score,
            )
            if self.enable_judge
            else None
        )
        self._generation_node = GenerationNode(model_name=self.model_name, provider=resolved_provider) if self.enable_generation else None

    def _build_graph(self, tools: list[Any], checkpointer: Any) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("input_safety", self._input_safety_node_fn)
        graph.add_node("conversation_compaction", self._conversation_compaction_node_fn)
        graph.add_node("router", self._router_node_fn)
        graph.add_node("greeting_response", self._greeting_response_node_fn)
        graph.add_node("off_topic_response", self._off_topic_response_node_fn)
        graph.add_node("agent", lambda state: self._agent_node_fn(state, tools))
        graph.add_node("tools", ToolNode(tools))
        graph.add_node("process_results", self._process_results_node_fn)
        graph.add_node("generate", self._generate_node_fn)
        graph.add_node("output_safety", self._output_safety_node_fn)
        graph.add_node("post_answer_compaction", self._post_answer_compaction_node_fn)
        graph.add_node("conversation_history_logger", self._conversation_history_logger_node_fn)

        graph.add_edge(START, "input_safety")
        graph.add_edge("input_safety", "conversation_compaction")
        graph.add_conditional_edges("conversation_compaction", self._route_after_compaction, {"rejected": "conversation_history_logger", "ok": "router"})
        graph.add_conditional_edges("router", self._route_after_router, {"greeting": "greeting_response", "off_topic": "off_topic_response", "related": "agent"})
        graph.add_edge("greeting_response", "post_answer_compaction")
        graph.add_edge("off_topic_response", "post_answer_compaction")
        graph.add_conditional_edges("agent", self._should_continue, {"continue": "tools", "end": "process_results"})
        graph.add_edge("tools", "agent")
        graph.add_edge("process_results", "generate")
        graph.add_edge("generate", "output_safety")
        graph.add_edge("output_safety", "post_answer_compaction")
        graph.add_edge("post_answer_compaction", "conversation_history_logger")
        graph.add_edge("conversation_history_logger", END)

        return graph.compile(name="agent_rag", checkpointer=checkpointer)

    def _input_safety_node_fn(self, state: AgentState) -> dict:
        if self._input_guard is None:
            return {"query": state["query"]}

        result = self._input_guard.run(state["query"])
        if result.rejected:
            refusal_text = result.node_result.response if result.node_result else "Input rejected by safety guard."
            logger.warning("Agentic input guard rejected query")
            return {
                "route": "rejected",
                "label": "safety_violation",
                "input_guard_result": result.to_dict(),
                "node_result": result.node_result.to_dict() if result.node_result else None,
                "final_answer": refusal_text,
                "messages": [AIMessage(content=refusal_text)],
            }
        return {
            "query": result.query,
            "input_guard_result": result.to_dict(),
        }

    def _conversation_compaction_node_fn(self, state: AgentState) -> dict:
        if self._compaction_node is None:
            return {"query": state["query"]}

        system_context = state.get("system_context")
        recent_turns = state.get("recent_turns", [])
        query = state["query"]
        active_thread_id = self.thread_id or ""

        try:
            result = self._compaction_node.pre_answer_check(
                system_context=system_context,
                recent_turns=recent_turns,
                query=query,
                thread_id=active_thread_id,
            )
        except Exception as exc:
            logger.warning("Pre-answer compaction failed (non-fatal): {}", exc)
            return {"query": state["query"]}

        return {
            "query": state["query"],
            "system_context": result.get("system_context"),
            "recent_turns": result.get("recent_turns", []),
            "summarized": result.get("summarized", False),
        }

    def _post_answer_compaction_node_fn(self, state: AgentState) -> dict:
        if self._compaction_node is None:
            return {}

        system_context = state.get("system_context")
        recent_turns = state.get("recent_turns", [])
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        active_thread_id = self.thread_id or ""

        if not query or not final_answer:
            return {}

        try:
            result = self._compaction_node.post_answer_check(
                system_context=system_context,
                recent_turns=recent_turns,
                query=query,
                answer=final_answer,
                thread_id=active_thread_id,
            )
        except Exception as exc:
            logger.warning("Post-answer compaction failed (non-fatal): {}", exc)
            return {}

        self._write_token_debug(
            thread_id=active_thread_id,
            query=query,
            final_answer=final_answer,
            system_context=system_context,
            recent_turns=recent_turns,
            result=result,
        )

        return {
            "system_context": result.get("system_context"),
            "recent_turns": result.get("recent_turns", []),
            "summarized": result.get("summarized", False),
        }

    def _write_token_debug(
        self,
        *,
        thread_id: str,
        query: str,
        final_answer: str,
        system_context: str | None,
        recent_turns: list[BaseMessage],
        result: dict[str, Any],
    ) -> None:
        if not self.enable_token_debug_log:
            return
        if not thread_id:
            return

        debug_dir = self.conversation_history_dir
        debug_path = Path(debug_dir) / f"{thread_id}_tokens.md"
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        turn_number = _get_next_turn_number(debug_path)

        sc_tokens = count_tokens(system_context, self.model_name) if system_context else 0
        rt_tokens = count_messages_tokens(recent_turns, self.model_name) if recent_turns else 0
        q_tokens = count_tokens(query, self.model_name)
        ans_tokens = count_tokens(final_answer, self.model_name)
        total_before = sc_tokens + rt_tokens + q_tokens + ans_tokens
        summarized = result.get("summarized", False)

        lines = []
        if not debug_path.exists() or debug_path.stat().st_size == 0:
            lines.append("# Token Debug Log\n")

        lines.append(f"## Turn {turn_number}\n")
        lines.append(f"- **Timestamp:** {timestamp}")
        lines.append(f"- **Query:** {query}")
        lines.append(f"- **Query tokens:** {q_tokens}")
        lines.append(f"- **Answer tokens:** {ans_tokens}")
        lines.append(f"- **system_context_tokens (before):** {sc_tokens}")
        lines.append(f"- **recent_turns_tokens (before):** {rt_tokens}")
        lines.append(f"- **Total (before compaction):** {total_before}")
        lines.append(f"- **Compaction triggered:** {summarized}")
        if summarized:
            new_sc = result.get("system_context", "")
            new_sc_tokens = count_tokens(new_sc, self.model_name) if new_sc else 0
            lines.append(f"- **system_context_tokens (after):** {new_sc_tokens}")
        lines.append(f"- **Threshold:** {self._compaction_node._threshold_tokens}")
        lines.append("")
        lines.append("### Full Final Answer\n")
        lines.append(final_answer)
        lines.append("\n---\n")

        existing = ""
        if debug_path.exists() and debug_path.stat().st_size > 0:
            existing = debug_path.read_text(encoding="utf-8").rstrip()
            debug_path.write_text(existing + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")
        else:
            debug_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

        logger.info("Token debug log written to {}", debug_path.resolve())

    @staticmethod
    def _route_after_compaction(state: AgentState) -> str:
        if state.get("route") == "rejected":
            return "rejected"
        return "ok"

    def _router_node_fn(self, state: AgentState) -> dict:
        decision = self._router.route(state["query"])
        logger.info(
            "Agentic router decision: query={!r}, route={}, label={}",
            state["query"],
            decision.route,
            decision.label,
        )
        system_context = state.get("system_context")
        recent_turns = state.get("recent_turns", [])
        messages = []
        if system_context:
            messages.append(SystemMessage(content=system_context))
        messages.extend(recent_turns)
        messages.append(HumanMessage(content=state["query"]))
        return {
            "route": decision.route,
            "label": decision.label,
            "router_message": decision.message,
            "router_raw_output": decision.raw_output,
            "router_prompt": decision.prompt,
            "messages": messages,
        }

    def _greeting_response_node_fn(self, state: AgentState) -> dict:
        node_result = self._greeting_node.run(state["query"])
        return {
            "node_result": node_result.to_dict(),
            "final_answer": node_result.response,
            "messages": [AIMessage(content=node_result.response)],
        }

    def _off_topic_response_node_fn(self, state: AgentState) -> dict:
        node_result = self._off_topic_node.run(state["query"])
        return {
            "node_result": node_result.to_dict(),
            "final_answer": node_result.response,
            "messages": [AIMessage(content=node_result.response)],
        }

    def _agent_node_fn(self, state: AgentState, tools: list[Any]) -> dict:
        llm_with_tools = self._llm.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                logger.info("Agent calling tool: {} with args={}", tc.get("name", "unknown"), tc.get("args", {}))
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def _process_results_node_fn(self, state: AgentState) -> dict:
        agent_output = {"messages": state["messages"]}
        tool_results = extract_tool_results(agent_output)
        chunks = tool_results.get("chunks", [])
        web_results = tool_results.get("web_results", [])

        # Filter to top-k chunks by score before judging to avoid judging too many chunks
        top_k = getattr(self, 'top_k', 5)
        if len(chunks) > top_k:
            # Sort chunks by score (descending) and take top-k
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.get("score", c.get("rerank_score", 0)),
                reverse=True
            )
            chunks = sorted_chunks[:top_k]
            logger.info("Filtered to top {} chunks by score (from {} total)", top_k, len(sorted_chunks))

        judged_chunks = chunks
        if self._judge is not None and chunks:
            logger.info("Judging {} chunks", len(chunks))
            judge_result = self._judge.run(state["query"], chunks)
            judged_chunks = judge_result.results
            logger.info("Judged {} chunks, filtered to top {} (min_score={})",
                       len(chunks), len(judged_chunks), self._judge.min_score)

        combined_context = judged_chunks + web_results
        return {"combined_context": combined_context}

    def _generate_node_fn(self, state: AgentState) -> dict:
        combined_context = state.get("combined_context") or []

        if self._generation_node is not None and combined_context:
            logger.info("Generating answer from {} context items", len(combined_context))
            node_result = self._generation_node.run(state["query"], combined_context)
        else:
            messages = state.get("messages", [])
            final_message = messages[-1] if messages else None
            final_answer = str(final_message.content).strip() if final_message else ""
            node_result = PromptNodeResult(
                route="agentic_rag",
                response=final_answer,
                raw_output=final_answer,
                prompt="",
            )

        return {
            "node_result": node_result.to_dict(),
            "final_answer": node_result.response,
            "messages": [AIMessage(content=node_result.response)],
        }

    def _output_safety_node_fn(self, state: AgentState) -> dict:
        final_answer = state.get("final_answer", "")
        node_result_dict = state.get("node_result")

        if node_result_dict:
            node_result = PromptNodeResult.from_dict(node_result_dict)
        else:
            node_result = PromptNodeResult(
                route=state.get("route") or "unknown",
                response=final_answer,
                raw_output=final_answer,
                prompt="",
            )

        if self._output_guard is None:
            return {}

        output_result = self._output_guard.run(node_result)
        safe_response = output_result.node_result.response
        updates: dict[str, Any] = {
            "output_guard_result": output_result.to_dict(),
            "final_answer": safe_response,
            "node_result": output_result.node_result.to_dict(),
            "safety_passed": output_result.safety_result.passed if output_result.safety_result else True,
        }

        if output_result.safety_result and not output_result.safety_result.passed:
            updates["route"] = "rejected"
            updates["label"] = "safety_violation"
            updates["messages"] = [AIMessage(content=safe_response)]
        elif safe_response != final_answer:
            updates["messages"] = [AIMessage(content=safe_response)]

        return updates

    def _conversation_history_logger_node_fn(self, state: AgentState) -> dict:
        if self._compaction_node is None:
            return {}

        active_thread_id = self.thread_id or ""
        if not active_thread_id:
            return {}

        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        system_context = state.get("system_context")

        history_file_path = self.conversation_history_dir / f"{active_thread_id}.md"

        if system_context:
            try:
                _replace_system_context_in_file(system_context, history_file_path)
            except Exception as exc:
                logger.warning("Failed to update system context in conversation history (non-fatal): {}", exc)

        if query or final_answer:
            try:
                append_turn_to_file(
                    query=query,
                    response=final_answer,
                    file_path=history_file_path,
                )
            except Exception as exc:
                logger.warning("Failed to append turn to conversation history (non-fatal): {}", exc)

        return {}

    @staticmethod
    def _route_after_router(state: AgentState) -> str:
        route = state.get("route") or "related"
        if route == "retrieval":
            return "related"
        return route

    async def invoke(self, query: str, *, thread_id: str | None = None, include_prompt: bool = False) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("User query must not be empty")

        active_thread_id = thread_id or self.thread_id or str(uuid.uuid4())
        self.thread_id = active_thread_id

        from pipelines.agent_pipeline.agent_rag.tool_client import AgentToolClient

        tool_client = AgentToolClient(
            tavily_api_key=self.tavily_api_key,
            enable_input_guard=self.enable_input_guard,
            enable_output_guard=self.enable_output_guard,
        )
        tools = await tool_client.connect()

        try:
            async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_db)) as checkpointer:
                compiled_graph = self._build_graph(tools, checkpointer)
                config = {"configurable": {"thread_id": active_thread_id}}
                initial_state: dict[str, Any] = {"query": query, "messages": []}
                final_state = await compiled_graph.ainvoke(initial_state, config=config)
        finally:
            await tool_client.close()

        QALogger(self.checkpoint_db).log(
            thread_id=active_thread_id,
            query=query,
            final_answer=final_state.get("final_answer") or "",
        )

        return self._finalize_from_state(final_state, include_prompt=include_prompt, thread_id=active_thread_id)

    def _finalize_from_state(self, state: AgentState, *, include_prompt: bool = False, thread_id: str | None = None) -> dict[str, Any]:
        route = state.get("route") or "unknown"
        label = state.get("label") or "unknown"
        router_message = state.get("router_message") or ""
        final_answer = state.get("final_answer") or ""
        node_result_dict = state.get("node_result") or {}
        node_prompt = node_result_dict.get("prompt", "")

        payload: dict[str, Any] = {
            "route": route,
            "label": label,
            "message": router_message,
            "raw_output": state.get("router_raw_output") or "",
            "response": final_answer,
            "node_raw_output": final_answer,
        }

        if state.get("input_guard_result") is not None:
            payload["input_guard_result"] = state["input_guard_result"]
        if state.get("output_guard_result") is not None:
            payload["output_guard_result"] = state["output_guard_result"]
        if thread_id is not None:
            payload["thread_id"] = thread_id
        if state.get("summarized"):
            payload["summarized"] = state["summarized"]
        if state.get("conversation_history_file"):
            payload["conversation_history_file"] = state["conversation_history_file"]
        if include_prompt:
            payload["prompt"] = state.get("router_prompt") or ""
            payload["node_prompt"] = node_prompt
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
        provider=getattr(args, 'hf_inference_provider', None),
        thread_id=getattr(args, 'thread_id', None),
        checkpoint_db=getattr(args, 'checkpoint_db', None),
        enable_conversation_compaction=getattr(args, 'enable_conversation_compaction', False),
        max_input_tokens=getattr(args, 'max_input_tokens', 256_000),
        context_token_threshold_pct=getattr(args, 'context_token_threshold_pct', 0.30),
        min_keep_recent_turns=getattr(args, 'min_keep_recent_turns', 1),
        conversation_history_dir=getattr(args, 'conversation_history_dir', None),
        compaction_max_summary_tokens=getattr(args, 'compaction_max_summary_tokens', 2048),
        enable_token_debug_log=getattr(args, 'enable_token_debug_log', False),
    )
    payload = await pipeline.invoke(
        args.query,
        thread_id=getattr(args, 'thread_id', None),
        include_prompt=args.print_prompt,
    )

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
    if payload.get("thread_id"):
        print("-" * 80)
        print(f"thread_id={payload['thread_id']}")
    if args.print_prompt:
        print("-" * 80)
        print(payload.get("prompt", ""))
        if payload.get("node_prompt"):
            print("-" * 80)
            print(payload["node_prompt"])
