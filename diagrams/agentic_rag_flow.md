# Agentic RAG Flow

Mermaid diagram of the actual implemented pipeline through `orchestration.py` → `AgentRAGPipeline` (`agent_graph.py`).

```mermaid
flowchart TB
    Start(["orchestration.py run --turn-on-agent-rag"])
    Start --> Invoke

    Invoke["AgentRAGPipeline.invoke(query, thread_id)<br/>SqliteSaver checkpoint → logs/agent_checkpoints.sqlite"]
    Invoke --> InSafety

    InSafety["input_safety 🛡<br/>SafetyInputNode — PII redact + safety check<br/>(--enable-input-guard only)"]
    InSafety --> InDecide{"decide_after_input_safety"}

    InDecide -- "rejected (PII / safety violation → refusal final_answer)" --> HistLog
    InDecide -- ok --> Compaction

    Compaction["conversation_compaction<br/>(--enable-conversation-compaction only)<br/>pre-answer token count<br/>over threshold ⇒ summarize old turns → SystemMessage"]
    Compaction --> Router

    Router["router<br/>PromptQueryRouter (LLM prompt) →<br/>greeting / related / off_topic"]
    Router --> RouteAfterRouter{"decide_after_router"}

    RouteAfterRouter -- greeting --> Greet
    RouteAfterRouter -- off_topic --> Off
    RouteAfterRouter -- related --> Agent

    Greet["greeting_response<br/>GreetingNode"]
    Greet --> PostComp

    Off["off_topic_response<br/>OffTopicNode"]
    Off --> PostComp

    Agent["agent<br/>InferenceClientChatAdapter.bind_tools(tools)<br/>LLM decides: tool_calls or final text"]
    Agent --> ToolDecision{"has tool_calls?"}

    ToolDecision -- continue --> Tools
    ToolDecision -- end --> Proc

    Tools["tools (ToolNode → MCP stdio servers)<br/>① retrieve_context — FAISS retrieval<br/>② rerank — cross-encoder rerank<br/>③ web_search — Tavily (guarded 🛡 if enabled)<br/>④ query_user_data — NOT wired"]
    Tools --> Agent

    Proc["process_results<br/>extract_tool_results: chunks + web_results<br/>filter top-k by score → RetrieverJudgeNode (if enabled)<br/>⇒ combined_context"]
    Proc --> Gen

    Gen["generate<br/>GenerationNode (if enabled)<br/>uses combined_context to produce answer"]
    Gen --> OutSafety

    OutSafety["output_safety 🛡<br/>SafetyOutputNode (--enable-output-guard only)<br/>violation ⇒ route=rejected, refusal message"]
    OutSafety --> PostComp

    PostComp["post_answer_compaction<br/>post-answer token check + summary (optional)"]
    PostComp --> HistLog

    HistLog["conversation_history_logger<br/>append turn → logs/conversation_history/{thread_id}.md"]
    HistLog --> EndOut

    QALog["QALogger.log(thread_id, query, answer)"]
    HistLog --> QALog
    QALog --> EndOut

    EndOut(["END → final payload JSON"])

    style Start fill:#d32f2f,color:#fff
    style EndOut fill:#1b5e20,color:#fff
    style InSafety fill:#ffebee
    style OutSafety fill:#ffebee
    style Tools fill:#e3f2fd
    style Agent fill:#fff3e0
    style Router fill:#f3e5f5
    style Proc fill:#e8f5e9
    style Gen fill:#e8f5e9
    style Compaction fill:#fffde7
    style PostComp fill:#fffde7
```

> **⚠ Code vs diagram note:** The rejection routing check lives at the `conversation_compaction` exit (`_route_after_compaction`), not at `input_safety`, because the graph always flows `input_safety → conversation_compaction` unconditionally. On the rejected path compaction is a no-op (no messages to summarize). The diagram shows the decision at `input_safety` to reflect *semantic ownership* — the rejection originates from the input guard, not from compaction.

## Key implementation details

| Node | Source | Behavior |
|------|--------|----------|
| `input_safety` | `SafetyInputNode` | PII redaction + NSFW/Toxicity check. Skipped if `--enable-input-guard` not set. On rejection, sets `route=rejected` + refusal `final_answer`. |
| `conversation_compaction` | `ConversationCompactionNode.pre_answer_check()` | Token count via `AutoTokenizer`. If total > threshold × max_input_tokens, summarizes older turns into a `SystemMessage`. |
| `router` | `PromptQueryRouter` | LLM call with `prompts/route_node_prompt.txt`. Maps `retrieval` → `related`, `greeting` → `greeting`, `off_topic` → `off_topic`. |
| `agent` | `InferenceClientChatAdapter.bind_tools(tools)` | LLM with OpenAI-compatible tool schema. Returns `tool_calls` or text content. |
| `tools` | `ToolNode` (LangGraph) | Spawns MCP stdio servers via `MultiServerMCPClient`. Only `web_search` is wrapped by `GuardTool` when guards enabled. |
| `process_results` | `extract_tool_results()` + optional `RetrieverJudgeNode` | Chunks sorted by score, top-k filtered, judged if enabled. Combined with web results. |
| `generate` | `GenerationNode` | Sends `combined_context` to LLM for final answer. Falls back to last AI message if disabled or no context. |
| `output_safety` | `SafetyOutputNode` | Checks generated answer. If violation detected, replaces with refusal and sets `route=rejected`. |
| `post_answer_compaction` | `ConversationCompactionNode.post_answer_check()` | Second compaction pass after generation. Writes token debug log if `--enable-token-debug-log`. |
| `conversation_history_logger` | `append_turn_to_file()` | Appends user/assistant/system messages to `logs/conversation_history/{thread_id}.md` on every turn. |

## MCP tools

| Tool | Server | Guards |
|------|--------|--------|
| `retrieve_context` | `mcp_servers/retrieve_context_server.py` | None (raw) |
| `rerank` | `mcp_servers/rerank_server.py` | None (raw) |
| `web_search` | `mcp_servers/web_search_server.py` | `GuardTool` wrapping `SafetyInputNode` + `SafetyOutputNode` (when `--enable-input-guard` or `--enable-output-guard`) |
| `query_user_data` | `mcp_servers/query_user_data_server.py` | Not wired into the agent |
