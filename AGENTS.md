# AGENTS.md

## Entrypoints

- **FAISS-based (local):** `python -m pipelines.indexing_pipeline {index,query,scope}`
- **Prompt-based + Pinecone (remote):** `python -m pipelines.agent_pipeline.orchestration run --pinecone-index-name <name>`
  - Also runnable as a script: `python pipelines/agent_pipeline/orchestration.py run --pinecone-index-name <name>`
  - Backward-compatible alias: `python -m pipelines.agent_pipeline.pipeline run --pinecone-index-name <name>`
  - The orchestration module also re-exports `index`, `query`, `scope` subcommands (same CLI as indexing pipeline)
- **LangSmith variant:** `python -m pipelines.agent_pipeline.orchestration run --enable-langsmith --pinecone-index-name <name>`
  - Tracing is off by default; pass `--enable-langsmith` to send LLM spans to LangSmith.
  - The old `pipeline_v2.py` has been moved to `.old_artifacts/`.
- **Agentic RAG (LangGraph + MCP, FAISS retrieval):** `python -m pipelines.agent_pipeline.orchestration run --turn-on-agent-rag --query "..."`
  - Set env var `TURN_ON_AGENT_RAG=true` or pass `--turn-on-agent-rag`.
  - The entire pipeline (input safety, routing, ReAct agent, generation, output safety) runs as a single checkpointed LangGraph `StateGraph`.
  - Conversation state is persisted via `SqliteSaver` to `logs/agent_checkpoints.sqlite` (override with `--checkpoint-db`).
  - Pass `--thread-id <id>` to continue a previous conversation; omitted, a new UUID is auto-generated.
  - `related` queries are handled by a LangGraph ReAct agent that calls MCP tools: `retrieve_context` (FAISS), `rerank`, and `web_search`.
  - `query_user_data` is registered as a draft MCP server but is not wired into the active tool loop.
  - Requires `TAVILY_API_KEY` for the external web search tool.

## Two routing systems (do not conflate)


| Router              | File                                                         | Mechanism                                         | Labels                                                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `QueryRouter`       | `pipelines/indexing_pipeline/query_router.py`                | regex greeting check → FAISS similarity threshold | `greeting` / `off_topic` / `retrieval`                                                  |
| `PromptQueryRouter` | `pipelines/agent_pipeline/routers/routing_classification.py` | LLM prompt classification                         | `greeting` / `related` / `off_topic` (mapped to `greeting` / `retrieval` / `off_topic`) |


A third runtime route, `rejected`, is produced by the input safety/PII guard when `--enable-input-guard` is used.

If asked to change routing behavior, confirm which router.

## Commands

```bash
# Index a PDF into FAISS
python -m pipelines.indexing_pipeline index --pdf path/to.pdf --index-dir storage

# Query FAISS
python -m pipelines.indexing_pipeline query --index-dir storage --query "..."

# Generate scope summary from metadata
python -m pipelines.indexing_pipeline scope --metadata-path storage/metadata.json

# Run prompt-based pipeline with Pinecone retrieval
python -m pipelines.agent_pipeline.orchestration run \
  --query "..." \
  --pinecone-index-name your-index \
  [--pinecone-namespace default]

# With reranking
python -m pipelines.agent_pipeline.orchestration run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-rerank

# With LLM-as-a-judge over reranked chunks
python -m pipelines.agent_pipeline.orchestration run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-rerank \
  --enable-judge

# With input/output safety + PII guards (requires Presidio services + guardrails)
python -m pipelines.agent_pipeline.orchestration run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-input-guard \
  --enable-output-guard

# With LangSmith tracing
python -m pipelines.agent_pipeline.orchestration run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-langsmith \
  [--langsmith-project my-project]

# With custom inference provider
python -m pipelines.agent_pipeline.orchestration run \
  --query "..." \
  --pinecone-index-name your-index \
  --hf-inference-provider together

# Sync FAISS vectors to Pinecone
python vector_databases/pinecone_sync.py \
  --pinecone-index-name your-index \
  [--index-dir storage] \
  [--namespace default] \
  [--cloud aws] \
  [--region us-east-1] \
  [--batch-size 100]

# Streamlit chunk explorer
streamlit run app/app.py

# Compare retriever vs reranker scores
python rerank/evaluate_retriever_node.py --query "..." --pinecone-index-name your-index

# Agentic RAG with FAISS retrieval and Tavily web search
python -m pipelines.agent_pipeline.orchestration run \
  --turn-on-agent-rag \
  --query "..." \
  --tavily-api-key "$TAVILY_API_KEY"

# Agentic RAG with safety guards enabled
python -m pipelines.agent_pipeline.orchestration run \
  --turn-on-agent-rag \
  --enable-input-guard \
  --enable-output-guard \
  --query "..." \
  --tavily-api-key "$TAVILY_API_KEY"

# Agentic RAG with conversation compaction (summarizes old turns when context grows)
python -m pipelines.agent_pipeline.orchestration run \
  --turn-on-agent-rag \
  --enable-conversation-compaction \
  --context-token-threshold-pct 0.30 \
  --min-keep-recent-turns 1 \
  --query "..."

# Run MCP servers individually for debugging
python mcp_servers/retrieve_context_server.py
python mcp_servers/rerank_server.pypro
python mcp_servers/web_search_server.py
python mcp_servers/query_user_data_server.py
```

## Verification

```bash
# Syntax smoke
python -m compileall pipelines/indexing_pipeline pipelines/agent_pipeline mcp_servers vector_databases app

# CLI help
python -m pipelines.indexing_pipeline --help
python -m pipelines.agent_pipeline.orchestration --help
python -m pipelines.agent_pipeline.orchestration run --help
python -m pipelines.agent_pipeline.pipeline --help
python vector_databases/pinecone_sync.py --help

# Hello-world routing test (no Pinecone needed)
python -m pipelines.indexing_pipeline query --index-dir storage --query "hello"

# Prompt routing test (needs HF_TOKEN)
python test/test_pipeline.py --query "hello"
```

`test/test_pipeline.py` is a CLI harness (subprocess wrapper), not a unit test suite. No pytest or real automated tests exist.

## Architecture

- `pipelines/indexing_pipeline/` — FAISS ingestion, embedding, retrieval, query router, scope gen
- `pipelines/agent_pipeline/` — LangChain prompt-based routing + retrieval + reranking + safety/PII guardrails + optional LangSmith tracing
  - `orchestration.py` — dispatcher that selects traditional or agentic RAG based on `TURN_ON_AGENT_RAG`
  - `traditional_rag/` — existing deterministic Pinecone-backed pipeline
    - `routers/` — `PromptQueryRouter` and response nodes (`GreetingNode`, `OffTopicNode`)
    - `retriever/` — Pinecone-backed `RetrieverNode`
    - `rerank/` — HuggingFace Hub reranker wrapper
    - `retriever_judge/` — LLM-as-a-judge scoring node
  - `agent_rag/` — LangGraph ReAct agentic RAG
    - `agent_graph.py` — full-pipeline `StateGraph` (input safety → compaction → router → ReAct agent → judge → generation → output safety) checkpointed with `SqliteSaver`
    - `conversation_compaction.py` — `ConversationCompactionNode` that summarizes older turns when context exceeds a token threshold; also provides `append_turn_to_file()` for continuous per-thread logging
    - `token_utils.py` — `AutoTokenizer`-based token counting with word-count fallback
    - `tool_client.py` — `MultiServerMCPClient` configuration for stdio MCP servers
    - `guardtool.py` — wraps every MCP tool call with `SafetyInputNode` + `SafetyOutputNode`
  - `commons/` — shared `SafetyInputNode`, `SafetyOutputNode`, `RejectedNode`
  - `safety_input_nodes.py` / `safety_output_nodes.py` / `rejected_nodes.py` — moved to `commons/`
- `mcp_servers/` — FastMCP servers for agentic RAG
  - `retrieve_context_server.py` — FAISS retrieval
  - `rerank_server.py` — cross-encoder rerank
  - `web_search_server.py` — Tavily web search
  - `query_user_data_server.py` — draft / not wired into the active agent loop
- `pipelines/agent_pipeline/shared/` — shared guardrails, safety nodes, and rejection node
  - `guardrails.py` — single re-export for `PresidioPIIGuard`, `GuardrailsSafetyGuard`, `PIIDetectionResult`, `SafetyCheckResult` (only file that imports from `commons.guardrails`)
  - `safety_input_nodes.py` / `safety_output_nodes.py` — input/output safety wrappers
  - `rejected_nodes.py` — refusal message node for failed safety checks
- `storage/faiss.index` + `storage/metadata.json` — local vector store (generated, do not edit)
- `results/scope_result_*.txt` — generated scope summaries (do not edit)
- `output_prompt.txt` — default rendered prompt snapshot from the `scope` command; path configurable via `--output-prompt-path`
- `logs/` — log files (`agent_pipeline.log`), conversation checkpoint DB (`agent_checkpoints.sqlite`), and per-thread conversation history (`conversation_history/{thread_id}.md`, appended on every turn)
- `vector_databases/pinecone_sync.py` — FAISS→Pinecone sync
- `rerank/` — HuggingFace Hub reranker (`BAAI/bge-reranker-v2-m3`) and evaluation
- `embedding_mmodel/` — untracked directory, do not remove
- `.old_artifacts/` — archived old code (e.g. `pipeline_v2.py`)

## Models (all via HuggingFace InferenceClient with provider)

- Embedding: `BAAI/bge-m3` (via `huggingface_hub.InferenceClient` with optional inference provider)
- LLM: `google/gemma-4-26B-A4B-it` (via `huggingface_hub.InferenceClient` with `provider` parameter)
- Reranker: `BAAI/bge-reranker-v2-m3` (via HF Inference API, raw HTTP)

All LLM calls use `InferenceClient.chat_completion()` with configurable provider (default: `scaleway`).

## Secrets

- `HF_TOKEN` — required for any LLM/routing/reranker call
- `HF_INFERENCE_PROVIDER` — optional; specifies which HuggingFace Inference Provider to use (e.g. `scaleway`, `together`, `fireworks-ai`). Defaults to `scaleway`. Can also be passed via `--hf-inference-provider` CLI flag.
- `PINECONE_API_KEY` — required for Pinecone operations
- `TAVILY_API_KEY` — required for the `web_search` MCP server when agentic RAG is enabled
- `TURN_ON_AGENT_RAG` — optional env flag; set to `true` to default to agentic RAG mode
- `LANGCHAIN_API_KEY`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT` — optional, read from `.env` by the agent pipeline; only used when `--enable-langsmith` is passed

Resolution order: process env → `.env` file in repo root.

## Footguns

- `pipelines/agent_pipeline/routers/routing_response.py:13` and `pipelines/agent_pipeline/pipeline.py:112` hardcode a specific scope file path (`results/scope_result_20260606_193507.txt`). If regenerated scope changes name, these break unless `--scope` / `--scope-file` is passed.
- Prompt templates use `str.format()` placeholders `{scope}`, `{user_question}`, `{context}`. Any prompt edit must preserve these.
- README's venv activation shows Windows syntax; this is a Linux environment.
- FAISS uses inner product on L2-normalized vectors (cosine-style similarity).
- Greeting detection in `QueryRouter` is regex-based and intentionally narrow.
- The Guardrails safety validators (especially `NSFWText`) can be noisy on long LLM-generated responses. The default `NSFWText` threshold is raised to 0.95 in the pipeline CLI; use `--safety-nsfw-threshold` to tune, or `--safety-toxic-threshold` for `ToxicLanguage`.
- `requirements.txt` previously omitted `loguru` and `langsmith`; both are now listed and required by `pipelines/agent_pipeline/pipeline.py`.
- `diagrams/agent_graph.md` describes a more advanced aspirational graph (memory, generation, verification) than the current `pipeline.py` implementation. The implemented pipeline only has router → greeting/off_topic/retrieval → optional rerank/judge/safety guards.
- The old `pipelines/agent_pipeline/pipeline_v2.py` has been moved to `.old_artifacts/pipeline_v2.py`; use the merged `pipeline.py` with `--enable-langsmith` instead.
- The agentic RAG path requires the MCP server scripts (`mcp_servers/*.py`) to be runnable as separate stdio processes. Ensure `python` is on `PATH` and the working directory is the repo root.
- `TAVILY_API_KEY` must be set for `web_search` to return real results; without it the server will raise a clear error.
- `query_user_data` MCP server is a draft and is intentionally omitted from the LangGraph tool list.
- Tool outputs from `web_search` are guarded by `SafetyOutputNode`; if a violation is detected the tool result is replaced by a refusal message.

## Graph diagram

See `diagrams/agent_graph.md` for the aspirational LangGraph agent flow and `chat_pipeline_flow.md` for the full ChatPipeline graph. The currently implemented flow is: input_safety → conversation_compaction → router → greeting/off_topic/agent (ReAct loop) → process_results → generate → output_safety.

## Conversation compaction

When `--enable-conversation-compaction` is passed, the pipeline inserts a `ConversationCompactionNode` between `input_safety` and `router`. On every turn it:

1. Counts tokens of the accumulated `messages` using `AutoTokenizer` for the configured LLM (fallback: word-count heuristic).
2. If total ≤ `--context-token-threshold-pct` × `--max-input-tokens` → no-op.
3. If total > threshold → walk backwards to find a split point that keeps at least `--min-keep-recent-turns` exchanges unsummarized, while pushing the total live context (summary + unsummarized) under the threshold.
4. Saves the full original conversation to `logs/conversation_history/{thread_id}.md`.
5. Calls the LLM with `prompts/summarization/history_context_summarization.txt` to produce a structured summary.
6. Replaces the old messages with `RemoveMessage` entries + a single `SystemMessage` containing the summary and a pointer to the saved file.

The `InferenceClientChatAdapter` was extended to handle `SystemMessage` → `role: "system"` so the summary is passed as system context to the LLM.

A final `conversation_history_logger` node (after `output_safety`) appends each turn's user/assistant/system messages to `logs/conversation_history/{thread_id}.md` on every turn, not just when compaction triggers.