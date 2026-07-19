# Assistant Context

## Purpose

This repository is a small PDF-focused RAG prototype with two routing layers:

1. An embedding-based router used by the main CLI in `pipelines/indexing_pipeline/indexing.py` (run via `python -m pipelines.indexing_pipeline`).
2. An LLM prompt-based router and response nodes used by `pipelines/agent_pipeline/pipeline.py` (run via `python -m pipelines.agent_pipeline.pipeline`). The pipeline now dispatches to either a traditional deterministic RAG path or a new agentic RAG path controlled by `TURN_ON_AGENT_RAG`.

The indexed document in the current workspace is a TYM tractor operator manual (`t130sp_na_operator_manual.pdf`). Most prompts, defaults, and example scope text assume that document.

## Main Workflows

### 1. Build or rebuild the vector index

Entry point: `pipelines/indexing_pipeline/indexing.py` (module entrypoint `python -m pipelines.indexing_pipeline`)

Command shape:

```bash
python -m pipelines.indexing_pipeline index --pdf t130sp_na_operator_manual.pdf --index-dir storage
```

What it does:

- Extracts page text from a PDF with PyMuPDF.
- Chunks text with LangChain `RecursiveCharacterTextSplitter`.
- Embeds chunks with `BAAI/bge-m3` via Hugging Face Hub Inference API (or an optional inference provider).
- L2-normalizes vectors and stores them in a FAISS inner-product index.
- Writes chunk metadata to `storage/metadata.json`.

### 2. Query the FAISS-backed RAG pipeline

Entry point: `pipelines/indexing_pipeline/indexing.py` (module entrypoint `python -m pipelines.indexing_pipeline`)

Command shape:

```bash
python -m pipelines.indexing_pipeline query --index-dir storage --query "What does the DPF warning lamp mean?"
```

Optional answer generation:

```bash
python -m pipelines.indexing_pipeline query --index-dir storage --query "..." --generate-answer
```

Behavior:

- Embeds the query with the same BGE-M3 model.
- Uses `pipelines/indexing_pipeline/query_router.py` for lightweight routing:
  - `greeting` if the query is a simple greeting.
  - `off_topic` if top similarity is below the threshold.
  - `retrieval` otherwise.
- On `retrieval`, fetches top-k chunks from FAISS.
- If `--generate-answer` is enabled, sends retrieved chunks to a Hugging Face hosted LLM.

### 3. Run the prompt-based routing pipeline

Entry point: `pipelines/agent_pipeline/orchestration.py` (module entrypoint `python -m pipelines.agent_pipeline.orchestration`)

Command shape:

```bash
python -m pipelines.agent_pipeline.orchestration run --query "hello" --as-json
```

With Pinecone retrieval:

```bash
python -m pipelines.agent_pipeline.orchestration run \
  --query "What does the DPF warning lamp mean?" \
  --pinecone-index-name your-index
```

Optional add-ons:

- `--enable-rerank` — rerank Pinecone matches with `BAAI/bge-reranker-v2-m3`.
- `--enable-judge` — score retrieved/reranked chunks with an LLM-as-a-judge.
- `--enable-input-guard` / `--enable-output-guard` — Presidio PII + Guardrails safety checks.
- `--enable-langsmith` / `--langsmith-project` — send LLM spans to LangSmith.
- `--turn-on-agent-rag` — switch to the LangGraph ReAct agentic RAG path.
- `--tavily-api-key` — Tavily API key for agentic web search.

Behavior:

- Uses `pipelines/agent_pipeline/routers/routing_classification.py` to classify into `greeting`, `related`, or `off_topic`.
- Maps `related` → route `retrieval`, `greeting` → route `greeting`, `off_topic` → route `off_topic`.
- Runs prompt-driven response nodes for `greeting` and `off_topic`.
- Runs Pinecone-backed retrieval for `related` by embedding the retrieval query with the same BGE-M3 path used in `pipelines/indexing_pipeline/pdf_rag.py`.
- Optionally reranks, judges, and/or runs safety guards.

### 4. Sync FAISS vectors into Pinecone

Entry point: `vector_databases/pinecone_sync.py`

Command shape:

```bash
python vector_databases/pinecone_sync.py --pinecone-index-name your-index
```

Full flags:

```bash
python vector_databases/pinecone_sync.py \
  --pinecone-index-name your-index \
  --index-dir storage \
  --namespace default \
  --cloud aws \
  --region us-east-1 \
  --batch-size 100
```

Behavior:

- Reconstructs dense vectors from `storage/faiss.index`.
- Loads aligned chunk metadata from `storage/metadata.json`.
- Creates a Pinecone serverless index if it does not already exist.
- Upserts vectors and chunk metadata into the configured Pinecone namespace.

Important:

- Pinecone data operations are performed against an index. Collections are a different Pinecone primitive and are not used here for querying or upserts.

### 5. Generate or refresh a document scope summary

Entry point: `pipelines/indexing_pipeline/scope.py` (module entrypoint `python -m pipelines.indexing_pipeline scope`)

Command shape:

```bash
python -m pipelines.indexing_pipeline scope --metadata-path storage/metadata.json
```

Behavior:

- Samples chunk records from `storage/metadata.json`.
- Renders `prompts/prompt_scope.txt`.
- Calls the Hugging Face LLM.
- Writes a rendered prompt snapshot to `output_prompt.txt` by default (configurable via `--output-prompt-path`).
- Writes a text summary into `results/scope_result_YYYYMMDD_HHMMSS.txt`.

This scope output is used as input for the prompt-based router and response nodes.

### 6. Explore chunk metadata visually

Entry point: `app/app.py`

Command shape:

```bash
streamlit run app/app.py
```

Behavior:

- Loads `storage/metadata.json` and `storage/faiss.index`.
- Shows chunk statistics, length distributions, and embedding-space projections.
- Includes a raw PyMuPDF chunk analyzer page.

### 7. Run the agentic RAG pipeline

Entry point: `pipelines/agent_pipeline/orchestration.py` (module entrypoint `python -m pipelines.agent_pipeline.orchestration`)

Command shape:

```bash
python -m pipelines.agent_pipeline.orchestration run \
  --turn-on-agent-rag \
  --query "What does the DPF warning lamp mean?"
```

With Tavily API key:

```bash
python -m pipelines.agent_pipeline.orchestration run \
  --turn-on-agent-rag \
  --query "..." \
  --tavily-api-key "$TAVILY_API_KEY"
```

Behavior:

- `orchestration.py` reads the `TURN_ON_AGENT_RAG` flag (CLI flag overrides env var).
- If `false`, it delegates to the existing deterministic pipeline in `traditional_rag/`.
- If `true`:
  - Runs `SafetyInputNode`. If rejected, returns the `RejectedNode` refusal.
  - Runs `PromptQueryRouter` (greeting / related / off_topic).
  - `greeting` and `off_topic` go to their response nodes, then `SafetyOutputNode`.
  - `related` goes to the LangGraph agent node.
  - The agent loop calls MCP tools wrapped by `GuardTool`:
    - `retrieve_context` (FAISS-based, from `storage/faiss.index`)
    - `rerank`
    - `web_search` (Tavily)
    - `query_user_data` is **not** exposed to the agent.
  - Every tool input is guarded by `SafetyInputNode` and every tool output by `SafetyOutputNode`.
  - After tool observations, the agent generates a final answer.
  - The final answer passes through `SafetyOutputNode`.

## Code Map

- `pipelines/indexing_pipeline/indexing.py`: main CLI for indexing and querying.
- `pipelines/indexing_pipeline/pdf_rag.py`: PDF extraction, whitespace normalization, chunking, embedding.
- `pipelines/indexing_pipeline/index_store.py`: FAISS persistence and metadata JSON serialization.
- `pipelines/indexing_pipeline/retriever.py`: top-k retrieval result assembly.
- `pipelines/indexing_pipeline/llm.py`: Hugging Face answer-generation wrapper and `.env` token loading.
- `pipelines/indexing_pipeline/scope.py`: scope-summary generation from sampled chunk metadata.
- `pipelines/indexing_pipeline/query_router.py`: embedding-similarity router for the main CLI.
- `pipelines/agent_pipeline/orchestration.py`: main CLI entrypoint and dispatcher that selects traditional or agentic RAG based on `TURN_ON_AGENT_RAG`.
- `pipelines/agent_pipeline/pipeline.py`: thin backward-compatible alias; delegates to `orchestration.py`.
- `pipelines/agent_pipeline/traditional_rag/pipeline.py`: existing deterministic routed pipeline (Pinecone retrieval).
- `pipelines/agent_pipeline/traditional_rag/routers/routing_classification.py`: LLM classification router using `prompts/route_node_prompt.txt`.
- `pipelines/agent_pipeline/traditional_rag/routers/routing_response.py`: greeting and off-topic node response generation.
- `pipelines/agent_pipeline/traditional_rag/retriever/retriever_node.py`: Pinecone-backed retrieval node for `related` route decisions.
- `pipelines/agent_pipeline/traditional_rag/rerank/rerank_node.py`: HuggingFace Hub reranker wrapper.
- `pipelines/agent_pipeline/traditional_rag/retriever_judge/retriever_judge_node.py`: LLM-as-a-judge relevance scoring.
- `pipelines/agent_pipeline/agent_rag/agent_graph.py`: LangGraph ReAct agent node + final answer generation.
- `pipelines/agent_pipeline/agent_rag/tool_client.py`: `MultiServerMCPClient` configuration for stdio MCP servers.
- `pipelines/agent_pipeline/agent_rag/guardtool.py`: wraps tool calls with `SafetyInputNode` and `SafetyOutputNode`.
- `pipelines/agent_pipeline/shared/`: shared safety nodes and rejection node.
- `mcp_servers/retrieve_context_server.py`: FastMCP server for FAISS retrieval.
- `mcp_servers/rerank_server.py`: FastMCP server for cross-encoder reranking.
- `mcp_servers/web_search_server.py`: FastMCP server wrapping Tavily web search.
- `mcp_servers/query_user_data_server.py`: draft FastMCP server for user data (not wired into the agent).
- `commons/guardrails/pii_presidio.py`: Presidio analyzer/anonymizer HTTP client.
- `commons/guardrails/guardrailsAI_safetycheck.py`: Guardrails Hub safety validators.
- `vector_databases/pinecone_sync.py`: FAISS-to-Pinecone sync script.
- `rerank/hf_reranker.py`: raw HTTP reranker client.
- `rerank/evaluate_retriever_node.py`: retriever vs reranker comparison harness.
- `app/app.py`: Streamlit inspection app for metadata and embeddings.
- `test/test_pipeline.py`: manual CLI-style pipeline test harness that executes `pipeline.py`, not a real unit test suite.
- `.old_artifacts/pipeline_v2.py`: archived LangSmith-only pipeline variant (functionality merged into `pipeline.py`).

## Models And External Dependencies

Embedding model:

- `BAAI/bge-m3`

Generation / routing model default:

- `google/gemma-4-26B-A4B-it`

Reranker model default:

- `BAAI/bge-reranker-v2-m3`

Libraries that matter most:

- `faiss-cpu`
- `PyMuPDF`
- `langchain-huggingface`
- `langchain-text-splitters`
- `pinecone`
- `huggingface_hub`
- `requests`
- `streamlit`
- `plotly`
- `umap-learn`
- `scikit-learn`
- `guardrails-ai`
- `loguru`
- `langsmith`
- `mcp` (FastMCP server SDK)
- `langchain-mcp-adapters` (converts MCP tools to LangChain tools)
- `langgraph` (ReAct agent graph)
- `tavily-python` (web search provider)

## Secrets And Environment

This repo expects a Hugging Face token in `HF_TOKEN`.

Pinecone-backed features expect a Pinecone API key in `PINECONE_API_KEY`.

Tavily-backed web search expects `TAVILY_API_KEY`.

`TURN_ON_AGENT_RAG` env flag selects agentic RAG mode when set to `true`.

LangSmith tracing (only active with `--enable-langsmith`) expects `LANGCHAIN_API_KEY`, `LANGCHAIN_ENDPOINT`, and `LANGCHAIN_PROJECT` in `.env` or the process environment.

Resolution order in code:

1. Process environment variable
2. `.env` file in the repo root

Relevant code:

- `pipelines/indexing_pipeline/llm.py`
- `pipelines/agent_pipeline/traditional_rag/retriever/retriever_node.py`
- `pipelines/agent_pipeline/traditional_rag/routers/routing_classification.py`
- `pipelines/agent_pipeline/traditional_rag/routers/routing_response.py`
- `pipelines/agent_pipeline/pipeline.py`
- `pipelines/agent_pipeline/orchestration.py`
- `pipelines/agent_pipeline/agent_rag/tool_client.py`
- `mcp_servers/web_search_server.py`
- `vector_databases/pinecone_sync.py`

If LLM-backed commands fail, verify `HF_TOKEN`.
If Pinecone-backed commands fail, verify `PINECONE_API_KEY`.
If LangSmith tracing fails, verify `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT`.

## Data Artifacts

Current generated artifacts:

- `storage/faiss.index`
- `storage/metadata.json`
- `results/scope_result_*.txt`
- `output_prompt.txt`
- `logs/agent_pipeline.log`

Remote vector storage, when configured:

- Pinecone index named via `--pinecone-index-name`
- Pinecone namespace, default `default`

Treat these as generated outputs unless the task is specifically about inspecting or improving them.

## Important Implementation Details

- Retrieval uses cosine-style similarity implemented as inner product on L2-normalized vectors.
- `QueryRouter` in `pipelines/indexing_pipeline/query_router.py` always embeds first, then checks greeting/off-topic routing.
- Greeting detection in the embedding-based router is regex-based and intentionally narrow.
- The prompt-based router expects the model to output exactly one label: `greeting`, `related`, or `off_topic`.
- `pipeline.py` routes `related` decisions into Pinecone retrieval through `retriever/retriever_node.py`.
- `pipeline.py` can also optionally rerank, judge, and apply input/output safety guards.
- Prompt templates rely on Python `str.format(...)` placeholders such as `{scope}`, `{user_question}`, `{context}`, and `{retrieved_chunk}`. Any prompt edit must preserve those placeholders.

## Repo-Specific Footguns

- `test/test_pipeline.py` is a runnable script, not a proper automated test module.
- Several defaults hardcode a specific generated scope file:
  - `pipelines/agent_pipeline/pipeline.py`
  - `pipelines/agent_pipeline/traditional_rag/routers/routing_response.py`
  - `pipelines/agent_pipeline/agent_rag/agent_graph.py`
- If that file is deleted or renamed, prompt-based routing can break unless `--scope` or `--scope-file` is provided.
- `pipelines/indexing_pipeline/scope.py` writes `output_prompt.txt` in the repo root by default, unless `--output-prompt-path` is provided.
- The README uses Windows-style virtualenv activation examples; the current environment here is Linux.
- The repo currently has an untracked `embedding_mmodel/` directory. Do not remove or rewrite it unless the task is explicitly about that folder.
- The user request called Pinecone storage a "collection", but the implemented path uses a Pinecone index because that is the queryable data-plane primitive in the current SDK.
- `pipelines/agent_pipeline/pipeline_v2.py` has been archived to `.old_artifacts/pipeline_v2.py`; use `pipeline.py --enable-langsmith` instead.

## Editing Guidance

- Preserve the separation between:
  - embedding-based routing in `pipelines/indexing_pipeline/query_router.py`
  - prompt-based routing in `pipelines/agent_pipeline/traditional_rag/routers/routing_classification.py`
- Keep the Pinecone retrieval path isolated in `pipelines/agent_pipeline/traditional_rag/retriever/retriever_node.py` and the ingestion path isolated in `vector_databases/pinecone_sync.py`.
- Keep FAISS retrieval for agentic RAG isolated in `mcp_servers/retrieve_context_server.py`.
- Keep MCP tool guards isolated in `pipelines/agent_pipeline/agent_rag/guardtool.py`.
- If changing chunking, embedding, or metadata shape, verify all downstream readers:
  - `pipelines/indexing_pipeline/index_store.py`
  - `pipelines/agent_pipeline/retriever/retriever_node.py`
  - `pipelines/indexing_pipeline/scope.py`
  - `app/app.py`
  - `vector_databases/pinecone_sync.py`
- If changing prompt files, verify the corresponding router/node code still formats them correctly.
- If changing defaults for scope handling, update:
  - `pipelines/agent_pipeline/pipeline.py`
  - `pipelines/agent_pipeline/traditional_rag/routers/routing_response.py`
  - `pipelines/agent_pipeline/agent_rag/agent_graph.py`
- If adding LangSmith tracing behavior, prefer the `--enable-langsmith` flag in `pipeline.py` rather than reviving `pipeline_v2.py`.
- Prefer small, explicit changes. This repo is compact, and regressions usually come from changing shared assumptions rather than from deep abstraction issues.

## Verification Checklist

After meaningful changes, use the smallest relevant check:

- Syntax smoke test:

```bash
python -m compileall pipelines/indexing_pipeline pipelines/agent_pipeline mcp_servers vector_databases app
```

- CLI help:

```bash
python -m pipelines.indexing_pipeline --help
python -m pipelines.agent_pipeline.orchestration --help
python -m pipelines.agent_pipeline.orchestration run --help
python -m pipelines.agent_pipeline.pipeline --help
python -m pipelines.indexing_pipeline scope --help
python vector_databases/pinecone_sync.py --help
```

- If indexing or retrieval changed:

```bash
python -m pipelines.indexing_pipeline query --index-dir storage --query "hello"
```

- If prompt-based routing changed, run:

```bash
python test/test_pipeline.py --query "hello"
```

- If LangSmith tracing changed:

```bash
python -m pipelines.agent_pipeline.orchestration run --query "hello" --enable-langsmith --as-json
```

- If agentic RAG changed:

```bash
python -m pipelines.agent_pipeline.orchestration run --turn-on-agent-rag --query "hello" --as-json
```

Note: any LLM-backed runtime check requires a valid `HF_TOKEN` and network access to Hugging Face services.
Note: any Pinecone-backed runtime check requires a valid `PINECONE_API_KEY` and network access to Pinecone.
Note: LangSmith checks require `LANGCHAIN_API_KEY` and a reachable LangSmith endpoint.

## Task Intake Template

Use this structure when starting a new implementation task:

### Task
What should be changed.

### Goal
What should be true when the task is complete.

### Scope
Which files, folders, or subsystems are in play.

### Constraints
What must not change.

### Verification
Which commands or checks prove the change works.

### Notes
Any special context, assumptions, or risks.

## Current Task Intake

### Task
Refactor `pipelines/agent_pipeline` into traditional and agentic RAG packages, add an `orchestration.py` dispatcher driven by `TURN_ON_AGENT_RAG`, implement four FastMCP servers (`retrieve_context`, `rerank`, `web_search`, `query_user_data`), and wire the agentic path through a LangGraph ReAct agent where every tool call is guarded by `SafetyInputNode`/`SafetyOutputNode`.

### Goal
- The existing deterministic pipeline is preserved under `traditional_rag/`.
- A new agentic RAG path works under `agent_rag/` with FAISS retrieval, reranking, Tavily web search, and guarded tool I/O.
- `pipeline.py` remains the CLI entrypoint and delegates to `orchestration.py`.
- `AGENTS.md` and `assistant-context.md` reflect the new structure.
- `requirements.txt` includes `mcp`, `langchain-mcp-adapters`, `langgraph`, and `tavily-python`.

### Scope
- `AGENTS.md`, `assistant-context.md`
- `requirements.txt`
- `pipelines/agent_pipeline/pipeline.py`
- `pipelines/agent_pipeline/orchestration.py` (new)
- `pipelines/agent_pipeline/shared/` (new, moved from top-level safety/rejected nodes)
- `pipelines/agent_pipeline/traditional_rag/` (new package, existing logic moved)
- `pipelines/agent_pipeline/agent_rag/` (new package)
- `mcp_servers/` (new package)

### Constraints
- Do not change the FAISS indexing or Pinecone sync logic.
- Do not remove or alter the untracked `embedding_mmodel/` folder.
- Keep existing prompt placeholders (`{scope}`, `{user_question}`, `{context}`).
- Preserve the old CLI commands (`run`, `index`, `query`, `scope`) at `pipeline.py`.
- `query_user_data` remains a draft and must not be added to the agent tool list.

### Verification
- `python -m compileall pipelines/indexing_pipeline pipelines/agent_pipeline mcp_servers vector_databases app`
- `python -m pipelines.agent_pipeline.orchestration --help`
- `python -m pipelines.agent_pipeline.orchestration run --help`
- `python -m pipelines.agent_pipeline.orchestration run --query "hello" --as-json` (greeting route smoke)
- `python -m pipelines.agent_pipeline.orchestration run --turn-on-agent-rag --query "hello" --as-json` (agentic greeting route smoke)
- `python mcp_servers/retrieve_context_server.py --help`
- `python mcp_servers/web_search_server.py --help`

### Notes
- MCP servers run as separate stdio processes; the agent spawns them via `MultiServerMCPClient`.
- `TAVILY_API_KEY` is required for real web search.
- `HF_TOKEN` is required for routing and final answer generation.
- `PINECONE_API_KEY` is still required for the traditional RAG path.
