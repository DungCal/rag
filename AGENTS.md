# AGENTS.md

## Entrypoints

- **FAISS-based (local):** `python -m pipelines.indexing_pipeline {index,query,scope}`
- **Prompt-based + Pinecone (remote):** `python -m pipelines.agent_pipeline.pipeline run --pinecone-index-name <name>`
  - Also runnable as a script: `python pipelines/agent_pipeline/pipeline.py run --pinecone-index-name <name>`
  - The pipeline module also re-exports `index`, `query`, `scope` subcommands (same CLI as indexing pipeline)
- **LangSmith variant:** `python -m pipelines.agent_pipeline.pipeline run --enable-langsmith --pinecone-index-name <name>`
  - Tracing is off by default; pass `--enable-langsmith` to send LLM spans to LangSmith.
  - The old `pipeline_v2.py` has been moved to `.old_artifacts/`.

## Two routing systems (do not conflate)

| Router | File | Mechanism | Labels |
|---|---|---|---|
| `QueryRouter` | `pipelines/indexing_pipeline/query_router.py` | regex greeting check → FAISS similarity threshold | `greeting` / `off_topic` / `retrieval` |
| `PromptQueryRouter` | `pipelines/agent_pipeline/routers/routing_classification.py` | LLM prompt classification | `greeting` / `related` / `off_topic` (mapped to `greeting` / `retrieval` / `off_topic`) |

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
python -m pipelines.agent_pipeline.pipeline run \
  --query "..." \
  --pinecone-index-name your-index \
  [--pinecone-namespace default]

# With reranking
python -m pipelines.agent_pipeline.pipeline run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-rerank

# With LLM-as-a-judge over reranked chunks
python -m pipelines.agent_pipeline.pipeline run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-rerank \
  --enable-judge

# With input/output safety + PII guards (requires Presidio services + guardrails)
python -m pipelines.agent_pipeline.pipeline run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-input-guard \
  --enable-output-guard

# With LangSmith tracing
python -m pipelines.agent_pipeline.pipeline run \
  --query "..." \
  --pinecone-index-name your-index \
  --enable-langsmith \
  [--langsmith-project my-project]

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
```

## Verification

```bash
# Syntax smoke
python -m compileall pipelines/indexing_pipeline pipelines/agent_pipeline vector_databases app

# CLI help
python -m pipelines.indexing_pipeline --help
python -m pipelines.agent_pipeline.pipeline --help
python -m pipelines.agent_pipeline.pipeline run --help
python vector_databases/pinecone_sync.py --help

# Hello-world routing test (no Pinecone needed)
python -m pipelines.indexing_pipeline query --index-dir storage --query "hello"

# Prompt routing test (needs HF_TOKEN)
python test/test_pipeline.py --query "hello"
```

`test/test_pipeline.py` is a CLI harness (subprocess wrapper), not a unit test suite. No pytest or real automated tests exist.

## Architecture

- `pipelines/indexing_pipeline/` — FAISS ingestion, embedding, retrieval, query router, scope gen
- `pipelines/agent_pipeline/` — LangChain prompt-based routing + Pinecone retrieval + reranking + safety/PII guardrails + optional LangSmith tracing
  - `routers/` — `PromptQueryRouter` and response nodes (`GreetingNode`, `OffTopicNode`)
  - `retriever/` — Pinecone-backed `RetrieverNode`
  - `rerank/` — HuggingFace Hub reranker wrapper
  - `retriever_judge/` — LLM-as-a-judge scoring node
  - `safety_input_nodes.py` / `safety_output_nodes.py` — Presidio + Guardrails safety wrappers
  - `rejected_nodes.py` — refusal message node for failed safety checks
- `commons/guardrails/` — reusable Presidio PII and Guardrails safety wrappers
- `storage/faiss.index` + `storage/metadata.json` — local vector store (generated, do not edit)
- `results/scope_result_*.txt` — generated scope summaries (do not edit)
- `output_prompt.txt` — default rendered prompt snapshot from the `scope` command; path configurable via `--output-prompt-path`
- `logs/` — log files written by the agent pipeline (`agent_pipeline.log`)
- `vector_databases/pinecone_sync.py` — FAISS→Pinecone sync
- `rerank/` — HuggingFace Hub reranker (`BAAI/bge-reranker-v2-m3`) and evaluation
- `embedding_mmodel/` — untracked directory, do not remove
- `.old_artifacts/` — archived old code (e.g. `pipeline_v2.py`)

## Models (all via HuggingFace Hub)

- Embedding: `BAAI/bge-m3` (via `huggingface_hub.InferenceClient` with optional inference provider)
- LLM: `google/gemma-4-26B-A4B-it` (via `langchain-huggingface`)
- Reranker: `BAAI/bge-reranker-v2-m3` (via HF Inference API, raw HTTP)

## Secrets

- `HF_TOKEN` — required for any LLM/routing/reranker call
- `PINECONE_API_KEY` — required for Pinecone operations
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

## Graph diagram

See `diagrams/agent_graph.md` for the aspirational LangGraph agent flow and `chat_pipeline_flow.md` for the full ChatPipeline graph. The currently implemented flow is simpler: router → greeting/off_topic/retrieval with optional rerank, judge, and safety guards.
