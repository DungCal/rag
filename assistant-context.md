# Assistant Context

## Purpose

This repository is a small PDF-focused RAG prototype with two routing layers:

1. An embedding-based router used by the main CLI in `pipelines/indexing_pipeline/indexing.py`.
2. An LLM prompt-based router and response nodes used by `pipeline.py` and `test/test_pipeline.py`.

The indexed document in the current workspace is a TYM tractor operator manual (`t130sp_na_operator_manual.pdf`). Most prompts, defaults, and example scope text assume that document.

## Main Workflows

### 1. Build or rebuild the vector index

Entry point: `pipelines/indexing_pipeline/indexing.py`

Command shape:

```bash
python -m pipelines.indexing_pipeline.indexing index --pdf t130sp_na_operator_manual.pdf --index-dir storage
```

What it does:

- Extracts page text from a PDF with PyMuPDF.
- Chunks text with LangChain `RecursiveCharacterTextSplitter`.
- Embeds chunks with `BAAI/bge-m3` via `FlagEmbedding`.
- L2-normalizes vectors and stores them in a FAISS inner-product index.
- Writes chunk metadata to `storage/metadata.json`.

### 2. Query the FAISS-backed RAG pipeline

Entry point: `pipelines/indexing_pipeline/indexing.py`

Command shape:

```bash
python -m pipelines.indexing_pipeline.indexing query --index-dir storage --query "What does the DPF warning lamp mean?"
```

Optional answer generation:

```bash
python -m pipelines.indexing_pipeline.indexing query --index-dir storage --query "..." --generate-answer
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

Entry point: `pipeline.py`

Command shape:

```bash
python pipeline.py --query "hello" --as-json --pinecone-index-name your-index
```

Behavior:

- Uses `routers/prompt_query_router.py` to classify into `greeting`, `related`, or `off_topic`.
- Runs prompt-driven response nodes for:
  - `greeting`
  - `off_topic`
- Runs Pinecone-backed retrieval for `related` by embedding the retrieval query with the same BGE-M3 path used in `pipelines/indexing_pipeline/pdf_rag.py`.

### 4. Sync FAISS vectors into Pinecone

Entry point: `vector_databases/pinecone_sync.py`

Command shape:

```bash
python vector_databases/pinecone_sync.py --pinecone-index-name your-index
```

Behavior:

- Reconstructs dense vectors from `storage/faiss.index`.
- Loads aligned chunk metadata from `storage/metadata.json`.
- Creates a Pinecone serverless index if it does not already exist.
- Upserts vectors and chunk metadata into the configured Pinecone namespace.

Important:

- Pinecone data operations are performed against an index. Collections are a different Pinecone primitive and are not used here for querying or upserts.

### 5. Generate or refresh a document scope summary

Entry point: `pipelines/indexing_pipeline/scope.py`

Command shape:

```bash
python -m pipelines.indexing_pipeline.scope --metadata-path storage/metadata.json
```

Behavior:

- Samples chunk records from `storage/metadata.json`.
- Renders `prompts/prompt_scope.txt`.
- Calls the Hugging Face LLM.
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
- Shows chunk statistics and distribution views.
- Includes embedding reconstruction and dimensionality reduction utilities.

## Code Map

- `pipelines/indexing_pipeline/indexing.py`: main CLI for indexing and querying.
- `pipelines/indexing_pipeline/pdf_rag.py`: PDF extraction, whitespace normalization, chunking, embedding.
- `pipelines/indexing_pipeline/index_store.py`: FAISS persistence and metadata JSON serialization.
- `pipelines/indexing_pipeline/retriever.py`: top-k retrieval result assembly.
- `pipelines/indexing_pipeline/llm.py`: Hugging Face answer-generation wrapper and `.env` token loading.
- `pipelines/agent_pipeline/retriever/retriever_node.py`: Pinecone-backed retrieval node for `related` route decisions.
- `pipelines/indexing_pipeline/query_router.py`: embedding-similarity router for the main CLI.
- `pipelines/agent_pipeline/routers/prompt_query_router.py`: LLM classification router using `prompts/route_node_prompt.txt`.
- `pipelines/agent_pipeline/routers/prompt_response_nodes.py`: greeting and off-topic node response generation.
- `pipelines/agent_pipeline/pipeline.py`: CLI wrapper around prompt-based routing plus Pinecone retrieval.
- `pipelines/indexing_pipeline/scope.py`: scope-summary generation from sampled chunk metadata.
- `vector_databases/pinecone_sync.py`: FAISS-to-Pinecone sync script.
- `app/app.py`: Streamlit inspection app for metadata and embeddings.
- `test/test_pipeline.py`: manual CLI-style pipeline test harness that executes `pipeline.py`, not a real unit test suite.

## Models And External Dependencies

Embedding model:

- `BAAI/bge-m3`

Generation / routing model default:

- `google/gemma-4-26B-A4B-it`

Libraries that matter most:

- `FlagEmbedding`
- `faiss-cpu`
- `PyMuPDF`
- `langchain-huggingface`
- `langchain-text-splitters`
- `streamlit`
- `plotly`
- `umap-learn`

## Secrets And Environment

This repo expects a Hugging Face token in `HF_TOKEN`.

Pinecone-backed features expect a Pinecone API key in `PINECONE_API_KEY`.

Resolution order in code:

1. Process environment variable
2. `.env` file in the repo root

Relevant code:

- `pipelines/indexing_pipeline/llm.py`
- `retriever/retriever_node.py`
- `routers/prompt_query_router.py`
- `routers/prompt_response_nodes.py`
- `vector_databases/pinecone_sync.py`

If LLM-backed commands fail, verify `HF_TOKEN`.
If Pinecone-backed commands fail, verify `PINECONE_API_KEY`.

## Data Artifacts

Current generated artifacts:

- `storage/faiss.index`
- `storage/metadata.json`
- `results/scope_result_*.txt`
- `output_prompt.txt`

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
- Prompt templates rely on Python `str.format(...)` placeholders such as `{scope}`, `{user_question}`, and `{context}`. Any prompt edit must preserve those placeholders.

## Repo-Specific Footguns

- `test/test_pipeline.py` is a runnable script, not a proper automated test module.
- Several defaults hardcode a specific generated scope file:
  - `pipeline.py`
  - `routers/prompt_response_nodes.py`
- If that file is deleted or renamed, prompt-based routing can break unless `--scope` or `--scope-file` is provided.
- `pipelines/indexing_pipeline/scope.py` writes `output_prompt.txt` in the repo root by default, unless `--output-prompt-path` is provided.
- The README uses Windows-style virtualenv activation examples; the current environment here is Linux.
- The repo currently has an untracked `embedding_mmodel/` directory. Do not remove or rewrite it unless the task is explicitly about that folder.
- The user request called Pinecone storage a "collection", but the implemented path uses a Pinecone index because that is the queryable data-plane primitive in the current SDK.

## Editing Guidance

- Preserve the separation between:
  - embedding-based routing in `pipelines/indexing_pipeline/query_router.py`
  - prompt-based routing in `pipelines/agent_pipeline/routers/prompt_query_router.py`
- Keep the Pinecone retrieval path isolated in `retriever/retriever_node.py` and the ingestion path isolated in `vector_databases/pinecone_sync.py`.
- If changing chunking, embedding, or metadata shape, verify all downstream readers:
  - `pipelines/indexing_pipeline/index_store.py`
  - `pipelines/agent_pipeline/retriever/retriever_node.py`
  - `pipelines/indexing_pipeline/scope.py`
  - `app/app.py`
  - `vector_databases/pinecone_sync.py`
- If changing prompt files, verify the corresponding router/node code still formats them correctly.
- If changing defaults for scope handling, update both:
  - `pipeline.py`
  - `routers/prompt_response_nodes.py`
- Prefer small, explicit changes. This repo is compact, and regressions usually come from changing shared assumptions rather than from deep abstraction issues.

## Verification Checklist

After meaningful changes, use the smallest relevant check:

- Syntax smoke test:

```bash
python -m compileall pipelines/indexing_pipeline pipelines/agent_pipeline vector_databases app
```

- CLI help:

```bash
python -m pipelines.indexing_pipeline.indexing --help
python pipeline.py --help
python -m pipelines.indexing_pipeline.scope --help
python vector_databases/pinecone_sync.py --help
```

- If indexing or retrieval changed:

```bash
python -m pipelines.indexing_pipeline.indexing query --index-dir storage --query "hello"
```

- If prompt-based routing changed, run:

```bash
python test/test_pipeline.py --query "hello"
```

Note: any LLM-backed runtime check requires a valid `HF_TOKEN` and network access to Hugging Face services.
Note: any Pinecone-backed runtime check requires a valid `PINECONE_API_KEY` and network access to Pinecone.

## When Assisting In This Repo

- Read `README.md` and this file first.
- Do not assume there is a mature automated test suite.
- Distinguish clearly between generated artifacts and source code.
- Be cautious with hardcoded paths and dated result filenames.
- If a user asks for routing behavior changes, confirm which router they mean before making broad edits:
  - embedding-based CLI router
  - prompt-based LLM router
- If a user asks for retrieval behavior changes, confirm whether they mean:
  - FAISS retrieval in `pipelines/indexing_pipeline/indexing.py`
  - Pinecone retrieval in `retriever/retriever_node.py`

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
Create `vector_databases/pinecone_sync.py` to reload vectors and metadata from FAISS and upsert them into Pinecone. Create `pipelines/agent_pipeline/retriever/retriever_node.py` so the routed pipeline passes `greeting` and `off_topic` decisions through, but sends `related` decisions into Pinecone retrieval using the embedding model from `pipelines/indexing_pipeline/pdf_rag.py`. Rename `route_node_pipeline.py` to `pipeline.py` and update the pipeline entrypoint accordingly.

### Goal
The repo has a Pinecone sync path, a Pinecone-backed retrieval node, and a single routed entrypoint in `pipeline.py`.

### Scope
`pipeline.py`, `retriever/`, `vector_databases/`, `test/test_pipeline.py`, `requirements.txt`, and this context file.

### Constraints
Keep the existing prompt-based routing model intact. Reuse the existing BGE-M3 embedding path for retrieval. Do not rewrite unrelated artifacts or the untracked `embedding_mmodel/` folder.

### Verification
Run syntax smoke checks, CLI help for the renamed pipeline and the new Pinecone sync script, and the pipeline test harness against greeting and retrieval scenarios as credentials allow.

### Notes
Pinecone collections are not the queryable primitive in the current SDK, so the implementation uses a Pinecone index for upserts and queries.
