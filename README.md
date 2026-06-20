# PDF RAG with FAISS, BGE-M3, and LangChain Hugging Face

This project builds a simple RAG indexing and retrieval pipeline for PDF files.
It now includes a lightweight query router before retrieval:

- Greeting node: detects inputs like `hi` or `hello`, then ends
- Off-topic node: rejects queries that are not relevant to the indexed document
- Retrieval node: runs normal similarity search for relevant document questions

- Input: PDF file
- Embedding model: `BAAI/bge-m3`
- Chunking: LangChain `RecursiveCharacterTextSplitter`
- Main LLM for answer generation: `google/gemma-4-26B-A4B-it` via `langchain-huggingface`
- Vector store: FAISS
- Output: top matching chunks with source page metadata, with optional generated answers

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```bash
HF_TOKEN=your_hugging_face_token_here
```

The app will read `HF_TOKEN` from the process environment first, then fall back to `.env`.

## Use the indexing pipeline

Show available commands:

```bash
python -m pipelines.indexing_pipeline --help
```

Core subcommands:

- `index`: build `storage/faiss.index` and `storage/metadata.json`
- `query`: search the FAISS index, with optional answer generation
- `scope`: sample metadata and write a scope summary under `results/`

## Index a PDF

```bash
python -m pipelines.indexing_pipeline index --pdf path\to\document.pdf --index-dir storage
```

Optional flags:

- `--chunk-size 900`
- `--chunk-overlap 150`
- `--use-fp16`

## Query the index

```bash
python -m pipelines.indexing_pipeline query --index-dir storage --query "What are the main topics?"
```

Generate an answer with the main LLM:

```bash
python -m pipelines.indexing_pipeline query --index-dir storage --query "What are the main topics?" --generate-answer
```

Optional router flag:

- `--off-topic-threshold 0.35`
- `--llm-model google/gemma-4-26B-A4B-it`

JSON output:

```bash
python -m pipelines.indexing_pipeline query --index-dir storage --query "What are the main topics?" --as-json
```

## Generate scope results

```bash
python -m pipelines.indexing_pipeline scope --metadata-path storage/metadata.json
```

This writes:

- `output_prompt.txt` by default, or the file passed via `--output-prompt-path`
- `results/scope_result_<timestamp>.txt`

Example routed outputs:

- `hi` -> `greeting`
- unrelated question -> `off_topic`
- document question -> `retrieval`

## Sync FAISS vectors into Pinecone

```bash
python vector_databases/pinecone.py --pinecone-index-name your-index
```

This reads vectors back from `storage/faiss.index`, aligns them with `storage/metadata.json`, and upserts them into a Pinecone index namespace.

## Run the routed pipeline

```bash
python pipelines/agent_pipeline/pipeline.py --query "What does the DPF warning lamp mean?" --pinecone-index-name your-index
```

Behavior:

- `greeting` -> returns the greeting node response
- `off_topic` -> returns the off-topic node response
- `related` -> queries Pinecone using the BGE-M3 embedding path from `pipelines/indexing_pipeline/pdf_rag.py`

Optional reranking with Qwen 0.6B:

```bash
python pipelines/agent_pipeline/pipeline.py \
  --query "What does the DPF warning lamp mean?" \
  --pinecone-index-name your-index \
  --enable-rerank
```

This keeps the original Pinecone retrieval score and adds:

- `retrieval_score`
- `rerank_score`
- `retrieval_rank`
- `rerank_rank`
- `rank_shift`

To compare the retriever node directly against the Qwen reranker:

```bash
python rerank/evaluate_retriever_node.py \
  --query "What does the DPF warning lamp mean?" \
  --pinecone-index-name your-index
```

As of May 30, 2026, this project is configured to use `langchain-huggingface` with `google/gemma-4-26B-A4B-it` and `HF_TOKEN`.

## Files written

- `storage/faiss.index`: FAISS dense vector index
- `storage/metadata.json`: chunk text and source metadata

## Run the chunk statistics app

This project also includes a Streamlit app for exploring chunk metadata and chunk size distribution.

```bash
streamlit run app/app.py
```

The app lets you:

- inspect summary statistics for indexed chunks
- visualize chunk counts by character-length bins
- browse chunks by source file and page number
- view each chunk's content and stored metadata
