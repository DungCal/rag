from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index_store import build_faiss_index, load_index, save_index
from .llm import DEFAULT_LLM_MODEL, HuggingFaceAnswerGenerator
from .pdf_rag import DEFAULT_MODEL_NAME, PDFRAG
from .query_router import QueryRouter
from .retriever import retrieve_results


DEFAULT_INDEX_DIR = Path("storage")


def command_index(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rag = PDFRAG(model_name=args.model_name, use_fp16=args.use_fp16)
    records = rag.build_records_from_pdf(
        pdf_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not records:
        raise ValueError(f"No text chunks could be extracted from {pdf_path}")

    vectors = rag.embed_texts(record.text for record in records)
    index = build_faiss_index(vectors)
    save_index(Path(args.index_dir), index, records)

    print(f"Indexed {len(records)} chunks from {pdf_path}")
    print(f"Saved FAISS index to {Path(args.index_dir).resolve()}")


def command_query(args: argparse.Namespace) -> None:
    index, records = load_index(Path(args.index_dir))
    rag = PDFRAG(model_name=args.model_name, use_fp16=args.use_fp16)
    query_vector = rag.embed_query(args.query)
    router = QueryRouter(off_topic_threshold=args.off_topic_threshold)
    decision = router.route(args.query, query_vector, index)

    if decision.route != "retrieval":
        payload = {
            "route": decision.route,
            "message": decision.message,
            "similarity_score": decision.similarity_score,
            "results": [],
        }
        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        print(f"route={decision.route}")
        if decision.similarity_score is not None:
            print(f"similarity_score={decision.similarity_score:.4f}")
        print(decision.message)
        return

    results = retrieve_results(index, records, query_vector, top_k=args.top_k)
    answer_text: str | None = None
    if args.generate_answer:
        generator = HuggingFaceAnswerGenerator(
            model_name=args.llm_model,
        )
        answer_text = generator.answer(args.query, results)

    if args.as_json:
        payload = {
            "route": decision.route,
            "message": decision.message,
            "similarity_score": decision.similarity_score,
            "answer": answer_text,
            "results": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"route={decision.route}")
    if decision.similarity_score is not None:
        print(f"similarity_score={decision.similarity_score:.4f}")
    if answer_text is not None:
        print("answer:")
        print(answer_text)
        print("-" * 80)
    for item in results:
        print(f"score={item['score']:.4f} page={item['page_number']} chunk={item['chunk_id']}")
        print(item["text"])
        print("-" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF RAG with FAISS and BGE-M3 embeddings")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a PDF into a FAISS vector store")
    index_parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    index_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory for FAISS index and metadata")
    index_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Embedding model name")
    index_parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters")
    index_parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    index_parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for faster inference on supported hardware")
    index_parser.set_defaults(func=command_index)

    query_parser = subparsers.add_parser("query", help="Query an existing FAISS vector store")
    query_parser.add_argument("--query", required=True, help="User query text")
    query_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory containing FAISS index and metadata")
    query_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Embedding model name")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    query_parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Call the main Hugging Face model after retrieval to synthesize an answer from the retrieved chunks",
    )
    query_parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Hugging Face model used for answer generation",
    )
    query_parser.add_argument(
        "--off-topic-threshold",
        type=float,
        default=0.35,
        help="Minimum similarity score required to treat a query as document-relevant",
    )
    query_parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for faster inference on supported hardware")
    query_parser.add_argument("--as-json", action="store_true", help="Print results as JSON")
    query_parser.set_defaults(func=command_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
