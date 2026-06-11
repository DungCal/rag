from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.llm import DEFAULT_LLM_MODEL
from rag.pdf_rag import DEFAULT_MODEL_NAME
from retriever.retriever_node import RetrieverNode
from routers.greeting_node import GreetingNode
from routers.off_topic_node import OffTopicNode
from routers.prompt_query_router import DEFAULT_DOCUMENT_SCOPE, PromptQueryRouter
from routers.prompt_response_nodes import load_default_scope


DEFAULT_SCOPE_FILE = Path("results/scope_result_20260606_193507.txt")
DEFAULT_PINECONE_NAMESPACE = "default"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the routed RAG pipeline")
    parser.add_argument("--query", required=True, help="User query to classify")
    parser.add_argument("--prompt-path", default="prompts/route_node_prompt.txt", help="Path to the router prompt template")
    parser.add_argument("--scope", default=None, help="Document scope text to inject into the router prompt")
    parser.add_argument("--scope-file", default=None, help="Path to a text file containing the document scope")
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL, help="Hugging Face model used for routing")
    parser.add_argument("--embedding-model-name", default=DEFAULT_MODEL_NAME, help="Embedding model used for Pinecone retrieval")
    parser.add_argument("--pinecone-index-name", required=False, help="Pinecone index name for retrieval route")
    parser.add_argument("--pinecone-namespace", default=DEFAULT_PINECONE_NAMESPACE, help="Pinecone namespace for retrieval route")
    parser.add_argument("--top-k", type=int, default=5, help="Number of Pinecone matches to return for retrieval route")
    parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for the embedding model when supported")
    parser.add_argument("--as-json", action="store_true", help="Print the pipeline result as JSON")
    parser.add_argument("--print-prompt", action="store_true", help="Include the rendered prompt in the output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scope = _load_scope(args.scope, args.scope_file)
    router = PromptQueryRouter(
        prompt_path=args.prompt_path,
        scope=scope,
        model_name=args.model_name,
    )
    decision = router.route(args.query)
    node_result = None
    retriever_result = None

    if decision.route == "greeting":
        node_result = GreetingNode(scope=scope, model_name=args.model_name).run(args.query)
    elif decision.route == "off_topic":
        node_result = OffTopicNode(scope=scope, model_name=args.model_name).run(args.query)
    else:
        if not args.pinecone_index_name:
            raise ValueError("--pinecone-index-name is required when the route is retrieval")
        retriever = RetrieverNode(
            index_name=args.pinecone_index_name,
            namespace=args.pinecone_namespace,
            model_name=args.embedding_model_name,
            top_k=args.top_k,
            use_fp16=args.use_fp16,
        )
        retriever_result = retriever.run(decision, args.query)

    payload = {
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
        payload["results"] = retriever_result.results
    if args.print_prompt:
        payload["prompt"] = decision.prompt
        if node_result is not None:
            payload["node_prompt"] = node_result.prompt

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"route={decision.route}")
    print(f"label={decision.label}")
    print(f"raw_output={decision.raw_output}")
    print(decision.message)
    if node_result is not None:
        print("-" * 80)
        print(node_result.response)
    if retriever_result is not None:
        print("-" * 80)
        print(f"retrieval_query={retriever_result.query}")
        for item in retriever_result.results:
            print(f"score={item['score']:.4f} page={item.get('page_number')} chunk={item.get('chunk_id')}")
            print(item.get("text", ""))
            print("-" * 80)
    if args.print_prompt:
        print("-" * 80)
        print(decision.prompt)
        if node_result is not None:
            print("-" * 80)
            print(node_result.prompt)


if __name__ == "__main__":
    main()
