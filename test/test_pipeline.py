from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ENTRYPOINT = PROJECT_ROOT / "pipelines" / "agent_pipeline" / "pipeline.py"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "route_node_prompt.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test the pipeline.py CLI end-to-end")
    parser.add_argument("--query", required=True, help="Query to send to pipeline.py")
    parser.add_argument("--expected-label", choices=["greeting", "related", "off_topic"], default=None, help="Optional expected router label")
    parser.add_argument("--expected-route", choices=["greeting", "retrieval", "off_topic"], default=None, help="Optional expected pipeline route")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH), help="Path to the router prompt template")
    parser.add_argument("--scope", default=None, help="Document scope text to inject into the router prompt")
    parser.add_argument("--scope-file", default=None, help="Path to a text file containing the document scope")
    parser.add_argument("--model-name", default=None, help="Hugging Face model used for routing")
    parser.add_argument("--embedding-model-name", default=None, help="Embedding model used for retrieval route")
    parser.add_argument("--pinecone-index-name", default=None, help="Pinecone index name for retrieval route")
    parser.add_argument("--pinecone-namespace", default=None, help="Pinecone namespace for retrieval route")
    parser.add_argument("--top-k", type=int, default=None, help="Number of Pinecone matches to return for retrieval route")
    parser.add_argument("--use-fp16", action="store_true", help="Use fp16 for the embedding model when supported")
    parser.add_argument("--enable-rerank", action="store_true", help="Rerank Pinecone matches with Hugging Face Hub")
    parser.add_argument("--reranker-model-name", default=None, help="Hugging Face reranker model used after Pinecone retrieval")
    parser.add_argument("--reranker-batch-size", type=int, default=None, help="Batch size for reranking")
    parser.add_argument("--reranker-max-length", type=int, default=None, help="Max sequence length for reranking")
    parser.add_argument("--reranker-instruction", default=None, help="Optional custom instruction for the reranker")
    parser.add_argument("--reranker-fp16", action="store_true", help="Use fp16 for the reranker when supported")
    parser.add_argument("--reranker-sigmoid", action="store_true", help="Convert reranker scores to 0-1 probabilities")
    parser.add_argument("--print-prompt", action="store_true", help="Request prompt output from pipeline.py")
    parser.add_argument("--as-json", action="store_true", help="Print the test result as JSON")
    return parser


def build_pipeline_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE_ENTRYPOINT),
        "--query",
        args.query,
        "--prompt-path",
        args.prompt_path,
        "--as-json",
    ]
    if args.scope is not None:
        command.extend(["--scope", args.scope])
    if args.scope_file is not None:
        command.extend(["--scope-file", args.scope_file])
    if args.model_name is not None:
        command.extend(["--model-name", args.model_name])
    if args.embedding_model_name is not None:
        command.extend(["--embedding-model-name", args.embedding_model_name])
    if args.pinecone_index_name is not None:
        command.extend(["--pinecone-index-name", args.pinecone_index_name])
    if args.pinecone_namespace is not None:
        command.extend(["--pinecone-namespace", args.pinecone_namespace])
    if args.top_k is not None:
        command.extend(["--top-k", str(args.top_k)])
    if args.use_fp16:
        command.append("--use-fp16")
    if args.enable_rerank:
        command.append("--enable-rerank")
    if args.reranker_model_name is not None:
        command.extend(["--reranker-model-name", args.reranker_model_name])
    if args.reranker_batch_size is not None:
        command.extend(["--reranker-batch-size", str(args.reranker_batch_size)])
    if args.reranker_max_length is not None:
        command.extend(["--reranker-max-length", str(args.reranker_max_length)])
    if args.reranker_instruction is not None:
        command.extend(["--reranker-instruction", args.reranker_instruction])
    if args.reranker_fp16:
        command.append("--reranker-fp16")
    if args.reranker_sigmoid:
        command.append("--reranker-sigmoid")
    if args.print_prompt:
        command.append("--print-prompt")
    return command


def evaluate_result(payload: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    failures: list[str] = []

    if args.expected_label is not None and payload.get("label") != args.expected_label:
        failures.append(f"expected label {args.expected_label!r}, got {payload.get('label')!r}")
    if args.expected_route is not None and payload.get("route") != args.expected_route:
        failures.append(f"expected route {args.expected_route!r}, got {payload.get('route')!r}")

    return (len(failures) == 0, failures)


def main() -> None:
    args = build_parser().parse_args()
    command = build_pipeline_command(args)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    payload: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

    if completed.returncode == 0:
        try:
            pipeline_payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            payload["passed"] = False
            payload["failures"] = [f"pipeline entrypoint returned non-JSON stdout: {exc}"]
        else:
            passed, failures = evaluate_result(pipeline_payload, args)
            payload["passed"] = passed
            payload["failures"] = failures
            payload["pipeline_result"] = pipeline_payload
    else:
        payload["passed"] = False
        payload["failures"] = [f"pipeline entrypoint exited with return code {completed.returncode}"]

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"passed={payload['passed']}")
    print(f"returncode={completed.returncode}")
    if "pipeline_result" in payload:
        result = payload["pipeline_result"]
        print(f"actual_label={result.get('label')}")
        print(f"route={result.get('route')}")
        print(f"raw_output={result.get('raw_output')}")
    for failure in payload.get("failures", []):
        print(f"failure={failure}")
    if completed.stderr.strip():
        print("-" * 80)
        print(completed.stderr.strip())


if __name__ == "__main__":
    main()
