from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.llm import DEFAULT_LLM_MODEL
from route_node_pipeline import _load_scope
from routers import PromptQueryRouter
from routers.greeting_node import GreetingNode
from routers.off_topic_node import OffTopicNode

DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "route_node_prompt.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test the prompt-driven query router node")
    parser.add_argument("--query", required=True, help="Query to send to the router node")
    parser.add_argument("--expected-label", choices=["greeting", "related", "off_topic"], default=None, help="Optional expected router label")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH), help="Path to the router prompt template")
    parser.add_argument("--scope", default=None, help="Document scope text to inject into the router prompt")
    parser.add_argument("--scope-file", default=None, help="Path to a text file containing the document scope")
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL, help="Hugging Face model used for routing")
    parser.add_argument("--as-json", action="store_true", help="Print the test result as JSON")
    parser.add_argument("--print-prompt", action="store_true", help="Include the rendered router and node prompts in the output")
    parser.add_argument("--skip-node", action="store_true", help="Only test the router label without running greeting/off-topic nodes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scope = _load_scope(args.scope, args.scope_file)
    router = PromptQueryRouter(
        prompt_path=Path(args.prompt_path),
        scope=scope,
        model_name=args.model_name,
    )
    decision = router.route(args.query)
    passed = args.expected_label is None or decision.label == args.expected_label
    node_result = None

    if not args.skip_node:
        if decision.route == "greeting":
            node_result = GreetingNode(scope=scope, model_name=args.model_name).run(args.query)
        elif decision.route == "off_topic":
            node_result = OffTopicNode(scope=scope, model_name=args.model_name).run(args.query)

    payload = {
        "passed": passed,
        "expected_label": args.expected_label,
        "actual_label": decision.label,
        "route": decision.route,
        "message": decision.message,
        "raw_output": decision.raw_output,
    }
    if node_result is not None:
        payload["response"] = node_result.response
        payload["node_raw_output"] = node_result.raw_output
    if args.print_prompt:
        payload["prompt"] = decision.prompt
        if node_result is not None:
            payload["node_prompt"] = node_result.prompt

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"passed={passed}")
    print(f"actual_label={decision.label}")
    print(f"route={decision.route}")
    print(f"raw_output={decision.raw_output}")
    if args.expected_label is not None:
        print(f"expected_label={args.expected_label}")
    print(decision.message)
    if node_result is not None:
        print("-" * 80)
        print(node_result.response)
    if args.print_prompt:
        print("-" * 80)
        print(decision.prompt)
        if node_result is not None:
            print("-" * 80)
            print(node_result.prompt)


if __name__ == "__main__":
    main()
