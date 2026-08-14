from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.agent_pipeline.traditional_rag.routers.routing_classification import (
    PromptQueryRouter,
    DEFAULT_ROUTE_PROMPT_PATH,
    DEFAULT_DOCUMENT_SCOPE,
)

LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

TEST_CASES: list[dict[str, str]] = [
    {"query": "Hi there, how do you do?", "expected": "greeting"},
    {"query": "Good morning!", "expected": "greeting"},
    {"query": "Thanks, talk to you later.", "expected": "greeting"},
    {"query": "How do I check the transmission oil level on the tractor?", "expected": "related"},
    {"query": "What does the DPF warning lamp mean and what should I do?", "expected": "related"},
    {"query": "How do I safely attach the PTO to an implement?", "expected": "related"},
    {"query": "How many cups of water do I need to boil pasta?", "expected": "off_topic"},
    {"query": "Write a python script to scrape a website.", "expected": "off_topic"},
]


def print_prompt(decision) -> None:
    print("--- Prompt ---")
    print(decision.prompt)
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the PromptQueryRouter using meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--model-name",
        default=LLAMA_MODEL,
        help="Hugging Face model used for routing (defaults to Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument("--query", default=None, help="Run a single query instead of the battery")
    parser.add_argument(
        "--expected-label",
        choices=["greeting", "related", "off_topic"],
        default=None,
        help="Expected label when running a single --query",
    )
    parser.add_argument("--scope", default=DEFAULT_DOCUMENT_SCOPE, help="Document scope text")
    parser.add_argument("--prompt-path", default=str(DEFAULT_ROUTE_PROMPT_PATH), help="Router prompt template path")
    parser.add_argument("--print-prompt", action="store_true", help="Print the built prompt for each case")
    parser.add_argument("--as-json", action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    router = PromptQueryRouter(
        prompt_path=args.prompt_path,
        scope=args.scope,
        model_name=args.model_name,
    )

    if args.query is not None:
        cases = [{"query": args.query, "expected": args.expected_label}]
    else:
        cases = TEST_CASES

    results = []
    for case in cases:
        decision = router.route(case["query"])
        result = {
            "query": case["query"],
            "expected": case["expected"],
            "label": decision.label,
            "route": decision.route,
            "raw_output": decision.raw_output,
            "passed": case["expected"] is not None and decision.label == case["expected"],
        }
        results.append(result)
        if args.print_prompt:
            print_prompt(decision)
        if args.as_json:
            continue
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] query={case['query']!r}")
        print(f"    expected_label={case['expected']!r} actual_label={decision.label!r} route={decision.route!r}")
        print(f"    raw_output={decision.raw_output!r}")

    checked = [r for r in results if r["expected"] is not None]
    all_passed = all(r["passed"] for r in checked)
    summary = {
        "model": args.model_name,
        "total": len(results),
        "passed": sum(1 for r in checked if r["passed"]),
        "results": results,
    }

    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("-" * 80)
        print(f"passed={summary['passed']}/{len(checked)} (all_passed={all_passed})")

    if not all_passed:
        for r in checked:
            if not r["passed"]:
                print(f"failure: expected_label={r['expected']!r} actual_label={r['label']!r} query={r['query']!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()