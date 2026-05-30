from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.llm import DEFAULT_LLM_MODEL, HuggingFaceAnswerGenerator


DEFAULT_METADATA_PATH = Path("storage/metadata.json")
DEFAULT_PROMPT_PATH = Path("prompts/prompt_scope.txt")
DEFAULT_RESULTS_DIR = Path("results")


def load_records(metadata_path: Path) -> list[dict[str, Any]]:
    raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("metadata.json must contain a JSON array of chunk records")

    records: list[dict[str, Any]] = []
    for item in raw_payload:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        records.append(item)

    if not records:
        raise ValueError(f"No valid chunk records found in {metadata_path}")

    return records


def sample_records(records: list[dict[str, Any]], sample_size: int, seed: int | None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    actual_size = min(sample_size, len(records))
    sampled = rng.sample(records, actual_size)
    return sorted(sampled, key=lambda item: int(item.get("chunk_id", 0)))


def build_context(records: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for item in records:
        blocks.append(
            "\n".join(
                [
                    f"Chunk ID: {item.get('chunk_id', 'unknown')}",
                    f"Source file: {item.get('source_file', 'unknown')}",
                    f"Page number: {item.get('page_number', 'unknown')}",
                    "Text:",
                    str(item.get("text", "")).strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(prompt_template_path: Path, context: str) -> str:
    template = prompt_template_path.read_text(encoding="utf-8")
    if "{context}" not in template:
        raise ValueError(f"Prompt template must contain '{{context}}': {prompt_template_path}")
    return template.format(context=context)


def write_result(
    results_dir: Path,
    *,
    prompt: str,
    response_text: str,
    sampled_records: list[dict[str, Any]],
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"scope_result_{timestamp}.txt"

    sampled_summary = "\n".join(
        [
            f"- chunk_id={item.get('chunk_id', 'unknown')} page={item.get('page_number', 'unknown')} source={item.get('source_file', 'unknown')}"
            for item in sampled_records
        ]
    )

    output_text = "\n\n".join(
        [
            "Generated summary:",
            response_text.strip(),
            "Sampled chunks:",
            sampled_summary,
            "Prompt used:",
            prompt,
        ]
    ).strip()

    output_path.write_text(output_text + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample chunk metadata and generate a scoped summary")
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH), help="Path to metadata.json")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH), help="Path to prompt template")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for generated output")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of random chunks to sample")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL, help="Hugging Face model used for generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    prompt_path = Path(args.prompt_path)
    results_dir = Path(args.results_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path.resolve()}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path.resolve()}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")

    records = load_records(metadata_path)
    sampled_records = sample_records(records, args.sample_size, args.seed)
    context = build_context(sampled_records)
    prompt = build_prompt(prompt_path, context)

    generator = HuggingFaceAnswerGenerator(
        model_name=args.model_name,
    )
    response_text = generator.generate_text(prompt)

    output_path = write_result(
        results_dir,
        prompt=prompt,
        response_text=response_text,
        sampled_records=sampled_records,
    )

    print(f"Sampled {len(sampled_records)} chunks from {metadata_path.resolve()}")
    print(f"Wrote result to {output_path.resolve()}")


if __name__ == "__main__":
    main()
