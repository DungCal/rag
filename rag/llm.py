from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_LLM_MODEL = "google/gemma-4-26B-A4B-it"
ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_api_key_from_env_file(env_path: Path = ENV_FILE_PATH) -> str | None:
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":

            resolved = value.strip().strip('"').strip("'")
            return resolved or None

    return None


class HuggingFaceAnswerGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
    ) -> None:
        try:
            from langchain_huggingface import HuggingFaceEndpoint
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for Hugging Face inference. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        resolved_api_key = api_key or os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        self.model_name = model_name
        self.client = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=resolved_api_key,
            task="text-generation",
        )

    def generate_text(self, prompt: str, max_tokens: int = 220) -> str:
        response = self.client.bind(max_new_tokens=max_tokens).invoke(prompt)
        return str(response).strip()

    def answer(self, query: str, results: list[dict[str, object]]) -> str:
        if not results:
            return "No retrieved context was available to answer the question."

        context_blocks: list[str] = []
        for item in results:
            context_blocks.append(
                "\n".join(
                    [
                        f"Source file: {item['source_file']}",
                        f"Page: {item['page_number']}",
                        f"Chunk: {item['chunk_id']}",
                        f"Similarity score: {item['score']:.4f}",
                        f"Text: {item['text']}",
                    ]
                )
            )

        prompt = "\n\n".join(
            [
                "You are answering questions about a document using only the retrieved context below.",
                "If the answer is not supported by the context, say that clearly.",
                f"User question: {query}",
                "Retrieved context:",
                "\n\n---\n\n".join(context_blocks),
            ]
        )
        return self.generate_text(prompt)
