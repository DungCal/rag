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
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for Hugging Face inference. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        resolved_api_key = api_key or os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        self.model_name = model_name
        self.client = InferenceClient(api_key=resolved_api_key)
        self.mode = "chat"

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return str(response).strip()

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if content is not None:
            return str(content).strip()

        return str(response).strip()

    def generate_text(self, prompt: str, max_tokens: int = 220) -> str:
        response = self.client.chat_completion(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
        )
        return self._extract_chat_text(response)

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
