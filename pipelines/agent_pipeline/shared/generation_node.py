from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from pipelines.agent_pipeline.shared.config import GENERATION_MODEL, GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE
from pipelines.agent_pipeline.shared.node_results import PromptNodeResult
from pipelines.indexing_pipeline.llm import DEFAULT_HF_INFERENCE_PROVIDER, _load_api_key_from_env_file, _load_provider_from_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GENERATION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "generation_node_prompt.txt"


class GenerationNode:
    """Generates a final answer from context using an LLM.

    This node takes a user query and context (which can be RAG chunks, web search
    results, or both) and generates a natural language answer using the configured
    LLM model.
    """

    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_GENERATION_PROMPT_PATH,
        model_name: str = GENERATION_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
        max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
        temperature: float = GENERATION_TEMPERATURE,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for the generation node. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Generation prompt template not found: {self.prompt_path.resolve()}")

        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        resolved_api_key = api_key or os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        resolved_provider = provider or os.getenv("HF_INFERENCE_PROVIDER") or _load_provider_from_env_file() or DEFAULT_HF_INFERENCE_PROVIDER

        self._client = InferenceClient(
            model=model_name,
            token=resolved_api_key,
            provider=resolved_provider,
        )
        self._model_name = model_name
        self._provider = resolved_provider
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def run(self, query: str, context: list[dict[str, Any]]) -> PromptNodeResult:
        """Generate an answer from the query and context.

        Args:
            query: The user's question.
            context: List of context items, which can be:
                - RAG chunks: dicts with 'text' field (and optionally 'score', 'page_number', etc.)
                - Web search results: dicts with 'title', 'snippet', 'url' fields

        Returns:
            PromptNodeResult with the generated answer.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        if not context:
            return PromptNodeResult(
                route="generation",
                response="I do not have enough information to answer this question.",
                raw_output="",
                prompt="",
            )

        formatted_context = self._format_context(context)
        prompt = self._build_prompt(query, formatted_context)

        logger.info("GenerationNode: generating answer from {} context items", len(context))
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat_completion(
            messages=messages,
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            stream=False,
        )
        raw_output = str(response.choices[0].message.content).strip()

        return PromptNodeResult(
            route="generation",
            response=raw_output,
            raw_output=raw_output,
            prompt=prompt,
        )

    def _format_context(self, context: list[dict[str, Any]]) -> str:
        """Format context items into a readable string for the prompt."""
        formatted_parts = []

        for idx, item in enumerate(context, start=1):
            # Check if this is a web search result
            if "url" in item and "snippet" in item:
                title = item.get("title", "No title")
                snippet = item.get("snippet", "")
                url = item.get("url", "")
                formatted_parts.append(
                    f"Web Result {idx}: {title} - {snippet}\n(URL: {url})"
                )
            # Otherwise treat as RAG chunk
            else:
                text = item.get("text", "")
                score = item.get("score")
                judge_score = item.get("judge", {}).get("final_score")

                if judge_score is not None:
                    formatted_parts.append(
                        f"Chunk {idx} (Judge Score: {judge_score}): {text}"
                    )
                elif score is not None:
                    formatted_parts.append(
                        f"Chunk {idx} (Score: {score:.2f}): {text}"
                    )
                else:
                    formatted_parts.append(f"Chunk {idx}: {text}")

        return "\n\n".join(formatted_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the generation prompt."""
        try:
            return self._prompt_template.format(
                question=query.strip(),
                context=context.strip(),
            )
        except KeyError as exc:
            raise ValueError(f"Generation prompt template is missing a required placeholder: {exc}") from exc
