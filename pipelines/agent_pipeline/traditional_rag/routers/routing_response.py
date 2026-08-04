from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from pipelines.agent_pipeline.shared.node_results import PromptNodeResult
from pipelines.agent_pipeline.shared.utils import is_degenerate_response
from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL, DEFAULT_HF_INFERENCE_PROVIDER, _load_api_key_from_env_file, _load_provider_from_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GREETING_PROMPT_PATH = PROJECT_ROOT / "prompts" / "greeting_node_prompt.txt"
DEFAULT_OFF_TOPIC_PROMPT_PATH = PROJECT_ROOT / "prompts" / "off_topic_node_prompt.txt"
DEFAULT_SCOPE_FILE_PATH = PROJECT_ROOT / "results" / "scope_result_20260606_193507.txt"

_FALLBACK_GREETING = (
    "Hello! I'm here to help you with the TYM diesel tractor operator manual. "
    "Feel free to ask about tractor operation, maintenance procedures, or safety warnings."
)

_FALLBACK_OFF_TOPIC = (
    "I can only answer questions about the TYM diesel tractor operator manual. "
    "Please ask me about tractor operation, maintenance, or safety procedures."
)


def load_default_scope(scope_file: str | Path = DEFAULT_SCOPE_FILE_PATH) -> str:
    scope_path = Path(scope_file)
    if not scope_path.exists():
        raise FileNotFoundError(f"Scope file not found: {scope_path.resolve()}")

    scope = scope_path.read_text(encoding="utf-8").strip()
    if not scope:
        raise ValueError(f"Scope file is empty: {scope_path.resolve()}")
    return scope


class _PromptResponseNode:
    def __init__(
        self,
        *,
        route_name: str,
        prompt_path: str | Path,
        scope: str,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
        max_new_tokens: int = 800,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for the prompt response nodes. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.route_name = route_name
        self.prompt_path = Path(prompt_path)
        self.scope = scope.strip()
        if not self.scope:
            raise ValueError("Document scope must not be empty")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_path.resolve()}")

        self._prompt_template = self.prompt_path.read_text(encoding="utf-8").strip()
        if not self._prompt_template:
            raise ValueError(f"Prompt template is empty: {self.prompt_path.resolve()}")

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

    def build_prompt(self, user_query: str) -> str:
        query = user_query.strip()
        if not query:
            raise ValueError("User query must not be empty")

        try:
            return self._prompt_template.format(
                scope=self.scope,
                user_question=query,
            )
        except KeyError as exc:
            raise ValueError(
                f"Prompt template is missing a required placeholder: {exc}"
            ) from exc

    def run(self, user_query: str) -> PromptNodeResult:
        max_retries = 2
        last_output = ""
        last_prompt = ""

        for attempt in range(max_retries + 1):
            prompt = self.build_prompt(user_query)
            last_prompt = prompt
            temperature = 0.2 + (attempt * 0.1)
            messages = [{"role": "user", "content": prompt}]
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=self._max_new_tokens,
                temperature=temperature,
                stream=False,
            )
            raw_output = str(response.choices[0].message.content).strip()
            last_output = raw_output

            if not is_degenerate_response(raw_output):
                return PromptNodeResult(
                    route=self.route_name,
                    response=raw_output,
                    raw_output=raw_output,
                    prompt=prompt,
                )
            logger.warning(
                "Degenerate response detected (attempt {}/{}, temperature={:.1f}). Response: {!r}",
                attempt + 1, max_retries + 1, temperature, raw_output[:80],
            )

        logger.warning("All {} retries exhausted. Returning fallback response for route '{}'.", max_retries + 1, self.route_name)
        fallback = _FALLBACK_GREETING if self.route_name == "greeting" else _FALLBACK_OFF_TOPIC
        return PromptNodeResult(
            route=self.route_name,
            response=fallback,
            raw_output=last_output,
            prompt=last_prompt,
        )


class GreetingNode(_PromptResponseNode):
    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_GREETING_PROMPT_PATH,
        scope: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            route_name="greeting",
            prompt_path=prompt_path,
            scope=scope or load_default_scope(),
            model_name=model_name,
            api_key=api_key,
            provider=provider,
        )


class OffTopicNode(_PromptResponseNode):
    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_OFF_TOPIC_PROMPT_PATH,
        scope: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            route_name="off_topic",
            prompt_path=prompt_path,
            scope=scope or load_default_scope(),
            model_name=model_name,
            api_key=api_key,
            provider=provider,
        )
