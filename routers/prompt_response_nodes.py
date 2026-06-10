from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag.llm import DEFAULT_LLM_MODEL, _load_api_key_from_env_file


DEFAULT_GREETING_PROMPT_PATH = Path("prompts/greeting_node_prompt.txt")
DEFAULT_OFF_TOPIC_PROMPT_PATH = Path("prompts/off_topic_node_prompt.txt")
DEFAULT_SCOPE_FILE_PATH = Path("results/scope_result_20260606_193507.txt")


def load_default_scope(scope_file: str | Path = DEFAULT_SCOPE_FILE_PATH) -> str:
    scope_path = Path(scope_file)
    if not scope_path.exists():
        raise FileNotFoundError(f"Scope file not found: {scope_path.resolve()}")

    scope = scope_path.read_text(encoding="utf-8").strip()
    if not scope:
        raise ValueError(f"Scope file is empty: {scope_path.resolve()}")
    return scope


@dataclass
class PromptNodeResult:
    route: str
    response: str
    raw_output: str
    prompt: str


class _PromptResponseNode:
    def __init__(
        self,
        *,
        route_name: str,
        prompt_path: str | Path,
        scope: str,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        max_new_tokens: int = 160,
    ) -> None:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        except ImportError as exc:
            raise ImportError(
                "langchain-huggingface is required for the prompt response nodes. "
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

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=resolved_api_key,
            task="conversational",
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
        )
        self._chat_model = ChatHuggingFace(llm=endpoint)

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
        prompt = self.build_prompt(user_query)
        raw_output = str(self._chat_model.invoke(prompt).content).strip()
        return PromptNodeResult(
            route=self.route_name,
            response=raw_output,
            raw_output=raw_output,
            prompt=prompt,
        )


class GreetingNode(_PromptResponseNode):
    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_GREETING_PROMPT_PATH,
        scope: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            route_name="greeting",
            prompt_path=prompt_path,
            scope=scope or load_default_scope(),
            model_name=model_name,
            api_key=api_key,
        )


class OffTopicNode(_PromptResponseNode):
    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_OFF_TOPIC_PROMPT_PATH,
        scope: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            route_name="off_topic",
            prompt_path=prompt_path,
            scope=scope or load_default_scope(),
            model_name=model_name,
            api_key=api_key,
        )
