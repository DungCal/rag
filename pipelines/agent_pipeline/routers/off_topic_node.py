from __future__ import annotations

from pathlib import Path

from rag.llm import DEFAULT_LLM_MODEL

from .prompt_response_nodes import DEFAULT_OFF_TOPIC_PROMPT_PATH, OffTopicNode as _OffTopicNode


class OffTopicNode(_OffTopicNode):
    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_OFF_TOPIC_PROMPT_PATH,
        scope: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            prompt_path=prompt_path,
            scope=scope,
            model_name=model_name,
            api_key=api_key,
        )
