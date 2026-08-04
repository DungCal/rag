from __future__ import annotations

import re
from typing import Any

from loguru import logger

from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL

_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(model_name: str = DEFAULT_LLM_MODEL) -> Any | None:
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_cache[model_name] = tokenizer
        logger.info("Loaded tokenizer for model={}", model_name)
        return tokenizer
    except Exception as exc:
        logger.warning(
            "Failed to load tokenizer for model={} ({}). Falling back to word-count heuristic.",
            model_name,
            exc,
        )
        _tokenizer_cache[model_name] = None
        return None


def count_tokens(text: str, model_name: str = DEFAULT_LLM_MODEL) -> int:
    tokenizer = _get_tokenizer(model_name)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    words = text.split()
    return max(1, int(len(words) * 1.3))


def _format_message_for_tokenization(message: Any) -> str:
    role = getattr(message, "type", "user") or "user"
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        content = " ".join(parts)
    return f"{role}: {content}"


def count_messages_tokens(messages: list[Any], model_name: str = DEFAULT_LLM_MODEL) -> int:
    if not messages:
        return 0
    formatted = "\n".join(_format_message_for_tokenization(msg) for msg in messages)
    return count_tokens(formatted, model_name)
