from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from pipelines.agent_pipeline.agent_rag.token_utils import count_messages_tokens, count_tokens
from pipelines.indexing_pipeline.llm import (
    DEFAULT_LLM_MODEL,
    DEFAULT_HF_INFERENCE_PROVIDER,
    _load_api_key_from_env_file,
    _load_provider_from_env_file,
)

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        RemoveMessage,
        SystemMessage,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-core is required for conversation compaction. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPACTON_PROMPT_PATH = PROJECT_ROOT / "prompts" / "summarization" / "history_context_summarization.txt"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "logs" / "conversation_history"


def _format_message_for_display(message: BaseMessage) -> str:
    role = getattr(message, "type", "user") or "user"
    role_map = {"human": "User", "ai": "Assistant", "system": "System", "tool": "Tool"}
    display_role = role_map.get(role, role.capitalize())
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        content = " ".join(parts)
    return f"**{display_role}:** {content}"


_TURN_HEADER_RE = re.compile(r"^## Turn (\d+)", re.MULTILINE)
_SYSTEM_CONTEXT_RE = re.compile(r"^### System Context\n\n.*?\n---\n?", re.MULTILINE | re.DOTALL)


def _get_next_turn_number(file_path: Path) -> int:
    if not file_path.exists():
        return 1
    text = file_path.read_text(encoding="utf-8")
    matches = _TURN_HEADER_RE.findall(text)
    if not matches:
        return 1
    return max(int(m) for m in matches) + 1


def _replace_system_context_in_file(system_context: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    system_block = f"### System Context\n\n{system_context}\n---\n"
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")
        if _SYSTEM_CONTEXT_RE.search(text):
            text = _SYSTEM_CONTEXT_RE.sub(system_block, text, count=1)
        else:
            text = text.rstrip() + "\n\n" + system_block
        file_path.write_text(text, encoding="utf-8")
    else:
        text = f"# Conversation History\n\n{system_block}"
        file_path.write_text(text, encoding="utf-8")


def append_turn_to_file(
    query: str,
    response: str,
    file_path: Path,
    *,
    system_message: str | None = None,
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    if system_message:
        lines.append("[System context injected due to compaction]")
        lines.append(f"**System:** {system_message}")
        lines.append("---")
        lines.append("")

    turn_number = _get_next_turn_number(file_path)
    lines.append(f"## Turn {turn_number}\n")

    if query:
        lines.append(f"**User:** {query}")
        lines.append("")

    if response:
        lines.append(f"**Assistant:** {response}")
        lines.append("")

    if file_path.exists() and file_path.stat().st_size > 0:
        existing = file_path.read_text(encoding="utf-8")
        new_content = existing.rstrip() + "\n\n" + "\n".join(lines).rstrip() + "\n"
        file_path.write_text(new_content, encoding="utf-8")
    else:
        header = "# Conversation History\n\n"
        file_path.write_text(header + "\n".join(lines).rstrip() + "\n", encoding="utf-8")

    logger.info("Appended turn {} to {}", turn_number, file_path.resolve())


def _save_conversation_to_file(messages: list[BaseMessage], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Conversation History\n\n"]
    turn_number = 0
    for message in messages:
        role = getattr(message, "type", "user") or "user"
        if role in ("human", "ai"):
            if role == "human":
                turn_number += 1
                lines.append(f"## Turn {turn_number}\n")
            lines.append(_format_message_for_display(message))
            lines.append("")
    file_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved original conversation history to {}", file_path.resolve())


class ConversationCompactionNode:
    """Manages conversation context with rolling summarization.

    Algorithm:
      1. Pre-answer: count (system_context + recent_turns + query).
         If > threshold: summarize (system_context + recent_turns), update system_context, clear recent_turns.
      2. Answer using (system_context + recent_turns + query).
      3. Post-answer: add (query + answer) to recent_turns.
         Count (system_context + recent_turns). If > threshold: summarize, update, clear.
      4. Log system_context to .md every turn.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_LLM_MODEL,
        provider: str | None = None,
        max_input_tokens: int = 256_000,
        threshold_pct: float = 0.30,
        min_keep_recent_turns: int = 1,
        history_dir: str | Path = DEFAULT_HISTORY_DIR,
        prompt_path: str | Path = DEFAULT_COMPACTON_PROMPT_PATH,
        max_summary_tokens: int = 2048,
        summary_temperature: float = 0.0,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for conversation compaction. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._model_name = model_name
        resolved_api_key = os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        resolved_provider = provider or os.getenv("HF_INFERENCE_PROVIDER") or _load_provider_from_env_file() or DEFAULT_HF_INFERENCE_PROVIDER

        self._client = InferenceClient(
            model=model_name,
            token=resolved_api_key,
            provider=resolved_provider,
        )
        self._provider = resolved_provider

        self._max_input_tokens = max_input_tokens
        self._threshold_tokens = int(max_input_tokens * threshold_pct)
        self._threshold_pct = threshold_pct
        self._min_keep_recent_turns = min_keep_recent_turns
        self._history_dir = Path(history_dir)
        self._max_summary_tokens = max_summary_tokens
        self._summary_temperature = summary_temperature

        prompt_path = Path(prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Compaction prompt template not found: {prompt_path.resolve()}")
        self._prompt_template = prompt_path.read_text(encoding="utf-8")

    def _build_prompt(self, messages_to_summarize: list[BaseMessage]) -> str:
        formatted = "\n\n".join(_format_message_for_display(m) for m in messages_to_summarize)
        try:
            return self._prompt_template.format(messages=formatted)
        except KeyError as exc:
            raise ValueError(f"Compaction prompt template is missing placeholder: {exc}") from exc

    def _call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat_completion(
            messages=messages,
            max_tokens=self._max_summary_tokens,
            temperature=self._summary_temperature,
            stream=False,
        )
        return str(response.choices[0].message.content).strip()

    def _count_system_context_tokens(self, system_context: str | None) -> int:
        if not system_context:
            return 0
        return count_tokens(system_context, self._model_name)

    def _count_recent_turns_tokens(self, recent_turns: list[BaseMessage]) -> int:
        if not recent_turns:
            return 0
        return count_messages_tokens(recent_turns, self._model_name)

    def _count_query_tokens(self, query: str) -> int:
        if not query:
            return 0
        return count_tokens(query, self._model_name)

    def pre_answer_check(
        self,
        system_context: str | None,
        recent_turns: list[BaseMessage],
        query: str,
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        """Check tokens before answering. If > threshold, summarize."""
        sc_tokens = self._count_system_context_tokens(system_context)
        rt_tokens = self._count_recent_turns_tokens(recent_turns)
        q_tokens = self._count_query_tokens(query)
        total_tokens = sc_tokens + rt_tokens + q_tokens

        logger.info(
            "Pre-answer check: system_context_tokens={}, recent_turns_tokens={}, query_tokens={}, total={}, threshold={}",
            sc_tokens, rt_tokens, q_tokens, total_tokens, self._threshold_tokens,
        )

        if total_tokens <= self._threshold_tokens:
            return {"system_context": system_context, "recent_turns": recent_turns, "summarized": False}

        logger.info("Pre-answer: threshold exceeded, summarizing system_context + recent_turns")

        messages_to_summarize = []
        if system_context:
            messages_to_summarize.append(SystemMessage(content=system_context))
        messages_to_summarize.extend(recent_turns)

        prompt = self._build_prompt(messages_to_summarize)
        summary_text = self._call_llm(prompt)

        history_file_path = self._history_dir / f"{thread_id}.md"
        _replace_system_context_in_file(summary_text, history_file_path)

        logger.info(
            "Pre-answer compaction: summarized {} messages into {} tokens",
            len(messages_to_summarize),
            self._count_system_context_tokens(summary_text),
        )

        return {"system_context": summary_text, "recent_turns": [], "summarized": True}

    def post_answer_check(
        self,
        system_context: str | None,
        recent_turns: list[BaseMessage],
        query: str,
        answer: str,
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        """Add (query + answer) to recent_turns, check tokens. If > threshold, summarize."""
        new_turn = [HumanMessage(content=query), AIMessage(content=answer)]
        updated_recent_turns = recent_turns + new_turn

        sc_tokens = self._count_system_context_tokens(system_context)
        rt_tokens = self._count_recent_turns_tokens(updated_recent_turns)
        total_tokens = sc_tokens + rt_tokens

        logger.info(
            "Post-answer check: system_context_tokens={}, recent_turns_tokens={}, total={}, threshold={}",
            sc_tokens, rt_tokens, total_tokens, self._threshold_tokens,
        )

        if total_tokens <= self._threshold_tokens:
            return {"system_context": system_context, "recent_turns": updated_recent_turns, "summarized": False}

        logger.info("Post-answer: threshold exceeded, summarizing system_context + recent_turns")

        messages_to_summarize = []
        if system_context:
            messages_to_summarize.append(SystemMessage(content=system_context))
        messages_to_summarize.extend(updated_recent_turns)

        prompt = self._build_prompt(messages_to_summarize)
        summary_text = self._call_llm(prompt)

        history_file_path = self._history_dir / f"{thread_id}.md"
        _replace_system_context_in_file(summary_text, history_file_path)

        logger.info(
            "Post-answer compaction: summarized {} messages into {} tokens",
            len(messages_to_summarize),
            self._count_system_context_tokens(summary_text),
        )

        return {"system_context": summary_text, "recent_turns": [], "summarized": True}
