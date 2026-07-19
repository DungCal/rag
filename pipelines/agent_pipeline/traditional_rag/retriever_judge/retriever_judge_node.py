from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL, _load_api_key_from_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JUDGE_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "llm-as-a-judge" / "llm-as-a-judge-context-relevance.txt"
)


DEFAULT_JUDGE_TOP_K = 3
DEFAULT_JUDGE_MIN_SCORE = 5


@dataclass
class RetrieverJudgeNodeResult:
    query: str
    results: list[dict[str, Any]]


class RetrieverJudgeNode:
    """Judge retrieved/reranked chunks for contextual relevance using an LLM."""

    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_JUDGE_PROMPT_PATH,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = DEFAULT_JUDGE_TOP_K,
        min_score: int = DEFAULT_JUDGE_MIN_SCORE,
    ) -> None:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        except ImportError as exc:
            raise ImportError(
                "langchain-huggingface is required for the retriever judge. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Retriever judge prompt template not found: {self.prompt_path.resolve()}"
            )

        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        resolved_api_key = api_key or os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError(
                "Missing HF_TOKEN in the environment or .env file for Hugging Face inference access"
            )

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=resolved_api_key,
            task="conversational",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )
        self._chat_model = ChatHuggingFace(llm=endpoint)
        self.top_k = max(1, top_k)
        self.min_score = max(0, min(10, min_score))

    def run(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        text_key: str = "text",
        top_k: int | None = None,
        min_score: int | None = None,
    ) -> RetrieverJudgeNodeResult:
        """Score every result for context relevance against the user query.

        Only chunks whose final judge score is at least ``min_score`` are kept.
        The surviving chunks are sorted by descending score and truncated to
        ``top_k`` items for downstream processing.
        """

        top_k = self.top_k if top_k is None else max(1, top_k)
        min_score = self.min_score if min_score is None else max(0, min(10, min_score))

        judged: list[dict[str, Any]] = []
        for item in results:
            judged_item = self._judge_item(query, item, text_key=text_key)
            final_score = judged_item.get("judge", {}).get("final_score", 0)
            if final_score >= min_score:
                judged.append(judged_item)

        # Sort by descending judge score so the most relevant chunks come first.
        judged.sort(
            key=lambda x: x.get("judge", {}).get("final_score", 0.0),
            reverse=True,
        )

        return RetrieverJudgeNodeResult(query=query, results=judged[:top_k])

    def _judge_item(
        self,
        query: str,
        item: dict[str, Any],
        text_key: str,
    ) -> dict[str, Any]:
        chunk_text = str(item.get(text_key, "")).strip()
        prompt = self._build_prompt(user_query=query, retrieved_chunk=chunk_text)

        raw_output = str(self._chat_model.invoke(prompt).content).strip()
        judgement = self._parse_judgement(raw_output)

        result = dict(item)
        result["judge"] = judgement
        result["judge_prompt"] = prompt
        return result

    def _build_prompt(self, user_query: str, retrieved_chunk: str) -> str:
        try:
            return self._prompt_template.format(
                user_query=user_query.strip(),
                retrieved_chunk=retrieved_chunk.strip(),
            )
        except KeyError as exc:
            raise ValueError(
                f"Retriever judge prompt template is missing a required placeholder: {exc}"
            ) from exc

    @staticmethod
    def _parse_judgement(raw_output: str) -> dict[str, Any]:
        """Extract the JSON object from the LLM output and normalize it."""

        cleaned = raw_output.strip()

        # If the model wrapped JSON in a markdown code fence, strip it.
        if "```" in cleaned:
            fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if fence_match:
                cleaned = fence_match.group(1).strip()
            else:
                cleaned = cleaned.replace("```", "").strip()

        # Extract the first JSON object if there is surrounding text.
        json_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return {
                "reasoning": f"Failed to parse judge JSON: {exc}. Raw output: {raw_output!r}",
                "score_band": "unknown",
                "final_score": 0,
                "raw_output": raw_output,
            }

        final_score = parsed.get("final_score")
        try:
            final_score = int(final_score)
        except (TypeError, ValueError):
            final_score = 0

        # Clamp the score to the declared 0-10 range.
        final_score = max(0, min(10, final_score))

        return {
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "score_band": str(parsed.get("score_band", "")).strip() or "unknown",
            "final_score": final_score,
            "raw_output": raw_output,
        }
