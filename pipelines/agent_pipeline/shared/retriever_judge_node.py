from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from pipelines.agent_pipeline.shared.config import JUDGE_MODEL, JUDGE_MAX_NEW_TOKENS, JUDGE_TEMPERATURE
from pipelines.indexing_pipeline.llm import DEFAULT_HF_INFERENCE_PROVIDER, _load_api_key_from_env_file, _load_provider_from_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
        model_name: str = JUDGE_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
        max_new_tokens: int = JUDGE_MAX_NEW_TOKENS,
        temperature: float = JUDGE_TEMPERATURE,
        top_k: int = DEFAULT_JUDGE_TOP_K,
        min_score: int = DEFAULT_JUDGE_MIN_SCORE,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for the retriever judge. "
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

        raw_output = self._call_llm(prompt)
        judgement = self._parse_judgement(raw_output)

        result = dict(item)
        result["judge"] = judgement
        result["judge_prompt"] = prompt
        
        # Log judge result
        chunk_id = item.get("chunk_id", "N/A")
        final_score = judgement.get("final_score", "N/A")
        score_band = judgement.get("score_band", "N/A")
        reasoning = judgement.get("reasoning", "")
        # Format reasoning to first 150 chars
        reasoning_preview = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
        
        logger.info(
            "Judge result for chunk_id={}: final_score={} score_band='{}' reasoning='{}'",
            chunk_id, final_score, score_band, reasoning_preview
        )
        
        return result

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry logic, falling back to reasoning field if content is empty."""
        messages = [{"role": "user", "content": prompt}]
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    stream=False,
                )
                content = response.choices[0].message.content
                reasoning = response.choices[0].message.reasoning
                
                if not content and reasoning:
                    return reasoning
                
                if content:
                    return str(content).strip()
                
                if attempt < max_retries - 1:
                    logger.warning("Judge LLM returned empty content and reasoning, retrying ({}/{})", attempt + 1, max_retries)
                    continue
                    
                return ""
            except Exception as e:
                logger.error("LLM call failed for judge (attempt {}/{}): {}", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    continue
                return ""

    def _build_prompt(self, user_query: str, retrieved_chunk: str) -> str:
        try:
            prompt = self._prompt_template.replace("{user_query}", user_query.strip()).replace(
                "{retrieved_chunk}", retrieved_chunk.strip()
            )
            return prompt
        except (KeyError, AttributeError) as exc:
            raise ValueError(
                f"Retriever judge prompt template is missing a required placeholder: {exc}"
            ) from exc

    @staticmethod
    def _strip_preamble(text: str) -> str:
        """Remove common meta-commentary prefixes that reasoning models emit before JSON."""
        preamble_patterns = [
            r"^(?:I understand|I will|Okay|Sure|Here is|Let me|Certainly|Right)[^\n]*\n",
            r"^I understand my role[^{]*",
            r"^\d+\.\s*\[User Query\][^{]*",
            r"^<\|channel\>thought\n<channel\|>",
        ]
        result = text
        for pattern in preamble_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.DOTALL).strip()
        return result

    @staticmethod
    def _extract_json(raw_output: str) -> tuple[dict[str, Any] | None, str]:
        """Extract the first valid JSON object from the output using balanced brace matching."""
        cleaned = raw_output.strip()

        if "```" in cleaned:
            fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if fence_match:
                cleaned = fence_match.group(1).strip()

        cleaned = RetrieverJudgeNode._strip_preamble(cleaned)

        candidates: list[str] = []
        i = 0
        while i < len(cleaned):
            if cleaned[i] == "{":
                depth = 0
                in_string = False
                escape_next = False
                for j in range(i, len(cleaned)):
                    ch = cleaned[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == "\\":
                        escape_next = True
                        continue
                    if ch == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(cleaned[i:j + 1])
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1

        for candidate in candidates:
            try:
                return json.loads(candidate), ""
            except json.JSONDecodeError:
                continue

        return None, cleaned

    def _parse_judgement(self, raw_output: str) -> dict[str, Any]:
        """Extract the JSON object from the LLM output and normalize it."""
        parsed, leftover = self._extract_json(raw_output)

        if parsed is None:
            try:
                fallback = json.loads(leftover)
                parsed = fallback
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

        final_score = max(0, min(10, final_score))

        return {
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "score_band": str(parsed.get("score_band", "")).strip() or "unknown",
            "final_score": final_score,
            "raw_output": raw_output,
        }
