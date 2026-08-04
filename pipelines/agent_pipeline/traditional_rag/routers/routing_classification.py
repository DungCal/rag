from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from pipelines.indexing_pipeline.llm import DEFAULT_LLM_MODEL, DEFAULT_HF_INFERENCE_PROVIDER, _load_api_key_from_env_file, _load_provider_from_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROUTE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "route_node_prompt.txt"
DEFAULT_DOCUMENT_SCOPE = (
    "The document is a TYM diesel tractor operator manual. "
    "It covers general tractor construction, safety precautions, instrument controls, "
    "touch monitor usage, telematics, maintenance, and troubleshooting. "
    "Topics within the scope include engine operation, PTO (Power Take-Off) use, "
    "towing, DPF regeneration, fluid replacement, and agricultural safety standards."
)
VALID_ROUTE_LABELS = {"greeting", "related", "off_topic"}
LABEL_TO_ROUTE = {
    "greeting": "greeting",
    "related": "retrieval",
    "off_topic": "off_topic",
}


@dataclass
class PromptRouteDecision:
    route: str
    label: str
    message: str
    raw_output: str
    prompt: str


class PromptQueryRouter:
    def __init__(
        self,
        *,
        prompt_path: str | Path = DEFAULT_ROUTE_PROMPT_PATH,
        scope: str = DEFAULT_DOCUMENT_SCOPE,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for the prompt router. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.prompt_path = Path(prompt_path)
        self.scope = scope.strip()
        if not self.scope:
            raise ValueError("Document scope must not be empty")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Router prompt template not found: {self.prompt_path.resolve()}")

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
                f"Router prompt template is missing a required placeholder: {exc}"
            ) from exc

    def route(self, user_query: str) -> PromptRouteDecision:
        prompt = self.build_prompt(user_query)
        
        # Retry logic for empty responses
        max_retries = 3
        raw_output = ""
        label = None
        
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=300,
                    temperature=0.0,
                    stream=False,
                )
                raw_output = str(response.choices[0].message.content or "").strip()
                logger.info("Router raw output (attempt {}): {!r}", attempt + 1, raw_output)
                
                if not raw_output:
                    logger.warning("Router returned empty response, retrying... ({}/{})", attempt + 1, max_retries)
                    continue
                
                # Try to normalize the label
                try:
                    label = self._normalize_label(raw_output)
                    break  # Success
                except ValueError as e:
                    logger.warning("Router returned invalid label '{}', retrying... ({}/{}): {}", 
                                 raw_output, attempt + 1, max_retries, e)
                    continue
                    
            except Exception as e:
                logger.error("Router invocation failed (attempt {}/{}): {}", attempt + 1, max_retries, e)
                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Router model failed after {max_retries} retries. "
                        "This may indicate an issue with the HuggingFace API, model availability, or network connectivity."
                    ) from e
        
        if label is None:
            raise ValueError(
                f"Router model returned empty or invalid response after {max_retries} retries. "
                "This may indicate an issue with the HuggingFace API or model availability."
            )
        
        route = LABEL_TO_ROUTE[label]

        return PromptRouteDecision(
            route=route,
            label=label,
            message=self._build_message(label),
            raw_output=raw_output,
            prompt=prompt,
        )

    def _normalize_label(self, raw_output: str) -> str:
        normalized = raw_output.strip().lower()
        if normalized in VALID_ROUTE_LABELS:
            return normalized

        first_line = normalized.splitlines()[0].strip() if normalized else ""
        if first_line in VALID_ROUTE_LABELS:
            return first_line

        first_token = first_line.split()[0] if first_line else ""
        cleaned_token = first_token.rstrip(".,:;!?")
        if cleaned_token in VALID_ROUTE_LABELS:
            return cleaned_token

        raise ValueError(
            "Router model returned an invalid label. "
            f"Expected one of {sorted(VALID_ROUTE_LABELS)}, got: {raw_output!r}"
        )

    def _build_message(self, label: str) -> str:
        if label == "greeting":
            return "Greeting query detected. Routing ended without document retrieval."
        if label == "related":
            return "Query is relevant to the document scope. Proceeding to retrieval."
        return "Query is outside the document scope. Routing ended without retrieval."
