from __future__ import annotations

import json
import math
import os
from typing import Any
from urllib import error, request

from pipelines.indexing_pipeline.llm import _load_api_key_from_env_file


DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_API_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model_name}"


class HuggingFaceHubReranker:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        batch_size: int = 8,
        max_length: int = 4096,
        instruction: str = "",
        use_fp16: bool = False,
        apply_sigmoid: bool = False,
        api_key: str | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("HF_TOKEN") or _load_api_key_from_env_file()
        if not resolved_api_key:
            raise ValueError("Missing HF_TOKEN in the environment or .env file for Hugging Face inference access")

        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max_length
        self.instruction = instruction.strip()
        self.use_fp16 = use_fp16
        self.apply_sigmoid = apply_sigmoid
        self._api_key = resolved_api_key
        self._api_url = DEFAULT_RERANK_API_URL_TEMPLATE.format(model_name=model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query must not be empty")
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            scores.extend(self._score_batch(query_text, batch))
        return scores

    def rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return []

        documents = [str(item.get("text", "")) for item in results]
        scores = self.score(query, documents)

        enriched: list[dict[str, Any]] = []
        for retrieval_rank, (item, rerank_score) in enumerate(zip(results, scores), start=1):
            payload = dict(item)
            payload["retrieval_rank"] = retrieval_rank
            payload["retrieval_score"] = float(item.get("score", 0.0))
            payload["rerank_score"] = rerank_score
            enriched.append(payload)

        reranked = sorted(enriched, key=lambda item: item["rerank_score"], reverse=True)
        for rerank_rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rerank_rank
            item["rank_shift"] = item["retrieval_rank"] - rerank_rank

        return reranked

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        payload = {
            "inputs": [
                {
                    "text": query,
                    "text_pair": document,
                }
                for document in documents
            ],
            "parameters": {
                "function_to_apply": "none",
                "top_k": 1,
            },
            "options": {
                "wait_for_model": True,
            },
        }

        try:
            response = self._post_json(payload)
            return self._parse_scores(response, len(documents))
        except ValueError:
            return [self._score_single(query, document) for document in documents]

    def _score_single(self, query: str, document: str) -> float:
        payload = {
            "inputs": {
                "text": query,
                "text_pair": document,
            },
            "parameters": {
                "function_to_apply": "none",
                "top_k": 1,
            },
            "options": {
                "wait_for_model": True,
            },
        }
        response = self._post_json(payload)
        return self._apply_score_transform(self._extract_score(response))

    def _post_json(self, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Hugging Face router reranker request failed with HTTP {exc.code} for {self.model_name}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach Hugging Face router for {self.model_name}: {exc}") from exc

    def _parse_scores(self, response: Any, expected_count: int) -> list[float]:
        if not isinstance(response, list) or len(response) != expected_count:
            raise ValueError("Unexpected batched reranker response shape")
        return [self._apply_score_transform(self._extract_score(item)) for item in response]

    def _extract_score(self, payload: Any) -> float:
        if isinstance(payload, (int, float)):
            return float(payload)

        if isinstance(payload, dict):
            if "score" in payload and isinstance(payload["score"], (int, float)):
                return float(payload["score"])
            if "scores" in payload:
                return self._extract_score(payload["scores"])
            if "label" in payload and "score" in payload:
                return float(payload["score"])

        if isinstance(payload, list):
            if not payload:
                return 0.0
            if all(isinstance(item, (int, float)) for item in payload):
                return float(max(payload))

            labeled_scores = [
                item for item in payload
                if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
            ]
            if labeled_scores:
                preferred_labels = {"true", "relevant", "yes", "positive", "label_1"}
                for item in labeled_scores:
                    label = str(item.get("label", "")).strip().lower()
                    if label in preferred_labels:
                        return float(item["score"])
                return float(max(item["score"] for item in labeled_scores))

            if len(payload) == 1:
                return self._extract_score(payload[0])

        raise ValueError(f"Unsupported Hugging Face reranker response payload: {payload!r}")

    def _apply_score_transform(self, score: float) -> float:
        if not self.apply_sigmoid:
            return score
        return 1.0 / (1.0 + math.exp(-score))
