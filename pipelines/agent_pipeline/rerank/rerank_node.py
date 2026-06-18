from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rerank.qwen_reranker import DEFAULT_RERANKER_MODEL_NAME, QwenReranker


@dataclass
class RerankNodeResult:
    query: str
    results: list[dict[str, Any]]


class RerankNode:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        batch_size: int = 8,
        max_length: int = 4096,
        instruction: str | None = None,
        use_fp16: bool = False,
        apply_sigmoid: bool = False,
    ) -> None:
        self._reranker = QwenReranker(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            instruction=instruction or "",
            use_fp16=use_fp16,
            apply_sigmoid=apply_sigmoid,
        )

    def run(self, query: str, results: list[dict[str, Any]]) -> RerankNodeResult:
        reranked = self._reranker.rerank(query, results)
        return RerankNodeResult(query=query, results=reranked)
