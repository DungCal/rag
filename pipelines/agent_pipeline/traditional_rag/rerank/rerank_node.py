from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rerank.hf_reranker import DEFAULT_RERANKER_MODEL_NAME, HuggingFaceHubReranker


DEFAULT_RERANK_INPUT_TOP_K = 20
DEFAULT_RERANK_OUTPUT_TOP_K = 5


@dataclass
class RerankNodeResult:
    query: str
    input_results: list[dict[str, Any]]
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
        input_top_k: int = DEFAULT_RERANK_INPUT_TOP_K,
        output_top_k: int = DEFAULT_RERANK_OUTPUT_TOP_K,
    ) -> None:
        self.input_top_k = input_top_k
        self.output_top_k = output_top_k
        self._reranker = HuggingFaceHubReranker(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            instruction=instruction or "",
            use_fp16=use_fp16,
            apply_sigmoid=apply_sigmoid,
        )

    def run(self, query: str, results: list[dict[str, Any]]) -> RerankNodeResult:
        rerank_inputs = [dict(item) for item in results[: self.input_top_k]]
        reranked = self._reranker.rerank(query, rerank_inputs)[: self.output_top_k]
        return RerankNodeResult(
            query=query,
            input_results=rerank_inputs,
            results=reranked,
        )
