from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrieverJudgeNodeResult:
    query: str
    retriever_results: list[dict[str, Any]]
    rerank_results: list[dict[str, Any]]
    pairwise_comparisons: list[dict[str, Any]]
    results: list[dict[str, Any]]


class RetrieverJudgeNode:
    def __init__(self, *, top_k: int = 5) -> None:
        self.top_k = top_k

    def run(
        self,
        *,
        query: str,
        retriever_results: list[dict[str, Any]],
        rerank_results: list[dict[str, Any]],
    ) -> RetrieverJudgeNodeResult:
        top_retriever = [dict(item) for item in retriever_results[: self.top_k]]
        top_rerank = [dict(item) for item in rerank_results[: self.top_k]]
        comparisons = self._build_pairwise_comparisons(top_retriever, top_rerank)
        merged = self._merge_top_results(top_retriever, top_rerank)
        return RetrieverJudgeNodeResult(
            query=query,
            retriever_results=top_retriever,
            rerank_results=top_rerank,
            pairwise_comparisons=comparisons,
            results=merged,
        )

    def _build_pairwise_comparisons(
        self,
        retriever_results: list[dict[str, Any]],
        rerank_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        max_len = max(len(retriever_results), len(rerank_results))
        for index in range(max_len):
            retriever_item = retriever_results[index] if index < len(retriever_results) else None
            rerank_item = rerank_results[index] if index < len(rerank_results) else None
            retriever_identity = self._result_identity(retriever_item) if retriever_item else None
            rerank_identity = self._result_identity(rerank_item) if rerank_item else None
            retriever_score = self._retriever_score(retriever_item) if retriever_item else None
            rerank_score = self._rerank_score(rerank_item) if rerank_item else None

            comparisons.append(
                {
                    "position": index + 1,
                    "same_chunk_and_page": bool(
                        retriever_identity is not None
                        and rerank_identity is not None
                        and retriever_identity == rerank_identity
                    ),
                    "retriever_chunk_id": retriever_item.get("chunk_id") if retriever_item else None,
                    "retriever_page_number": retriever_item.get("page_number") if retriever_item else None,
                    "retriever_score": retriever_score,
                    "rerank_chunk_id": rerank_item.get("chunk_id") if rerank_item else None,
                    "rerank_page_number": rerank_item.get("page_number") if rerank_item else None,
                    "rerank_score": rerank_score,
                    "rerank_score_higher": bool(
                        rerank_score is not None
                        and retriever_score is not None
                        and rerank_score > retriever_score
                    ),
                }
            )

        return comparisons

    def _merge_top_results(
        self,
        retriever_results: list[dict[str, Any]],
        rerank_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged_by_identity: dict[tuple[Any, Any, Any], dict[str, Any]] = {}

        for source_name, items in (("retriever", retriever_results), ("rerank", rerank_results)):
            for item in items:
                normalized = self._normalize_candidate(item, source_name)
                identity = self._result_identity(normalized)
                existing = merged_by_identity.get(identity)
                if existing is None or normalized["relevant_score"] > existing["relevant_score"]:
                    merged_by_identity[identity] = normalized

        merged = sorted(
            merged_by_identity.values(),
            key=lambda item: item["relevant_score"],
            reverse=True,
        )
        for rank, item in enumerate(merged[: self.top_k], start=1):
            item["judge_rank"] = rank
        return merged[: self.top_k]

    @staticmethod
    def _normalize_candidate(item: dict[str, Any], source_name: str) -> dict[str, Any]:
        candidate = dict(item)
        candidate["source"] = source_name
        if source_name == "rerank":
            candidate["relevant_score"] = float(item.get("rerank_score", 0.0))
        else:
            candidate["relevant_score"] = float(item.get("score", item.get("retrieval_score", 0.0)))
        return candidate

    @staticmethod
    def _result_identity(item: dict[str, Any] | None) -> tuple[Any, Any, Any] | None:
        if item is None:
            return None
        return (
            item.get("id"),
            item.get("chunk_id"),
            item.get("page_number"),
        )

    @staticmethod
    def _retriever_score(item: dict[str, Any]) -> float:
        return float(item.get("score", item.get("retrieval_score", 0.0)))

    @staticmethod
    def _rerank_score(item: dict[str, Any]) -> float:
        return float(item.get("rerank_score", 0.0))
