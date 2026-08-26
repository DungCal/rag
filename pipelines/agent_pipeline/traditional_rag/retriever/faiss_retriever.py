from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipelines.indexing_pipeline.index_store import load_index
from pipelines.indexing_pipeline.pdf_rag import DEFAULT_MODEL_NAME, PDFRAG
from pipelines.indexing_pipeline.retriever import retrieve_results
from pipelines.agent_pipeline.traditional_rag.routers.routing_classification import PromptRouteDecision

from .retriever_node import RetrieverNodeResult, ROUTING_MESSAGES

logger = logging.getLogger(__name__)


class FAISSRetrieverNode:
    """Retriever that searches a local FAISS index (from recursive or hierarchical indexing)."""

    def __init__(
        self,
        *,
        index_dir: str = "storage_hierarchical",
        model_name: str = DEFAULT_MODEL_NAME,
        provider: str | None = None,
        top_k: int = 5,
        use_fp16: bool = False,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self._rag = PDFRAG(model_name=model_name, provider=provider, use_fp16=use_fp16)
        self._index = None
        self._records = None

        try:
            self._index, self._records = load_index(self.index_dir)
        except FileNotFoundError:
            logger.warning("FAISS index not found at {}; returning empty results", self.index_dir)

    def run(self, decision: PromptRouteDecision, original_query: str) -> RetrieverNodeResult | None:
        if decision.route in {"greeting", "off_topic"}:
            return None

        if self._index is None:
            retrieval_query = self._resolve_query(decision, original_query)
            return RetrieverNodeResult(
                route="retrieval",
                query=retrieval_query,
                results=[],
            )

        retrieval_query = self._resolve_query(decision, original_query)
        query_vector = self._rag.embed_query(retrieval_query)
        results = retrieve_results(self._index, self._records, query_vector, top_k=self.top_k)
        return RetrieverNodeResult(
            route="retrieval",
            query=retrieval_query,
            results=results,
        )

    def _resolve_query(self, decision: PromptRouteDecision, original_query: str) -> str:
        candidate = decision.message.strip()
        if candidate and candidate not in ROUTING_MESSAGES:
            return candidate
        return original_query.strip()
