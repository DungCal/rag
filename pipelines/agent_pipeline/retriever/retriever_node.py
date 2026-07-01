from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.indexing_pipeline.pdf_rag import DEFAULT_MODEL_NAME, PDFRAG
from pipelines.agent_pipeline.routers.routing_classification import PromptRouteDecision


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
DEFAULT_PINECONE_NAMESPACE = "default"
ROUTING_MESSAGES = {
    "Greeting query detected. Routing ended without document retrieval.",
    "Query is relevant to the document scope. Proceeding to retrieval.",
    "Query is outside the document scope. Routing ended without retrieval.",
}


def _load_env_value(name: str, env_path: Path = ENV_FILE_PATH) -> str | None:
    if value := os.getenv(name):
        return value
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            resolved = value.strip().strip('"').strip("'")
            return resolved or None

    return None


@dataclass
class RetrieverNodeResult:
    route: str
    query: str
    results: list[dict[str, Any]]


class RetrieverNode:
    def __init__(
        self,
        *,
        index_name: str,
        namespace: str = DEFAULT_PINECONE_NAMESPACE,
        model_name: str = DEFAULT_MODEL_NAME,
        provider: str | None = None,
        top_k: int = 5,
        api_key: str | None = None,
        use_fp16: bool = False,
    ) -> None:
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError(
                "pinecone is required for Pinecone-backed retrieval. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        resolved_api_key = api_key or _load_env_value("PINECONE_API_KEY")
        if not resolved_api_key:
            raise ValueError("Missing PINECONE_API_KEY in the environment or .env file")

        self.index_name = index_name
        self.namespace = namespace
        self.top_k = top_k
        self._rag = PDFRAG(model_name=model_name, provider=provider, use_fp16=use_fp16)
        self._client = Pinecone(api_key=resolved_api_key)
        self._index = self._client.Index(name=index_name)

    def run(self, decision: PromptRouteDecision, original_query: str) -> RetrieverNodeResult | None:
        if decision.route in {"greeting", "off_topic"}:
            return None

        retrieval_query = self._resolve_query(decision, original_query)
        query_vector = self._rag.embed_query(retrieval_query)[0].tolist()
        response = self._index.query(
            vector=query_vector,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])

        return RetrieverNodeResult(
            route="retrieval",
            query=retrieval_query,
            results=[self._normalize_match(match) for match in matches or []],
        )

    def _resolve_query(self, decision: PromptRouteDecision, original_query: str) -> str:
        candidate = decision.message.strip()
        if candidate and candidate not in ROUTING_MESSAGES:
            return candidate
        return original_query.strip()

    @staticmethod
    def _normalize_match(match: Any) -> dict[str, Any]:
        metadata = getattr(match, "metadata", None)
        if metadata is None and isinstance(match, dict):
            metadata = match.get("metadata", {})
        metadata = dict(metadata or {})

        score = getattr(match, "score", None)
        if score is None and isinstance(match, dict):
            score = match.get("score", 0.0)

        vector_id = getattr(match, "id", None)
        if vector_id is None and isinstance(match, dict):
            vector_id = match.get("id")

        payload: dict[str, Any] = {
            "id": vector_id,
            "score": float(score or 0.0),
        }
        payload.update(metadata)
        return payload
