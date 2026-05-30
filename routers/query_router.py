from __future__ import annotations

import re
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class RouteDecision:
    route: str
    message: str
    similarity_score: float | None = None


class QueryRouter:
    GREETING_PATTERNS = (
        r"hi",
        r"hello",
        r"hey",
        r"good morning",
        r"good afternoon",
        r"good evening",
    )

    def __init__(self, off_topic_threshold: float = 0.35) -> None:
        self.off_topic_threshold = off_topic_threshold
        greeting_group = "|".join(self.GREETING_PATTERNS)
        self._greeting_regex = re.compile(rf"^\s*(?:{greeting_group})[!.?,\s]*$", re.IGNORECASE)

    def route(self, query: str, query_vector: np.ndarray, index: faiss.Index) -> RouteDecision:
        if self._is_greeting(query):
            return RouteDecision(
                route="greeting",
                message="Greeting query detected. Routing ended without document retrieval.",
            )

        scores, _ = index.search(query_vector, 1)
        top_score = float(scores[0][0]) if scores.size else -1.0
        if top_score < self.off_topic_threshold:
            return RouteDecision(
                route="off_topic",
                message="Query appears unrelated to the indexed documents. Routing ended without retrieval.",
                similarity_score=top_score,
            )

        return RouteDecision(
            route="retrieval",
            message="Query is relevant to the indexed documents. Proceeding to retrieval.",
            similarity_score=top_score,
        )

    def _is_greeting(self, query: str) -> bool:
        return bool(self._greeting_regex.fullmatch(query.strip()))
