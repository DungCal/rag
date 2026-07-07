from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class PIIDetectionResult:
    """Result of a Presidio PII analyze + anonymize pass."""

    original_text: str
    anonymized_text: str
    entities: list[dict[str, Any]] = field(default_factory=list)


class PresidioPIIGuard:
    """HTTP client for a running Presidio Analyzer + Anonymizer service."""

    def __init__(
        self,
        analyzer_url: str = "http://localhost:5002/analyze",
        anonymizer_url: str = "http://localhost:5001/anonymize",
        language: str = "en",
        timeout: float = 30.0,
    ) -> None:
        self.analyzer_url = analyzer_url
        self.anonymizer_url = anonymizer_url
        self.language = language
        self.timeout = timeout

    def check(self, text: str) -> PIIDetectionResult:
        """Analyze and anonymize ``text``. Returns the anonymized text plus entities."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        analyzer_payload = {
            "text": text,
            "language": self.language,
        }
        try:
            analyzer_response = requests.post(
                self.analyzer_url,
                json=analyzer_payload,
                timeout=self.timeout,
            )
            analyzer_response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Presidio analyzer request failed: {exc}") from exc

        entities = analyzer_response.json()
        if not isinstance(entities, list):
            raise ValueError(f"Unexpected analyzer response format: {entities!r}")

        anonymizer_payload = {
            "text": text,
            "analyzer_results": entities,
        }
        try:
            anonymizer_response = requests.post(
                self.anonymizer_url,
                json=anonymizer_payload,
                timeout=self.timeout,
            )
            anonymizer_response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Presidio anonymizer request failed: {exc}") from exc

        anonymized_data = anonymizer_response.json()
        if isinstance(anonymized_data, dict):
            anonymized_text = anonymized_data.get("text", text)
        else:
            anonymized_text = text

        return PIIDetectionResult(
            original_text=text,
            anonymized_text=anonymized_text,
            entities=entities,
        )

    def sanitize(self, text: str) -> str:
        """Convenience shortcut that returns only the anonymized text."""
        return self.check(text).anonymized_text
