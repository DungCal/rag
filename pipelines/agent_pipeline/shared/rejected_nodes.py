from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from pipelines.agent_pipeline.shared.node_results import PromptNodeResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFUSAL_MESSAGES_PATH = PROJECT_ROOT / "prompts" / "guardrails" / "refusal_messages.json"


class RejectedNode:
    """Returns context-aware refusal messages based on violation types."""

    def __init__(
        self,
        refusal_messages_path: str | Path | None = None,
    ) -> None:
        self.refusal_messages_path = Path(
            refusal_messages_path if refusal_messages_path is not None else DEFAULT_REFUSAL_MESSAGES_PATH
        )
        if not self.refusal_messages_path.exists():
            raise FileNotFoundError(
                f"Refusal messages file not found: {self.refusal_messages_path.resolve()}"
            )

        # Load structured messages
        with open(self.refusal_messages_path, encoding="utf-8") as f:
            self._messages = json.load(f)
        
        if not self._messages:
            raise ValueError(
                f"Refusal messages file is empty: {self.refusal_messages_path.resolve()}"
            )

    def _extract_violation_types(self, violations: list[dict] | None) -> list[str]:
        """Extract validator names from violations list."""
        if not violations:
            return []
        
        violation_types = []
        for violation in violations:
            validator = violation.get("validator")
            if validator:
                violation_types.append(validator)
        
        return violation_types

    def _select_message(self, violation_types: list[str]) -> str:
        """Select appropriate message based on violation types."""
        if not violation_types:
            return self._messages.get("default", "I'm sorry, but I can't help with that request.")
        
        # If multiple violations, use the multiple_violations message
        if len(violation_types) > 1:
            return self._messages.get(
                "multiple_violations",
                self._messages.get("default", "I'm sorry, but I can't help with that request.")
            )
        
        # Single violation - use specific message if available
        violation_type = violation_types[0]
        return self._messages.get(
            violation_type,
            self._messages.get("default", "I'm sorry, but I can't help with that request.")
        )

    def run(self, text: str, violations: list[dict] | None = None) -> PromptNodeResult:
        """Return a rejection result with context-aware message."""
        violation_types = self._extract_violation_types(violations)
        message = self._select_message(violation_types)
        
        logger.info(
            "Returning safety refusal for text={!r}, violations={}, message_type={}",
            text,
            violation_types,
            "multiple_violations" if len(violation_types) > 1 else (violation_types[0] if violation_types else "default")
        )
        
        return PromptNodeResult(
            route="rejected",
            response=message,
            raw_output=message,
            prompt=str(self.refusal_messages_path),
        )
