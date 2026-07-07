from __future__ import annotations

from pathlib import Path

from loguru import logger

from pipelines.agent_pipeline.routers.routing_response import PromptNodeResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFUSAL_MESSAGES_PATH = PROJECT_ROOT / "prompts" / "guardrails" / "refusal_messages.txt"


class RejectedNode:
    """Returns a fixed refusal message for queries/responses that fail safety checks."""

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

        self._message = self.refusal_messages_path.read_text(encoding="utf-8").strip()
        if not self._message:
            raise ValueError(
                f"Refusal messages file is empty: {self.refusal_messages_path.resolve()}"
            )

    def run(self, text: str) -> PromptNodeResult:
        """Return a rejection result for the provided text."""
        logger.info("Returning safety refusal for text={!r}", text)
        return PromptNodeResult(
            route="rejected",
            response=self._message,
            raw_output=self._message,
            prompt=str(self.refusal_messages_path),
        )
