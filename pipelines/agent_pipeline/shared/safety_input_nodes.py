from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipelines.agent_pipeline.shared.guardrails import PIIDetectionResult, PresidioPIIGuard, GuardrailsSafetyGuard, SafetyCheckResult
from pipelines.agent_pipeline.shared.rejected_nodes import RejectedNode, PromptNodeResult


@dataclass
class SafetyInputResult:
    """Result of applying the input safety + PII guard to a user query."""

    query: str
    rejected: bool = False
    node_result: PromptNodeResult | None = None
    pii_result: PIIDetectionResult | None = None
    safety_result: SafetyCheckResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rejected": self.rejected,
            "pii_result": {
                "original_text": self.pii_result.original_text,
                "anonymized_text": self.pii_result.anonymized_text,
                "entities": self.pii_result.entities,
            }
            if self.pii_result
            else None,
            "safety_result": {
                "text": self.safety_result.text,
                "passed": self.safety_result.passed,
                "violations": self.safety_result.violations,
            }
            if self.safety_result
            else None,
        }


class SafetyInputNode:
    """Guards the incoming user query.

    Order of operations:
      1. Safety check. If it fails, reject immediately and return a refusal node.
      2. PII check. If PII is found, anonymize the query and continue routing.
    """

    def __init__(
        self,
        pii_guard: PresidioPIIGuard | None = None,
        safety_guard: GuardrailsSafetyGuard | None = None,
        rejected_node: RejectedNode | None = None,
    ) -> None:
        self.pii_guard = pii_guard or PresidioPIIGuard()
        self.safety_guard = safety_guard or GuardrailsSafetyGuard()
        self.rejected_node = rejected_node or RejectedNode()

    def run(self, query: str) -> SafetyInputResult:
        if not query or not query.strip():
            raise ValueError("User query must not be empty")

        original_query = query.strip()

        # 1. Safety check first.
        safety_result = self.safety_guard.check(original_query)
        if not safety_result.passed:
            for violation in safety_result.violations:
                logger.warning(
                    "Input safety violation: text={!r}, validator={}, error={}",
                    original_query,
                    violation.get("validator"),
                    violation.get("error"),
                )
            return SafetyInputResult(
                query=original_query,
                rejected=True,
                node_result=self.rejected_node.run(original_query, safety_result.violations),
                safety_result=safety_result,
            )

        logger.info("Input safety passed: text={!r}", original_query)

        # 2. PII check and anonymize if needed.
        pii_result = self.pii_guard.check(original_query)
        if pii_result.entities:
            logger.info(
                "Input PII detected and anonymized: entities={}, "
                "original_length={}, anonymized_length={}",
                pii_result.entities,
                len(original_query),
                len(pii_result.anonymized_text),
            )
            return SafetyInputResult(
                query=pii_result.anonymized_text,
                rejected=False,
                pii_result=pii_result,
                safety_result=safety_result,
            )

        logger.info("Input PII check passed: text={!r}", original_query)
        return SafetyInputResult(
            query=original_query,
            rejected=False,
            pii_result=pii_result,
            safety_result=safety_result,
        )
