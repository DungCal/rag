from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from commons.guardrails.pii_presidio import PIIDetectionResult, PresidioPIIGuard
from commons.guardrails.guardrailsAI_safetycheck import GuardrailsSafetyGuard, SafetyCheckResult
from pipelines.agent_pipeline.rejected_nodes import RejectedNode, PromptNodeResult


@dataclass
class SafetyOutputResult:
    """Result of applying the output safety + PII guard to a generated response."""

    node_result: PromptNodeResult
    pii_result: PIIDetectionResult | None = None
    safety_result: SafetyCheckResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
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


class SafetyOutputNode:
    """Guards a generated textual response (e.g. greeting/off-topic answers).

    Order of operations:
      1. Safety check. If it fails, replace the response with a refusal message.
      2. PII check. If PII is found, anonymize the response and return it.
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

    def run(self, node_result: PromptNodeResult) -> SafetyOutputResult:
        if not node_result.response:
            return SafetyOutputResult(node_result=node_result)

        text = node_result.response

        # 1. Safety check first.
        safety_result = self.safety_guard.check(text)
        if not safety_result.passed:
            for violation in safety_result.violations:
                logger.warning(
                    "Output safety violation: text={!r}, validator={}, error={}",
                    text,
                    violation.get("validator"),
                    violation.get("error"),
                )
            rejected = self.rejected_node.run(text)
            return SafetyOutputResult(
                node_result=PromptNodeResult(
                    route="rejected",
                    response=rejected.response,
                    raw_output=rejected.raw_output,
                    prompt=node_result.prompt,
                ),
                safety_result=safety_result,
            )

        logger.info("Output safety passed: text={!r}", text)

        # 2. PII check and anonymize if needed.
        pii_result = self.pii_guard.check(text)
        if pii_result.entities:
            logger.info(
                "Output PII detected and anonymized: entities={}",
                pii_result.entities,
            )
            return SafetyOutputResult(
                node_result=PromptNodeResult(
                    route=node_result.route,
                    response=pii_result.anonymized_text,
                    raw_output=pii_result.anonymized_text,
                    prompt=node_result.prompt,
                ),
                pii_result=pii_result,
                safety_result=safety_result,
            )

        logger.info("Output PII check passed: text={!r}", text)
        return SafetyOutputResult(
            node_result=node_result,
            pii_result=pii_result,
            safety_result=safety_result,
        )
