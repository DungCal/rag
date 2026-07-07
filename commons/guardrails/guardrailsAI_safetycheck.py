from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SafetyCheckResult:
    """Result of a Guardrails safety check against one or more validators."""

    text: str
    passed: bool
    violations: list[dict[str, Any]] = field(default_factory=list)
    validated_output: str | None = None


class GuardrailsSafetyGuard:
    """Multi-validator safety guard using Guardrails Hub validators.

    Each validator is run independently so that the caller knows exactly which
    validator triggered a failure.
    """

    def __init__(
        self,
        nsfw_threshold: float = 0.95,
        toxic_threshold: float = 0.5,
        validation_method: str = "sentence",
        on_fail: str = "exception",
    ) -> None:
        self.nsfw_threshold = nsfw_threshold
        self.toxic_threshold = toxic_threshold
        self.validation_method = validation_method
        self.on_fail = on_fail

        try:
            from guardrails.hub import NSFWText, ProfanityFree, ToxicLanguage
            from guardrails import Guard
        except ImportError as exc:
            raise ImportError(
                "guardrails is required for safety checking. "
                "Install it with `pip install guardrails-ai` or add it to requirements.txt."
            ) from exc

        self._Guard = Guard
        self._validators: list[tuple[str, Callable[[], Any]]] = []

        if NSFWText is not None:
            self._validators.append(
                (
                    "NSFWText",
                    lambda: NSFWText(
                        threshold=self.nsfw_threshold,
                        validation_method=self.validation_method,
                        on_fail=self.on_fail,
                    ),
                )
            )

        if ProfanityFree is not None:
            self._validators.append(
                (
                    "ProfanityFree",
                    lambda: ProfanityFree(on_fail=self.on_fail),
                )
            )

        if ToxicLanguage is not None:
            self._validators.append(
                (
                    "ToxicLanguage",
                    lambda: ToxicLanguage(
                        threshold=self.toxic_threshold,
                        validation_method=self.validation_method,
                        on_fail=self.on_fail,
                    ),
                )
            )

    def check(self, text: str) -> SafetyCheckResult:
        """Run all configured validators and return aggregated results."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        violations: list[dict[str, Any]] = []
        validated_output = text

        for validator_name, factory in self._validators:
            try:
                guard = self._Guard().use(factory())
                result = guard.validate(text)
                output = getattr(result, "validated_output", None)
                if output is not None:
                    validated_output = output
            except Exception as exc:  # noqa: BLE001 - guardrails raises varied exceptions
                violations.append(
                    {
                        "validator": validator_name,
                        "error": str(exc),
                    }
                )

        return SafetyCheckResult(
            text=text,
            passed=len(violations) == 0,
            violations=violations,
            validated_output=validated_output,
        )
