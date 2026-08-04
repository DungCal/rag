from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PromptNodeResult:
    route: str
    response: str
    raw_output: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptNodeResult":
        return cls(
            route=data.get("route", "unknown"),
            response=data.get("response", ""),
            raw_output=data.get("raw_output", ""),
            prompt=data.get("prompt", ""),
        )
