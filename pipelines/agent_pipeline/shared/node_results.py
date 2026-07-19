from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptNodeResult:
    route: str
    response: str
    raw_output: str
    prompt: str
