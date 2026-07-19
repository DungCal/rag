from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger

from pipelines.agent_pipeline.shared import PromptNodeResult, SafetyInputNode, SafetyOutputNode


class GuardTool:
    """Wraps a LangChain tool so every invocation is guarded by safety nodes.

    Order of operations:
      1. SafetyInputNode guards the serialized tool input.
         - Safety violations cause an immediate rejection result.
         - PII is anonymized and the anonymized input is forwarded.
      2. The underlying tool is invoked.
      3. SafetyOutputNode guards the serialized tool output.
         - Safety violations replace the output with a refusal message.
         - PII is anonymized in the output.
    """

    def __init__(
        self,
        tool: BaseTool,
        *,
        input_guard: SafetyInputNode | None = None,
        output_guard: SafetyOutputNode | None = None,
    ) -> None:
        self.tool = tool
        self.input_guard = input_guard
        self.output_guard = output_guard

    async def ainvoke(self, tool_input: dict[str, Any]) -> Any:
        guarded_input = tool_input
        if self.input_guard is not None:
            serialized_input = json.dumps(tool_input, ensure_ascii=False)
            logger.debug("GuardTool input guard for tool={}", self.tool.name)
            input_result = self.input_guard.run(serialized_input)
            if input_result.rejected:
                logger.warning("Tool input rejected by safety guard: tool={}", self.tool.name)
                return {
                    "error": "Tool input rejected by safety guard",
                    "tool": self.tool.name,
                    "result": None,
                }
            # Use the anonymized input if PII was detected.
            guarded_input = json.loads(input_result.query)

        logger.debug("GuardTool invoking tool={}", self.tool.name)
        raw_output = await self.tool.ainvoke(guarded_input)

        if self.output_guard is None:
            return {
                "tool": self.tool.name,
                "result": raw_output,
            }

        serialized_output = json.dumps(raw_output, ensure_ascii=False)
        node_result = PromptNodeResult(
            route="tool_output",
            response=serialized_output,
            raw_output=serialized_output,
            prompt="",
        )
        output_result = self.output_guard.run(node_result)
        logger.debug("GuardTool output guard for tool={}", self.tool.name)

        if output_result.safety_result and not output_result.safety_result.passed:
            logger.warning("Tool output rejected by safety guard: tool={}", self.tool.name)
            return {
                "error": "Tool output rejected by safety guard",
                "tool": self.tool.name,
                "result": json.loads(output_result.node_result.response),
            }

        return {
            "tool": self.tool.name,
            "result": json.loads(output_result.node_result.response),
            "pii_anonymized": output_result.pii_result is not None,
        }


def wrap_tool_with_guard(
    tool: BaseTool,
    *,
    input_guard: SafetyInputNode | None = None,
    output_guard: SafetyOutputNode | None = None,
) -> StructuredTool:
    """Return a LangChain StructuredTool whose calls are routed through GuardTool."""
    guard = GuardTool(tool, input_guard=input_guard, output_guard=output_guard)

    async def _arun(**kwargs: Any) -> Any:
        return await guard.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=_arun,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )
