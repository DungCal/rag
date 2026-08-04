from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger

from pipelines.agent_pipeline.shared import PromptNodeResult, SafetyInputNode, SafetyOutputNode
from pipelines.agent_pipeline.shared.rejected_nodes import RejectedNode


def _guard_text(
    text: str,
    *,
    tool_name: str,
    item_index: int,
    output_guard: SafetyOutputNode,
) -> tuple[str, bool, bool]:
    """Guard a single text string with SafetyOutputNode.

    Returns:
        (text, passed_safety, pii_anonymized)
        - text: the (possibly anonymized) text, or "" if safety violated
        - passed_safety: True if safety check passed
        - pii_anonymized: True if PII was detected and text was anonymized
    """
    if not text:
        return text, True, False

    node_result = PromptNodeResult(
        route="tool_output",
        response=text,
        raw_output=text,
        prompt="",
    )
    output_result = output_guard.run(node_result)

    if output_result.safety_result and not output_result.safety_result.passed:
        for violation in output_result.safety_result.violations:
            logger.warning(
                "Tool {} item #{} safety violation: validator={}, error={}",
                tool_name,
                item_index,
                violation.get("validator"),
                violation.get("error"),
            )
        return "", False, False

    if output_result.pii_result and output_result.pii_result.entities:
        logger.info(
            "Tool {} item #{} PII anonymized: entities={}",
            tool_name,
            item_index,
            output_result.pii_result.entities,
        )
        return output_result.pii_result.anonymized_text, True, True

    return text, True, False


def _all_violated_rejection(tool_name: str) -> dict[str, Any]:
    """Build a rejection result when all tool results violated safety policy."""
    rejection = RejectedNode().run("", None)
    logger.warning(
        "All results from tool {} violated safety policy; returning rejection",
        tool_name,
    )
    return {
        "error": "All tool results violated safety policy",
        "tool": tool_name,
        "result": rejection.response,
    }


class GuardTool:
    """Wraps a LangChain tool so every invocation is guarded by safety nodes.

    Order of operations:
      1. SafetyInputNode guards the serialized tool input.
         - Safety violations cause an immediate rejection result.
         - PII is anonymized and the anonymized input is forwarded.
      2. The underlying tool is invoked.
      3. SafetyOutputNode guards the tool output.
         - For web_search (list of result dicts): ``title + snippet`` is
           checked per result. Safety-violating results are dropped;
           PII is anonymized in place.
         - For other tools: the raw output is JSON-serialized and checked
           as a single string.
         - If every result is safety-rejected, a RejectedNode refusal
           message is returned.
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

        # Per-result guarding for web_search.
        if self.tool.name == "web_search" and isinstance(raw_output, list):
            return self._guard_web_results(raw_output)

        # Fallback: JSON-serialize the whole output and guard as a string.
        return self._guard_serialized(raw_output)

    def _guard_web_results(self, raw_output: list[dict[str, Any]]) -> dict[str, Any]:
        """Guard each web result's title+snippet individually for web_search."""
        assert self.output_guard is not None

        # Handle LangChain content block format: [{"type": "text", "text": "[{...results...}]"}]
        if len(raw_output) == 1 and isinstance(raw_output[0], dict):
            block = raw_output[0]
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                inner_text = block["text"]
                try:
                    parsed = json.loads(inner_text)
                    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                        raw_output = parsed
                except json.JSONDecodeError:
                    pass

        # Defensive: if elements are JSON strings or dicts with a "text" field
        # containing a JSON string, parse them back to proper dicts.
        parsed_output: list[dict[str, Any]] = []
        for item in raw_output:
            if isinstance(item, str):
                try:
                    parsed_output.append(json.loads(item))
                except json.JSONDecodeError:
                    parsed_output.append({"text": item})
            elif isinstance(item, dict):
                text_val = item.get("text", "")
                if isinstance(text_val, str):
                    try:
                        inner = json.loads(text_val)
                        if isinstance(inner, dict):
                            merged = {k: v for k, v in item.items() if k != "text"}
                            merged.update(inner)
                            parsed_output.append(merged)
                            continue
                    except json.JSONDecodeError:
                        pass
                parsed_output.append(item)
            else:
                parsed_output.append(item)
        raw_output = parsed_output

        checked_results: list[dict[str, Any]] = []
        violation_count = 0
        pii_anonymized_any = False

        for idx, result in enumerate(raw_output):
            if not isinstance(result, dict):
                checked_results.append(result)
                continue

            title = result.get("title", "")
            snippet = result.get("snippet", "")
            combined_text = (title + " " + snippet).strip()
            new_text, passed, pii_anonymized = _guard_text(
                combined_text,
                tool_name=self.tool.name,
                item_index=idx,
                output_guard=self.output_guard,
            )

            if not passed:
                violation_count += 1
                continue

            if pii_anonymized:
                result = dict(result)
                # Best-effort split back into title / snippet. If the
                # original title is preserved at the start of the anonymized
                # text (because no PII was in it), use that as the
                # boundary; otherwise split at half the original length.
                title_len = len(title)
                if title_len > 0 and new_text.startswith(title[: min(title_len, 50)]):
                    result["title"] = title
                    result["snippet"] = new_text[title_len:].strip()
                else:
                    mid = len(combined_text) // 2
                    result["title"] = new_text[:mid].strip()
                    result["snippet"] = new_text[mid:].strip()
                pii_anonymized_any = True

            checked_results.append(result)

        if raw_output and violation_count == len(raw_output):
            return _all_violated_rejection(self.tool.name)

        return {
            "tool": self.tool.name,
            "result": checked_results,
            "pii_anonymized": pii_anonymized_any,
        }

    def _guard_serialized(self, raw_output: Any) -> dict[str, Any]:
        """Fallback: JSON-serialize the whole output and guard as a string."""
        assert self.output_guard is not None
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
                "result": output_result.node_result.response,
            }

        try:
            parsed_result = json.loads(output_result.node_result.response)
        except (json.JSONDecodeError, TypeError):
            parsed_result = output_result.node_result.response

        return {
            "tool": self.tool.name,
            "result": parsed_result,
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
