"""Utility functions for the agent pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


def _format_text_preview(text: str, max_length: int = 100) -> str:
    """Format text preview for logging (first max_length chars + '...')."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _format_source_file(source_file: str) -> str:
    """Format source file for logging (just filename, not full path)."""
    return Path(source_file).name


_DEGENERATE_PATTERNS = [
    r"own-heading",
    r"-heading-heading",
    r"-series-series",
    r"knowledge-heading",
    r"thought-series",
    r"similarity-series",
]


def is_degenerate_response(
    text: str,
    *,
    min_word_length: int = 20,
    max_repetition_ratio: float = 0.35,
) -> bool:
    """
    Detect model-generated garbage responses (repetition loops, known degenerate patterns).

    Returns True if the response is flagged as degenerate.
    """
    if not text or not text.strip():
        return True

    stripped = text.strip().lower()

    for pattern in _DEGENERATE_PATTERNS:
        if pattern in stripped:
            logger.info("Degeneracy detected by known pattern: {!r}", pattern)
            return True

    words = [w for w in re.split(r"\W+", stripped) if w]
    if not words:
        return True

    word_freq: dict[str, int] = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    most_common_count = max(word_freq.values())
    total_words = len(words)

    if most_common_count >= 5 and most_common_count / total_words > (1 - max_repetition_ratio):
        logger.info(
            "Degeneracy detected by repetition ratio: most_common_word_count={}, total={}, ratio={:.2f}",
            most_common_count, total_words, most_common_count / total_words,
        )
        return True

    if re.search(r"(.+?)\1{5,}", stripped):
        logger.info("Degeneracy detected by repeated substring pattern")
        return True

    alpha_chars = re.sub(r"[^\w\s]", "", stripped)
    if len(alpha_chars.strip()) < min_word_length:
        logger.info("Degeneracy detected by short meaningful length: {}", len(alpha_chars.strip()))
        return True

    return False


def extract_tool_results(agent_output: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract tool results from LangGraph agent output.

    Parses the agent's message history to find ToolMessage objects from the CURRENT TURN
    and extract results from retrieve_context, rerank, and web_search tools.

    Only processes tool messages that appear after the last HumanMessage to avoid
    extracting duplicate chunks from previous conversation turns.

    Args:
        agent_output: The output from agent.ainvoke(), containing "messages" list.

    Returns:
        Dictionary with two keys:
        - "chunks": List of RAG chunks from retrieve_context/rerank tools
        - "web_results": List of web search results from web_search tool
    """
    chunks = []
    web_results = []

    messages = agent_output.get("messages", [])

    # Find the index of the last HumanMessage to only process tool messages from the current turn
    last_human_msg_idx = -1
    for i, message in enumerate(messages):
        message_type = getattr(message, "type", None)
        if message_type == "human":
            last_human_msg_idx = i

    # Only process messages after the last HumanMessage (current turn only)
    start_idx = last_human_msg_idx + 1 if last_human_msg_idx >= 0 else 0

    for msg_idx in range(start_idx, len(messages)):
        message = messages[msg_idx]
        # Check if this is a ToolMessage by checking the type attribute
        message_type = getattr(message, "type", None)
        if message_type != "tool":
            continue

        # Check if this has tool name and content
        if not hasattr(message, "name") or not hasattr(message, "content"):
            continue

        tool_name = message.name
        content = message.content

        # Parse content (might be string, list of text objects, or already parsed)
        try:
            if isinstance(content, str):
                tool_result = json.loads(content)
                logger.debug("Parsed string content from {}: {}", tool_name, type(tool_result))
            elif isinstance(content, list):
                # Handle list of text objects from LangChain
                parsed_items = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        parsed_item = json.loads(text_content)
                        # If parsed_item is a list, extend parsed_items; otherwise append
                        if isinstance(parsed_item, list):
                            parsed_items.extend(parsed_item)
                        else:
                            parsed_items.append(parsed_item)
                        logger.debug("Parsed text item from {}: {}", tool_name, type(parsed_item))
                tool_result = parsed_items
                logger.debug("Parsed {} items from list content for {}", len(parsed_items), tool_name)
            else:
                tool_result = content
                logger.debug("Using content as-is from {}: {}", tool_name, type(tool_result))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse tool result from {}: {} - {}", tool_name, content, e)
            continue

        logger.debug("Successfully parsed tool result from {}: type={}", tool_name, type(tool_result))

        # Track chunks/web_results added by this tool message to avoid duplicate logging
        chunks_before = len(chunks)
        web_results_before = len(web_results)

        # Extract based on tool type
        if tool_name in ["retrieve_context", "rerank"]:
            # These tools return chunks
            if isinstance(tool_result, dict):
                # Check if this is a rejection message from GuardTool
                if "error" in tool_result:
                    logger.warning(
                        "Tool {} returned rejection: {}",
                        tool_name,
                        tool_result.get("error")
                    )
                    # Don't try to extract chunks from rejection messages
                else:
                    # Handle structured response with "result" field
                    result_list = tool_result.get("result", [])
                    if isinstance(result_list, list):
                        for item in result_list:
                            if isinstance(item, str):
                                # Parse JSON string
                                try:
                                    parsed_item = json.loads(item)
                                    chunks.append(parsed_item)
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse chunk JSON: {}", item)
                            else:
                                chunks.append(item)
                    elif isinstance(result_list, dict):
                        # Single chunk object
                        if "text" in result_list:
                            chunks.append(result_list)
                        else:
                            logger.warning("Expected chunk with 'text' field, got: {}", result_list.keys())
                    else:
                        logger.warning("Expected list or dict in result field, got: {}", type(result_list))
            elif isinstance(tool_result, list):
                # Handle list of parsed JSON objects
                logger.debug("Processing list of {} items from {}", len(tool_result), tool_name)
                for idx, item in enumerate(tool_result):
                    logger.debug("Item {} from {}: type={}, keys={}", idx, tool_name, type(item), item.keys() if isinstance(item, dict) else "N/A")
                    if isinstance(item, dict):
                        # Check if this is a structured response with "result" field
                        if "result" in item:
                            result_list = item.get("result", [])
                            if isinstance(result_list, list):
                                for result_item in result_list:
                                    if isinstance(result_item, str):
                                        # Parse JSON string
                                        try:
                                            parsed_item = json.loads(result_item)
                                            chunks.append(parsed_item)
                                        except json.JSONDecodeError:
                                            logger.warning("Failed to parse chunk JSON: {}", result_item)
                                    else:
                                        chunks.append(result_item)
                            else:
                                logger.warning("Expected list in result field, got: {}", type(result_list))
                        # Check if this is a direct chunk object (has 'text' field)
                        elif "text" in item:
                            chunks.append(item)
                        else:
                            logger.warning("Unexpected chunk format: {}", item.keys())
            else:
                logger.warning("Unexpected tool result format from {}: {}", tool_name, type(tool_result))

            # Log only the NEW chunks added by this tool message
            new_chunks = chunks[chunks_before:]
            for chunk_idx, chunk in enumerate(new_chunks, start=1):
                if isinstance(chunk, dict):
                    score = chunk.get("score", chunk.get("rerank_score", "N/A"))
                    page_number = chunk.get("page_number", "N/A")
                    chunk_id = chunk.get("chunk_id", "N/A")
                    source_file = chunk.get("source_file", "N/A")
                    text = chunk.get("text", "")
                    text_preview = _format_text_preview(text)
                    
                    # Check if this is a reranked chunk
                    if "rerank_score" in chunk:
                        retrieval_score = chunk.get("score", "N/A")
                        logger.info(
                            "Reranked chunk #{}: rerank_score={} retrieval_score={} page={} chunk_id={} text_preview='{}'",
                            chunk_idx, score, retrieval_score, page_number, chunk_id, text_preview
                        )
                    else:
                        logger.info(
                            "Retrieved chunk #{}: score={} page={} chunk_id={} source_file='{}' text_preview='{}'",
                            chunk_idx, score, page_number, chunk_id, _format_source_file(source_file), text_preview
                        )

        elif tool_name == "web_search":
            # Web search returns results with title, snippet, url
            if isinstance(tool_result, list):
                # Handle list of parsed JSON objects
                for item in tool_result:
                    if isinstance(item, dict):
                        # Check if this is a structured response with "result" field
                        if "result" in item:
                            result_list = item.get("result", [])
                            if isinstance(result_list, list):
                                for result_item in result_list:
                                    if isinstance(result_item, str):
                                        # Parse JSON string
                                        try:
                                            parsed_item = json.loads(result_item)
                                            web_results.append(parsed_item)
                                        except json.JSONDecodeError:
                                            logger.warning("Failed to parse web result JSON: {}", result_item)
                                    else:
                                        web_results.append(result_item)
                            else:
                                logger.warning("Expected list in result field, got: {}", type(result_list))
                        # Check if this is a direct web result object (has 'title' or 'snippet' field)
                        elif "title" in item or "snippet" in item or "url" in item:
                            web_results.append(item)
                        else:
                            logger.warning("Unexpected web result format: {}", item.keys())
            elif isinstance(tool_result, dict):
                # Handle structured response
                result_list = tool_result.get("result", [])
                if isinstance(result_list, list):
                    for item in result_list:
                        if isinstance(item, str):
                            # Parse JSON string
                            try:
                                parsed_item = json.loads(item)
                                web_results.append(parsed_item)
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse web result JSON: {}", item)
                        else:
                            web_results.append(item)
                else:
                    logger.warning("Expected list in result field, got: {}", type(result_list))
            else:
                logger.warning("Unexpected tool result format from {}: {}", tool_name, type(tool_result))

            # Log only the NEW web results added by this tool message
            new_web_results = web_results[web_results_before:]
            for result_idx, result in enumerate(new_web_results, start=1):
                if isinstance(result, dict):
                    title = result.get("title", "N/A")
                    url = result.get("url", "N/A")
                    snippet = result.get("snippet", result.get("content", ""))
                    snippet_preview = _format_text_preview(snippet)
                    
                    logger.info(
                        "Web result #{}: title='{}' url='{}' snippet_preview='{}'",
                        result_idx, title, url, snippet_preview
                    )

    logger.info(
        "Extracted tool results: {} chunks, {} web results",
        len(chunks),
        len(web_results),
    )

    return {
        "chunks": chunks,
        "web_results": web_results,
    }
