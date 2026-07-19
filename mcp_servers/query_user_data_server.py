from __future__ import annotations

import argparse
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "mcp is required for the MCP server. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


mcp = FastMCP("query_user_data")


@mcp.tool()
async def query_user_data(
    query: str,
    user_id: str,
) -> list[dict[str, Any]]:
    """Query user-specific data stored in the memory / user-data store.

    Use this tool when the answer depends on the user's personal historical data,
    saved preferences, or prior interactions. This tool is currently a draft and
    is not wired into the active agentic RAG loop.

    Args:
        query: The question about the user's data.
        user_id: The identifier of the current user.

    Returns:
        Relevant user data records, or an empty list if no user data store is configured.
    """
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Query user data MCP server (draft)")
    parser.parse_args()
    mcp.run()


if __name__ == "__main__":
    main()
