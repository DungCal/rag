from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "mcp is required for the MCP server. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


mcp = FastMCP("web_search")


def _load_env_value(name: str, env_path: Path = PROJECT_ROOT / ".env") -> str | None:
    if value := os.getenv(name):
        return value
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _load_tavily_api_key(api_key: str | None = None) -> str:
    resolved = api_key or os.getenv("TAVILY_API_KEY") or _load_env_value("TAVILY_API_KEY")
    if not resolved:
        raise ValueError(
            "Missing TAVILY_API_KEY. Set the environment variable, add it to .env, "
            "or pass --tavily-api-key."
        )
    return resolved


@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search the web for up-to-date information not contained in the indexed document.

    Use this tool when the user's question requires current, external, or factual
    information that is outside the scope of the tractor operator manual
    (for example: current weather, news, prices, recent product updates, or
    general knowledge not covered in the manual).

    Do not use this tool for questions that can be answered directly from
    retrieve_context.

    Args:
        query: The web search query.
        max_results: Maximum number of search results to return (default: 5).

    Returns:
        A list of search results. Each result contains:
        - title: result title
        - url: result URL
        - content: result snippet or summary
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise ImportError(
            "tavily-python is required for web search. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    api_key = _load_tavily_api_key()
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)
    results = response.get("results", [])
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in results
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Web search MCP server")
    parser.add_argument(
        "--tavily-api-key",
        default=None,
        help="Tavily API key (defaults to TAVILY_API_KEY env / .env)",
    )
    args = parser.parse_args()
    if args.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = args.tavily_api_key
    mcp.run()


if __name__ == "__main__":
    main()
