from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.agent_pipeline.agent_rag.guardtool import wrap_tool_with_guard
from pipelines.agent_pipeline.shared import SafetyInputNode, SafetyOutputNode

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError as exc:  # pragma: no cover - optional until deps installed
    raise ImportError(
        "langchain-mcp-adapters is required for agentic RAG. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc


DEFAULT_MCP_SERVERS: dict[str, dict[str, Any]] = {
    "retrieve_context": {
        "command": "python",
        "args": ["mcp_servers/retrieve_context_server.py"],
        "transport": "stdio",
        "env": None,
    },
    "rerank": {
        "command": "python",
        "args": ["mcp_servers/rerank_server.py"],
        "transport": "stdio",
        "env": None,
    },
    "web_search": {
        "command": "python",
        "args": ["mcp_servers/web_search_server.py"],
        "transport": "stdio",
        "env": None,
    },
}

# query_user_data is intentionally omitted from the active agent tool set.


class AgentToolClient:
    """Manages the lifecycle of the MCP stdio servers used by the agentic RAG agent."""

    def __init__(
        self,
        servers: dict[str, dict[str, Any]] | None = None,
        tavily_api_key: str | None = None,
        enable_input_guard: bool = False,
        enable_output_guard: bool = False,
    ) -> None:
        self.servers = dict(servers) if servers else dict(DEFAULT_MCP_SERVERS)
        if tavily_api_key:
            for server in self.servers.values():
                if server["args"][0] == "mcp_servers/web_search_server.py":
                    env = dict(server.get("env") or os.environ.copy())
                    env["TAVILY_API_KEY"] = tavily_api_key
                    server["env"] = env
        self._client: MultiServerMCPClient | None = None
        self._enable_input_guard = enable_input_guard
        self._enable_output_guard = enable_output_guard

    async def connect(self) -> list[Any]:
        """Connect to the MCP servers and return LangChain tools.

        Tools are wrapped by GuardTool only when input or output guards are enabled.
        """
        self._client = MultiServerMCPClient(self.servers)
        tools = await self._client.get_tools()
        if not self._enable_input_guard and not self._enable_output_guard:
            return tools
        return [
            wrap_tool_with_guard(
                tool,
                input_guard=SafetyInputNode() if self._enable_input_guard else None,
                output_guard=SafetyOutputNode() if self._enable_output_guard else None,
            )
            for tool in tools
        ]

    async def get_guarded_tools(self) -> list[Any]:
        """Return MCP tools wrapped by GuardTool."""
        return await self.connect()

    async def get_raw_tools(self) -> list[Any]:
        """Return raw MCP tools without guards."""
        self._client = MultiServerMCPClient(self.servers)
        return await self._client.get_tools()

    async def close(self) -> None:
        """Release the MCP client.

        Note: langchain-mcp-adapters does not expose an explicit close method,
        so we drop the reference and rely on process cleanup.
        """
        self._client = None
