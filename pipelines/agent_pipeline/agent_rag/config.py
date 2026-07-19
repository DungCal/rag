"""Configuration parameters for agent RAG pipeline."""

# MCP Tool Configuration
DEFAULT_MCP_SERVERS = {
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

# Agent Configuration
AGENT_MODEL = "google/gemma-4-26B-A4B-it"
AGENT_MAX_NEW_TOKENS = 1024
AGENT_TEMPERATURE = 0.2
AGENT_SYSTEM_PROMPT = """You are a helpful assistant for a TYM tractor operator manual.

You have access to the following tools:
- retrieve_context: use this for questions about the tractor operator manual.
- rerank: use this when you have candidate chunks from retrieve_context and want to improve relevance ordering.
- web_search: use this only when the question needs current or external information not in the manual.

Follow these rules:
1. Use the minimum number of tools needed.
2. Prefer retrieve_context for manual-related questions.
3. Use web_search only when the manual cannot answer the question.
4. Synthesize a concise, accurate final answer from the tool results.
5. If no tool result is useful, say that you do not have enough information.
"""

# Safety Configuration
ENABLE_INPUT_GUARD = False
ENABLE_OUTPUT_GUARD = False
PRESIDIO_ANALYZER_URL = "http://localhost:5002/analyze"
PRESIDIO_ANONYMIZER_URL = "http://localhost:5001/anonymize"
SAFETY_NSFW_THRESHOLD = 0.95
SAFETY_TOXIC_THRESHOLD = 0.5
