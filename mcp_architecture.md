````mermaid
flowchart TB
    user(["User / CLI"])
    cd(["External MCP clients<br/>Claude Desktop · Claude Code"])

    subgraph CS["chat_service (FastAPI :8000)"]
        api["/chat · /chat/stream"]
        mcpsrv["/mcp · FastMCP server<br/>retrieve_context · rerank · search_documents · query_documents<br/>query_user_data · list_collections · index_document · get_job_status"]
        guard["common/guardrails<br/>PII · injection · toxicity"]
        mcpcli["MCP client (MultiServerMCPClient)"]
        subgraph G["LangGraph agent"]
            si["safety_input 🛡"]
            agent["agent_node<br/>agentic RAG tool-loop"]
            so["safety_output 🛡"]
            si --> agent --> so
        end
    end

    subgraph EXT["External MCP servers"]
        ws["web search (fallback)"]
        fs["filesystem · GitHub · DB"]
    end

    os[("OpenSearch")]
    pg[("Postgres checkpointer")]
    userdb[("Postgres user-data<br/>RLS + read-only role")]
    sqs[["SQS / ElasticMQ"]]
    lf[("Langfuse<br/>redacted traces")]

    user --> api --> si
    cd --> mcpsrv
    agent -->|"GuardedTool 🛡"| mcpcli
    mcpcli -->|"RAG loopback in-proc"| mcpsrv
    mcpcli --> ws & fs
    mcpsrv --- os
    mcpsrv -->|"query_user_data · SET app.current_user_id"| userdb
    mcpsrv --> sqs
    G --- pg
    guard -. cross-cutting .- G
    so -.-> lf
````
