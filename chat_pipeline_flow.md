```mermaid
flowchart TB
    Start(["ChatPipeline.chat / POST /chat<br/>(message, thread_id, user_id)"]) --> Compile
    Compile["build() compile LangGraph<br/>+ checkpointer (memory/sqlite/postgres)"] --> Restore
    Restore["checkpointer loads state for thread_id"] --> Invoke["invoke {messages, user_id}"]
    Invoke --> SafetyIn

    subgraph SafetyIn["safety_input_node — step0 🛡"]
        direction TB
        Redact["redact PII (replace in HumanMessage by id)"]
        Inj["check_injection (regex) + SafetyJudge (toxicity)"]
        Redact --> Inj
    end
    SafetyIn --> SafeEdge{"decide_after_input_safety"}
    SafeEdge -- unsafe --> Reject["reject_node<br/>neutral refusal"] --> EndReject([END])
    SafeEdge -- safe --> Router

    subgraph Router["router_node — step1"]
        direction TB
        Scope["get_active_scope (manifest)"]
        RouterLLM["LLM → on_topic / greeting / off_topic"]
        Scope --> RouterLLM
    end
    Router --> RouterEdge{"decide_after_router"}
    RouterEdge -- greeting --> Greet["greeting_node"] --> SafetyOut
    RouterEdge -- off_topic --> Off["off_topic_node"] --> SafetyOut
    RouterEdge -- on_topic --> Memory

    Memory["memory_node — step3<br/>sliding-window summarization (3a)"] --> Agent

    subgraph Agent["agent_node — agentic RAG tool-loop (async bridge)"]
        direction TB
        Bind["LLM.bind_tools(guarded tools)"]
        Loop{"tool_calls?"}
        Tools["GuardedTool 🛡 invoke<br/>retrieve_context · rerank (RAG MCP)<br/>web_search (gated) · calculator · query_user_data"]
        Acc["accumulate docs → state.context + [n] citations<br/>set rag_insufficient if top_score &lt; min"]
        Ans["answer + citations footer"]
        Bind --> Loop
        Loop -- yes --> Tools --> Acc --> Bind
        Loop -- no --> Ans
    end
    Agent --> Verify

    subgraph Verify["verification_node — step3"]
        direction TB
        VJudge["GenerationJudge: grounded vs accumulated context"]
    end
    Verify --> VEdge{"decide_after_verification"}
    VEdge -- regenerate --> Agent
    VEdge -- accept --> SafetyOut

    subgraph SafetyOut["safety_output_node — step0 🛡"]
        direction TB
        OutTox["toxicity check on answer"]
        OutPII["PII net (redact)"]
        OutTox --> OutPII
    end
    SafetyOut --> EndOK([END])

    EndOK --> Persist["checkpointer saves GraphState"]
    EndReject --> Persist
````