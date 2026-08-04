"""Configuration parameters for shared agent pipeline components."""

# Retriever Judge Configuration
ENABLE_JUDGE = True
JUDGE_MODEL = "google/gemma-4-26B-A4B-it"
JUDGE_TOP_K = 3
JUDGE_MIN_SCORE = 5
JUDGE_MAX_NEW_TOKENS = 2048
JUDGE_TEMPERATURE = 0.0
JUDGE_PROMPT_PATH = "prompts/llm-as-a-judge/llm-as-a-judge-context-relevance.txt"

# Generation Node Configuration
ENABLE_GENERATION = True
GENERATION_MODEL = "google/gemma-4-26B-A4B-it"

GENERATION_MAX_NEW_TOKENS = 1024
GENERATION_TEMPERATURE = 0.2
GENERATION_PROMPT_PATH = "prompts/generation_node_prompt.txt"

# Conversation Compaction Configuration
ENABLE_CONVERSATION_COMPACTION = False
MAX_INPUT_TOKENS = 256_000
CONTEXT_TOKEN_THRESHOLD_PCT = 0.30
MIN_KEEP_RECENT_TURNS = 1
COMPACTION_MAX_SUMMARY_TOKENS = 2048
COMPACTION_SUMMARY_TEMPERATURE = 0.0
