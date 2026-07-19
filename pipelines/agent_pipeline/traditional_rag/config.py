"""Configuration parameters for traditional RAG pipeline."""

# Retrieval Configuration
DEFAULT_PINECONE_NAMESPACE = "default"
DEFAULT_TOP_K = 5

# Rerank Configuration
ENABLE_RERANK = False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_INPUT_TOP_K = 20
RERANK_OUTPUT_TOP_K = 5

# Judge Configuration
ENABLE_JUDGE = False
JUDGE_TOP_K = 3
JUDGE_MIN_SCORE = 5

# Generation Configuration
ENABLE_GENERATION = False

# Safety Configuration
ENABLE_INPUT_GUARD = False
ENABLE_OUTPUT_GUARD = False
PRESIDIO_ANALYZER_URL = "http://localhost:5002/analyze"
PRESIDIO_ANONYMIZER_URL = "http://localhost:5001/anonymize"
SAFETY_NSFW_THRESHOLD = 0.95
SAFETY_TOXIC_THRESHOLD = 0.5
