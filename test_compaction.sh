#!/bin/bash
# Test script for conversation compaction with related queries
# This demonstrates the full flow: pre-answer check, post-answer check, and system context logging

set -e

THREAD_ID="test-compaction"
MAX_INPUT_TOKENS=1500
THRESHOLD_PCT=0.30

echo "============================================"
echo "Cleaning up old state..."
echo "============================================"
rm -f logs/agent_checkpoints.sqlite
rm -f logs/conversation_history/${THREAD_ID}.md

# Array of related queries about TYM tractor
QUERIES=(
  "hello"
  "What is a TYM tractor and what is it used for?"
  "Tell me about the safety features of the TYM tractor"
  "How do I perform a pre-operation inspection?"
  "What should I check before starting the engine?"
  "How do I check the engine oil level?"
  "What is the DPF warning lamp and what does it mean?"
  "How do I drive the tractor on slopes safely?"
  "What maintenance is required for long-term storage?"
  "How do I adjust the touch monitor settings?"
)

echo "============================================"
echo "Starting compaction test with ${#QUERIES[@]} queries"
echo "Thread ID: ${THREAD_ID}"
echo "Max input tokens: ${MAX_INPUT_TOKENS}"
echo "Threshold: ${THRESHOLD_PCT} (${MAX_INPUT_TOKENS} * ${THRESHOLD_PCT} = $((MAX_INPUT_TOKENS * 30 / 100)) tokens)"
echo "============================================"

for i in "${!QUERIES[@]}"; do
  QUERY="${QUERIES[$i]}"
  TURN=$((i + 1))

  echo ""
  echo "--------------------------------------------"
  echo "Turn ${TURN}: ${QUERY}"
  echo "--------------------------------------------"

  python -m pipelines.agent_pipeline.orchestration run \
    --turn-on-agent-rag \
    --enable-conversation-compaction \
    --thread-id "${THREAD_ID}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --context-token-threshold-pct "${THRESHOLD_PCT}" \
    --query "${QUERY}" \
    --as-json 2>&1 | grep -E '"summarized"|"thread_id"|"response"' | head -3 || true

  echo ""
  echo "Current conversation history:"
  echo "--------------------------------------------"
  cat logs/conversation_history/${THREAD_ID}.md 2>/dev/null | head -50 || echo "(no history yet)"
  echo "..."
  echo ""

  # Brief pause between queries
  sleep 1
done

echo ""
echo "============================================"
echo "Test complete!"
echo "============================================"
echo ""
echo "Full conversation history saved to:"
echo "  logs/conversation_history/${THREAD_ID}.md"
echo ""
echo "To view the full history:"
echo "  cat logs/conversation_history/${THREAD_ID}.md"
echo ""
echo "To check logs for compaction events:"
echo "  grep -E 'Pre-answer check|Post-answer check|Compaction' logs/agent_pipeline.log | tail -20"
