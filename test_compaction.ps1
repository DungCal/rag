# Test script for conversation compaction with related queries
# This demonstrates the full flow: pre-answer check, post-answer check, and system context logging

# Replicates bash 'set -e' (stops execution if an error occurs)
$ErrorActionPreference = "Stop"

$THREAD_ID = "test-compaction"
$MAX_INPUT_TOKENS = 1500
$THRESHOLD_PCT = 0.30

Write-Host "============================================"
Write-Host "Cleaning up old state..."
Write-Host "============================================"
# SilentlyContinue prevents errors if the files don't exist yet
Remove-Item "logs\agent_checkpoints.sqlite" -Force -ErrorAction SilentlyContinue
Remove-Item "logs\conversation_history\${THREAD_ID}.md" -Force -ErrorAction SilentlyContinue
Remove-Item "logs\conversation_history\${THREAD_ID}_tokens.md" -Force -ErrorAction SilentlyContinue

# Array of related queries about TYM tractor
$QUERIES = @(
    # "hello",
    # "What is a TYM tractor and what is it used for?",
    # "Tell me about the safety features of the TYM tractor",
    # "How do I perform a pre-operation inspection?",
    # "What should I check before starting the engine?",
    # "How do I check the engine oil level?",
    # "What is the DPF warning lamp and what does it mean?",
    # "How do I drive the tractor on slopes safely?",
    "What maintenance is required for long-term storage?",
    "How do I adjust the touch monitor settings?"
)

Write-Host "============================================"
Write-Host "Starting compaction test with $($QUERIES.Count) queries"
Write-Host "Thread ID: $THREAD_ID"
Write-Host "Max input tokens: $MAX_INPUT_TOKENS"
$thresholdTokens = [math]::Floor($MAX_INPUT_TOKENS * $THRESHOLD_PCT)
Write-Host "Threshold: $THRESHOLD_PCT ($MAX_INPUT_TOKENS * $THRESHOLD_PCT = $thresholdTokens tokens)"
Write-Host "============================================"

for ($i = 0; $i -lt $QUERIES.Count; $i++) {
    $QUERY = $QUERIES[$i]
    $TURN = $i + 1

    Write-Host ""
    Write-Host "--------------------------------------------"
    Write-Host "Turn ${TURN}: ${QUERY}"
    Write-Host "--------------------------------------------"

    # Try/Catch block replicates '|| true' from bash, preventing pipeline crashes
    try {
        # Note: the backtick ` is used for line continuation in PowerShell
        python -m pipelines.agent_pipeline.orchestration run `
            --turn-on-agent-rag `
            --enable-conversation-compaction `
            --enable-token-debug-log `
            --thread-id "$THREAD_ID" `
            --max-input-tokens "$MAX_INPUT_TOKENS" `
            --context-token-threshold-pct "$THRESHOLD_PCT" `
            --query "$QUERY" `
            --as-json 2>&1 | Select-String -Pattern '"summarized"|"thread_id"|"response"' | Select-Object -First 3
    } catch {
        # Ignore errors
    }

    Write-Host ""
    Write-Host "Current conversation history:"
    Write-Host "--------------------------------------------"
    $historyPath = "logs\conversation_history\${THREAD_ID}.md"
    if (Test-Path $historyPath) {
        Get-Content $historyPath -TotalCount 50
    } else {
        Write-Host "(no history yet)"
    }
    Write-Host "..."
    Write-Host ""

    # Brief pause between queries
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "============================================"
Write-Host "Test complete!"
Write-Host "============================================"
Write-Host ""
Write-Host "Full conversation history saved to:"
Write-Host "  logs\conversation_history\${THREAD_ID}.md"
Write-Host ""
Write-Host "Token debug log saved to:"
Write-Host "  logs\conversation_history\${THREAD_ID}_tokens.md"
Write-Host ""
Write-Host "To view the full history:"
Write-Host "  Get-Content logs\conversation_history\${THREAD_ID}.md"
Write-Host ""
Write-Host "To view token debug:"
Write-Host "  Get-Content logs\conversation_history\${THREAD_ID}_tokens.md"
Write-Host ""
Write-Host "To check logs for compaction events:"
Write-Host "  Select-String -Path logs\agent_pipeline.log -Pattern 'Pre-answer check|Post-answer check|Compaction' | Select-Object -Last 20"