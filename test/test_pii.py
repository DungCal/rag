from guardrails.hub import GuardrailsPII, DetectPII
from guardrails import Guard

# ==========================================
# 1. Initialize the Guard with DetectPII
# ==========================================
# Instantiate DetectPII with its parameters BEFORE passing it to .use()
pii_guard = Guard().use(
    DetectPII(
        pii_entities='pii',
        on_fail="fix"
    )
)

# ==========================================
# 2. Example A: Standard RAG Output Validation
# ==========================================
rag_output = "The user John Smith requested a password reset link sent to john.smith@example.com or texted to 555-019-2837."

# Validate the output
standard_result = pii_guard.validate(rag_output)

print("--- Standard Validation ---")
print(f"Original: {rag_output}")
print(f"Sanitized: {standard_result.validated_output}\n")


# ==========================================
# 3. Example B: Using Dynamic Runtime Metadata
# ==========================================
# In some RAG chatbot turns, you might want to dynamically change what PII to block.
# You can pass 'pii_entities' into the `metadata` dictionary during validation to override defaults.

runtime_metadata = {
    "pii_entities": ["US_SSN"]
}

payment_rag_output = "Payment processed using card 4089-0410-6126-9940 tied to SSN 000-00-0000."

# Validate using the runtime metadata
dynamic_result = pii_guard.validate(
    payment_rag_output,
    metadata=runtime_metadata
)

print("--- Dynamic Metadata Validation ---")
print(f"Original: {payment_rag_output}")
print(f"Sanitized: {dynamic_result.validated_output}")