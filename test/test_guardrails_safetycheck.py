from guardrails.hub import NSFWText, ProfanityFree, ToxicLanguage
from guardrails import Guard

# ==========================================
# 1. Initialize the Guard with 3 Validators
# ==========================================
# Pass all instantiated validators directly into .use()
safety_guard = Guard().use(
    NSFWText(threshold=0.8, validation_method="sentence", on_fail="exception"),
    ProfanityFree(on_fail="exception"),
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception")
)

# ==========================================
# 2. Test Cases
# ==========================================
safe_text = "The new sci-fi movie was an absolute masterpiece with incredible visual effects."
unsafe_text = "You are a stupid idiot who can't do anything right. Go to hell."

# ==========================================
# 3. Run the Tests
# ==========================================
print("--- Testing Safe Text ---")
try:
    result = safety_guard.validate(safe_text)
    print("✅ Validation Passed!")
    print(f"Output: {result.validated_output}\n")
except Exception as e:
    print(f"❌ Validation Failed: {e}\n")

print("--- Testing Unsafe Text ---")
try:
    result = safety_guard.validate(unsafe_text)
    print("✅ Validation Passed!")
    print(f"Output: {result.validated_output}\n")
except Exception as e:
    # This will trigger because the text is toxic/profane
    print(f"❌ Validation Failed:\n{e}")