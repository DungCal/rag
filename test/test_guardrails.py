# Import Guard and Validator
from guardrails.hub import RegexMatch
from guardrails import Guard

# Initialize the Guard with
guard = Guard().use(
    RegexMatch(regex="^[A-Z][a-z]*$")
)

print(guard.parse("0345936523").validation_passed)  # Guardrail Passes
#print(guard.parse("Caesar Salad").validation_passed)  # Guardrail Fails