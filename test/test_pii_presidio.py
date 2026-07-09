import requests
import json

# ==========================================
# 1. Configuration
# ==========================================
ANALYZER_URL = "http://localhost:5002/analyze"
ANONYMIZER_URL = "http://localhost:5001/anonymize"

# Notice we use the valid test credit card so it triggers the detection
raw_text = 'Hey i have my card 408904106126 9940 please use it to buy tractor'

print(f"Original Text: {raw_text}\n")

# ==========================================
# 2. Step 1: Analyze the Text
# ==========================================
analyzer_payload = {
    "text": raw_text,
    "language": "en"
}

# Send text to the Analyzer
analyzer_response = requests.post(ANALYZER_URL, json=analyzer_payload)
analyzer_results = analyzer_response.json()

print("--- Step 1: Analyzer Results (Coordinates) ---")
print(json.dumps(analyzer_results, indent=2))
print("\n")

# ==========================================
# 3. Step 2: Anonymize the Text
# ==========================================
# The anonymizer needs BOTH the original text and the results from the analyzer
anonymizer_payload = {
    "text": raw_text,
    "analyzer_results": analyzer_results
}

# Send text + coordinates to the Anonymizer
anonymizer_response = requests.post(ANONYMIZER_URL, json=anonymizer_payload)
anonymized_data = anonymizer_response.json()

print("--- Step 2: Final Anonymized Output ---")
print(anonymized_data.get("text"))