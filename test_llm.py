import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load variables from the .env file
load_dotenv()

def main():
    # 1. Initialize InferenceClient for Gemma 4
    print("Initializing HuggingFace InferenceClient for Gemma 4...")
    client = InferenceClient(
        model="google/gemma-4-26B-A4B-it",
        token=os.getenv("HF_TOKEN"),
    )

    # 2. Define the test context and question
    test_context = (
        "The Aethelgard Protocol was a fictional treaty signed in 2104 on Mars. "
        "It established a zero-gravity trade zone spanning three lunar colonies. "
        "Its primary architect was Dr. Elias Vance, who later resigned in protest over asteroid mining regulations."
    )
    
    test_question = "Who created the Aethelgard Protocol and why did he resign?"

    # 3. Build the chat messages
    messages = [
        {
            "role": "system",
            "content": "You are an analytical assistant. Answer the user's question strictly using the provided context.\n\nContext:\n" + test_context,
        },
        {
            "role": "user",
            "content": test_question,
        },
    ]

    # 4. Execute the inference
    print("\nSending context and question to Gemma 4...")
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            stream=False,
        )
        
        print("\n--- Gemma 4 Output ---")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Ensure your HF_TOKEN is valid and you have accepted the Gemma 4 terms on Hugging Face.")

if __name__ == "__main__":
    main()