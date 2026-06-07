import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

# Set your Hugging Face API token (replace with your actual token)
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hugging_face_token_here"

import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

def main():
    # 1. Initialize the Base Endpoint
    # We specify the 27B instruction-tuned model. 
    print("Initializing HuggingFaceEndpoint...")
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-4-26B-A4B-it",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.1, # Low temperature for more factual, context-bound answers
        do_sample=False
    )

    # 2. Wrap with the Chat Interface
    # This automatically handles Gemma's specific chat templating requirements.
    chat_model = ChatHuggingFace(llm=llm)

    # 3. Create a Prompt Template with Context
    # We instruct the model to act as an assistant relying purely on the context.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analytical assistant. Answer the user's question strictly using the provided context.\n\nContext:\n{context}"),
        ("user", "{question}")
    ])

    # 4. Chain the prompt and the model together using LCEL (LangChain Expression Language)
    chain = prompt | chat_model

    # 5. Define the Test Context and Question
    test_context = (
        "The Aethelgard Protocol was a fictional treaty signed in 2104 on Mars. "
        "It established a zero-gravity trade zone spanning three lunar colonies. "
        "Its primary architect was Dr. Elias Vance, who later resigned in protest over asteroid mining regulations."
    )
    
    test_question = "Who created the Aethelgard Protocol and why did he resign?"

    # 6. Execute the Chain
    print("\nSending context and question to Gemma 2 27B...")
    try:
        response = chain.invoke({
            "context": test_context, 
            "question": test_question
        })
        
        print("\n--- Gemma's Output ---")
        print(response.content)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Ensure your HF token is valid and you have accepted the Gemma 2 terms on Hugging Face.")

if __name__ == "__main__":
    main()