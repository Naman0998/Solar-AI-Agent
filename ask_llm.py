from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),  # API
    base_url=os.getenv("OPENROUTER_BASE_URL")
)
def ask_llm(user_query: str) -> str:
    """
    Sends a prompt to the selected OpenRouter LLM and returns the response.
    """
    model = "mistralai/mistral-7b-instruct"
    prompt = f"{user_query}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for answering questions strictly based on "
                        "the internal document chunks provided below. Only use the information "
                        "strictly relevant to the named finance company in the question. "
                        "Do NOT make up answers or rely on outside knowledge."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"❌ Error: {e}"