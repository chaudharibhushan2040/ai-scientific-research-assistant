from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY

print("API KEY:", GROQ_API_KEY)

def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )