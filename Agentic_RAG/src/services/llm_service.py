from langchain_openai import ChatOpenAI
from core.config import Config

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENROUTER_MODEL,
            api_key=Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7
        )
