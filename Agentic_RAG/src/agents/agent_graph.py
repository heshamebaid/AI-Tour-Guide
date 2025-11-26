from langchain.agents import initialize_agent, AgentType
from services.llm_service import LLMService
from services.retriever_service import RetrieverService
from services.memory_service import MemoryService

from agents.tools.document_search_tool import document_search
from agents.tools.web_search_tool import web_search
from agents.tools.image_search_tool import image_search

class AgenticRAGBuilder:

    def __init__(self):
        self.llm = LLMService().llm
        self.retriever = RetrieverService().get_retriever()
        self.memory = MemoryService().memory

    def create_agent(self):
        """
        Creates an agent with document search, web search, and image search tools.
        The agent will:
        1. First search local documents for information
        2. Use web search for current/missing information
        3. Find relevant images to illustrate answers
        """
        tools = [document_search, web_search, image_search]

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        return agent
