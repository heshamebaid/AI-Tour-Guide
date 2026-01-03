"""
RAG Service - Integration layer between Django and Agentic_RAG pipeline.
Provides document search, web search, image search, and full agent capabilities.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add Agentic_RAG to Python path
AGENTIC_RAG_PATH = Path(__file__).resolve().parent.parent.parent / "Agentic_RAG"
AGENTIC_RAG_SRC_PATH = AGENTIC_RAG_PATH / "src"

sys.path.insert(0, str(AGENTIC_RAG_SRC_PATH))

# Load environment variables from Agentic_RAG
env_path = AGENTIC_RAG_PATH / ".env"
if env_path.exists():
    load_dotenv(str(env_path))


class RAGService:
    """
    Service class that wraps all Agentic_RAG functionality for Django integration.
    Provides document search, web search, image search, and conversational agent.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if RAGService._initialized:
            return
        
        self.retriever = None
        self.llm = None
        self.agent = None
        self.memory = None
        self._init_error = None
        
        try:
            self._initialize_services()
            RAGService._initialized = True
        except Exception as e:
            self._init_error = str(e)
            print(f"RAG Service initialization error: {e}")
    
    def _initialize_services(self):
        """Initialize all RAG services."""
        from services.retriever_service import RetrieverService
        from services.llm_service import LLMService
        from services.memory_service import MemoryService
        
        # Initialize core services
        self.retriever = RetrieverService().get_retriever(k=5)
        self.llm = LLMService().llm
        self.memory = MemoryService().memory
        
        print("✓ RAG Service initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if RAG service is ready."""
        return RAGService._initialized and self._init_error is None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "initialized": RAGService._initialized,
            "error": self._init_error,
            "has_retriever": self.retriever is not None,
            "has_llm": self.llm is not None,
        }
    
    def document_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search documents in the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with search results and metadata
        """
        if not self.is_ready():
            return {"success": False, "error": self._init_error or "Service not initialized"}
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "id": i + 1,
                    "content": doc.page_content,
                    "preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                })
            
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "documents": results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def web_search(self, query: str) -> Dict[str, Any]:
        """
        Perform web search using DuckDuckGo.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with web search results
        """
        try:
            from agents.tools.web_search_tool import web_search as web_search_tool
            result = web_search_tool.invoke(query)
            
            return {
                "success": True,
                "query": query,
                "results": result
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def image_search(self, query: str) -> Dict[str, Any]:
        """
        Search for relevant images using SerpAPI.
        
        Args:
            query: Image search query
            
        Returns:
            Dictionary with image URLs and metadata
        """
        try:
            from agents.tools.image_search_tool import image_search as image_search_tool
            result = image_search_tool.invoke(query)
            
            # Parse image URLs from the result
            images = self._parse_image_results(result)
            
            return {
                "success": True,
                "query": query,
                "images": images,
                "raw_result": result
            }
        except Exception as e:
            return {"success": False, "error": str(e), "images": []}
    
    def _parse_image_results(self, result: str) -> List[Dict[str, str]]:
        """Parse image search results into structured data."""
        images = []
        if "Error" in result or "No images found" in result:
            return images
        
        lines = result.split('\n')
        current_image = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_image:
                    images.append(current_image)
                current_image = {"title": line[2:].strip()}
            elif line.startswith('URL:'):
                current_image["url"] = line[4:].strip()
            elif line.startswith('Source:'):
                current_image["source"] = line[7:].strip()
        
        if current_image and "url" in current_image:
            images.append(current_image)
        
        return images
    
    def chat(self, user_input: str, use_agent: bool = True, include_images: bool = True) -> Dict[str, Any]:
        """
        Process a chat message using the full RAG pipeline.
        
        Args:
            user_input: User's message/question
            use_agent: Whether to use the full agent (with web/image search)
            include_images: Whether to search for relevant images
            
        Returns:
            Dictionary with response, sources, and optional images
        """
        if not self.is_ready():
            return {
                "success": False,
                "error": self._init_error or "Service not initialized",
                "answer": "Sorry, the AI service is currently unavailable."
            }
        
        try:
            # Step 1: Search documents
            doc_results = self.document_search(user_input, k=5)
            
            # Step 2: Build context from documents
            context = ""
            sources = []
            if doc_results["success"] and doc_results["documents"]:
                for doc in doc_results["documents"]:
                    context += doc["content"] + "\n\n"
                    sources.append(doc["preview"])
            
            # Step 3: Generate response using LLM
            prompt = self._build_prompt(user_input, context)
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Step 4: Optionally search for relevant images
            images = []
            if include_images:
                # Create an image search query based on user input
                image_query = f"Ancient Egypt {user_input}"
                image_results = self.image_search(image_query)
                if image_results["success"]:
                    images = image_results["images"][:3]  # Limit to 3 images
            
            # Step 5: Optionally enhance with web search if documents lack info
            web_info = None
            if use_agent and len(sources) < 2:
                web_results = self.web_search(user_input)
                if web_results["success"]:
                    web_info = web_results["results"]
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "images": images,
                "web_info": web_info,
                "documents_found": len(sources)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": f"An error occurred while processing your request: {str(e)}"
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM with context."""
        return f"""You are a knowledgeable tour guide specializing in Ancient Egyptian history, culture, and archaeology. 
Answer the user's question based on the provided context. If the context doesn't contain relevant information, 
use your knowledge to provide a helpful answer. Be informative, engaging, and educational.

Context from documents:
{context[:4000] if context else "No specific documents found."}

User Question: {question}

Please provide a detailed, informative answer:"""

    def get_agent(self):
        """
        Get the full conversational agent with all tools.
        
        Returns:
            Initialized LangChain agent
        """
        if not self.is_ready():
            return None
        
        try:
            from agents.agent_graph import AgenticRAGBuilder
            builder = AgenticRAGBuilder()
            return builder.create_agent()
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None
    
    def chat_with_agent(self, user_input: str) -> Dict[str, Any]:
        """
        Use the full LangChain agent for complex queries.
        The agent automatically decides which tools to use.
        
        Args:
            user_input: User's message
            
        Returns:
            Dictionary with agent response
        """
        try:
            agent = self.get_agent()
            if agent is None:
                return self.chat(user_input)  # Fallback to simple chat
            
            response = agent.invoke({"input": user_input})
            
            return {
                "success": True,
                "answer": response.get("output", str(response)),
                "agent_used": True
            }
        except Exception as e:
            # Fallback to simple chat if agent fails
            return self.chat(user_input)


# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create the singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
