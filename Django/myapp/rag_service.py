"""
RAG Service - Integration layer between Django and Agentic_RAG pipeline.
Provides document search, web search, image search, and full agent capabilities.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dotenv import load_dotenv

# CRITICAL: Disable CUDA and set environment BEFORE any torch imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA completely

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
    
    def __init__(self):
        self.retriever = None
        self.llm_service = None  # Store service, not LLM object
        self.agent = None
        self.memory = None
        self.retriever_service = None
        self.query_rewriter = None  # Query rewriter for optimized retrieval
        self._init_error = None
        self._initialized = False
        
        try:
            self._initialize_services()
            self._initialized = True
        except Exception as e:
            self._init_error = str(e)
            print(f"RAG Service initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    def __getstate__(self):
        """Custom pickle serialization - exclude unpicklable objects."""
        state = self.__dict__.copy()
        # Don't pickle LangChain objects that have classmethods
        state['retriever'] = None
        state['agent'] = None
        state['memory'] = None
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization - restore state."""
        self.__dict__.update(state)
        # Services will be reinitialized on first use if needed
    
    @property
    def llm(self):
        """Lazy access to LLM to avoid pickle issues."""
        if self.llm_service is not None:
            return self.llm_service.llm
        return None
    
    def _initialize_services(self):
        """Initialize all RAG services."""
        try:
            # Set environment variables to help with model loading
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            from services.retriever_service import RetrieverService
            from services.query_rewriter_service import QueryRewriterService
            
            # Initialize query rewriter for optimized retrieval
            print("Initializing query rewriter service...")
            self.query_rewriter = QueryRewriterService()
            print("✓ Query Rewriter initialized")
            
            # Initialize retriever with hybrid search (LLM optional)
            print("Initializing retriever service...")
            retriever_service = RetrieverService()
            print("Getting retriever with hybrid search...")
            self.retriever = retriever_service.get_retriever(k=5, search_type="hybrid")
            self.retriever_service = retriever_service
            
            print("✓ RAG Retriever initialized with Qdrant + Hybrid Search")
            
            # Try to initialize LLM (optional - will fail gracefully if not configured)
            try:
                from services.llm_service import LLMService
                self.llm_service = LLMService()
                print("✓ LLM Service initialized (lazy loading)")
            except Exception as llm_error:
                print(f"⚠️  LLM Service not available: {llm_error}")
                print("   Continuing in document-only mode (retrieval still works)")
                self.llm_service = None
            
            # Try to initialize memory (optional)
            try:
                from services.memory_service import MemoryService
                self.memory = MemoryService().memory
                print("✓ Memory Service initialized")
            except Exception as mem_error:
                print(f"⚠️  Memory Service not available: {mem_error}")
                self.memory = None
                
        except Exception as e:
            print(f"❌ Critical error initializing RAG services: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def is_ready(self) -> bool:
        """Check if RAG service is ready (at least retriever must work)."""
        return self._initialized and self.retriever is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        status = {
            "initialized": self._initialized,
            "error": self._init_error,
            "has_retriever": self.retriever is not None,
            "has_llm": self.llm is not None,
            "vector_db": "Qdrant with Hybrid Search"
        }
        
        # Get Qdrant collection info
        if self.retriever is not None:
            try:
                from services.vectorstore_service import VectorStoreService
                vs_service = VectorStoreService()
                info = vs_service.get_collection_info()
                status["collection_info"] = {
                    "exists": info.get("exists"),
                    "name": info.get("name"),
                    "documents": info.get("points_count", 0),
                    "hybrid_search": info.get("hybrid_search_enabled", False)
                }
            except Exception as e:
                status["collection_info"] = {"error": str(e)}
        
        return status
    
    def set_search_type(self, search_type: str = "hybrid", k: int = 5) -> Dict[str, Any]:
        """
        Switch between different search types.
        
        Args:
            search_type: "hybrid", "dense", "sparse", or "mmr"
            k: Number of documents to retrieve
            
        Returns:
            Status dictionary
        """
        if not self.is_ready():
            return {"success": False, "error": "Service not initialized"}
        
        try:
            if search_type == "mmr":
                self.retriever = self.retriever_service.get_mmr_retriever(k=k)
            else:
                self.retriever = self.retriever_service.get_retriever(k=k, search_type=search_type)
            
            return {
                "success": True,
                "search_type": search_type,
                "k": k
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def document_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search documents in the vector store.
        Uses query rewriter for optimized retrieval.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with search results and metadata
        """
        if not self.is_ready():
            return {"success": False, "error": self._init_error or "Service not initialized"}
        
        try:
            # Stage 1: Rewrite query for optimal retrieval
            optimized_query = query
            if self.query_rewriter:
                optimized_query = self.query_rewriter.rewrite_for_retrieval(query)
                print(f"Query rewriting: '{query}' → '{optimized_query}'")
            
            # Search with optimized query
            docs = self.retriever.invoke(optimized_query)
            
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
            # Step 1: Search documents using hybrid search
            doc_results = self.document_search(user_input, k=5)
            
            # Step 2: Build context from documents
            context = ""
            sources = []
            if doc_results["success"] and doc_results["documents"]:
                for doc in doc_results["documents"]:
                    context += doc["content"] + "\n\n"
                    sources.append(doc["preview"])
            
            # Step 3: Generate response using LLM (with fallback if LLM not available)
            if self.llm is not None:
                try:
                    prompt = self._build_prompt(user_input, context)
                    response = self.llm.invoke(prompt)
                    answer = response.content
                except Exception as llm_error:
                    # LLM failed - provide document excerpts as fallback
                    if sources:
                        answer = f"**Retrieved from documents (LLM error):**\n\n"
                        for i, source in enumerate(sources[:3], 1):
                            answer += f"{i}. {source}\n\n"
                        answer += f"\n*Note: LLM service error: {str(llm_error)}. Showing document excerpts using Qdrant hybrid search.*"
                    else:
                        raise
            else:
                # No LLM available - use document excerpts directly
                if sources:
                    answer = f"**Found in documents using Qdrant Hybrid Search:**\n\n"
                    for i, source in enumerate(sources[:3], 1):
                        answer += f"**{i}.** {source}\n\n"
                    answer += f"\n*Note: LLM service not configured. Configure OPENROUTER_API_KEY for AI-generated answers.*"
                else:
                    answer = "No relevant documents found for your query. Please try rephrasing or check if the index is built."
            
            # Step 4: Optionally search for relevant images
            images = []
            if include_images:
                try:
                    # Create an image search query based on user input
                    image_query = f"Ancient Egypt {user_input}"
                    image_results = self.image_search(image_query)
                    if image_results["success"]:
                        images = image_results["images"][:3]  # Limit to 3 images
                except:
                    pass  # Image search is optional
            
            # Step 5: Optionally enhance with web search if documents lack info
            web_info = None
            if use_agent and len(sources) < 2:
                try:
                    web_results = self.web_search(user_input)
                    if web_results["success"]:
                        web_info = web_results["results"]
                except:
                    pass  # Web search is optional
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "images": images,
                "web_info": web_info,
                "documents_found": len(sources)
            }
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n\nPlease ensure:\n- Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)\n- Collection is built (cd Agentic_RAG && python manage_qdrant.py build)"
            return {
                "success": False,
                "error": str(e),
                "answer": error_msg
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
    
    def chat_streaming(self, user_input: str, include_images: bool = True, search_web: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Process a chat message with streaming response.
        Uses two-stage query rewriting for optimal retrieval and response quality.
        
        Args:
            user_input: User's message/question
            include_images: Whether to search for relevant images
            search_web: Whether to search the web for additional context
            
        Yields:
            Dictionary chunks with type and content
        """
        if not self.is_ready():
            yield {"type": "error", "content": self._init_error or "Service not initialized"}
            return
        
        try:
            # Step 1: Search documents using hybrid search (with query rewriting in document_search)
            doc_results = self.document_search(user_input, k=5)
            
            # Step 2: Build context from documents
            context_parts = []
            sources = []
            if doc_results["success"] and doc_results["documents"]:
                for doc in doc_results["documents"]:
                    context_parts.append(doc["content"])
                    sources.append(doc["preview"])
            
            # Step 2.5: Optionally add web search context
            if search_web and context_parts:
                try:
                    web_results = self.web_search(user_input)
                    if web_results.get("success") and web_results.get("results"):
                        context_parts.append(f"Web Search Results:\n{web_results['results']}")
                except:
                    pass  # Web search is optional
            
            # Step 3: Use Stage 2 query rewriting to create conversational prompt
            if self.query_rewriter and context_parts:
                prompt = self.query_rewriter.rewrite_for_response(
                    user_query=user_input,
                    retrieved_context=context_parts,
                    language="en"  # Could be detected from user_input
                )
            else:
                # Fallback to old prompt building
                context = "\n\n".join(context_parts)
                prompt = self._build_prompt(user_input, context)
            
            # Step 4: Generate streaming response using LLM
            if self.llm is not None and hasattr(self.llm, 'stream'):
                try:
                    for token in self.llm.stream(prompt):
                        yield {"type": "token", "content": token}
                except Exception as llm_error:
                    # LLM failed - yield document excerpts
                    if sources:
                        answer = f"**Retrieved from documents (LLM error):**\n\n"
                        for i, source in enumerate(sources[:3], 1):
                            answer += f"{i}. {source}\n\n"
                        yield {"type": "token", "content": answer}
                    else:
                        yield {"type": "error", "content": str(llm_error)}
                        return
            elif self.llm is not None:
                # Fallback to non-streaming
                response = self.llm.invoke(prompt)
                yield {"type": "token", "content": response.content}
            else:
                # No LLM - use document excerpts
                if sources:
                    answer = f"**Found in documents using Qdrant Hybrid Search:**\n\n"
                    for i, source in enumerate(sources[:3], 1):
                        answer += f"**{i}.** {source}\n\n"
                    yield {"type": "token", "content": answer}
                else:
                    yield {"type": "token", "content": "No relevant documents found."}
            
            # Step 5: Optionally search for relevant images
            images = []
            if include_images:
                try:
                    image_query = f"Ancient Egypt {user_input}"
                    image_results = self.image_search(image_query)
                    if image_results["success"]:
                        images = image_results["images"][:3]
                except:
                    pass
            
            yield {"type": "images", "content": images}
            yield {"type": "done", "content": ""}
            
        except Exception as e:
            yield {"type": "error", "content": str(e)}
    
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
