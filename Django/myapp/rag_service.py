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

# Load environment from project root .env only
env_path = AGENTIC_RAG_PATH.parent / ".env"
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
        self.reranker = None  # Reranker for top-k selection
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
            from services.reranker_service import RerankerService
            
            # Try to initialize LLM first (needed for query rewriter)
            try:
                from services.llm_service import LLMService
                self.llm_service = LLMService()
                print("✓ LLM Service initialized (lazy loading)")
            except Exception as llm_error:
                print(f"⚠️  LLM Service not available: {llm_error}")
                print("   Continuing in document-only mode (retrieval still works)")
                self.llm_service = None
            
            # Initialize query rewriter with LLM for intelligent rewriting
            print("Initializing query rewriter service...")
            self.query_rewriter = QueryRewriterService(llm_client=self.llm if self.llm_service else None)
            if self.llm_service:
                print("✓ Query Rewriter initialized with LLM-based rewriting")
            else:
                print("✓ Query Rewriter initialized with rule-based rewriting")
            
            # Initialize retriever with hybrid search
            print("Initializing retriever service...")
            retriever_service = RetrieverService()
            print("Getting retriever with hybrid search...")
            self.retriever = retriever_service.get_retriever(k=25, search_type="hybrid")  # Top 25 for reranking
            self.retriever_service = retriever_service
            print("✓ RAG Retriever initialized with Qdrant + Hybrid Search (top-25)")
            
            # Initialize reranker for top-k selection
            print("Initializing reranker service...")
            self.reranker = RerankerService()
            print("✓ Reranker initialized")
            
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

**IMPORTANT SCOPE GUIDELINES:**
- You are an expert ONLY in Ancient Egypt (pharaohs, pyramids, temples, hieroglyphs, mummies, gods, dynasties, archaeology, culture, daily life, etc.)
- If the question is about Ancient Egypt: Answer it enthusiastically with the provided context or your knowledge
- If the question is NOT about Ancient Egypt (modern topics, other civilizations, unrelated subjects): Politely explain that you specialize in Ancient Egypt and offer your services

**Example out-of-scope response:**
"I appreciate your question, but I specialize exclusively in Ancient Egyptian history and culture. I'd be happy to help you explore topics like:
- Pharaohs and their dynasties
- Pyramids and ancient monuments
- Gods and religious beliefs
- Hieroglyphs and ancient writing
- Daily life in ancient Egypt
- Archaeological discoveries

What would you like to know about Ancient Egypt?"

Context from documents:
{context[:4000] if context else "No specific documents found."}

User Question: {question}

Please provide a detailed, informative answer:"""

    def _check_query_scope(self, query: str) -> tuple[bool, str]:
        """
        Check if the query is relevant to Ancient Egypt BEFORE doing any retrieval.
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (is_relevant: bool, rewritten_query: str or rejection_message: str)
        """
        if not self.llm:
            # No LLM available, assume relevant
            return True, query
        
        scope_check_prompt = f"""You are a scope checker for an Ancient Egypt chatbot.

User Question: "{query}"

Task: Determine if this question is about Ancient Egypt topics such as:
- Pharaohs, kings, queens, rulers
- Pyramids, temples, monuments, tombs
- Gods, goddesses, mythology, religion
- Hieroglyphs, papyrus, writing systems
- Mummies, burial practices, afterlife beliefs
- Daily life, culture, society in Ancient Egypt
- Archaeology, discoveries, excavations
- Ancient Egyptian history, dynasties, time periods
- Nile River, geography of Ancient Egypt
- Art, architecture, engineering of Ancient Egypt

Answer with ONLY:
- "YES" if the question is about Ancient Egypt
- "NO" if it's about something else (modern topics, other civilizations, unrelated subjects)

Answer:"""

        try:
            response = self.llm.invoke(scope_check_prompt)
            answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            is_in_scope = "YES" in answer
            
            if is_in_scope:
                print(f"✓ Query is in scope: '{query}'")
                # Rewrite query for retrieval if we have query rewriter
                if self.query_rewriter:
                    rewritten = self.query_rewriter.rewrite_for_retrieval(query)
                    return True, rewritten
                return True, query
            else:
                print(f"✗ Query out of scope: '{query}'")
                rejection_message = """I appreciate your question, but I specialize exclusively in Ancient Egyptian history and culture. 

I'd be delighted to help you explore topics such as:
• **Pharaohs and Dynasties** - Learn about Tutankhamun, Ramses II, Cleopatra, and other rulers
• **Pyramids and Monuments** - Discover how these incredible structures were built
• **Gods and Mythology** - Explore the fascinating world of Egyptian deities like Ra, Osiris, and Isis
• **Hieroglyphs and Writing** - Understand ancient Egyptian script and language
• **Daily Life and Culture** - See how ancient Egyptians lived, worked, and celebrated
• **Mummies and the Afterlife** - Learn about burial practices and beliefs about the afterlife

What would you like to know about Ancient Egypt? 🏛️"""
                return False, rejection_message
                
        except Exception as e:
            print(f"Scope check failed: {e}")
            # On error, assume in scope to avoid blocking legitimate questions
            return True, query

    def _check_context_relevance(self, query: str, context: str) -> bool:
        """
        Use LLM to check if retrieved context is relevant to the query.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            
        Returns:
            True if context is relevant, False otherwise
        """
        if not self.llm or not context:
            return False
        
        relevance_prompt = f"""You are a relevance checker for a question-answering system about Ancient Egypt.

User Question: {query}

Retrieved Context: {context[:1000]}

Task: Determine if BOTH of these conditions are true:
1. The user's question is about Ancient Egypt (pharaohs, pyramids, gods, culture, archaeology, hieroglyphs, etc.)
2. The retrieved context contains information that can help answer the question

Answer with ONLY "YES" if BOTH conditions are met, or "NO" if either:
- The question is NOT about Ancient Egypt (e.g., modern topics, other civilizations, unrelated subjects)
- The context doesn't help answer the question

Answer:"""

        try:
            response = self.llm.invoke(relevance_prompt)
            answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            # Check if answer contains YES
            is_relevant = "YES" in answer
            print(f"Relevance check: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
            return is_relevant
            
        except Exception as e:
            print(f"Relevance check failed: {e}")
            # On error, assume context is relevant to avoid unnecessary web searches
            return True
    
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
        Advanced RAG workflow with scope checking, reranking and relevance checking:
        0. Check if query is about Ancient Egypt (scope check)
        1. Query → Query Rewriter (Stage 1) 
        2. Hybrid Search → Top 25 documents
        3. Rerank → Top 3 documents
        4. LLM Relevance Check
        5. If not relevant → Web Search
        6. Generate final answer with LLM
        
        Args:
            user_input: User's message/question
            include_images: Whether to search for relevant images
            search_web: Whether to force web search (overrides relevance check)
            
        Yields:
            Dictionary chunks with type and content
        """
        if not self.is_ready():
            yield {"type": "error", "content": self._init_error or "Service not initialized"}
            return
        
        try:
            # STEP 0: Check if query is about Ancient Egypt (scope check with LLM)
            is_in_scope, processed_query = self._check_query_scope(user_input)
            
            if not is_in_scope:
                # Query is out of scope - return polite rejection
                yield {"type": "token", "content": processed_query}  # processed_query contains rejection message
                return
            
            # Query is in scope, proceed with RAG workflow
            # processed_query now contains the rewritten query for retrieval
            
            # STEP 1 & 2: Hybrid search with rewritten query → Top 25 documents
            doc_results = self.document_search(processed_query, k=25)
            
            # STEP 3: Rerank to extract top 3 documents
            context_parts = []
            sources = []
            
            if doc_results["success"] and doc_results["documents"] and self.reranker:
                # Extract Document objects for reranking
                documents = []
                for doc in doc_results["documents"]:
                    from langchain_core.documents import Document
                    documents.append(Document(
                        page_content=doc["content"],
                        metadata=doc.get("metadata", {})
                    ))
                
                # Rerank and get top 3
                print(f"Reranking {len(documents)} documents to top 3...")
                reranked_docs = self.reranker.rerank(user_input, documents, top_k=3)
                
                for doc in reranked_docs:
                    context_parts.append(doc.page_content)
                    # Create preview from first 200 chars
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    sources.append(preview)
                    
            elif doc_results["success"] and doc_results["documents"]:
                # Fallback: no reranker, use top 3 from search
                for doc in doc_results["documents"][:3]:
                    context_parts.append(doc["content"])
                    sources.append(doc["preview"])
            
            # Combine context
            context = "\n\n".join(context_parts)
            
            # STEP 4: Check context relevance with LLM (unless forced web search)
            use_web_search = search_web  # Start with user preference
            
            if not search_web and context:  # Only check if not forced and we have context
                is_relevant = self._check_context_relevance(user_input, context)
                if not is_relevant:
                    print("Context not relevant - triggering web search")
                    use_web_search = True
            
            # STEP 5: Conditional web search
            if use_web_search:
                try:
                    web_results = self.web_search(user_input)
                    if web_results.get("success") and web_results.get("results"):
                        web_context = f"\n\nWeb Search Results:\n{web_results['results']}"
                        context = context + web_context if context else web_context
                        print("Web search added to context")
                except Exception as e:
                    print(f"Web search failed: {e}")
            
            # STEP 6: Generate final answer with LLM
            # Use Stage 2 query rewriting for conversational prompt if we have context
            if self.query_rewriter and context:
                prompt = self.query_rewriter.rewrite_for_response(
                    user_query=user_input,
                    retrieved_context=[context],
                    language="en"
                )
            else:
                # Fallback prompt
                prompt = self._build_prompt(user_input, context)
            
            # Stream the LLM response
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
                    answer = f"**Found in documents (Top 3 after reranking):**\n\n"
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
