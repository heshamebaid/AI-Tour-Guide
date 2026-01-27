"""
Example: Integrating Query Rewriter into RAG Pipeline

This shows how to use the two-stage query rewriting system
in your existing Agentic RAG architecture.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.query_rewriter_service import QueryRewriterService
from services.retriever_service import RetrieverService
from services.llm_service import LLMService


class EnhancedRAGPipeline:
    """
    RAG Pipeline with Two-Stage Query Rewriting.
    
    Flow:
    1. User Query → Rewrite for Retrieval → Vector DB Search
    2. Retrieved Docs + Original Query → Rewrite for Response → LLM → Answer
    """
    
    def __init__(self):
        self.query_rewriter = QueryRewriterService()
        self.retriever_service = RetrieverService()
        self.llm_service = LLMService()
    
    def chat(self, user_query: str, include_images: bool = True, language: str = "en"):
        """
        Process user query through enhanced RAG pipeline.
        
        Args:
            user_query: Original user question
            include_images: Whether to include images in response
            language: Response language ('en' or 'ar')
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")
        print(f"{'='*60}")
        
        # STAGE 1: Rewrite query for optimal retrieval
        retrieval_query = self.query_rewriter.rewrite_for_retrieval(user_query)
        print(f"\n[Stage 1] Retrieval Query: {retrieval_query}")
        
        # Perform vector database search with optimized query
        retriever = self.retriever_service.get_retriever(k=5, search_type="hybrid")
        retrieved_docs = retriever.invoke(retrieval_query)
        
        print(f"[Retrieval] Found {len(retrieved_docs)} relevant documents")
        
        # Extract text from documents
        retrieved_context = [doc.page_content for doc in retrieved_docs]
        
        # STAGE 2: Rewrite prompt for conversational response
        llm_prompt = self.query_rewriter.rewrite_for_response(
            user_query=user_query,
            retrieved_context=retrieved_context,
            language=language
        )
        
        print(f"\n[Stage 2] Generated conversational prompt")
        print(f"Prompt length: {len(llm_prompt)} characters")
        
        # Generate response using LLM
        response = self.llm_service.generate(llm_prompt)
        
        print(f"\n[LLM] Generated response")
        print(f"Response length: {len(response)} characters")
        
        return {
            'response': response,
            'original_query': user_query,
            'retrieval_query': retrieval_query,
            'sources': retrieved_context[:3],  # Top 3 sources
            'num_sources': len(retrieved_docs)
        }
    
    def chat_streaming(self, user_query: str, include_images: bool = True, language: str = "en"):
        """
        Streaming version for real-time responses.
        
        Yields:
            Dictionary chunks with type and content
        """
        # STAGE 1: Rewrite for retrieval
        retrieval_query = self.query_rewriter.rewrite_for_retrieval(user_query)
        yield {"type": "retrieval_query", "content": retrieval_query}
        
        # Retrieve documents
        retriever = self.retriever_service.get_retriever(k=5, search_type="hybrid")
        retrieved_docs = retriever.invoke(retrieval_query)
        retrieved_context = [doc.page_content for doc in retrieved_docs]
        
        yield {"type": "sources", "content": retrieved_context[:3]}
        
        # STAGE 2: Rewrite for response
        llm_prompt = self.query_rewriter.rewrite_for_response(
            user_query=user_query,
            retrieved_context=retrieved_context,
            language=language
        )
        
        # Stream LLM response
        for token in self.llm_service.generate_streaming(llm_prompt):
            yield {"type": "token", "content": token}
        
        yield {"type": "done", "content": ""}


def demo_basic_usage():
    """Demonstrate basic query rewriting without full RAG pipeline."""
    print("\n" + "="*60)
    print("DEMO: Basic Query Rewriting")
    print("="*60)
    
    rewriter = QueryRewriterService()
    
    # Example 1: Simple query
    query1 = "Can you tell me about the pharaohs?"
    rewritten1 = rewriter.rewrite_for_retrieval(query1)
    print(f"\nQuery: {query1}")
    print(f"Rewritten: {rewritten1}")
    
    # Example 2: Complex query
    query2 = "I'm really curious about how ancient Egyptians managed to build those massive pyramids without modern technology"
    rewritten2 = rewriter.rewrite_for_retrieval(query2)
    print(f"\nQuery: {query2}")
    print(f"Rewritten: {rewritten2}")
    
    # Example 3: Create response prompt
    print("\n" + "="*60)
    print("DEMO: Response Prompt Generation")
    print("="*60)
    
    mock_context = [
        "The pyramids were built using limestone blocks quarried nearby.",
        "Workers used copper tools and wooden sledges to move stones.",
        "Ramps were likely used to raise blocks to higher levels."
    ]
    
    prompt = rewriter.rewrite_for_response(query2, mock_context, language="en")
    print(f"\nGenerated Prompt:\n{prompt[:500]}...")


def demo_comparison():
    """Compare retrieval with and without query rewriting."""
    print("\n" + "="*60)
    print("COMPARISON: With vs Without Query Rewriting")
    print("="*60)
    
    rewriter = QueryRewriterService()
    
    test_queries = [
        "Hey! What's the deal with mummies? Why did they do that?",
        "I'd love to learn more about the Valley of the Kings if you don't mind",
        "Could you possibly explain to me how hieroglyphic writing worked?",
    ]
    
    for query in test_queries:
        rewritten = rewriter.rewrite_for_retrieval(query)
        print(f"\nOriginal:  {query}")
        print(f"Optimized: {rewritten}")
        print(f"Reduction: {len(query)} → {len(rewritten)} chars")


if __name__ == "__main__":
    # Run demos
    demo_basic_usage()
    demo_comparison()
    
    print("\n" + "="*60)
    print("To integrate into your RAG pipeline:")
    print("="*60)
    print("""
1. Import QueryRewriterService in your RAG service
2. Stage 1: Rewrite user query before retrieval
3. Stage 2: Generate conversational prompt with context
4. Send prompt to LLM for final response

Example code:
    from services.query_rewriter_service import QueryRewriterService
    
    rewriter = QueryRewriterService()
    
    # Stage 1
    optimized_query = rewriter.rewrite_for_retrieval(user_input)
    docs = retriever.invoke(optimized_query)
    
    # Stage 2
    prompt = rewriter.rewrite_for_response(user_input, docs)
    response = llm.generate(prompt)
    """)
