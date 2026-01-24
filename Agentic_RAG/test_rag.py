#!/usr/bin/env python3
"""
Test script to verify the RAG pipeline is working correctly.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("1. Testing Configuration")
    print("="*60)
    
    from core.config import Config
    
    print(f"   OPENROUTER_API_KEY: {'✓ Set' if Config.OPENROUTER_API_KEY else '✗ Missing'}")
    print(f"   OPENROUTER_MODEL: {Config.OPENROUTER_MODEL or '✗ Missing'}")
    print(f"   EMBEDDING_MODEL: {Config.EMBEDDING_MODEL or '✗ Missing'}")
    print(f"   VECTORSTORE_PATH: {Config.VECTORSTORE_PATH}")
    print(f"   DOCUMENTS_PATH: {Config.DOCUMENTS_PATH}")
    
    # Check if documents exist
    if os.path.exists(Config.DOCUMENTS_PATH):
        files = os.listdir(Config.DOCUMENTS_PATH)
        print(f"   Documents found: {len(files)} files")
        for f in files[:5]:
            print(f"      - {f}")
    else:
        print(f"   ✗ Documents path does not exist!")
    
    return bool(Config.OPENROUTER_API_KEY and Config.EMBEDDING_MODEL)


def test_embeddings():
    """Test embedding service."""
    print("\n" + "="*60)
    print("2. Testing Embedding Service")
    print("="*60)
    
    try:
        from services.embeddings_service import EmbeddingService
        
        emb_service = EmbeddingService()
        test_text = "Ancient Egyptian pyramids"
        embedding = emb_service.embed(test_text)
        
        print(f"   ✓ Embedding model loaded successfully")
        print(f"   ✓ Test embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"   ✗ Embedding service error: {e}")
        return False


def test_build_index():
    """Test building the vector store index."""
    print("\n" + "="*60)
    print("3. Building Vector Store Index")
    print("="*60)
    
    try:
        from services.indexing_service import IndexingService
        
        indexing_service = IndexingService()
        success = indexing_service.build_index(verbose=True)
        
        if success:
            print("   ✓ Index built successfully!")
        else:
            print("   ✗ Index building failed!")
        
        return success
    except Exception as e:
        print(f"   ✗ Index building error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """Test document retrieval."""
    print("\n" + "="*60)
    print("4. Testing Document Retrieval")
    print("="*60)
    
    try:
        from services.retriever_service import RetrieverService
        
        retriever_service = RetrieverService()
        retriever = retriever_service.get_retriever(k=3)
        
        # Test query
        test_query = "Tell me about ancient Egypt"
        docs = retriever.invoke(test_query)
        
        print(f"   ✓ Retriever loaded successfully")
        print(f"   ✓ Query: '{test_query}'")
        print(f"   ✓ Retrieved {len(docs)} documents")
        
        for i, doc in enumerate(docs, 1):
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f"\n   Document {i}:")
            print(f"   {content_preview}...")
        
        return len(docs) > 0
    except Exception as e:
        print(f"   ✗ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm():
    """Test LLM service connection."""
    print("\n" + "="*60)
    print("5. Testing LLM Service")
    print("="*60)
    
    try:
        from services.llm_service import LLMService
        
        llm_service = LLMService()
        
        # Test a simple query
        response = llm_service.llm.invoke("Say 'Hello, RAG is working!' in one short sentence.")
        
        print(f"   ✓ LLM connected successfully")
        print(f"   ✓ Response: {response.content[:200]}")
        
        return True
    except Exception as e:
        print(f"   ✗ LLM service error: {e}")
        return False


def test_full_rag_query():
    """Test full RAG pipeline with a query."""
    print("\n" + "="*60)
    print("6. Testing Full RAG Query")
    print("="*60)
    
    try:
        from services.retriever_service import RetrieverService
        from services.llm_service import LLMService
        
        # Get retriever and LLM
        retriever = RetrieverService().get_retriever(k=3)
        llm = LLMService().llm
        
        # Test query
        query = "What do you know about Egyptian pharaohs?"
        
        print(f"   Query: '{query}'")
        print("   Retrieving relevant documents...")
        
        # Retrieve documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print(f"   ✓ Retrieved {len(docs)} documents")
        print("   Generating response...")
        
        # Create prompt with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context[:2000]}

Question: {query}

Answer:"""
        
        # Get response
        response = llm.invoke(prompt)
        
        print(f"\n   ✓ RAG Response:")
        print("-" * 50)
        print(response.content)
        print("-" * 50)
        
        return True
    except Exception as e:
        print(f"   ✗ Full RAG query error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("       RAG Pipeline Test Suite")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['config'] = test_config()
    
    if results['config']:
        results['embeddings'] = test_embeddings()
        
        if results['embeddings']:
            results['index'] = test_build_index()
            
            if results['index']:
                results['retrieval'] = test_retrieval()
                results['llm'] = test_llm()
                
                if results['retrieval'] and results['llm']:
                    results['full_rag'] = test_full_rag_query()
    
    # Summary
    print("\n" + "="*60)
    print("       Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("   All tests passed! RAG pipeline is working correctly.")
    else:
        print("   Some tests failed. Please check the errors above.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
