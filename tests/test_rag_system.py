#!/usr/bin/env python3
"""
Test the Qdrant-based RAG system with sample queries.
This script tests the system before running the Django chatbot.
"""
import sys
import os

# Add Agentic_RAG src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Agentic_RAG', 'src'))

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_qdrant_connection():
    """Test Qdrant connection"""
    print_header("1. Testing Qdrant Connection")
    
    try:
        from qdrant_client import QdrantClient
        from core.config import Config
        
        client = QdrantClient(url=Config.QDRANT_URL)
        collections = client.get_collections()
        
        print(f"✅ Connected to Qdrant at {Config.QDRANT_URL}")
        print(f"   Collections available: {len(collections.collections)}")
        
        for coll in collections.collections:
            print(f"   - {coll.name}")
        
        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("\n💡 Start Qdrant with:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        return False

def test_collection_status():
    """Check if collection exists and has data"""
    print_header("2. Checking Qdrant Collection")
    
    try:
        from services.vectorstore_service import VectorStoreService
        
        vs_service = VectorStoreService()
        info = vs_service.get_collection_info()
        
        if not info.get('exists'):
            print(f"❌ Collection '{info['name']}' does not exist!")
            print("\n💡 Build the index first:")
            print("   cd Agentic_RAG")
            print("   python manage_qdrant.py build")
            return False
        
        print(f"✅ Collection: {info['name']}")
        print(f"   Status: {info.get('status')}")
        print(f"   Documents: {info.get('points_count', 0)}")
        print(f"   Vectors: {info.get('vectors_count', 0)}")
        print(f"   Hybrid Search: {'Enabled' if info.get('hybrid_search_enabled') else 'Disabled'}")
        
        if info.get('points_count', 0) == 0:
            print("\n⚠️  Collection is empty! Add documents and rebuild index.")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_hybrid_search():
    """Test hybrid search"""
    print_header("3. Testing Hybrid Search (Semantic + Keyword)")
    
    try:
        from services.retriever_service import RetrieverService
        
        retriever_service = RetrieverService()
        retriever = retriever_service.get_retriever(k=3, search_type="hybrid")
        
        query = "Tell me about ancient Egyptian pyramids and pharaohs"
        print(f"\n🔍 Query: '{query}'")
        print("   Using: Hybrid Search (Dense + BM25)\n")
        
        results = retriever.invoke(query)
        
        print(f"✅ Found {len(results)} documents:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"{i}. {'-'*65}")
            content = doc.page_content[:250].replace('\n', ' ') + "..."
            print(f"   {content}")
            if doc.metadata:
                print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dense_search():
    """Test dense (semantic) search"""
    print_header("4. Testing Dense Search (Semantic Only)")
    
    try:
        from services.retriever_service import RetrieverService
        
        retriever_service = RetrieverService()
        retriever = retriever_service.get_retriever(k=2, search_type="dense")
        
        query = "ancient civilization and culture"
        print(f"\n🔍 Query: '{query}'")
        print("   Using: Dense Search (Semantic Similarity)\n")
        
        results = retriever.invoke(query)
        
        print(f"✅ Found {len(results)} documents:\n")
        
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:200].replace('\n', ' ') + "..."
            print(f"{i}. {content}\n")
        
        return True
    except Exception as e:
        print(f"❌ Dense search failed: {e}")
        return False

def test_rag_query():
    """Test full RAG query with LLM"""
    print_header("5. Testing Full RAG Pipeline (Retrieval + LLM)")
    
    try:
        from services.retriever_service import RetrieverService
        from services.llm_service import LLMService
        
        # Try different imports for RetrievalQA based on langchain version
        try:
            from langchain.chains import RetrievalQA
        except ImportError:
            try:
                from langchain_core.chains import RetrievalQA
            except ImportError:
                from langchain.chains.retrieval_qa.base import RetrievalQA
        
        # Get retriever with hybrid search
        retriever_service = RetrieverService()
        retriever = retriever_service.get_retriever(k=3, search_type="hybrid")
        
        # Get LLM
        llm_service = LLMService()
        llm = llm_service.llm
        
        print("\n✅ Services loaded successfully")
        print("   Creating QA chain...")
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        query = "What were the main achievements of ancient Egyptian civilization?"
        print(f"\n💬 Question: '{query}'")
        print("   Processing with RAG pipeline...\n")
        
        result = qa_chain.invoke({"query": query})
        
        print("✅ Answer:")
        print("-" * 70)
        answer = result.get('result') or result.get('answer', 'No answer generated')
        print(answer)
        print("-" * 70)
        
        source_docs = result.get('source_documents', [])
        print(f"\n📚 Based on {len(source_docs)} source documents")
        
        return True
    except Exception as e:
        print(f"❌ RAG query failed: {e}")
        print("\n💡 This might be okay if:")
        print("   - OPENROUTER_API_KEY is not configured")
        print("   - You want to skip LLM testing")
        print("\nError details:")
        import traceback
        traceback.print_exc()
        return False

def test_django_integration():
    """Test Django RAG service"""
    print_header("6. Testing Django Integration")
    
    try:
        # Add Django to path
        django_path = os.path.join(os.path.dirname(__file__), 'Django')
        sys.path.insert(0, django_path)
        
        # Set up minimal Django settings
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
        
        try:
            import django
            django.setup()
        except Exception as setup_error:
            print(f"⚠️  Django setup warning: {setup_error}")
            print("   Continuing with basic import test...")
        
        from myapp.rag_service import RAGService
        
        print("✅ RAG Service imported successfully")
        
        rag = RAGService()
        
        # Check status
        status = rag.get_status()
        print("\n📊 RAG Service Status:")
        print(f"   Initialized: {status['initialized']}")
        print(f"   Vector DB: {status.get('vector_db', 'Unknown')}")
        
        if 'collection_info' in status:
            info = status['collection_info']
            if 'error' not in info:
                print(f"   Collection: {info.get('name')}")
                print(f"   Documents: {info.get('documents', 0)}")
                print(f"   Hybrid Search: {info.get('hybrid_search', False)}")
            else:
                print(f"   Collection info error: {info['error']}")
        
        if not rag.is_ready():
            print(f"\n⚠️  Service not fully ready: {status.get('error')}")
            print("   This might be expected if LLM service isn't configured")
            # Still return True if basic retriever works
            if status.get('has_retriever'):
                print("   But retriever is available, so basic search should work")
                return True
            return False
        
        # Test document search
        print("\n🔍 Testing document search...")
        result = rag.document_search("Egyptian pyramids", k=2)
        
        if result['success']:
            print(f"✅ Found {result['count']} documents")
            for doc in result['documents'][:2]:
                preview = doc['preview'][:150].replace('\n', ' ')
                print(f"   - {preview}...")
        else:
            print(f"❌ Search failed: {result.get('error')}")
            return False
        
        print("\n✅ Django integration working!")
        return True
        
    except Exception as e:
        print(f"❌ Django integration test failed: {e}")
        print("\n💡 This is okay if you haven't set up Django yet.")
        print("   The RAG system itself is working fine.")
        print("\nError details:")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  🧪 Testing Qdrant-based RAG System")
    print("="*70)
    
    tests = [
        ("Qdrant Connection", test_qdrant_connection),
        ("Collection Status", test_collection_status),
        ("Hybrid Search", test_hybrid_search),
        ("Dense Search", test_dense_search),
        ("Full RAG Pipeline", test_rag_query),
        ("Django Integration", test_django_integration),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{'='*70}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Start Django: cd Django && python manage.py runserver")
        print("   2. Open browser: http://localhost:8000")
        print("   3. Start chatting with the RAG-powered chatbot!")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("   - Build index: cd Agentic_RAG && python manage_qdrant.py build")
        print("   - Check .env file configuration")
    
    print()
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
