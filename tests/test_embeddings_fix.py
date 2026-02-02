"""
Test the fixed embeddings service to ensure meta tensor issue is resolved.
"""
import os
import sys
from pathlib import Path

# Add Agentic_RAG to path
AGENTIC_RAG_PATH = Path(__file__).resolve().parent / "Agentic_RAG"
sys.path.insert(0, str(AGENTIC_RAG_PATH / "src"))

# Load environment from project root .env
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent / ".env"))

print("=" * 60)
print("Testing Fixed Embeddings Service")
print("=" * 60)

try:
    print("\n1. Testing Embeddings Service...")
    from services.embeddings_service import EmbeddingService
    
    embeddings_service = EmbeddingService()
    print("   ✓ Embeddings service initialized")
    
    # Test embedding
    test_text = "Hello world"
    embedding = embeddings_service.embed(test_text)
    print(f"   ✓ Generated embedding of length: {len(embedding)}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. Testing Retriever Service...")
    from services.retriever_service import RetrieverService
    
    retriever_service = RetrieverService()
    print("   ✓ Retriever service initialized")
    
    retriever = retriever_service.get_retriever(k=3, search_type="hybrid")
    print("   ✓ Hybrid retriever created")
    
    # Test search
    results = retriever.invoke("pyramids")
    print(f"   ✓ Retrieved {len(results)} documents")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Testing LLM Service...")
    from services.llm_service import LLMService
    
    llm_service = LLMService()
    print("   ✓ LLM service initialized")
    
    # Test pickle
    import pickle
    pickled = pickle.dumps(llm_service)
    unpickled = pickle.loads(pickled)
    print("   ✓ LLM service is picklable")
    
    # Test LLM
    llm = llm_service.llm
    response = llm.invoke("Say 'test'")
    print(f"   ✓ LLM responded: {response.content[:50]}")
    
except Exception as e:
    print(f"   ❌ LLM Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All critical services working!")
print("=" * 60)
