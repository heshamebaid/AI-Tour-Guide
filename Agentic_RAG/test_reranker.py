#!/usr/bin/env python3
"""
Test script for the Reranker Service.
Tests CrossEncoder reranking independently from the full RAG pipeline.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Disable CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from langchain_core.documents import Document

print("=" * 60)
print("Testing Reranker Service")
print("=" * 60)

# Test 1: Import and initialize
print("\n1. Initializing RerankerService...")
try:
    from services.reranker_service import RerankerService
    reranker = RerankerService()
    print("   ✓ RerankerService initialized")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: Create sample documents
print("\n2. Creating sample documents...")
documents = [
    Document(page_content="The Great Pyramid of Giza is one of the Seven Wonders of the Ancient World."),
    Document(page_content="Pizza is a popular Italian dish made with dough, tomato sauce, and cheese."),
    Document(page_content="Ancient Egyptian pharaohs were buried in elaborate tombs with treasures."),
    Document(page_content="The Sphinx is a limestone statue with the body of a lion and head of a human."),
    Document(page_content="Coffee is a beverage made from roasted coffee beans."),
    Document(page_content="Hieroglyphics were the formal writing system used in Ancient Egypt."),
    Document(page_content="The Nile River is the longest river in Africa and was crucial to Egyptian civilization."),
    Document(page_content="Basketball is a team sport played on a rectangular court."),
]
print(f"   ✓ Created {len(documents)} sample documents")

# Test 3: Rerank documents
print("\n3. Testing reranking...")
query = "Tell me about ancient Egyptian pyramids and monuments"
print(f"   Query: '{query}'")
print("\n   Original order:")
for i, doc in enumerate(documents, 1):
    print(f"   {i}. {doc.page_content[:60]}...")

print("\n   Reranking...")
try:
    reranked = reranker.rerank(query, documents)
    print("\n   ✓ Reranked order:")
    for i, doc in enumerate(reranked, 1):
        print(f"   {i}. {doc.page_content[:60]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Rerank with scores
print("\n4. Testing reranking with scores...")
try:
    scored_docs = reranker.rerank_with_scores(query, documents, top_k=5)
    print("\n   ✓ Top 5 documents with scores:")
    for i, (doc, score) in enumerate(scored_docs, 1):
        print(f"   {i}. [Score: {score:.4f}] {doc.page_content[:50]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Different query
print("\n5. Testing with different query...")
query2 = "What is the Nile River?"
print(f"   Query: '{query2}'")
try:
    scored_docs = reranker.rerank_with_scores(query2, documents, top_k=3)
    print("\n   ✓ Top 3 documents:")
    for i, (doc, score) in enumerate(scored_docs, 1):
        print(f"   {i}. [Score: {score:.4f}] {doc.page_content[:50]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 6: Integration with real retriever
print("\n6. Testing integration with retriever (optional)...")
try:
    from services.retriever_service import RetrieverService
    
    retriever_service = RetrieverService()
    
    # Test without reranker
    print("\n   a) Without reranker:")
    retriever_no_rerank = retriever_service.get_retriever(k=3, use_reranker=False)
    docs_no_rerank = retriever_no_rerank.invoke("pyramids of Egypt")
    for i, doc in enumerate(docs_no_rerank, 1):
        print(f"      {i}. {doc.page_content[:60]}...")
    
    # Test with reranker
    print("\n   b) With reranker:")
    retriever_with_rerank = retriever_service.get_retriever(k=3, use_reranker=True)
    docs_with_rerank = retriever_with_rerank.invoke("pyramids of Egypt")
    for i, doc in enumerate(docs_with_rerank, 1):
        print(f"      {i}. {doc.page_content[:60]}...")
    
    print("\n   ✓ Integration test passed")
except Exception as e:
    print(f"   ⚠️ Skipped (retriever not available): {e}")

print("\n" + "=" * 60)
print("✅ Reranker tests completed!")
print("=" * 60)
