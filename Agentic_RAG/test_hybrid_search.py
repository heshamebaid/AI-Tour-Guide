#!/usr/bin/env python3
"""
Test script to verify Hybrid Search is working correctly.
Tests: Dense (semantic), Sparse (BM25), and Hybrid (combined) search.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Disable CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("=" * 70)
print("Testing Hybrid Search Implementation")
print("=" * 70)

# Test 1: Check BM25 model exists
print("\n1. Checking BM25 model...")
try:
    from services.sparse_encoder_service import SparseEncoderService
    from core.config import Config
    
    sparse_encoder = SparseEncoderService()
    bm25_exists = sparse_encoder.load()
    
    if bm25_exists:
        print(f"   ✓ BM25 model loaded from: {sparse_encoder.model_path}")
        print(f"   ✓ Corpus size: {len(sparse_encoder.corpus_texts)} documents")
    else:
        print(f"   ❌ BM25 model NOT found at: {sparse_encoder.model_path}")
        print("   ⚠️  Hybrid search will fall back to dense-only mode!")
        print("   ℹ️  To enable hybrid search, rebuild the index.")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Test Dense Search
print("\n2. Testing Dense (Semantic) Search...")
try:
    from services.retriever_service import RetrieverService
    
    retriever_service = RetrieverService()
    
    query = "ancient Egyptian pyramids"
    print(f"   Query: '{query}'")
    
    # Get dense-only retriever (no reranker for fair comparison)
    dense_retriever = retriever_service.get_retriever(k=3, search_type="dense", use_reranker=False)
    dense_results = dense_retriever.invoke(query)
    
    print(f"   ✓ Dense search returned {len(dense_results)} documents:")
    for i, doc in enumerate(dense_results, 1):
        print(f"      {i}. {doc.page_content[:70]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test Sparse (BM25) Search
print("\n3. Testing Sparse (BM25) Search...")
try:
    if bm25_exists:
        sparse_retriever = retriever_service.get_retriever(k=3, search_type="sparse", use_reranker=False)
        sparse_results = sparse_retriever.invoke(query)
        
        print(f"   ✓ Sparse search returned {len(sparse_results)} documents:")
        for i, doc in enumerate(sparse_results, 1):
            print(f"      {i}. {doc.page_content[:70]}...")
    else:
        print("   ⚠️ Skipped - BM25 model not available")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Hybrid Search
print("\n4. Testing Hybrid Search (Dense + Sparse)...")
try:
    hybrid_retriever = retriever_service.get_retriever(k=3, search_type="hybrid", use_reranker=False)
    hybrid_results = hybrid_retriever.invoke(query)
    
    print(f"   ✓ Hybrid search returned {len(hybrid_results)} documents:")
    for i, doc in enumerate(hybrid_results, 1):
        print(f"      {i}. {doc.page_content[:70]}...")
    
    # Check if it's actually an EnsembleRetriever
    retriever_type = type(hybrid_retriever).__name__
    print(f"\n   Retriever type: {retriever_type}")
    
    if hasattr(hybrid_retriever, 'retrievers'):
        print(f"   ✓ EnsembleRetriever with {len(hybrid_retriever.retrievers)} sub-retrievers")
        print(f"   ✓ Weights: Dense={Config.DENSE_WEIGHT}, Sparse={Config.SPARSE_WEIGHT}")
    elif retriever_type == "VectorStoreRetriever":
        print("   ⚠️ Using dense-only fallback (BM25 not loaded)")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Compare results between search types
print("\n5. Comparing Search Types (keyword-heavy query)...")
try:
    keyword_query = "Sphinx limestone statue lion"
    print(f"   Query: '{keyword_query}'")
    
    print("\n   Dense results:")
    dense_r = retriever_service.get_retriever(k=2, search_type="dense", use_reranker=False)
    for i, doc in enumerate(dense_r.invoke(keyword_query), 1):
        print(f"      {i}. {doc.page_content[:60]}...")
    
    if bm25_exists:
        print("\n   Sparse (BM25) results:")
        sparse_r = retriever_service.get_retriever(k=2, search_type="sparse", use_reranker=False)
        for i, doc in enumerate(sparse_r.invoke(keyword_query), 1):
            print(f"      {i}. {doc.page_content[:60]}...")
    
    print("\n   Hybrid results:")
    hybrid_r = retriever_service.get_retriever(k=2, search_type="hybrid", use_reranker=False)
    for i, doc in enumerate(hybrid_r.invoke(keyword_query), 1):
        print(f"      {i}. {doc.page_content[:60]}...")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 6: Test with Reranker
print("\n6. Testing Hybrid + Reranker...")
try:
    reranked_retriever = retriever_service.get_retriever(k=3, search_type="hybrid", use_reranker=True)
    reranked_results = reranked_retriever.invoke("Tell me about the pyramids of Giza")
    
    retriever_type = type(reranked_retriever).__name__
    print(f"   Retriever type: {retriever_type}")
    print(f"   ✓ Reranked results ({len(reranked_results)} documents):")
    for i, doc in enumerate(reranked_results, 1):
        print(f"      {i}. {doc.page_content[:70]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print(f"  - BM25 Model: {'✓ Loaded' if bm25_exists else '❌ Not found'}")
print(f"  - Dense Search: {'✓ Working' if 'dense_results' in dir() else '❌ Failed'}")
print(f"  - Sparse Search: {'✓ Working' if bm25_exists and 'sparse_results' in dir() else '⚠️ Not available'}")
print(f"  - Hybrid Search: {'✓ Working' if 'hybrid_results' in dir() else '❌ Failed'}")
print(f"  - Reranker: {'✓ Working' if 'reranked_results' in dir() else '❌ Failed'}")

if not bm25_exists:
    print("\n⚠️  To enable full hybrid search, rebuild the index:")
    print("   python -c \"from services.indexing_service import IndexingService; IndexingService().build_index()\"")

print("=" * 70)
