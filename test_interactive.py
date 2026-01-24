#!/usr/bin/env python3
"""
Simple interactive test for the Qdrant RAG system.
Tests retrieval without requiring LLM API keys.
"""
import sys
import os

# Add Agentic_RAG src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Agentic_RAG', 'src'))

def test_retrieval():
    """Interactive retrieval test"""
    from services.retriever_service import RetrieverService
    
    print("\n" + "="*70)
    print("  🔍 Qdrant Hybrid Search - Interactive Test")
    print("="*70)
    
    retriever_service = RetrieverService()
    
    print("\nAvailable search types:")
    print("  1. Hybrid (Semantic + Keyword) - Best overall")
    print("  2. Dense (Semantic only) - Conceptual similarity")
    print("  3. Sparse (BM25 keyword) - Exact term matching")
    print("  4. MMR (Diverse results) - Avoid redundancy")
    
    while True:
        print("\n" + "-"*70)
        choice = input("\nSelect search type (1-4) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            break
        
        search_types = {
            '1': 'hybrid',
            '2': 'dense', 
            '3': 'sparse',
            '4': 'mmr'
        }
        
        if choice not in search_types:
            print("Invalid choice!")
            continue
        
        search_type = search_types[choice]
        
        query = input("\nEnter your question: ").strip()
        if not query:
            continue
        
        k = input("How many results? (default 3): ").strip()
        k = int(k) if k.isdigit() else 3
        
        try:
            print(f"\n🔍 Searching with {search_type.upper()} search...")
            
            if search_type == 'mmr':
                retriever = retriever_service.get_mmr_retriever(k=k, lambda_mult=0.5)
            else:
                retriever = retriever_service.get_retriever(k=k, search_type=search_type)
            
            results = retriever.invoke(query)
            
            print(f"\n✅ Found {len(results)} results:\n")
            
            for i, doc in enumerate(results, 1):
                print(f"{i}. {'-'*65}")
                content = doc.page_content[:400].replace('\n', ' ')
                if len(doc.page_content) > 400:
                    content += "..."
                print(f"   {content}")
                
                if doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"\n   📄 Source: {source}")
                print()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        test_retrieval()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
