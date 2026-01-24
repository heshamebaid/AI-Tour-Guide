"""
Example script demonstrating Qdrant vector store with hybrid search.
This script shows how to build the index and perform different types of searches.
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.indexing_service import IndexingService
from services.retriever_service import RetrieverService
from services.vectorstore_service import VectorStoreService


def build_index_example():
    """Example: Build the Qdrant vector index"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Building Qdrant Vector Index")
    print("="*60)
    
    indexing = IndexingService()
    success = indexing.build_index(verbose=True)
    
    if success:
        print("\n✓ Index built successfully!")
        
        # Get index info
        info = indexing.get_index_info()
        print(f"\nCollection Info:")
        print(f"  - Name: {info.get('name')}")
        print(f"  - Exists: {info.get('exists')}")
        print(f"  - Documents: {info.get('points_count', 'N/A')}")
        print(f"  - Hybrid Search: {info.get('hybrid_search_enabled', False)}")
    else:
        print("\n❌ Failed to build index")


def hybrid_search_example():
    """Example: Perform hybrid search"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Hybrid Search (Dense + Sparse)")
    print("="*60)
    
    retriever_service = RetrieverService()
    
    # Create hybrid retriever
    retriever = retriever_service.get_retriever(k=4, search_type="hybrid")
    
    # Perform search
    query = "What are the main attractions in Egypt?"
    print(f"\nQuery: {query}")
    print("\nSearching with hybrid retriever (semantic + keyword)...")
    
    results = retriever.invoke(query)
    
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"\n{i}. {content}")
        if doc.metadata:
            print(f"   Metadata: {doc.metadata}")


def dense_search_example():
    """Example: Perform dense (semantic) search only"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Dense Search (Semantic Only)")
    print("="*60)
    
    retriever_service = RetrieverService()
    
    # Create dense retriever
    retriever = retriever_service.get_retriever(k=4, search_type="dense")
    
    # Perform search
    query = "ancient pyramids and pharaohs"
    print(f"\nQuery: {query}")
    print("\nSearching with dense retriever (semantic similarity)...")
    
    results = retriever.invoke(query)
    
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"\n{i}. {content}")


def sparse_search_example():
    """Example: Perform sparse (BM25 keyword) search only"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Sparse Search (BM25 Keyword)")
    print("="*60)
    
    retriever_service = RetrieverService()
    
    try:
        # Create sparse retriever
        retriever = retriever_service.get_retriever(k=4, search_type="sparse")
        
        # Perform search
        query = "pyramid Egypt pharaoh"
        print(f"\nQuery: {query}")
        print("\nSearching with sparse retriever (BM25 keyword matching)...")
        
        results = retriever.invoke(query)
        
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"\n{i}. {content}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def mmr_search_example():
    """Example: Perform MMR search for diverse results"""
    print("\n" + "="*60)
    print("EXAMPLE 5: MMR Search (Maximum Marginal Relevance)")
    print("="*60)
    
    retriever_service = RetrieverService()
    
    # Create MMR retriever for diverse results
    retriever = retriever_service.get_mmr_retriever(
        k=4,
        fetch_k=20,
        lambda_mult=0.5  # Balance between relevance and diversity
    )
    
    # Perform search
    query = "Egyptian history and culture"
    print(f"\nQuery: {query}")
    print("\nSearching with MMR retriever (diverse results)...")
    
    results = retriever.invoke(query)
    
    print(f"\nFound {len(results)} diverse results:")
    for i, doc in enumerate(results, 1):
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"\n{i}. {content}")


def collection_info_example():
    """Example: Get collection information"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Collection Information")
    print("="*60)
    
    vs_service = VectorStoreService()
    info = vs_service.get_collection_info()
    
    print(f"\nQdrant Collection Information:")
    print(f"  - Collection Name: {info.get('name')}")
    print(f"  - Exists: {info.get('exists')}")
    
    if info.get('exists'):
        print(f"  - Status: {info.get('status')}")
        print(f"  - Vectors Count: {info.get('vectors_count')}")
        print(f"  - Points Count: {info.get('points_count')}")
        print(f"  - Hybrid Search Enabled: {info.get('hybrid_search_enabled')}")
    else:
        print("\n⚠️ Collection does not exist. Run build_index_example() first.")


def comparison_example():
    """Example: Compare different search types"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Comparing Search Types")
    print("="*60)
    
    retriever_service = RetrieverService()
    query = "ancient Egyptian civilization"
    
    print(f"\nQuery: {query}\n")
    
    # Try each search type
    search_types = ["hybrid", "dense", "sparse"]
    
    for search_type in search_types:
        print(f"\n--- {search_type.upper()} SEARCH ---")
        try:
            retriever = retriever_service.get_retriever(k=2, search_type=search_type)
            results = retriever.invoke(query)
            
            for i, doc in enumerate(results, 1):
                content = doc.page_content[:150] + "..."
                print(f"{i}. {content}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run examples"""
    print("\n" + "="*60)
    print("Qdrant Vector Store with Hybrid Search - Examples")
    print("="*60)
    
    print("\nAvailable examples:")
    print("1. Build Index")
    print("2. Hybrid Search")
    print("3. Dense Search")
    print("4. Sparse Search")
    print("5. MMR Search")
    print("6. Collection Info")
    print("7. Compare Search Types")
    print("0. Run All")
    
    choice = input("\nSelect example (0-7): ").strip()
    
    examples = {
        "1": build_index_example,
        "2": hybrid_search_example,
        "3": dense_search_example,
        "4": sparse_search_example,
        "5": mmr_search_example,
        "6": collection_info_example,
        "7": comparison_example,
    }
    
    if choice == "0":
        # Run all examples
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("Invalid choice!")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
