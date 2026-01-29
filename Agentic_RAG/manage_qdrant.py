"""
Utility script for managing Qdrant collections and indexes.
Provides convenient commands for common operations.
"""
import sys
import os
import argparse

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.indexing_service import IndexingService
from services.vectorstore_service import VectorStoreService
from services.retriever_service import RetrieverService


def build_index(args):
    """Build the Qdrant index from documents"""
    print("\n🔨 Building Qdrant index...")
    indexing = IndexingService()
    success = indexing.build_index(verbose=True)
    
    if success:
        print("\n✅ Index built successfully!")
        return 0
    else:
        print("\n❌ Failed to build index")
        return 1


def rebuild_index(args):
    """Rebuild the Qdrant index (deletes existing collection)"""
    print("\n🔄 Rebuilding Qdrant index (this will delete existing data)...")
    
    if not args.force:
        confirm = input("Are you sure? This will delete all existing data. (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return 0
    
    indexing = IndexingService()
    success = indexing.rebuild_index(verbose=True)
    
    if success:
        print("\n✅ Index rebuilt successfully!")
        return 0
    else:
        print("\n❌ Failed to rebuild index")
        return 1


def delete_collection(args):
    """Delete the Qdrant collection"""
    print("\n🗑️  Deleting Qdrant collection...")
    
    if not args.force:
        confirm = input("Are you sure? This will delete all data. (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return 0
    
    vs_service = VectorStoreService()
    success = vs_service.delete_collection()
    
    if success:
        print("\n✅ Collection deleted successfully!")
        return 0
    else:
        print("\n❌ Failed to delete collection")
        return 1


def show_info(args):
    """Show information about the Qdrant collection"""
    print("\n📊 Qdrant Collection Information")
    print("=" * 60)
    
    vs_service = VectorStoreService()
    info = vs_service.get_collection_info()
    
    print(f"\nCollection Name: {info.get('name')}")
    print(f"Exists: {info.get('exists')}")
    
    if info.get('exists'):
        print(f"Status: {info.get('status')}")
        print(f"Vectors Count: {info.get('vectors_count')}")
        print(f"Points Count: {info.get('points_count')}")
        print(f"Hybrid Search Enabled: {info.get('hybrid_search_enabled')}")
    else:
        print("\n⚠️  Collection does not exist. Run 'build' command first.")
    
    print("=" * 60)
    return 0


def test_search(args):
    """Test the search functionality"""
    query = args.query or "Tell me about Egypt"
    search_type = args.type or "hybrid"
    k = args.k or 4
    
    print(f"\n🔍 Testing {search_type} search...")
    print(f"Query: {query}")
    print(f"Results to fetch: {k}")
    print("=" * 60)
    
    try:
        retriever_service = RetrieverService()
        retriever = retriever_service.get_retriever(k=k, search_type=search_type)
        
        results = retriever.invoke(query)
        
        print(f"\n✅ Found {len(results)} results:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"{i}. {'-' * 55}")
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            print(f"   {content}")
            
            if doc.metadata:
                print(f"\n   Metadata:")
                for key, value in doc.metadata.items():
                    print(f"     - {key}: {value}")
            print()
        
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def check_health(args):
    """Check the health of Qdrant and the system"""
    print("\n🏥 System Health Check")
    print("=" * 60)
    
    # Check Qdrant connection
    print("\n1. Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        from core.config import Config
        
        client = QdrantClient(url=Config.QDRANT_URL)
        collections = client.get_collections()
        print(f"   ✅ Qdrant is running at {Config.QDRANT_URL}")
        print(f"   Collections: {len(collections.collections)}")
    except Exception as e:
        print(f"   ❌ Cannot connect to Qdrant: {e}")
        return 1
    
    # Check collection
    print("\n2. Checking collection...")
    vs_service = VectorStoreService()
    info = vs_service.get_collection_info()
    
    if info.get('exists'):
        print(f"   ✅ Collection '{info['name']}' exists")
        print(f"   Documents: {info.get('points_count', 0)}")
    else:
        print(f"   ⚠️  Collection '{info['name']}' does not exist")
    
    # Check BM25 model
    print("\n3. Checking BM25 model...")
    from services.sparse_encoder_service import SparseEncoderService
    
    sparse_encoder = SparseEncoderService()
    if sparse_encoder.load():
        print("   ✅ BM25 model loaded successfully")
        print(f"   Corpus size: {len(sparse_encoder.corpus_texts)}")
    else:
        print("   ⚠️  BM25 model not found (hybrid search unavailable)")
    
    # Check embeddings
    print("\n4. Checking embeddings service...")
    try:
        from services.embeddings_service import EmbeddingService
        
        emb_service = EmbeddingService()
        test_embedding = emb_service.embed("test")
        print(f"   ✅ Embeddings service working")
        print(f"   Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"   ❌ Embeddings service error: {e}")
    
    print("\n" + "=" * 60)
    print("Health check completed!")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Qdrant Management Utility for Agentic RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_qdrant.py build              # Build index from documents
  python manage_qdrant.py rebuild --force    # Rebuild index without confirmation
  python manage_qdrant.py info               # Show collection info
  python manage_qdrant.py test --query "Egypt pyramids"  # Test search
  python manage_qdrant.py health             # Check system health
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build the Qdrant index')
    build_parser.set_defaults(func=build_index)
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild the Qdrant index')
    rebuild_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    rebuild_parser.set_defaults(func=rebuild_index)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete the collection')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    delete_parser.set_defaults(func=delete_collection)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show collection information')
    info_parser.set_defaults(func=show_info)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test search functionality')
    test_parser.add_argument('--query', '-q', type=str, help='Search query')
    test_parser.add_argument('--type', '-t', choices=['hybrid', 'dense', 'sparse'], 
                           help='Search type (default: hybrid)')
    test_parser.add_argument('--k', '-k', type=int, help='Number of results (default: 4)')
    test_parser.set_defaults(func=test_search)
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    health_parser.set_defaults(func=check_health)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
