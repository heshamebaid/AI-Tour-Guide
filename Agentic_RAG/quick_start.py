#!/usr/bin/env python3
"""
Quick Start Script for Qdrant-based Agentic RAG
Helps you get started quickly with the new system.
"""

import sys
import os
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_qdrant():
    """Check if Qdrant is running"""
    print_header("Step 1: Checking Qdrant")
    
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=2)
        if response.status_code == 200:
            print("✅ Qdrant is running!")
            return True
    except:
        pass
    
    print("❌ Qdrant is not running!")
    print("\nTo start Qdrant, run one of these commands:")
    print("\n  Docker (Recommended):")
    print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
    print("\n  Docker with persistent storage:")
    print("  docker run -p 6333:6333 -p 6334:6334 \\")
    print("    -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant")
    return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Step 2: Checking Dependencies")
    
    required = [
        'qdrant_client',
        'langchain_qdrant',
        'rank_bm25',
        'sentence_transformers'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing packages detected!")
        print("\nTo install, run:")
        print("  cd Agentic_RAG")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_config():
    """Check if configuration exists"""
    print_header("Step 3: Checking Configuration")
    
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    env_example = os.path.join(os.path.dirname(__file__), '.env.example')
    
    if os.path.exists(env_file):
        print(f"✅ Configuration file found: {env_file}")
        
        # Check for required settings
        with open(env_file, 'r') as f:
            content = f.read()
            
        required_settings = [
            'QDRANT_URL',
            'EMBEDDING_MODEL',
            'OPENROUTER_API_KEY'
        ]
        
        for setting in required_settings:
            if setting in content:
                print(f"  ✅ {setting}")
            else:
                print(f"  ⚠️  {setting} not found")
        
        return True
    else:
        print(f"❌ Configuration file not found!")
        print(f"\nTo create, copy the example file:")
        print(f"  cp {env_example} {env_file}")
        print(f"\nThen edit {env_file} with your settings.")
        return False

def check_documents():
    """Check if documents are available"""
    print_header("Step 4: Checking Documents")
    
    # Check multiple possible locations
    base_dir = os.path.dirname(__file__)
    possible_dirs = [
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data')
    ]
    
    files = []
    used_dir = None
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            found_files = [f for f in os.listdir(data_dir) 
                          if f.endswith(('.pdf', '.txt')) and not f.startswith('~')]
            if found_files:
                files = found_files
                used_dir = data_dir
                break
    
    if files:
        print(f"✅ Found {len(files)} document(s) in:")
        print(f"   {used_dir}")
        for f in files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        return True
    else:
        # Create raw directory if it doesn't exist
        raw_dir = os.path.join(base_dir, 'data', 'raw')
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir, exist_ok=True)
        
        print(f"⚠️  No documents found!")
        print(f"\nChecked locations:")
        for d in possible_dirs:
            print(f"  - {d}")
        print(f"\nAdd PDF or TXT files to: {raw_dir}")
        return False

def build_index():
    """Build the Qdrant index"""
    print_header("Step 5: Building Index")
    
    print("This will:")
    print("  1. Process all documents in data/raw/")
    print("  2. Create embeddings")
    print("  3. Build Qdrant collection")
    print("  4. Create BM25 index for hybrid search")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        try:
            # Add src to path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
            
            from services.indexing_service import IndexingService
            
            indexing = IndexingService()
            success = indexing.build_index(verbose=True)
            
            if success:
                print("\n✅ Index built successfully!")
                return True
            else:
                print("\n❌ Failed to build index")
                return False
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Skipped.")
        return False

def test_search():
    """Test the search functionality"""
    print_header("Step 6: Testing Search")
    
    query = input("Enter a test query (or press Enter to skip): ").strip()
    
    if not query:
        print("Skipped.")
        return
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from services.retriever_service import RetrieverService
        
        retriever_service = RetrieverService()
        
        print(f"\n🔍 Searching for: '{query}'")
        print("Using hybrid search (semantic + keyword)...\n")
        
        retriever = retriever_service.get_retriever(k=3, search_type="hybrid")
        results = retriever.invoke(query)
        
        print(f"✅ Found {len(results)} result(s):\n")
        
        for i, doc in enumerate(results, 1):
            print(f"{i}. {'-'*55}")
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"   {content}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main quick start flow"""
    print("\n" + "="*60)
    print("  🚀 Qdrant-based Agentic RAG - Quick Start")
    print("="*60)
    
    # Run checks
    qdrant_ok = check_qdrant()
    deps_ok = check_dependencies()
    config_ok = check_config()
    docs_ok = check_documents()
    
    # Summary
    print_header("Setup Summary")
    
    checks = {
        "Qdrant Running": qdrant_ok,
        "Dependencies Installed": deps_ok,
        "Configuration Ready": config_ok,
        "Documents Available": docs_ok
    }
    
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check}")
    
    # Determine next steps
    if not all([qdrant_ok, deps_ok, config_ok]):
        print("\n⚠️  Please fix the issues above before proceeding.")
        print("\nFor help, see:")
        print("  - QDRANT_SETUP.md (detailed setup guide)")
        print("  - MIGRATION_GUIDE.md (migration from FAISS)")
        return 1
    
    if not docs_ok:
        print("\n⚠️  No documents found, but you can still build an empty index.")
    
    # Offer to build index
    if qdrant_ok and deps_ok and config_ok:
        build_index()
    
    # Offer to test search
    if docs_ok:
        test_search()
    
    # Final message
    print_header("Next Steps")
    print("You're ready to use the system! Here are some things to try:")
    print("\n1. Manage Qdrant:")
    print("   python manage_qdrant.py info       # Show collection info")
    print("   python manage_qdrant.py health     # Check system health")
    print("   python manage_qdrant.py rebuild    # Rebuild index")
    print("\n2. Try examples:")
    print("   python example_qdrant_usage.py     # Interactive examples")
    print("\n3. Use in your code:")
    print("   from services.retriever_service import RetrieverService")
    print("   retriever = RetrieverService().get_retriever()")
    print("\n4. Read documentation:")
    print("   - QDRANT_SETUP.md    (setup guide)")
    print("   - MIGRATION_GUIDE.md (migration help)")
    print("   - UPGRADE_SUMMARY.md (all changes)")
    print("\n" + "="*60)
    print("  Happy searching! 🔍")
    print("="*60 + "\n")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
