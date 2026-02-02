#!/usr/bin/env python
"""
Rebuild Qdrant index with semantic chunking.
Run from Agentic_RAG folder: python rebuild_index.py
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.indexing_service import IndexingService

def main():
    print("\n" + "=" * 60)
    print("REBUILDING INDEX WITH SEMANTIC CHUNKING")
    print("=" * 60 + "\n")
    
    svc = IndexingService()
    success = svc.rebuild_index()
    
    if success:
        print("\n✅ Index rebuilt successfully with semantic chunking!")
    else:
        print("\n❌ Index rebuild failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
