"""
Indexing Service - Build and manage Qdrant vector store indexes with hybrid search.
"""
import os
from typing import List, Optional
from langchain_core.documents import Document

from services.preprocessing import DocumentPreprocessor
from services.vectorstore_service import VectorStoreService


class IndexingService:
    """
    Service for building and managing Qdrant vector store indexes.
    Handles document preprocessing and index creation with hybrid search support.
    """
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.vectorstore_service = VectorStoreService()
    
    def build_index(self, verbose: bool = True) -> bool:
        """
        Build the Qdrant vector store index from documents.
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if verbose:
                print("=" * 60)
                print("Building Qdrant Vector Store Index with Hybrid Search")
                print("=" * 60)
            
            # Step 1: Load and preprocess documents
            if verbose:
                print("\n[1/2] Loading and preprocessing documents...")
            
            chunks = self.preprocessor.preprocess()
            
            if not chunks:
                if verbose:
                    print("\n❌ No documents found!")
                    print("   Please add PDF or TXT files to data/raw folder.")
                return False
            
            if verbose:
                print(f"✓ Created {len(chunks)} document chunks")
            
            # Step 2: Build Qdrant vector store
            if verbose:
                print("\n[2/2] Building Qdrant vector store...")
            
            vectorstore = self.vectorstore_service.build_vectorstore(chunks)
            
            if verbose:
                info = self.vectorstore_service.get_collection_info()
                print(f"✓ Qdrant collection created: {info['name']}")
                print(f"✓ Documents indexed: {info.get('points_count', len(chunks))}")
                print(f"✓ Hybrid search enabled: {info.get('hybrid_search_enabled', False)}")
                print("\n" + "=" * 60)
                print("✓ Index built successfully!")
                print("=" * 60 + "\n")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"\n❌ Error building index: {e}")
            return False
    
    def rebuild_index(self, verbose: bool = True) -> bool:
        """
        Rebuild the vector store index (deletes existing collection first).
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove existing collection if it exists
            info = self.vectorstore_service.get_collection_info()
            if info.get('exists'):
                if verbose:
                    print(f"Removing existing collection: {info['name']}")
                self.vectorstore_service.delete_collection()
            
            return self.build_index(verbose=verbose)
            
        except Exception as e:
            if verbose:
                print(f"\n❌ Error rebuilding index: {e}")
            return False
    
    def get_index_info(self) -> dict:
        """
        Get information about the current Qdrant collection.
        
        Returns:
            dict: Collection information including status and document count
        """
        return self.vectorstore_service.get_collection_info()
