"""
Indexing Service - Build and manage vector store indexes.
"""
import os
from typing import List, Optional
from langchain.schema import Document

from services.preprocessing import DocumentPreprocessor
from services.vectorstore_service import VectorStoreService


class IndexingService:
    """
    Service for building and managing FAISS vector store indexes.
    Handles document preprocessing and index creation.
    """
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.vectorstore_service = VectorStoreService()
    
    def build_index(self, verbose: bool = True) -> bool:
        """
        Build the FAISS vector store index from documents.
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if verbose:
                print("=" * 60)
                print("Building FAISS Vector Store Index")
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
            
            # Step 2: Build vector store
            if verbose:
                print("\n[2/2] Building FAISS vector store...")
            
            vectorstore = self.vectorstore_service.build_vectorstore(chunks)
            
            if verbose:
                print(f"✓ Vector store saved to: {self.vectorstore_service.index_path}")
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
        Rebuild the vector store index (deletes existing index first).
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove existing index if it exists
            index_path = self.vectorstore_service.index_path
            if os.path.exists(index_path):
                if verbose:
                    print(f"Removing existing index at: {index_path}")
                import shutil
                shutil.rmtree(index_path)
            
            return self.build_index(verbose=verbose)
            
        except Exception as e:
            if verbose:
                print(f"\n❌ Error rebuilding index: {e}")
            return False
    
    def get_index_info(self) -> dict:
        """
        Get information about the current index.
        
        Returns:
            dict: Index information including path, exists, and file sizes
        """
        index_path = self.vectorstore_service.index_path
        
        info = {
            'path': index_path,
            'exists': os.path.exists(index_path),
            'files': []
        }
        
        if info['exists']:
            index_file = os.path.join(index_path, 'index.faiss')
            pkl_file = os.path.join(index_path, 'index.pkl')
            
            if os.path.exists(index_file):
                info['files'].append({
                    'name': 'index.faiss',
                    'size': os.path.getsize(index_file)
                })
            
            if os.path.exists(pkl_file):
                info['files'].append({
                    'name': 'index.pkl',
                    'size': os.path.getsize(pkl_file)
                })
        
        return info
