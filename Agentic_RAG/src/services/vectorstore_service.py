"""
Vector Store Service - Qdrant-based vector storage with hybrid search support.
"""
import os
from typing import List, Optional
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document

from services.embeddings_service import EmbeddingService
from services.sparse_encoder_service import SparseEncoderService
from core.config import Config


class VectorStoreService:
    """
    Service for managing Qdrant vector store with hybrid search capabilities.
    Supports both dense (semantic) and sparse (BM25) search.
    """

    def __init__(self):
        self.emb_service = EmbeddingService()
        self.sparse_encoder = SparseEncoderService()
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        
        # Initialize Qdrant client
        self.client = self._init_client()
        self.vectorstore = None

    def _init_client(self) -> QdrantClient:
        """
        Initialize Qdrant client with configuration.
        
        Returns:
            QdrantClient instance
        """
        if Config.QDRANT_API_KEY:
            # Cloud instance
            return QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY
            )
        else:
            # Local instance
            return QdrantClient(url=Config.QDRANT_URL)

    def build_vectorstore(self, documents: List[Document]) -> QdrantVectorStore:
        """
        Build Qdrant vector store from documents with hybrid search support.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            QdrantVectorStore instance
        """
        # Extract texts from documents for sparse encoding
        texts = [doc.page_content for doc in documents]
        
        # Fit and save BM25 model for sparse search
        if Config.ENABLE_HYBRID_SEARCH:
            print("Building BM25 index for hybrid search...")
            self.sparse_encoder.fit(texts)
            self.sparse_encoder.save()
        
        # Create Qdrant vector store using LangChain
        print(f"Creating Qdrant collection: {self.collection_name}")
        vectorstore = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.emb_service.model,
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            collection_name=self.collection_name,
            force_recreate=True  # Recreate collection if it exists
        )
        
        self.vectorstore = vectorstore
        print(f"✓ Vector store created with {len(documents)} documents")
        
        return vectorstore

    def load_vectorstore(self) -> QdrantVectorStore:
        """
        Load existing Qdrant vector store.
        
        Returns:
            QdrantVectorStore instance
            
        Raises:
            ValueError: If collection doesn't exist
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if not collection_exists:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' not found. "
                f"Please run the indexing service first to build the vector store."
            )
        
        # Load BM25 model for hybrid search
        if Config.ENABLE_HYBRID_SEARCH:
            if not self.sparse_encoder.load():
                print("⚠️ Warning: BM25 model not found. Hybrid search will not be available.")
        
        # Create vectorstore instance
        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.emb_service.model
        )
        
        self.vectorstore = vectorstore
        return vectorstore

    def delete_collection(self) -> bool:
        """
        Delete the Qdrant collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
            
            # Also delete BM25 model
            bm25_path = os.path.join(Config.VECTORSTORE_PATH, "bm25_model.pkl")
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
                print("✓ Deleted BM25 model")
            
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def get_collection_info(self) -> dict:
        """
        Get information about the Qdrant collection.
        
        Returns:
            dict: Collection information
        """
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                return {
                    'exists': False,
                    'name': self.collection_name
                }
            
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'exists': True,
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'status': collection_info.status,
                'hybrid_search_enabled': Config.ENABLE_HYBRID_SEARCH
            }
        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }

