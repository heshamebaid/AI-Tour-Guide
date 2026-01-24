"""
Retriever Service - Hybrid search retriever using Qdrant and BM25.
"""
from typing import List
from langchain.schema import Document, BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from services.vectorstore_service import VectorStoreService
from services.sparse_encoder_service import SparseEncoderService
from core.config import Config


class BM25Retriever(BaseRetriever):
    """
    Custom BM25-based retriever for sparse search.
    """
    
    sparse_encoder: SparseEncoderService
    documents: List[Document]
    k: int = 4
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            run_manager: Callback manager (optional)
            
        Returns:
            List of relevant documents
        """
        results = self.sparse_encoder.search(query, top_k=self.k)
        return [self.documents[idx] for idx, score in results if idx < len(self.documents)]


class RetrieverService:
    """
    Service for creating retrievers with hybrid search support.
    Combines dense (semantic) and sparse (BM25) search for better results.
    """

    def __init__(self):
        self.vs_service = VectorStoreService()
        self.sparse_encoder = SparseEncoderService()
        self.documents = []

    def get_retriever(self, k: int = 4, search_type: str = "hybrid"):
        """
        Get a retriever instance from the vectorstore.
        
        Args:
            k: Number of documents to retrieve (default: 4)
            search_type: Type of search - "hybrid", "dense", or "sparse" (default: "hybrid")
            
        Returns:
            A retriever instance (EnsembleRetriever for hybrid, regular retriever otherwise)
        """
        vectorstore = self.vs_service.load_vectorstore()
        
        if search_type == "hybrid" and Config.ENABLE_HYBRID_SEARCH:
            # Load sparse encoder (BM25)
            if not self.sparse_encoder.load():
                print("⚠️ Warning: BM25 model not found. Falling back to dense search only.")
                return vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
            
            # Get documents from vectorstore for BM25
            # We need to retrieve all documents to create the BM25 retriever
            # In production, consider caching this
            all_results = vectorstore.similarity_search("", k=1000)  # Get many docs
            self.documents = all_results
            
            # Create dense retriever (semantic search)
            dense_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # Create sparse retriever (BM25)
            sparse_retriever = BM25Retriever(
                sparse_encoder=self.sparse_encoder,
                documents=self.documents,
                k=k
            )
            
            # Combine with ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[Config.DENSE_WEIGHT, Config.SPARSE_WEIGHT]
            )
            
            return ensemble_retriever
        
        elif search_type == "sparse":
            # BM25 only
            if not self.sparse_encoder.load():
                raise ValueError("BM25 model not found. Please build the index first.")
            
            all_results = vectorstore.similarity_search("", k=1000)
            self.documents = all_results
            
            return BM25Retriever(
                sparse_encoder=self.sparse_encoder,
                documents=self.documents,
                k=k
            )
        
        else:
            # Dense (semantic) search only
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

    def get_mmr_retriever(self, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5):
        """
        Get a Maximum Marginal Relevance (MMR) retriever for diverse results.
        
        Args:
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before MMR (default: 20)
            lambda_mult: Diversity factor (0=max diversity, 1=min diversity, default: 0.5)
            
        Returns:
            A retriever instance configured for MMR
        """
        vectorstore = self.vs_service.load_vectorstore()
        
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )

