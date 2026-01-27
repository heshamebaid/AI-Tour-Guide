"""
Reranker Service - Re-ranks retrieved documents using CrossEncoder.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for high-quality reranking.
"""
import os
from typing import List, Tuple
from langchain_core.documents import Document

from core.config import Config

# Disable CUDA to avoid meta tensor issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class RerankerService:
    """
    Service for reranking retrieved documents using CrossEncoder.
    Improves retrieval quality by scoring query-document pairs.
    """
    
    def __init__(self, model_name: str = None, max_length: int = 512):
        """
        Initialize the reranker with CrossEncoder model.
        
        Args:
            model_name: HuggingFace model name for CrossEncoder (defaults to config)
            max_length: Maximum sequence length for the model
        """
        self.model_name = model_name or Config.RERANKER_MODEL
        self.max_length = max_length
        self._reranker = None
        print(f"Reranker service ready (model: {self.model_name})")
    
    def _load_model(self):
        """Lazy load the CrossEncoder model."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.model_name, max_length=self.max_length)
            print(f"✓ CrossEncoder model loaded: {self.model_name}")
        return self._reranker
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None = return all, reranked)
            
        Returns:
            Reranked list of documents (most relevant first)
        """
        if not documents:
            return []
        
        if len(documents) == 1:
            return documents
        
        # Load model
        reranker = self._load_model()
        
        # Create query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked documents
        reranked_docs = [doc for doc, score in scored_docs]
        
        # Return top_k if specified
        if top_k is not None and top_k < len(reranked_docs):
            return reranked_docs[:top_k]
        
        return reranked_docs
    
    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        if len(documents) == 1:
            return [(documents[0], 1.0)]
        
        # Load model
        reranker = self._load_model()
        
        # Create query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None and top_k < len(scored_docs):
            return scored_docs[:top_k]
        
        return scored_docs
