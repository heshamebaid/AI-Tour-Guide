"""
Sparse Encoder Service - BM25-based sparse encoding for hybrid search.
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import pickle
import os
from core.config import Config


class SparseEncoderService:
    """
    Service for BM25-based sparse encoding to support hybrid search.
    Combines with dense embeddings for better retrieval accuracy.
    """
    
    def __init__(self):
        self.bm25 = None
        self.corpus_texts = []
        self.model_path = os.path.join(Config.VECTORSTORE_PATH, "bm25_model.pkl")
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the BM25 model on a corpus of texts.
        
        Args:
            texts: List of document texts to index
        """
        self.corpus_texts = texts
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def encode_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Encode queries to sparse vectors for search.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of sparse vector representations
        """
        sparse_vectors = []
        for query in queries:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            
            # Convert to sparse format (only non-zero indices)
            sparse_dict = {
                "indices": [i for i, score in enumerate(scores) if score > 0],
                "values": [score for score in scores if score > 0]
            }
            sparse_vectors.append(sparse_dict)
        
        return sparse_vectors
    
    def encode_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Encode documents to sparse vectors.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of sparse vector representations
        """
        sparse_vectors = []
        tokenized_corpus = [doc.lower().split() for doc in documents]
        
        for i, tokens in enumerate(tokenized_corpus):
            # For documents, we create a simple bag-of-words representation
            # This is a simplified sparse vector based on term frequency
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Create sparse vector (indices are token positions in vocabulary)
            sparse_dict = {
                "indices": list(range(len(term_freq))),
                "values": list(term_freq.values())
            }
            sparse_vectors.append(sparse_dict)
        
        return sparse_vectors
    
    def save(self) -> None:
        """Save the BM25 model to disk."""
        os.makedirs(Config.VECTORSTORE_PATH, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'corpus_texts': self.corpus_texts
            }, f)
    
    def load(self) -> bool:
        """
        Load the BM25 model from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.corpus_texts = data['corpus_texts']
            return True
        except Exception as e:
            print(f"Error loading BM25 model: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Search using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 model not initialized. Call fit() or load() first.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [(idx, scores[idx]) for idx in top_indices]
