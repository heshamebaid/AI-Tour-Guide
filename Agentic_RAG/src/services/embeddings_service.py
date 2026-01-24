import os
from typing import List

# CRITICAL: Set these BEFORE importing torch or sentence_transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA completely

from langchain_core.embeddings import Embeddings
from core.config import Config


class SentenceTransformerEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings wrapper for SentenceTransformer.
    Uses lazy loading to avoid meta tensor issues during import.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load model only when needed."""
        if self._model is None:
            # Import here to delay loading
            from sentence_transformers import SentenceTransformer
            # Don't specify device - let it use default CPU
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


class EmbeddingService:
    def __init__(self):
        """
        Initialize embedding service with lazy model loading.
        """
        print(f"Preparing embedding model: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformerEmbeddings(Config.EMBEDDING_MODEL)
        print("✓ Embedding service ready (model will load on first use)")

    def embed(self, text: str):
        """Embed a single text."""
        return self.model.embed_query(text)
