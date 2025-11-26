import os
import faiss
from langchain_community.vectorstores import FAISS
from services.embeddings_service import EmbeddingService
from core.config import Config

class VectorStoreService:

    def __init__(self):
        self.emb = EmbeddingService()
        self.index_path = Config.VECTORSTORE_PATH

    def build_vectorstore(self, documents):
        vect = FAISS.from_documents(documents, self.emb.model)
        vect.save_local(self.index_path)
        return vect

    def load_vectorstore(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Vectorstore not found at {self.index_path}. "
                f"Please run 'python build_vectorstore.py' first."
            )
        
        # For older langchain_community versions, don't use allow_dangerous_deserialization
        try:
            return FAISS.load_local(
                self.index_path,
                self.emb.model,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            # Fallback for older versions
            return FAISS.load_local(
                self.index_path,
                self.emb.model
            )
