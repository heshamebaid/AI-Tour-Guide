from langchain_community.embeddings import HuggingFaceEmbeddings
from core.config import Config

class EmbeddingService:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

    def embed(self, text: str):
        return self.model.embed_query(text)
