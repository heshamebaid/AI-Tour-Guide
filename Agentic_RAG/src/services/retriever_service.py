from services.vectorstore_service import VectorStoreService

class RetrieverService:

    def __init__(self):
        self.vs = VectorStoreService()

    def get_retriever(self, k=4):
        """
        Get a retriever instance from the vectorstore.
        
        Args:
            k: Number of documents to retrieve (default: 4)
            
        Returns:
            A retriever instance
        """
        vectorstore = self.vs.load_vectorstore()
        return vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
