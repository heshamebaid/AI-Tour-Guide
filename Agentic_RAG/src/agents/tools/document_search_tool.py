from langchain.tools import tool
from services.retriever_service import RetrieverService

# Initialize retriever once
_retriever_service = None

def get_retriever():
    """Get or initialize the retriever service."""
    global _retriever_service
    if _retriever_service is None:
        _retriever_service = RetrieverService()
    return _retriever_service.get_retriever()

@tool
def document_search(query: str) -> str:
    """
    Searches the local document knowledge base for relevant information.
    Use this FIRST to find information from uploaded PDF documents.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        Relevant information from the document database
    """
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant documents found in the knowledge base."
        
        # Format the results
        result = f"Found {len(docs)} relevant document(s):\n\n"
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:300]  # Limit to 300 chars per doc
            result += f"{i}. {content}...\n\n"
        
        return result
        
    except Exception as e:
        return f"Document search error: {str(e)}"
