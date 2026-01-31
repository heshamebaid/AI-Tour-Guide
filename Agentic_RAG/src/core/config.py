import os
from pathlib import Path
from dotenv import load_dotenv

# Single .env at repo root (4 levels up from Agentic_RAG/src/core/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Get the Agentic_RAG root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Same default as Place Details: OpenRouter model used by RAG, Pharos, Chatbot, Place Details
_DEFAULT_OPENROUTER_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"


class Config:
    OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    _raw_model = os.getenv("OPEN_ROUTER_MODEL") or os.getenv("OPENROUTER_MODEL") or _DEFAULT_OPENROUTER_MODEL
    # Qwen free model has no endpoints on OpenRouter
    OPENROUTER_MODEL = _DEFAULT_OPENROUTER_MODEL if _raw_model and "qwen" in _raw_model.lower() else _raw_model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "tour_guide_documents")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Optional for cloud deployments
    
    # Hybrid Search Settings
    ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))  # Weight for semantic search
    SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.5"))  # Weight for keyword search
    
    # Reranker Settings
    ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_INITIAL_K = int(os.getenv("RERANKER_INITIAL_K", "20"))
    
    # Use absolute paths
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH") or str(PROJECT_ROOT / "vectorstore")
    
    # Check for documents in data/raw/ first, fallback to data/ if raw is empty
    _raw_path = PROJECT_ROOT / "data" / "raw"
    _data_path = PROJECT_ROOT / "data"
    if os.getenv("DOCUMENTS_PATH"):
        DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
    elif _raw_path.exists() and any(_raw_path.iterdir()):
        DOCUMENTS_PATH = str(_raw_path)
    else:
        DOCUMENTS_PATH = str(_data_path)
    
    PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH") or str(PROJECT_ROOT / "data" / "processed")
