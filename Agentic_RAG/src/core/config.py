import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "tour_guide_documents")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Optional for cloud deployments
    
    # Hybrid Search Settings
    ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))  # Weight for semantic search
    SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.5"))  # Weight for keyword search
    
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
