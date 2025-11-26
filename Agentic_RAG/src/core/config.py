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
    
    # Use absolute paths
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH") or str(PROJECT_ROOT / "data" / "vectorstore" / "faiss_index")
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH") or str(PROJECT_ROOT / "data" / "raw")
    PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH") or str(PROJECT_ROOT / "data" / "processed")
