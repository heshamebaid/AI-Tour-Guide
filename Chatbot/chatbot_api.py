import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from project root .env only
try:
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

# Add the Agentic_RAG src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Agentic_RAG', 'src'))

# Import from RAG system
from pipeline.model import rag_query, load_documents_from_data_dir, get_document_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PharaohGuide Chatbot API",
    description="API for PharaohGuide RAG Chatbot using Agentic_RAG folder system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global for RAG system status
rag_initialized = False

class ChatRequest(BaseModel):
    query: str
    k: int = 5

@app.on_event("startup")
async def startup_event():
    global rag_initialized
    try:
        logger.info("Initializing RAG system from Agentic_RAG folder...")
        
        # Load documents from RAG data directory
        load_result = load_documents_from_data_dir()
        
        if load_result["success"]:
            logger.info(f"Successfully loaded {load_result['files_processed']} files with {load_result['total_chunks']} chunks")
            rag_initialized = True
        else:
            logger.error(f"Failed to load documents: {load_result.get('error', 'Unknown error')}")
            rag_initialized = False
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        rag_initialized = False

@app.get("/")
async def root():
    return {
        "message": "PharaohGuide Chatbot API",
        "status": "running",
        "endpoints": {
            "/chat": "Chat with the PharaohGuide RAG chatbot",
            "/health": "Check API health",
            "/config": "Get current configuration"
        }
    }

@app.get("/health")
async def health_check():
    if not rag_initialized:
        return {"status": "error", "message": "RAG system not initialized"}
    return {"status": "healthy", "message": "RAG system ready"}

@app.get("/config")
async def get_config():
    stats = get_document_stats()
    return {
        "rag_system": "Agentic_RAG folder system",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "x-ai/grok-4-fast:free (via OpenRouter)",
        "total_documents": stats.get("total_documents", 0),
        "total_chunks": stats.get("total_chunks", 0),
        "files": stats.get("files", []),
        "has_index": stats.get("has_index", False)
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_initialized:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    query = request.query
    try:
        # Use the RAG system's rag_query function
        answer = rag_query(query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=8080, reload=True, log_level="info") 