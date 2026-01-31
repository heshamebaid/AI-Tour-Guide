# Disable TF/Flax before any transformers/sentence_transformers import
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_IMPORT_ERROR = None
except Exception as exc:  # pylint: disable=broad-except
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMER_IMPORT_ERROR = exc
try:
    import faiss
    FAISS_IMPORT_ERROR = None
except ImportError as exc:
    faiss = None  # type: ignore
    FAISS_IMPORT_ERROR = exc
import numpy as np
import requests
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
try:
    import fitz  # PyMuPDF for PDF processing
    FITZ_IMPORT_ERROR = None
except ImportError as exc:
    fitz = None  # type: ignore
    FITZ_IMPORT_ERROR = exc
import logging

# Load .env: project root first (Docker mounts .env at /app/.env), then Agentic_RAG/.env
_agentic_root = Path(__file__).resolve().parents[2]  # Agentic_RAG/src -> Agentic_RAG
_project_root_env = _agentic_root.parent / ".env"
_agentic_env = _agentic_root / ".env"
if _project_root_env.exists():
    load_dotenv(dotenv_path=_project_root_env)
if _agentic_env.exists():
    load_dotenv(dotenv_path=_agentic_env)  # can override with local Agentic_RAG/.env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store documents and embeddings
uploaded_documents = []
document_metadata = []  # Store metadata about each document chunk
model_embeddings = None

# Initialize empty FAISS index (will be populated when documents are loaded)
index = None

def load_documents_from_data_dir(data_dir: str = None) -> Dict[str, Any]:
    """
    Load and process all documents from the data directory
    
    Args:
        data_dir: Path to data directory (defaults to controllers/data)
        
    Returns:
        Dict with loading results
    """
    global uploaded_documents, document_metadata, index
    
    if data_dir is None:
        # Default to controllers/data directory
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "controllers" / "data"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {"success": False, "error": f"Data directory not found: {data_dir}"}
    
    logger.info(f"Loading documents from: {data_dir}")
    
    # Find all supported files
    supported_extensions = ['.pdf', '.txt', '.md']
    files_to_process = []
    for ext in supported_extensions:
        files_to_process.extend(data_dir.rglob(f"*{ext}"))
    
    if not files_to_process:
        logger.warning(f"No supported files found in {data_dir}")
        return {"success": False, "error": "No supported files found"}
    
    # Process each file
    all_chunks = []
    processed_files = 0
    
    for file_path in files_to_process:
        try:
            logger.info(f"Processing: {file_path.name}")
            chunks = _extract_text_and_chunk(file_path)
            all_chunks.extend(chunks)
            processed_files += 1
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
    
    if not all_chunks:
        return {"success": False, "error": "No text chunks created from documents"}
    
    # Update global storage
    uploaded_documents = [chunk['content'] for chunk in all_chunks]
    document_metadata = all_chunks
    
    # Create FAISS index
    _build_faiss_index()
    
    logger.info(f"Successfully loaded {processed_files} files with {len(all_chunks)} total chunks")
    
    return {
        "success": True,
        "files_processed": processed_files,
        "total_chunks": len(all_chunks),
        "files": [f.name for f in files_to_process[:processed_files]]
    }

def _extract_text_and_chunk(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a file and split into chunks
    
    Returns:
        List of chunk dictionaries with content and metadata
    """
    # Extract text based on file type
    text_content = _extract_text_from_file(file_path)
    
    if not text_content.strip():
        raise ValueError("No text content extracted")
    
    # Simple chunking (split by paragraphs or max length)
    chunks = _chunk_text(text_content, chunk_size=1000, overlap=200)
    
    # Create chunk objects with metadata
    chunk_objects = []
    for i, chunk_text in enumerate(chunks):
        if chunk_text.strip():
            chunk_objects.append({
                'content': chunk_text.strip(),
                'file_name': file_path.name,
                'file_path': str(file_path),
                'chunk_index': i,
                'chunk_id': f"{file_path.stem}_chunk_{i}"
            })
    
    return chunk_objects

def _extract_text_from_file(file_path: Path) -> str:
    """Extract text from different file types"""
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return _extract_pdf_text(file_path)
    elif extension in ['.txt', '.md']:
        return _extract_txt_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF using PyMuPDF"""
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed. Install pymupdf to process PDF files."
        ) from FITZ_IMPORT_ERROR
    try:
        text_content = ""
        with fitz.open(str(file_path)) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text
        return text_content
    except Exception as e:
        # Fallback to basic PDF reading if PyMuPDF fails
        logger.warning(f"PyMuPDF failed for {file_path}, using fallback")
        return f"Content from {file_path.name} (PDF processing failed: {e})"

def _extract_txt_text(file_path: Path) -> str:
    """Extract text from TXT/MD files"""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not decode {file_path} with any encoding")

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple text chunking"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence or paragraph boundaries
        chunk = text[start:end]
        for i in range(len(chunk) - 1, max(0, len(chunk) - 100), -1):
            if chunk[i] in '.!?\n':
                chunk = chunk[:i + 1]
                end = start + i + 1
                break
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c.strip()]

def _get_embeddings_model():
    """Lazy-load SentenceTransformer to avoid import-time failures during tests."""

    global model_embeddings  # noqa: PLW0603
    if model_embeddings is None:
        if SentenceTransformer is None:
            err = SENTENCE_TRANSFORMER_IMPORT_ERROR
            raise RuntimeError(
                "SentenceTransformer not available. Install: pip install sentence-transformers faiss-cpu. "
                f"Original error: {err}"
            ) from err
        model_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    return model_embeddings


def _get_faiss():
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. Install faiss-cpu or faiss-gpu to enable retrieval."
        ) from FAISS_IMPORT_ERROR
    return faiss


def _build_faiss_index():
    """Build FAISS index from uploaded documents"""
    global index
    
    if not uploaded_documents:
        logger.warning("No documents to index")
        return
    
    # Create embeddings
    logger.info("Creating embeddings for documents...")
    all_embeddings = _get_embeddings_model().encode(uploaded_documents, convert_to_tensor=True)
    
    # Create FAISS index
    index = _get_faiss().IndexFlatL2(all_embeddings.shape[1])
    index.add(np.array(all_embeddings))
    
    logger.info(f"FAISS index built with {len(uploaded_documents)} documents")

def update_documents_with_chunks(chunks):
    """
    Update the document store and FAISS index with new chunks from uploaded files
    Args:
        chunks: List of Document objects from langchain text splitter
    """
    global uploaded_documents, index
    
    # Extract text content from chunks
    new_documents = [chunk.page_content for chunk in chunks]
    
    # Add to our uploaded documents store
    uploaded_documents.extend(new_documents)
    
    # Create/recreate FAISS index with all uploaded documents
    if uploaded_documents:
        all_embeddings = _get_embeddings_model().encode(uploaded_documents, convert_to_tensor=True)
        
        # Create new FAISS index
        index = _get_faiss().IndexFlatL2(all_embeddings.shape[1])
        index.add(np.array(all_embeddings))
    
    return len(new_documents)

def get_current_documents():
    """Get all current uploaded documents"""
    return uploaded_documents

def clear_documents():
    """Clear all uploaded documents (useful for testing or reset)"""
    global uploaded_documents, document_metadata, index
    uploaded_documents = []
    document_metadata = []
    index = None

def get_document_stats() -> Dict[str, Any]:
    """Get statistics about loaded documents"""
    if not document_metadata:
        return {"total_documents": 0, "total_chunks": 0, "files": []}
    
    # Count unique files
    unique_files = list(set(chunk['file_name'] for chunk in document_metadata))
    
    # Count chunks per file
    file_chunks = {}
    for chunk in document_metadata:
        filename = chunk['file_name']
        file_chunks[filename] = file_chunks.get(filename, 0) + 1
    
    return {
        "total_documents": len(unique_files),
        "total_chunks": len(document_metadata),
        "files": unique_files,
        "chunks_per_file": file_chunks,
        "has_index": index is not None
    }

# OpenRouter Setup
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
_raw_model = os.getenv("OPEN_ROUTER_MODEL") or os.getenv("LLM_MODEL") or "liquid/lfm-2.5-1.2b-thinking:free"
FALLBACK_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"
# When primary free model hits daily limit, try another free model once
RATE_LIMIT_FALLBACK_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
# Qwen free model has no endpoints on OpenRouter; force Liquid
if _raw_model and "qwen" in _raw_model.lower():
    OPENROUTER_MODEL = FALLBACK_MODEL
    logger.info("Using %s (qwen free model has no endpoints)", OPENROUTER_MODEL)
else:
    OPENROUTER_MODEL = _raw_model or FALLBACK_MODEL
url = "https://openrouter.ai/api/v1/chat/completions"
if not OPENROUTER_API_KEY:
    logger.warning("OPEN_ROUTER_API_KEY not found in environment variables. RAG queries will return retrieved context only.")


# Chat History 
chat_history = []  # Will store conversation turns

def rag_query(user_query, top_k: int = 2):
    # Auto-load documents if none are present (may fail if sentence_transformers etc. missing)
    if not uploaded_documents or index is None:
        logger.info("No documents loaded, attempting to load from data directory...")
        try:
            load_result = load_documents_from_data_dir()
            if not load_result.get("success"):
                logger.info("No RAG documents available; will answer from LLM general knowledge.")
            else:
                logger.info(f"Auto-loaded {load_result['files_processed']} files with {load_result['total_chunks']} chunks")
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("RAG load failed (%s); will answer from LLM general knowledge.", e)

    # 1. Retrieve relevant chunks if we have an index; otherwise use LLM-only (no context)
    if uploaded_documents and index is not None:
        try:
            top_k = max(1, min(int(top_k), 5))
        except Exception:
            top_k = 2
        query_embedding = _get_embeddings_model().encode([user_query])
        D, I = index.search(query_embedding, k=top_k)
        retrieved_chunks = [uploaded_documents[i] for i in I[0]]
        retrieved_context = "\n".join(retrieved_chunks)
        max_context_chars = 12000
        if len(retrieved_context) > max_context_chars:
            retrieved_context = retrieved_context[:max_context_chars] + "\n[Context truncated]"
        has_context = bool(retrieved_context.strip())
    else:
        retrieved_context = ""
        has_context = False

    # 2. Build messages for OpenRouter (use context when available, else LLM general knowledge)
    if has_context:
        system_content = "You are a helpful assistant. Answer questions based on the provided context from uploaded documents when possible; you may add general knowledge if the context is incomplete."
    else:
        system_content = "No document context was provided. Answer from your general knowledge. Be helpful and accurate."
    messages = [{"role": "system", "content": system_content}]
    # Keep only the last 2 user+assistant turns
    if len(chat_history) > 4:
        history = chat_history[-4:]
    else:
        history = chat_history
    messages.extend(history)
    # Truncate user query to avoid oversized prompts
    safe_user_query = user_query[:2000]
    messages.append({
        "role": "user",
        "content": f"Context from uploaded documents:\n{retrieved_context}\n\nQuestion: {safe_user_query}"
    })

    # 3. If no API key, return retrieved context and guidance instead of calling API
    if not OPENROUTER_API_KEY:
        answer = (
            "OpenRouter API key is not configured. Showing relevant context only.\n\n"
            + retrieved_context[:800]
            + ("\n\n[Context truncated]" if len(retrieved_context) > 800 else "")
            + "\n\nSet environment variable OPEN_ROUTER_API_KEY to enable LLM responses."
        )
        # Save this turn into history and return
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": answer})
        return answer

    # 3. Send request to OpenRouter (retry with fallback model if "No endpoints found")
    model_to_use = OPENROUTER_MODEL
    # Never send qwen (no free endpoints on OpenRouter); resolve at request time
    if model_to_use and "qwen" in (model_to_use or "").lower():
        model_to_use = FALLBACK_MODEL
        logger.debug("Using fallback model (qwen has no endpoints): %s", model_to_use)
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model_to_use,
            "messages": messages,
        }),
        timeout=60,
    )

    # If "No endpoints found" for this model, retry once with Liquid free model
    err_msg = ""
    if response.text:
        try:
            err_body = response.json()
            err_msg = (err_body.get("error") or {}).get("message", response.text or "") or ""
        except Exception:  # pylint: disable=broad-except
            err_msg = response.text or ""
    if not response.ok and model_to_use != FALLBACK_MODEL:
        if "no endpoints found" in (err_msg or "").lower() or "no endpoints found" in (response.text or "").lower():
            logger.warning("Model %s unavailable (%s), retrying with %s", model_to_use, err_msg[:80], FALLBACK_MODEL)
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": FALLBACK_MODEL,
                    "messages": messages,
                }),
                timeout=60,
            )
            model_to_use = FALLBACK_MODEL

    # If primary/fallback hit daily free limit, retry once with another free model
    is_rate_limited = False
    if not response.ok and response.text:
        try:
            err_body = response.json()
            err_msg = (err_body.get("error") or {}).get("message", response.text or "") or ""
            is_rate_limited = (
                response.status_code == 429 or "free-models-per-day" in (err_msg or "").lower()
            )
        except Exception:  # pylint: disable=broad-except
            is_rate_limited = response.status_code == 429
    if is_rate_limited and model_to_use != RATE_LIMIT_FALLBACK_MODEL:
        logger.warning(
            "Daily free limit reached for %s (see OpenRouter credits). Retrying with %s.",
            model_to_use,
            RATE_LIMIT_FALLBACK_MODEL,
        )
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": RATE_LIMIT_FALLBACK_MODEL,
                "messages": messages,
            }),
            timeout=60,
        )
        model_to_use = RATE_LIMIT_FALLBACK_MODEL

    # 4. Parse response
    try:
        if response.ok:
            answer_json = response.json()
            answer = answer_json["choices"][0]["message"]["content"]
        else:
            err_body = response.json() if response.text else {}
            err_msg = (err_body.get("error") or {}).get("message", response.text or f"HTTP {response.status_code}") or ""
            # User-friendly message for rate limit / free tier
            if response.status_code == 429 or "free-models-per-day" in (err_msg or "").lower():
                logger.info(
                    "OpenRouter rate limit: model=%s status=%s message=%s",
                    model_to_use,
                    response.status_code,
                    (err_msg or response.text or "")[:200],
                )
                answer = (
                    "The daily free limit for this model has been reached. "
                    "You can add credits at https://openrouter.ai/credits to unlock more requests, or try again tomorrow."
                )
            elif "no endpoints found" in (err_msg or "").lower():
                answer = (
                    "The configured AI model is not available right now. "
                    "Set OPEN_ROUTER_MODEL=liquid/lfm-2.5-1.2b-thinking:free in your .env and restart the service."
                )
            else:
                answer = f"Error: {(err_msg or str(response.status_code))[:300]}"
    except Exception as e:
        answer = f"Error: {str(e)[:300]}"

    # 5. Save this turn into history
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer



# ==== Step 4: Interactive Loop ====
def start_terminal_chat():
    """
    Start an interactive terminal chat session.
    This is now optional and won't run automatically.
    """
    print("🏺 Pharaoh Tour Guide - RAG Chatbot")
    print("=" * 40)
    
    # Try to load documents first
    print("📚 Loading documents from data directory...")
    load_result = load_documents_from_data_dir()
    
    if load_result["success"]:
        print(f"✅ Loaded {load_result['files_processed']} files:")
        for filename in load_result['files']:
            print(f"   • {filename}")
        print(f"✅ Total chunks: {load_result['total_chunks']}")
    else:
        print(f"⚠️  Could not load documents: {load_result.get('error', 'Unknown error')}")
        print("You can still chat, but responses will be limited.")
    
    print("\n💬 Chat started (type 'exit' to quit)")
    print("Ask me anything about Pharaoh history!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Ending chat. Goodbye!")
                break
                
            answer = rag_query(user_input)
            print(f"🤖 Bot: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Chat ended by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending chat.")
            break
        try:
            answer = rag_query(user_input)
            print(f"Bot: {answer}\n")
        except Exception as e:
            print("Error:", e)


# Only run interactive mode if this file is executed directly (not imported)
if __name__ == "__main__":
    start_terminal_chat()