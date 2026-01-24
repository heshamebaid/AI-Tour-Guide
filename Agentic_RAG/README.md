# 🤖 Agentic RAG Tour Guide

An intelligent chatbot powered by **Retrieval-Augmented Generation (RAG)** using LangChain, Qdrant, and AI agents with **hybrid search capabilities**. The system combines document search, web search, and image search capabilities to provide comprehensive answers with visual context.

## ✨ Features

- 📄 **Hybrid Document Search** - Query documents using combined semantic (dense) and keyword (sparse/BM25) search
- 🔍 **Advanced Retrieval** - Multiple search strategies: hybrid, dense, sparse, and MMR
- 🌐 **Web Search** - Access current information from the internet via DuckDuckGo
- 🖼️ **Image Search** - Find and display relevant images using SerpAPI
- 🤖 **Smart Agent** - AI agent that automatically selects the best tools for your query
- 💬 **Interactive Web UI** - Beautiful Django-based chat interface
- 📊 **Admin Dashboard** - Django admin panel for chat history management
- ⚡ **Qdrant Vector DB** - High-performance, scalable vector database
- 🎯 **LangChain Integration** - Full LangChain framework integration for flexibility

## 🏗️ Architecture

```
Agentic_RAG/
├── src/
│   ├── agents/              # LangChain agents and tools
│   │   ├── agent_graph.py   # Main agent orchestration
│   │   └── tools/           # Document, web, and image search tools
│   ├── services/            # Core services
│   │   ├── llm_service.py           # LLM integration (OpenRouter)
│   │   ├── embeddings_service.py    # HuggingFace embeddings
│   │   ├── vectorstore_service.py   # Qdrant vector store
│   │   ├── sparse_encoder_service.py # BM25 sparse encoding
│   │   ├── retriever_service.py     # Hybrid retrieval
│   │   ├── memory_service.py        # Conversation memory
│   │   ├── preprocessing.py         # Document processing
│   │   └── indexing_service.py      # Index building
│   └── core/
│       └── config.py        # Configuration management
├── data/
│   ├── raw/                # Upload your PDFs here
│   └── processed/          # Processed data
├── vectorstore/            # Qdrant data and BM25 model
├── example_qdrant_usage.py # Example scripts
└── QDRANT_SETUP.md        # Detailed setup guide

```

## 🚀 Quick Start

### 1. Start Qdrant Database

**Using Docker (Recommended):**
```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Installation

```bash
# Navigate to Agentic_RAG directory
cd Agentic_RAG

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the Agentic_RAG directory:

```env
# LLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct

# SerpAPI (for image search)
SERPAPI_API_KEY=your_serpapi_key

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=tour_guide_documents

# Hybrid Search Settings
ENABLE_HYBRID_SEARCH=true
DENSE_WEIGHT=0.5
SPARSE_WEIGHT=0.5
```

### 4. Prepare Your Documents

```bash
# Add PDF or TXT files to data/raw folder
cp your_documents.pdf data/raw/

# Build the Qdrant vector store index
python example_qdrant_usage.py
# Select option 1 to build index
```

Or programmatically:
```python
from services.indexing_service import IndexingService
indexing_service = IndexingService()
indexing_service.build_index()
```

## 📖 Usage

### Web Interface

1. Navigate to `http://localhost:8000`
2. Select a mode:
   - **🤖 Smart Agent** - AI decides which tools to use
   - **📄 Documents** - Search only your PDFs
   - **🌐 Web Search** - Search the internet
   - **🖼️ Images** - Find and display images
3. Start chatting!

### Rebuild Index

Rebuild the vector store from the Django admin panel or using Django shell:

```bash
python manage.py shell
```

```python
from services.indexing_service import IndexingService
indexing_service = IndexingService()

# Rebuild from scratch
indexing_service.rebuild_index()

# Or just build (if not exists)
indexing_service.build_index()

# Check index info
info = indexing_service.get_index_info()
print(info)
```

## 🛠️ Technology Stack

### Core Technologies
- **LangChain** - Agent framework and RAG implementation
- **FAISS** - Vector similarity search
- **HuggingFace** - Sentence transformers for embeddings
- **OpenRouter** - LLM API integration

### Web Technologies
- **Django** - Web framework
- **Django REST Framework** - API endpoints
- **HTML/CSS/JavaScript** - Frontend

### Search & Retrieval
- **PyPDF** - PDF document processing
- **BeautifulSoup4** - Web scraping
- **SerpAPI** - Image search

## 🔧 Advanced Configuration

### Vector Store Management

Use the `IndexingService` for programmatic index management:

```python
from services.indexing_service import IndexingService

indexing_service = IndexingService()

# Build index
indexing_service.build_index(verbose=True)

# Rebuild (removes existing index first)
indexing_service.rebuild_index(verbose=True)

# Get index information
info = indexing_service.get_index_info()
print(f"Index exists: {info['exists']}")
print(f"Index path: {info['path']}")
```

### Custom LLM Models

Edit `.env` to use different models:
```env
OPENROUTER_MODEL=openai/gpt-4
# or
OPENROUTER_MODEL=anthropic/claude-2
```

### Chunk Size Configuration

Edit `src/services/preprocessing.py`:
```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,      # Adjust chunk size
    chunk_overlap=100    # Adjust overlap
)
```

## 📊 Project Structure Details

### Services Layer (`src/services/`)
- **Modular design** - Each service handles a specific responsibility
- **Reusable components** - Can be imported by web app
- **Clean interfaces** - Well-defined APIs between components

### Agents Layer (`src/agents/`)
- **Tool-based architecture** - Each tool is a standalone function
- **Smart orchestration** - Agent decides which tools to use
- **Extensible** - Easy to add new tools

### Web Application (`webapp/`)
- **RESTful API** - Clean separation of frontend/backend
- **Modern UI** - Responsive design with smooth animations
- **Real-time chat** - Instant response display

## 🧪 Testing

Test the system through the web interface at `http://localhost:8000` or use Django shell:

```bash
python manage.py shell
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the terms included in the LICENSE file.

## 🔗 Links

- **Repository**: https://github.com/EyadAmgad/agentic_rag_tourguide
- **OpenRouter**: https://openrouter.ai/
- **SerpAPI**: https://serpapi.com/
