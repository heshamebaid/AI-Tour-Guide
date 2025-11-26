# 🤖 Agentic RAG Tour Guide

An intelligent chatbot powered by **Retrieval-Augmented Generation (RAG)** using LangChain, FAISS, and AI agents. The system combines document search, web search, and image search capabilities to provide comprehensive answers with visual context.

## ✨ Features

- 📄 **Document Search** - Query your uploaded PDF documents using semantic search
- 🌐 **Web Search** - Access current information from the internet via DuckDuckGo
- 🖼️ **Image Search** - Find and display relevant images using SerpAPI
- 🤖 **Smart Agent** - AI agent that automatically selects the best tools for your query
- 💬 **Interactive Web UI** - Beautiful Django-based chat interface
- 📊 **Admin Dashboard** - Django admin panel for chat history management

## 🏗️ Architecture

```
agentic_rag_tourguide/
├── src/
│   ├── agents/              # LangChain agents and tools
│   │   ├── agent_graph.py   # Main agent orchestration
│   │   └── tools/           # Document, web, and image search tools
│   ├── services/            # Core services
│   │   ├── llm_service.py        # LLM integration (OpenRouter)
│   │   ├── embeddings_service.py # HuggingFace embeddings
│   │   ├── vectorstore_service.py # FAISS vector store
│   │   ├── retriever_service.py  # Document retrieval
│   │   ├── memory_service.py     # Conversation memory
│   │   ├── preprocessing.py      # Document processing
│   │   └── indexing_service.py   # Index building
│   └── core/
│       └── config.py        # Configuration management
├── webapp/                  # Django web application
│   ├── chatbot/            # Chatbot Django app
│   ├── static/             # CSS, JavaScript
│   └── templates/          # HTML templates
├── data/
│   ├── raw/                # Upload your PDFs here
│   ├── processed/          # Processed data
│   └── vectorstore/        # FAISS index storage
└── manage.py               # Django management script

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/EyadAmgad/agentic_rag_tourguide.git
cd agentic_rag_tourguide

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct

# SerpAPI (for image search)
SERPAPI_API_KEY=your_serpapi_key

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Prepare Your Documents

```bash
# Add PDF files to data/raw folder
cp your_documents.pdf data/raw/

# Build the vector store index using Django shell
python manage.py shell
```

Then in the Django shell:
```python
from services.indexing_service import IndexingService
indexing_service = IndexingService()
indexing_service.build_index()
```

### 4. Run the Web Application

```bash
# Run migrations
python manage.py migrate

# Start the server
python manage.py runserver

# Open browser at http://localhost:8000
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
