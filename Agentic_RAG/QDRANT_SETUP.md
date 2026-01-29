# Qdrant Setup Guide

## Overview
The Agentic RAG system has been upgraded to use Qdrant vector database with hybrid search capabilities, powered by LangChain framework.

## Key Features
- **Qdrant Vector Database**: High-performance vector search with better scalability than FAISS
- **Hybrid Search**: Combines dense (semantic) and sparse (BM25 keyword) search for improved retrieval
- **LangChain Integration**: Full integration with LangChain framework for better flexibility
- **Ensemble Retriever**: Weighted combination of multiple search strategies

## Installation

### 1. Install Qdrant (Docker - Recommended)
```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Or on Windows:
```bash
docker run -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Install Python Dependencies
```bash
cd Agentic_RAG
pip install -r requirements.txt
```

## Configuration

### Environment Variables (.env file)
Add these to your `.env` file in the Agentic_RAG directory:

```env
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=tour_guide_documents
QDRANT_API_KEY=  # Optional, for Qdrant Cloud

# Hybrid Search Settings
ENABLE_HYBRID_SEARCH=true
DENSE_WEIGHT=0.5   # Weight for semantic search (0.0-1.0)
SPARSE_WEIGHT=0.5  # Weight for BM25 keyword search (0.0-1.0)

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration (existing)
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=your_model
```

## Usage

### Build the Vector Index
```python
from services.indexing_service import IndexingService

indexing = IndexingService()
indexing.build_index()  # Build new index
# Or
indexing.rebuild_index()  # Rebuild existing index
```

### Query with Hybrid Search
```python
from services.retriever_service import RetrieverService

retriever_service = RetrieverService()

# Hybrid search (default)
retriever = retriever_service.get_retriever(k=4, search_type="hybrid")

# Dense (semantic) search only
retriever = retriever_service.get_retriever(k=4, search_type="dense")

# Sparse (BM25) search only
retriever = retriever_service.get_retriever(k=4, search_type="sparse")

# Use the retriever
results = retriever.get_relevant_documents("your query here")
```

### MMR (Maximum Marginal Relevance) for Diversity
```python
# Get diverse results to avoid redundancy
mmr_retriever = retriever_service.get_mmr_retriever(
    k=4,           # Number of documents to return
    fetch_k=20,    # Number to fetch before applying MMR
    lambda_mult=0.5  # 0=max diversity, 1=min diversity
)
```

## Architecture

### Components
1. **VectorStoreService**: Manages Qdrant collections and vector operations
2. **SparseEncoderService**: Handles BM25 encoding for keyword search
3. **EmbeddingService**: Provides dense embeddings via HuggingFace
4. **RetrieverService**: Creates hybrid retrievers combining dense + sparse
5. **IndexingService**: Builds and manages the vector index

### Hybrid Search Flow
```
Query → Dense Embedding (HuggingFace) → Qdrant Semantic Search
     → Sparse Encoding (BM25)        → Keyword Search
     → Ensemble Retriever            → Weighted Combination
     → Final Results
```

## Tuning Hybrid Search

### Adjusting Weights
- `DENSE_WEIGHT=0.7, SPARSE_WEIGHT=0.3`: Favor semantic understanding
- `DENSE_WEIGHT=0.3, SPARSE_WEIGHT=0.7`: Favor exact keyword matches
- `DENSE_WEIGHT=0.5, SPARSE_WEIGHT=0.5`: Balanced (default)

### When to Use Each Mode
- **Hybrid**: Best for general use, combines semantic + keywords
- **Dense**: When you need semantic/conceptual similarity
- **Sparse**: When exact keywords/terms are critical

## Migration from FAISS

### What Changed
- ✅ FAISS → Qdrant (better scalability, more features)
- ✅ Added BM25 sparse search for hybrid retrieval
- ✅ Full LangChain integration
- ✅ Collection-based storage (easier management)
- ✅ Support for distributed deployments

### Data Migration
No automatic migration is provided. You need to:
1. Rebuild the index with the new system
2. Delete old FAISS index if desired

## Troubleshooting

### Qdrant Connection Issues
```python
# Check if Qdrant is running
curl http://localhost:6333/collections

# Or in Python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())
```

### BM25 Model Not Found
If you see "BM25 model not found", rebuild the index:
```python
indexing = IndexingService()
indexing.rebuild_index()
```

### Collection Already Exists
Use `rebuild_index()` to delete and recreate:
```python
indexing.rebuild_index()
```

## Performance Tips

1. **Batch Size**: For large datasets, Qdrant handles batching automatically
2. **Memory**: BM25 loads corpus in memory - consider document limits
3. **Caching**: Qdrant has built-in caching for better performance
4. **Indexing**: HNSW algorithm provides fast approximate nearest neighbor search

## Qdrant Cloud (Optional)

To use Qdrant Cloud instead of local:
```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_api_key
```

## Additional Resources
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Qdrant Integration](https://python.langchain.com/docs/integrations/vectorstores/qdrant)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
