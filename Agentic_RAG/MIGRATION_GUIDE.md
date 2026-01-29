# Migration Guide: FAISS to Qdrant with Hybrid Search

This guide will help you migrate from the old FAISS-based system to the new Qdrant-based system with hybrid search capabilities.

## What's Changed?

### Core Changes
- **Vector Database**: FAISS → Qdrant
- **Search Method**: Single semantic search → Hybrid search (semantic + keyword)
- **Framework**: Enhanced LangChain integration
- **Storage**: File-based (index.faiss) → Collection-based (Qdrant)

### New Features
- ✅ Hybrid search combining dense (semantic) and sparse (BM25 keyword) retrieval
- ✅ Multiple search strategies: hybrid, dense, sparse, MMR
- ✅ Better scalability and performance
- ✅ Distributed deployment support
- ✅ Real-time updates and filtering
- ✅ Built-in metrics and monitoring

## Migration Steps

### Step 1: Backup Your Data (Optional)
If you want to keep your old FAISS index:
```bash
cd Agentic_RAG
cp -r vectorstore/faiss_index vectorstore/faiss_index.backup
```

### Step 2: Install Qdrant

**Option A: Docker (Recommended)**
```bash
# Start Qdrant in Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Option B: Download Binary**
Visit https://qdrant.tech/documentation/guides/installation/ for platform-specific instructions.

### Step 3: Update Dependencies
```bash
cd Agentic_RAG
pip install -r requirements.txt
```

This will install:
- `qdrant-client` (replaces `faiss-cpu`)
- `langchain-qdrant` (Qdrant integration)
- `rank-bm25` (for hybrid search)
- Updated LangChain packages

### Step 4: Update Configuration

Update your `.env` file with new Qdrant settings:

```env
# Add these new settings
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=tour_guide_documents

# Hybrid search configuration
ENABLE_HYBRID_SEARCH=true
DENSE_WEIGHT=0.5
SPARSE_WEIGHT=0.5

# Update this path (remove /faiss_index)
VECTORSTORE_PATH=./vectorstore
```

### Step 5: Rebuild Your Index

**Using the management utility:**
```bash
python manage_qdrant.py build
```

**Or programmatically:**
```python
from services.indexing_service import IndexingService

indexing = IndexingService()
indexing.build_index()
```

This will:
1. Read documents from `data/raw/`
2. Create embeddings
3. Build BM25 index for keyword search
4. Create Qdrant collection with all documents

### Step 6: Verify Migration

Check that everything is working:
```bash
python manage_qdrant.py health
```

Or test search:
```bash
python manage_qdrant.py test --query "your test query"
```

### Step 7: Update Your Code

**Old Code (FAISS):**
```python
from services.retriever_service import RetrieverService

retriever_service = RetrieverService()
retriever = retriever_service.get_retriever(k=4)
results = retriever.get_relevant_documents(query)
```

**New Code (Qdrant with Hybrid Search):**
```python
from services.retriever_service import RetrieverService

retriever_service = RetrieverService()

# Hybrid search (recommended)
retriever = retriever_service.get_retriever(k=4, search_type="hybrid")

# Or use dense only (like old behavior)
retriever = retriever_service.get_retriever(k=4, search_type="dense")

# Or MMR for diverse results
retriever = retriever_service.get_mmr_retriever(k=4, lambda_mult=0.5)

results = retriever.get_relevant_documents(query)
```

## Code Changes Required

### If You Import VectorStoreService Directly

**Before:**
```python
from services.vectorstore_service import VectorStoreService

vs = VectorStoreService()
vectorstore = vs.load_vectorstore()  # Returns FAISS
```

**After:**
```python
from services.vectorstore_service import VectorStoreService

vs = VectorStoreService()
vectorstore = vs.load_vectorstore()  # Returns QdrantVectorStore
# API is the same, no other changes needed!
```

### If You Use IndexingService

**Before:**
```python
from services.indexing_service import IndexingService

indexing = IndexingService()
indexing.build_index()  # Creates FAISS index
```

**After:**
```python
from services.indexing_service import IndexingService

indexing = IndexingService()
indexing.build_index()  # Creates Qdrant collection + BM25 index
# API is the same!
```

## Troubleshooting

### "Cannot connect to Qdrant"
**Problem**: Qdrant server is not running.

**Solution**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# If not, start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant
```

### "Collection not found"
**Problem**: Index hasn't been built yet.

**Solution**:
```bash
python manage_qdrant.py build
```

### "BM25 model not found"
**Problem**: Sparse encoder wasn't saved during indexing.

**Solution**:
```bash
# Rebuild the index to create BM25 model
python manage_qdrant.py rebuild --force
```

### Hybrid search not working
**Problem**: BM25 model missing or hybrid search disabled.

**Solution**:
```env
# Check your .env file
ENABLE_HYBRID_SEARCH=true

# Rebuild index
python manage_qdrant.py rebuild --force
```

### Performance Issues
**Problem**: Slow retrieval with hybrid search.

**Solution**:
```env
# Adjust weights to favor dense search (faster)
DENSE_WEIGHT=0.8
SPARSE_WEIGHT=0.2

# Or disable hybrid search completely
ENABLE_HYBRID_SEARCH=false
```

## Configuration Tuning

### Search Type Selection

**Use Hybrid Search When:**
- You want best overall results
- Queries mix concepts and specific terms
- Default for most use cases

**Use Dense Search When:**
- Semantic/conceptual similarity is most important
- You want fastest retrieval
- Queries are conceptual

**Use Sparse Search When:**
- Exact keyword matching is critical
- Working with technical terms
- Need precise term-based search

### Weight Tuning

Adjust in `.env`:
```env
# More semantic, less keyword
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3

# More keyword, less semantic
DENSE_WEIGHT=0.3
SPARSE_WEIGHT=0.7

# Balanced (default)
DENSE_WEIGHT=0.5
SPARSE_WEIGHT=0.5
```

## Cleanup (Optional)

After successful migration, you can remove old FAISS data:
```bash
# Remove old FAISS index
rm -rf vectorstore/faiss_index
rm -rf vectorstore/faiss_index.backup

# Uninstall FAISS (if not used elsewhere)
pip uninstall faiss-cpu
```

## Rollback (If Needed)

If you need to rollback to FAISS:

1. **Restore old code** from git:
```bash
git checkout <previous-commit> -- src/services/
```

2. **Restore old dependencies**:
```bash
pip install faiss-cpu==1.8.0
```

3. **Restore old index** (if you backed it up):
```bash
cp -r vectorstore/faiss_index.backup vectorstore/faiss_index
```

## Benefits Summary

After migration, you'll have:

- 🚀 **Better Performance**: Qdrant's HNSW algorithm is optimized for speed
- 🎯 **Better Accuracy**: Hybrid search combines semantic + keyword matching
- 📈 **Scalability**: Easy to scale horizontally with Qdrant cluster
- 🔧 **Flexibility**: Multiple search strategies (hybrid, dense, sparse, MMR)
- 📊 **Monitoring**: Built-in metrics and collection stats
- 🌐 **Cloud Ready**: Easy deployment to Qdrant Cloud

## Need Help?

- Check [QDRANT_SETUP.md](QDRANT_SETUP.md) for detailed setup instructions
- Run `python manage_qdrant.py health` to diagnose issues
- Test search with `python example_qdrant_usage.py`
- Review [Qdrant Documentation](https://qdrant.tech/documentation/)
