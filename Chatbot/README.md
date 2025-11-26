# PharaohGuide Chatbot

This chatbot uses the RAG (Retrieval-Augmented Generation) system to provide expert knowledge about ancient Egyptian history, pharaohs, monuments, and culture.

## 🏺 Features

- **RAG-Powered**: Uses historical documents from the Agentic_RAG folder system
- **Expert Knowledge**: Trained on Egyptian historical texts and documents
- **Multiple Interfaces**: Streamlit web interface and FastAPI server
- **Real-time Chat**: Interactive conversation about ancient Egypt

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have:
- Python 3.8+ installed
- Virtual environment activated
- Dependencies installed: `pip install -r requirements.txt`
- RAG system documents loaded in `../Agentic_RAG/src/controllers/data/`

### 2. Set up API Keys

Create a `.env` file in `../Agentic_RAG/src/` with your API keys:
```
OPEN_ROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Test the System

```bash
# Test RAG system integration
python test_rag_system.py
```

### 4. Run the Chatbot

**Option A: Streamlit Web Interface**
```bash
streamlit run main.py
```
Open: http://localhost:8501

**Option B: FastAPI Server**
```bash
python chatbot_api.py
```
API available at: http://localhost:8080

## 📚 How It Works

1. **Document Loading**: Loads historical documents from the Agentic_RAG folder
2. **Query Processing**: Processes user questions about Egyptian history
3. **RAG Retrieval**: Finds relevant information from historical documents
4. **Response Generation**: Generates expert answers using AI

## 🔧 Configuration

The chatbot automatically:
- Loads documents from `../Agentic_RAG/src/controllers/data/`
- Uses the RAG system for document retrieval
- Provides historical context for answers

## 📊 API Endpoints (FastAPI)

- `GET /` - API information
- `GET /health` - Check system health
- `GET /config` - View configuration and loaded documents
- `POST /chat` - Chat with the Egyptologist bot

### Example API Usage

```bash
# Test the chatbot API
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Tutankhamun"}'
```

## 🧪 Testing

```bash
# Test RAG system integration
python test_rag_system.py

# Test specific functionality
python test_rag_integration.py
```

## 📁 Files

- `main.py` - Streamlit web interface
- `chatbot_api.py` - FastAPI server
- `test_rag_system.py` - System integration tests
- `test_rag_integration.py` - RAG system tests
- `requirements.txt` - Python dependencies

## 🎯 Sample Queries

Try asking:
- "Who was Tutankhamun and why is he famous?"
- "What are the main pyramids at Giza?"
- "Tell me about ancient Egyptian gods and religion"
- "How were pharaohs mummified?"
- "What was the role of pharaohs in ancient Egypt?"

## 🔍 Troubleshooting

### Common Issues

1. **RAG System Not Initialized**
   - Check if documents exist in `../Agentic_RAG/src/controllers/data/`
   - Verify API keys are set in `.env` file
   - Run `python test_rag_system.py` to diagnose

2. **Import Errors**
   - Ensure you're in the Chatbot directory
   - Check that Agentic_RAG folder exists in parent directory
   - Verify all dependencies are installed

3. **API Key Issues**
   - Check `.env` file in `../Agentic_RAG/src/`
   - Verify OpenRouter API key is valid
   - Test API key with a simple request

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🏺 Historical Documents

The chatbot uses these historical sources:
- A Short History of Egypt
- Egyptian Pharaohs (Penn Museum)
- Life in Ancient Egypt
- Ancient Egypt: Gods and Pharaohs
- History of Ancient Egypt

## 🎉 Success Indicators

You'll know it's working when:
- ✅ RAG system loads documents successfully
- ✅ Chatbot responds to questions about Egyptian history
- ✅ Answers include historical context and details
- ✅ Health check returns "healthy" status

## 📞 Support

If you encounter issues:
1. Check the test results: `python test_rag_system.py`
2. Verify RAG system status: `curl http://localhost:8080/health`
3. Check configuration: `curl http://localhost:8080/config`
4. Review logs for error messages

Happy exploring ancient Egypt! 🏺