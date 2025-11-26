import streamlit as st
import sys
import os

# Add the Agentic_RAG src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Agentic_RAG', 'src'))

# Import from RAG system
from pipeline.model import rag_query, load_documents_from_data_dir, get_document_stats

# Initialize RAG system
@st.cache(allow_output_mutation=True)
def initialize_rag():
    """Initialize the RAG system from the Agentic_RAG folder"""
    load_result = load_documents_from_data_dir()
    if load_result["success"]:
        return True, load_result
    else:
        return False, load_result

# Initialize the system
rag_initialized, load_result = initialize_rag()

# Streamlit UI
st.title("🏺 Welcome to PharaohGuide Chatbot")
st.subheader("Powered by Agentic_RAG System with Egyptian Historical Documents")

# Show system status
if rag_initialized:
    st.success(f"✅ Agentic_RAG System Ready! Loaded {load_result['files_processed']} files with {load_result['total_chunks']} chunks")
    st.info(f"📚 Files loaded: {', '.join(load_result['files'])}")
else:
    st.error(f"❌ RAG System failed to initialize: {load_result.get('error', 'Unknown error')}")

# Chat interface
query = st.text_input("Ask me anything about ancient Egyptian history, pharaohs, or monuments:")

if st.button("Get Answer") and query:
    if rag_initialized:
        with st.spinner("🔍 Searching through historical documents..."):
            try:
                response = rag_query(query)
                st.markdown("### 💬 Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")
    else:
        st.error("Agentic_RAG system is not initialized. Please check the system status.")
