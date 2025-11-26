import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from core.config import Config

class DocumentPreprocessor:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

    def load_documents(self):
        """Load both PDF and TXT files from the documents directory."""
        documents = []
        docs_path = Config.DOCUMENTS_PATH
        
        # Load PDF files
        pdf_loader = DirectoryLoader(
            docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        # Load TXT files
        txt_loader = DirectoryLoader(
            docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        try:
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"Error loading PDFs: {e}")
        
        try:
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} TXT documents")
        except Exception as e:
            print(f"Error loading TXT files: {e}")
        
        return documents

    def clean_text(self, text: str):
        """Clean and normalize text."""
        return text.replace("\n", " ").strip()

    def preprocess(self):
        """Load, clean, and chunk documents."""
        raw_docs = self.load_documents()
        
        if not raw_docs:
            print("Warning: No documents found!")
            return []
        
        print(f"Processing {len(raw_docs)} documents...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(raw_docs)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
