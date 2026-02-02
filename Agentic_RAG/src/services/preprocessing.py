import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from core.config import Config


class DocumentPreprocessor:

    def __init__(self, chunking_method: str = "semantic"):
        """
        Initialize the document preprocessor.
        
        Args:
            chunking_method: "semantic" for semantic chunking, "simple" for fixed-size chunking
        """
        self.chunking_method = chunking_method
        
        if chunking_method == "semantic":
            # Initialize embeddings for semantic chunking
            embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            
            # Semantic chunker - splits based on semantic similarity
            # breakpoint_threshold_type options:
            # - "percentile": splits at points where similarity drops below percentile
            # - "standard_deviation": splits when similarity drops by X std devs
            # - "interquartile": splits based on IQR of similarities
            # - "gradient": splits at steepest drops in similarity
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,  # Higher = fewer, larger chunks
                min_chunk_size=200,  # Minimum chunk size in characters
            )
            print(f"✓ Initialized Semantic Chunker (threshold: 85th percentile)")
        else:
            # Fallback to simple recursive character splitting
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )
            print(f"✓ Initialized Simple Chunker (700 chars, 100 overlap)")

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
        # Remove excessive whitespace and newlines
        text = ' '.join(text.split())
        return text.strip()

    def preprocess(self):
        """Load, clean, and chunk documents using semantic or simple chunking."""
        raw_docs = self.load_documents()
        
        if not raw_docs:
            print("Warning: No documents found!")
            return []
        
        print(f"Processing {len(raw_docs)} documents...")
        print(f"Chunking method: {self.chunking_method.upper()}")
        
        # Clean document text before chunking
        for doc in raw_docs:
            doc.page_content = self.clean_text(doc.page_content)
        
        # Split documents into chunks
        if self.chunking_method == "semantic":
            print("Performing semantic chunking (this may take a moment)...")
            try:
                chunks = self.text_splitter.split_documents(raw_docs)
                print(f"✓ Created {len(chunks)} semantic chunks")
                
                # Print chunk size statistics
                chunk_sizes = [len(c.page_content) for c in chunks]
                avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                print(f"  Average chunk size: {avg_size:.0f} characters")
                print(f"  Min chunk size: {min(chunk_sizes) if chunk_sizes else 0} characters")
                print(f"  Max chunk size: {max(chunk_sizes) if chunk_sizes else 0} characters")
                
            except Exception as e:
                print(f"⚠️ Semantic chunking failed: {e}")
                print("Falling back to simple chunking...")
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=100
                )
                chunks = fallback_splitter.split_documents(raw_docs)
                print(f"✓ Created {len(chunks)} chunks (simple fallback)")
        else:
            chunks = self.text_splitter.split_documents(raw_docs)
            print(f"✓ Created {len(chunks)} chunks")
        
        return chunks
