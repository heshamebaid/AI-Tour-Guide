#!/usr/bin/env python3
"""
Environment setup script for the Hieroglyph Translation System
"""

import os
import sys
import subprocess
from pathlib import Path

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    if not Path("venv").exists():
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    # Install main requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install chatbot requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "Chatbot/requirements.txt"])
    
    print("✅ Dependencies installed")

def setup_environment_files():
    """Set up environment files"""
    print("🔧 Setting up environment files...")
    
    # Create Agentic_RAG src directory
    rag_src = Path("Agentic_RAG/src")
    rag_src.mkdir(parents=True, exist_ok=True)
    
    # Create .env file
    env_file = rag_src / ".env"
    if not env_file.exists():
        env_content = """# OpenRouter API Key for Agentic_RAG Chatbot
# Get your API key from: https://openrouter.ai/keys
OPEN_ROUTER_API_KEY=your_api_key_here

# Optional: Hugging Face API Key
# HUGGINGFACE_API_KEY=your_hf_api_key_here
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Logs directory created")

def check_models():
    """Check if required models exist"""
    print("🔍 Checking AI models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("✅ Models directory created")
    
    # Check for SAM model
    sam_model = models_dir / "sam_vit_b.pth"
    if not sam_model.exists():
        print("⚠️  SAM model not found. You may need to download it.")
        print("   Run: python -m translation_service.model_downloader")
    else:
        print("✅ SAM model found")
    
    # Check for InceptionV3 model
    inception_model = models_dir / "InceptionV3_model.h5"
    if not inception_model.exists():
        print("⚠️  InceptionV3 model not found. You may need to download it.")
        print("   Run: python -m translation_service.model_downloader")
    else:
        print("✅ InceptionV3 model found")

def check_data():
    """Check if required data exists"""
    print("🔍 Checking data files...")
    
    # Check for Gardiner's list
    gardiner_file = Path("data/Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx")
    if not gardiner_file.exists():
        print("⚠️  Gardiner's Sign List not found")
        print("   Please ensure the Excel file is in the data/ directory")
    else:
        print("✅ Gardiner's Sign List found")
    
    # Check for Agentic_RAG documents
    rag_data_dir = Path("Agentic_RAG/src/controllers/data")
    if not rag_data_dir.exists():
        print("⚠️  Agentic_RAG data directory not found")
        print("   Please ensure PDF documents are in Agentic_RAG/src/controllers/data/")
    else:
        pdf_files = list(rag_data_dir.glob("*.pdf"))
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF documents in Agentic_RAG data")
        else:
            print("⚠️  No PDF documents found in Agentic_RAG data directory")

def main():
    """Main setup function"""
    print("🏺 Hieroglyph Translation System Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("translation_service/api_server.py").exists():
        print("❌ Please run this script from the AI-Tour-Guide directory")
        return False
    
    try:
        # Setup steps
        create_virtual_environment()
        install_dependencies()
        setup_environment_files()
        check_models()
        check_data()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit Agentic_RAG/src/.env and add your API keys")
        print("2. Download models if needed: python -m translation_service.model_downloader")
        print("3. Run the system: python start_system.py")
        print("4. Test the system: python test_system.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)







