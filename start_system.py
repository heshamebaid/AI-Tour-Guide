#!/usr/bin/env python3
"""
Simple startup script for the Hieroglyph Translation System
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

def _print_gpu_status():
    """Detect and display GPU availability for PyTorch and TensorFlow, and enable TF memory growth."""
    try:
        import torch
        torch_available = torch.cuda.is_available()
        if torch_available:
            device_count = torch.cuda.device_count()
            names = ", ".join([torch.cuda.get_device_name(i) for i in range(device_count)])
            print(f"🟢 PyTorch CUDA: available ({device_count} device(s): {names})")
        else:
            print("🟡 PyTorch CUDA: not available (CPU fallback)")
    except Exception as e:
        print(f"🟡 PyTorch check failed: {e}")

    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"🟢 TensorFlow GPU: {len(gpus)} device(s) detected; memory growth enabled")
        else:
            print("🟡 TensorFlow GPU: not detected (CPU fallback)")
    except Exception as e:
        print(f"🟡 TensorFlow check failed: {e}")

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if we're in the right directory
    if not Path("translation_service/api_server.py").exists():
        print("❌ Please run this script from the AI-Tour-Guide directory")
        return False
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("⚠️  Virtual environment not found. Creating one...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    
    # Check dependencies
    try:
        import requests
        import streamlit
        import fastapi
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "Chatbot/requirements.txt"])
        print("✅ Dependencies installed")
    
    return True

def setup_environment():
    """Set up environment variables"""
    print("🔧 Setting up environment...")
    
    # Create Agentic_RAG src directory if it doesn't exist
    rag_src = Path("Agentic_RAG/src")
    rag_src.mkdir(parents=True, exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = rag_src / ".env"
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# OpenRouter API Key for Agentic_RAG Chatbot
# Get your API key from: https://openrouter.ai/keys
OPEN_ROUTER_API_KEY=your_api_key_here

# Optional: Hugging Face API Key
# HUGGINGFACE_API_KEY=your_hf_api_key_here
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit Agentic_RAG/src/.env and add your actual API keys")
    else:
        print("✅ .env file already exists")

def start_component(name, command, cwd=None):
    """Start a system component"""
    print(f"🚀 Starting {name}...")
    
    try:
        if cwd:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Give it a moment to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ {name} started successfully")
            return process
        else:
            print(f"❌ {name} failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def main():
    """Main startup function"""
    print("🏺 Hieroglyph Translation System Startup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        return
    
    # Setup environment
    setup_environment()
    
    # Show GPU status
    print("\n🧮 Hardware Acceleration:")
    _print_gpu_status()
    
    print("\n🚀 Starting system components...")
    
    # Start components
    processes = []
    
    # 1. Main API Server
    api_process = start_component("Main API Server", "python -m translation_service.api_server")
    if api_process:
        processes.append(("Main API Server", api_process))
    
    # 2. Chatbot API Server
    chatbot_process = start_component("Chatbot API Server", "python chatbot_api.py", "Chatbot")
    if chatbot_process:
        processes.append(("Chatbot API Server", chatbot_process))
    
    # 3. Streamlit Chatbot (optional)
    try:
        streamlit_process = start_component("Streamlit Chatbot", "streamlit run main.py --server.port 8501 --server.headless true", "Chatbot")
        if streamlit_process:
            processes.append(("Streamlit Chatbot", streamlit_process))
    except Exception as e:
        print(f"⚠️  Streamlit chatbot not started: {e}")
    
    # Show status
    print("\n📊 System Status:")
    print("=" * 30)
    print("🌐 Main API Server:     http://localhost:8000")
    print("🤖 Chatbot API:         http://localhost:8080")
    print("💬 Streamlit Chatbot:   http://localhost:8501")
    
    print("\n📚 Available Endpoints:")
    print("   • POST /translate    - Upload hieroglyph image")
    print("   • POST /chat         - Chat with Egyptologist bot")
    print("   • GET  /health       - Check system health")
    print("   • GET  /config       - View configuration")
    
    print("\n🎯 Quick Tests:")
    print("   • Test translation:  curl -X POST http://localhost:8000/translate -F 'file=@image.jpg'")
    print("   • Test chatbot:      curl -X POST http://localhost:8080/chat -d '{\"query\":\"Hello\"}'")
    
    print("\n⏹️  Press Ctrl+C to stop all components")
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} stopped unexpectedly")
                    
    except KeyboardInterrupt:
        print("\n🛑 Stopping all components...")
        for name, process in processes:
            try:
                process.terminate()
                print(f"✅ {name} stopped")
            except:
                pass
        print("👋 All components stopped")

if __name__ == "__main__":
    main()

