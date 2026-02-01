"""
Test OpenRouter LLM connection with free models
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
load_dotenv()

from services.llm_service import LLMService

print("🔍 Testing OpenRouter LLM Connection\n")
print("="*70)

# Get current settings
api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")

print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT SET")
print(f"Model: {model}")
print("="*70 + "\n")

# Test the LLM
try:
    print("Initializing LLM service...")
    llm_service = LLMService()
    llm = llm_service.llm
    
    print("✅ LLM service initialized\n")
    
    print("Testing simple prompt...")
    test_prompt = "Say 'Hello, I am working!' in one short sentence."
    
    response = llm.invoke(test_prompt)
    
    print("✅ LLM Response:")
    print("-" * 70)
    print(response.content)
    print("-" * 70)
    print("\n✅ SUCCESS! Your LLM is working correctly!\n")
    
except Exception as e:
    print("❌ LLM FAILED:")
    print("-" * 70)
    print(str(e))
    print("-" * 70)
    
    print("\n💡 SOLUTIONS:\n")
    print("1. Try these working FREE models in your .env:")
    print("   OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free")
    print("   OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free")
    print("   OPENROUTER_MODEL=google/gemini-flash-1.5:free")
    print("   OPENROUTER_MODEL=qwen/qwen-2-7b-instruct:free")
    print("   OPENROUTER_MODEL=microsoft/phi-3-mini-128k-instruct:free")
    
    print("\n2. Check your API key is valid:")
    print("   Visit: https://openrouter.ai/keys")
    
    print("\n3. Or run without LLM (retrieval only):")
    print("   The evaluation can work with just context retrieval\n")
