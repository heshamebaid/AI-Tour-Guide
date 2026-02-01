"""
Debug OpenRouter API - Check why "User not found" error
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")

print("🔍 Debugging OpenRouter API\n")
print("="*70)
print(f"API Key: {api_key[:30]}..." if api_key else "NO API KEY")
print(f"Model: {model}")
print("="*70 + "\n")

# Test 1: Check API key validity
print("1️⃣ Testing API key with simple request...\n")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "RAG Evaluation"
}

# Try a simple working free model first
test_model = "google/gemini-2.0-flash-exp:free"
data = {
    "model": test_model,
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.7,
    "max_tokens": 50
}

print(f"Testing with model: {test_model}\n")

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{response.text}\n")
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("✅ SUCCESS! API key is working!")
        print(f"Response: {content}\n")
        
        # Now test their model
        print("="*70)
        print(f"2️⃣ Testing YOUR model: {model}\n")
        
        data["model"] = model
        response2 = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"Status Code: {response2.status_code}")
        print(f"Response:\n{response2.text}\n")
        
        if response2.status_code == 200:
            print(f"✅ Model {model} is working!")
        else:
            print(f"❌ Model {model} failed!")
            print("\n💡 Your API key works, but this model doesn't.")
            print("Try one of these FREE working models:\n")
            print("   google/gemini-2.0-flash-exp:free")
            print("   meta-llama/llama-3.2-3b-instruct:free")
            print("   qwen/qwen-2-7b-instruct:free")
            print("   microsoft/phi-3-mini-128k-instruct:free\n")
            
    else:
        print("❌ FAILED!")
        error_data = response.json() if response.text else {}
        error_msg = error_data.get("error", {}).get("message", response.text)
        
        print(f"\nError: {error_msg}\n")
        
        if "User not found" in error_msg or "Invalid" in error_msg:
            print("💡 This means your API key is INVALID or EXPIRED\n")
            print("Solutions:")
            print("1. Get a new API key from: https://openrouter.ai/keys")
            print("2. Make sure you copied the FULL key")
            print("3. Check if your account is active\n")
        
except Exception as e:
    print(f"❌ Request failed: {e}\n")
    print("Check your internet connection or OpenRouter status")
