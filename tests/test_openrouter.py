"""
Test OpenRouter API directly to diagnose LLM issues.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment from project root .env
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(str(REPO_ROOT / ".env"))

api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPEN_ROUTER_MODEL") or os.getenv("OPENROUTER_MODEL")

print("=" * 60)
print("Testing OpenRouter API")
print("=" * 60)
print(f"API Key: {api_key[:20]}...")
print(f"Model: {model}")
print()

# Test 1: Check if API key is valid
print("1. Testing API endpoint...")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Say 'hello' only"}
    ],
    "max_tokens": 50
}

try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    print(f"   Status Code: {response.status_code}")
    
    if response.ok:
        result = response.json()
        if "choices" in result:
            answer = result["choices"][0]["message"]["content"]
            print(f"   ✓ Response: {answer}")
        else:
            print(f"   ✓ Raw response: {result}")
    else:
        print(f"   ❌ Error: {response.text}")
        
        # Try to get more info about available models
        print("\n2. Checking model availability...")
        models_url = "https://openrouter.ai/api/v1/models"
        models_response = requests.get(models_url, headers=headers)
        if models_response.ok:
            models_data = models_response.json()
            print(f"   Found {len(models_data.get('data', []))} models")
            
            # Check if our model exists
            our_model = model
            found = False
            for m in models_data.get('data', []):
                if m.get('id') == our_model:
                    found = True
                    print(f"   ✓ Model '{our_model}' is available")
                    break
            
            if not found:
                print(f"   ❌ Model '{our_model}' not found")
                print("\n   Available free models:")
                for m in models_data.get('data', []):
                    if 'free' in m.get('id', '').lower() or m.get('pricing', {}).get('prompt') == '0':
                        print(f"      - {m.get('id')}")
        
except Exception as e:
    print(f"   ❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
