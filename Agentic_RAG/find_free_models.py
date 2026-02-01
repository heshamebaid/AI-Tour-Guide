"""
Find working FREE chat models on OpenRouter
"""
import requests

print("🔍 Finding FREE chat models on OpenRouter...\n")

try:
    response = requests.get('https://openrouter.ai/api/v1/models', timeout=10)
    models = response.json().get('data', [])
    
    print(f"Total models found: {len(models)}\n")
    print("="*70)
    print("FREE TEXT/CHAT MODELS:")
    print("="*70 + "\n")
    
    free_chat_models = []
    for m in models:
        model_id = m.get('id', '')
        # Check if it's free
        if ':free' in model_id:
            # Skip image models
            if any(x in model_id.lower() for x in ['flux', 'dall', 'stable', 'image', 'sd-', 'sdxl']):
                continue
            free_chat_models.append(model_id)
    
    for model in free_chat_models[:20]:
        print(f"  {model}")
    
    print(f"\n{'='*70}")
    print(f"Found {len(free_chat_models)} free chat models")
    print("="*70 + "\n")
    
    if free_chat_models:
        print("💡 Copy one of these to your .env file:")
        print(f"   OPENROUTER_MODEL={free_chat_models[0]}\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTry these known working free models:")
    print("  meta-llama/llama-3.2-3b-instruct:free")
    print("  qwen/qwen-2.5-7b-instruct:free")
    print("  mistralai/mistral-7b-instruct:free")
    print("  google/gemma-2-9b-it:free")
