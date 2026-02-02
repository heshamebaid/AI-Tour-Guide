#!/usr/bin/env python3
"""
Minimal test to check LLM service import and functionality.
Run from: /mnt/a/AI-Tour-Guide
"""
import sys
sys.path.insert(0, '/mnt/a/AI-Tour-Guide/Agentic_RAG/src')

print("Testing LLM Service Import...")

try:
    from services.llm_service import LLMService
    print("✓ Import successful")
    
    service = LLMService()
    print("✓ Service instantiated")
    
    # Test pickle
    import pickle
    pickled = pickle.dumps(service)
    print("✓ Pickle successful")
    
    unpickled = pickle.loads(pickled)
    print("✓ Unpickle successful")
    
    # Test LLM
    print("Testing LLM call...")
    llm = service.llm
    response = llm.invoke("Say hello")
    print(f"✓ LLM response: {response.content[:100]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
