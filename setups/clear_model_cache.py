#!/usr/bin/env python3
"""
Clear sentence-transformers cache and fix PyTorch meta tensor issues.
Run this if you get "Cannot copy out of meta tensor" errors.
"""
import os
import shutil
from pathlib import Path

def clear_model_cache():
    """Clear the sentence-transformers model cache."""
    cache_paths = [
        Path.home() / ".cache" / "torch" / "sentence_transformers",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    
    print("Clearing model cache to fix PyTorch issues...")
    
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"Removing: {cache_path}")
            try:
                shutil.rmtree(cache_path)
                print(f"✓ Cleared: {cache_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {cache_path}: {e}")
        else:
            print(f"⚠️  Path doesn't exist: {cache_path}")
    
    print("\n" + "="*70)
    print("Cache cleared! Now run:")
    print("  pip install --upgrade torch sentence-transformers")
    print("  cd Django && python manage.py runserver")
    print("="*70)

if __name__ == "__main__":
    clear_model_cache()
