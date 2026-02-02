#!/usr/bin/env python3
"""
Unified launcher for the Hieroglyph Translation System.
Starts:
- Main FastAPI server (translation_service/api_server.py)
- Django web app (Django/manage.py)

Reports GPU status and prints service URLs.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run(cmd: str, cwd: str | None = None):
    return subprocess.Popen(cmd, shell=True, cwd=cwd)

def main():
    print("🏁 Starting Hieroglyph Translation System (all services)")
    print("=" * 60)

    # Ensure we're at project root
    if not Path("translation_service/api_server.py").exists():
        print("❌ Please run this script from the project root (AI-Tour-Guide)")
        sys.exit(1)

    # Optional: quick GPU status
    try:
        import torch
        print(f"🟢 PyTorch CUDA: {'available' if torch.cuda.is_available() else 'not available'}")
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
            print(f"🟢 TensorFlow GPU: {len(gpus)} device(s) detected")
        else:
            print("🟡 TensorFlow GPU: not detected")
    except Exception as e:
        print(f"🟡 TensorFlow check failed: {e}")

    # Start services (API + Django only)
    api = run("python -m translation_service.api_server")
    django = run("python manage.py runserver 8000", cwd="Django")

    print("\n📍 Service URLs:")
    print("- 🌐 API:        http://localhost:8000")
    print("- 🕸️ Django:     http://localhost:8000")

    print("\n⏹️  Press Ctrl+C here to stop all services.")
    try:
        while True:
            time.sleep(1)
            # Optionally, monitor processes and print if any stopped
            for name, proc in (("API", api), ("Django", django)):
                if proc.poll() is not None:
                    print(f"⚠️  {name} process exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        for proc in (api, django):
            try:
                proc.terminate()
            except Exception:
                pass
        print("👋 Done.")

if __name__ == "__main__":
    main()


