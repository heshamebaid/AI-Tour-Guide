#!/usr/bin/env python3
"""
Setup OpenRouter configuration for all services (Translation API, Chatbot API, Agentic_RAG).

Usage examples:
  python setup_openrouter.py --key sk-or-v1-XXXXX --model qwen/qwen3-30b-a3b:free

This will:
  - Write Agentic_RAG/.env with OPEN_ROUTER_API_KEY and optional OPEN_ROUTER_MODEL
  - Load the key into the current process env
  - Verify visibility for Agentic_RAG and report masked key/model

Note: To persist across new terminals on Windows, also run:
  setx OPEN_ROUTER_API_KEY "YOUR_KEY" && setx OPEN_ROUTER_MODEL "your/model"
Then open a new terminal before starting services.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv


def mask(value: str, show: int = 6) -> str:
    if not value:
        return "(none)"
    if len(value) <= show:
        return value
    return value[:show] + "…" + value[-4:]


def write_env_file(rag_env_path: Path, key: str, model: str | None) -> None:
    rag_env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"OPEN_ROUTER_API_KEY={key}\n"]
    if model:
        lines.append(f"OPEN_ROUTER_MODEL={model}\n")
    rag_env_path.write_text("".join(lines), encoding="utf-8")


def verify_rag_visibility(project_root: Path) -> tuple[str, str]:
    try:
        sys.path.append(str(project_root / "Agentic_RAG" / "src"))
        # Import after sys.path change so the module reads dotenv in its __init__ path
        import pipeline.model as rag_model  # type: ignore
        key = os.getenv("OPEN_ROUTER_API_KEY") or getattr(rag_model, "OPENROUTER_API_KEY", None)
        # Model name may be read from env; default inside module if not set
        model = os.getenv("OPEN_ROUTER_MODEL") or os.getenv("LLM_MODEL") or "qwen/qwen3-30b-a3b:free"
        return key or "", model
    except Exception:
        return os.getenv("OPEN_ROUTER_API_KEY") or "", os.getenv("OPEN_ROUTER_MODEL") or ""


def main():
    parser = argparse.ArgumentParser(description="Configure OpenRouter API for all services")
    parser.add_argument("--key", required=True, help="OpenRouter API key (sk-or-v1-…)")
    parser.add_argument("--model", default=None, help="Optional OpenRouter model id (e.g., qwen/qwen3-30b-a3b:free)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    rag_env = project_root / "Agentic_RAG" / ".env"

    # 1) Write Agentic_RAG/.env
    write_env_file(rag_env, args.key, args.model)

    # 2) Load into current process env
    load_dotenv(rag_env)
    os.environ["OPEN_ROUTER_API_KEY"] = args.key
    if args.model:
        os.environ["OPEN_ROUTER_MODEL"] = args.model

    # 3) Verify visibility (Agentic_RAG)
    vis_key, vis_model = verify_rag_visibility(project_root)

    print("Configured OpenRouter:")
    print(f"- Agentic_RAG/.env: {rag_env}")
    print(f"- Key (masked): {mask(args.key)}")
    print(f"- Model: {args.model or '(default)'}")
    print("\nDetected at runtime:")
    print(f"- ENV OPEN_ROUTER_API_KEY (masked): {mask(os.getenv('OPEN_ROUTER_API_KEY') or '')}")
    print(f"- ENV OPEN_ROUTER_MODEL: {os.getenv('OPEN_ROUTER_MODEL') or '(default)'}")
    print(f"- Agentic_RAG module sees key (masked): {mask(vis_key)}")
    print(f"- Agentic_RAG module model: {vis_model or '(default)'}")

    print("\nNext steps:")
    print("- Restart services from this terminal so they inherit the key:")
    print("  python -m translation_service.api_server")
    print("  cd Chatbot && $env:TRANSFORMERS_NO_TF='1'; python chatbot_api.py")
    print("  python Django/manage.py runserver 8000")
    print("- To persist for new terminals (Windows):")
    print("  setx OPEN_ROUTER_API_KEY \"YOUR_KEY\"")
    print("  setx OPEN_ROUTER_MODEL \"your/model\"")


if __name__ == "__main__":
    main()


