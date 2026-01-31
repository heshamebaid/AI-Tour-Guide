from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from .persona_registry import get_persona, list_personas, persona_count
from .prompt_builder import build_persona_prompt
from .schemas import ConverseRequest, ConverseResponse, PersonaResponse

# Ensure Agentic_RAG src directory is on the path
SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_DIR.parent
RAG_SRC = PROJECT_ROOT / "Agentic_RAG" / "src"
sys.path.append(str(RAG_SRC))

# Load env vars from either Agentic_RAG/.env or Agentic_RAG/src/.env to reach OpenRouter creds
for env_candidate in (RAG_SRC / ".env", PROJECT_ROOT / "Agentic_RAG" / ".env"):
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate)

from pipeline.model import (  # type: ignore  # pylint: disable=wrong-import-position
    rag_query,
    load_documents_from_data_dir,
    get_document_stats,
)

logger = logging.getLogger("talk_to_pharos_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

rag_initialized = False

app = FastAPI(
    title="Talk To Pharos Service",
    description="Persona-aware FastAPI service backed by the Hieroglyph Agentic_RAG system.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("PHAROS_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _persona_to_response(persona) -> PersonaResponse:
    return PersonaResponse(**persona.to_public_dict())


@app.on_event("startup")
async def bootstrap_rag():
    global rag_initialized  # noqa: PLW0603
    logger.info("Bootstrapping Agentic_RAG document store for Talk To Pharos service...")
    if os.getenv("PHAROS_SKIP_STARTUP") == "1":
        rag_initialized = True
        logger.info("PHAROS_SKIP_STARTUP set. Skipping Agentic_RAG loading (tests).")
        return
    try:
        load_result = load_documents_from_data_dir()
        if load_result.get("success"):
            rag_initialized = True
            logger.info(
                "Loaded %s files with %s chunks",
                load_result.get("files_processed"),
                load_result.get("total_chunks"),
            )
        else:
            logger.warning(
                "No RAG documents loaded (%s); service will use LLM general knowledge only.",
                load_result.get("error", "no files"),
            )
            rag_initialized = True  # Service ready; converse uses LLM when no RAG data
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error while initializing Agentic_RAG: %s", exc)
        rag_initialized = True  # Service ready; converse uses LLM when no RAG data


@app.get("/")
def root():
    return {
        "service": "Talk To Pharos",
        "status": "running" if rag_initialized else "degraded",
        "available_endpoints": [
            "/health",
            "/config",
            "/pharos",
            "/converse",
        ],
    }


@app.get("/health")
def health():
    if not rag_initialized:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Agentic_RAG system not initialized"},
        )
    return {"status": "healthy", "message": "Agentic_RAG system ready"}


@app.get("/config")
def config():
    stats = get_document_stats()
    return {
        "rag_initialized": rag_initialized,
        "documents": stats,
        "persona_count": persona_count(),
    }


@app.get("/pharos", response_model=List[PersonaResponse])
def pharos():
    return [_persona_to_response(p) for p in list_personas()]


@app.post("/converse", response_model=ConverseResponse)
def converse(request: ConverseRequest):
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="Agentic_RAG system is still loading")

    persona = get_persona(request.pharaoh_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Unknown pharaoh id")

    try:
        persona_prompt = build_persona_prompt(persona, request.user_query, request.history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        answer = rag_query(persona_prompt, top_k=request.top_k or 2)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Agentic_RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agentic_RAG query failed") from exc

    return ConverseResponse(
        pharaoh_id=persona.id,
        answer=answer,
        persona=_persona_to_response(persona),
        used_history_turns=len(request.history),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "talk_to_pharos_service.app:app",
        host=os.getenv("PHAROS_HOST", "0.0.0.0"),
        port=int(os.getenv("PHAROS_PORT", "8050")),
        reload=True,
    )

