# Talk To Pharos Service

Persona-aware microservice that lets clients converse with specific Pharaohs
backed by the Agentic Retrieval-Augmented Generation (Agentic_RAG) pipeline.

## Features

- Serves persona metadata via `GET /pharos`.
- Accepts persona-scoped chat turns via `POST /converse`.
- Proxies all questions through the shared Agentic_RAG stack (FAISS + OpenRouter LLM).
- Ready for browser clients thanks to permissive CORS defaults.

## Setup

```bash
cd talk_to_pharos_service
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
```

The service depends on the existing `Agentic_RAG` folder. Ensure the project root is on
the PYTHONPATH (the included `app.py` manipulates `sys.path` automatically) and
that `OPEN_ROUTER_API_KEY` is configured in either `Agentic_RAG/.env` or `Agentic_RAG/src/.env`.

## Running

```bash
uvicorn talk_to_pharos_service.app:app --host 0.0.0.0 --port 8050 --reload
```

Env vars:

- `PHAROS_ALLOWED_ORIGINS` – CSV of allowed origins (defaults to `*`)
- `PHAROS_HOST` / `PHAROS_PORT` – override server binding when using `python app.py`

## Docker

A simple Dockerfile is provided:

```bash
docker build -t talk-to-pharos .
docker run -p 8050:8050 --env OPEN_ROUTER_API_KEY=... talk-to-pharos
```

## Endpoints

- `GET /health` – confirms Agentic_RAG readiness
- `GET /config` – reports document stats + persona count
- `GET /pharos` – lists personas (currently Ramesses II)
- `POST /converse` – persona conversation `{pharaoh_id, user_query, history[]}`

## Voice-to-Voice (frontend)

The Django page **Talk To Pharos** is built for **voice-to-voice**:

- **pharos-voice.js** – Voice module: Web Speech API (recognition + synthesis), start/stop listening, speak pharaoh response, callbacks for transcript and speak-end.
- **talk_to_pharos.js** – App: API calls, personas, UI, and wiring: mic → listen → final transcript → `POST /converse` → speak answer → auto-restart listening.

Flow: user clicks mic → continuous listening → on pause, user speech is sent to `/converse` → pharaoh reply is spoken → listening restarts. Text input still works; pharaoh replies are also spoken when sent from the text box.