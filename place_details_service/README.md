# Place Details Service

Standalone API service that returns visitor-friendly place information using:

- **Input:** `lat`/`lng` (reverse-geocoded via Nominatim) or `place_name`
- **Output:** LLM-generated description (OpenRouter) with significance, what to see, and tips

## Endpoints

- `GET /health` — health check
- `POST /details` — body: `{ "lat", "lng" }` or `{ "place_name": "..." }`  
  Response: `{ "success", "place_name", "details" }` or `{ "success": false, "error" }`

## Environment

- `OPEN_ROUTER_API_KEY` — required for LLM responses
- `OPEN_ROUTER_MODEL` — optional (default: `openai/gpt-4o-mini`)
- `PLACE_DETAILS_ALLOWED_ORIGINS` — optional CORS origins (comma-separated)

## Run locally

```bash
pip install -r place_details_service/requirements.txt
export OPEN_ROUTER_API_KEY=your_key
uvicorn place_details_service.app:app --host 0.0.0.0 --port 8060
```

## Docker

Built via `docker/Dockerfile.place_details`; exposed as `place-details-service` on port 8060 in `docker-compose.yml`. The Django app proxies `/api/place-details/` to this service.
