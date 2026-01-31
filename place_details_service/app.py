"""
Place Details Service — standalone API for location → visitor-friendly info via OpenRouter LLM.
Accepts lat/lng (reverse-geocoded via Nominatim) or place_name; returns LLM-generated details.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("place_details_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")
OPEN_ROUTER_MODEL = os.environ.get("OPEN_ROUTER_MODEL") or os.environ.get("LLM_MODEL") or "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

app = FastAPI(
    title="Place Details Service",
    description="Returns visitor-friendly place info from location or place name using OpenRouter LLM.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("PLACE_DETAILS_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlaceDetailsRequest(BaseModel):
    lat: float | None = None
    lng: float | None = None
    place_name: str | None = None


class PlaceDetailsResponse(BaseModel):
    success: bool
    place_name: str | None = None
    details: str | None = None
    error: str | None = None


def _reverse_geocode(lat: float, lon: float) -> str:
    """Resolve lat/lon to a human-readable place name using Nominatim (no API key)."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "AI-Tour-Guide-PlaceDetails/1.0 (Educational)"},
            timeout=5,
        )
        if not r.ok:
            return f"Location ({lat:.4f}, {lon:.4f})"
        data = r.json()
        return data.get("display_name") or data.get("name") or f"Location ({lat:.4f}, {lon:.4f})"
    except Exception as e:
        logger.warning("Nominatim request failed: %s", e)
        return f"Location ({lat:.4f}, {lon:.4f})"


def _place_details_llm(place_name_or_address: str) -> str:
    """Call OpenRouter to generate visitor-friendly place details."""
    if not OPEN_ROUTER_API_KEY:
        return "Place details are unavailable: OPEN_ROUTER_API_KEY is not set."
    system = (
        "You are an AI tour guide for visitors. Given a place name or address, provide a concise, "
        "visitor-friendly overview: name, significance, what to see, practical tips, and any "
        "ancient Egyptian or historical connection if relevant. Use clear short paragraphs. "
        "If the place is unknown, say so politely and suggest checking the name or trying a nearby landmark."
    )
    payload: dict[str, Any] = {
        "model": OPEN_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Tell me about this place for a visitor:\n\n{place_name_or_address[:2000]}"},
        ],
    }
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        if not r.ok:
            return f"Sorry, the guide service returned an error (HTTP {r.status_code})."
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or "No description generated."
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except Exception as e:
        logger.exception("OpenRouter request failed")
        return f"Error: {str(e)}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "place-details"}


@app.post("/details", response_model=PlaceDetailsResponse)
def get_place_details(body: PlaceDetailsRequest) -> PlaceDetailsResponse:
    """
    POST JSON: { "lat", "lng" } or { "place_name" }.
    Returns { "success", "place_name", "details" } or { "success": false, "error" }.
    """
    place_name = (body.place_name or "").strip()
    if body.lat is not None and body.lng is not None:
        try:
            lat, lng = float(body.lat), float(body.lng)
        except (TypeError, ValueError):
            return PlaceDetailsResponse(success=False, error="Invalid lat/lng")
        place_name = _reverse_geocode(lat, lng)
    if not place_name:
        return PlaceDetailsResponse(success=False, error="Provide lat/lng or place_name")
    details = _place_details_llm(place_name)
    return PlaceDetailsResponse(
        success=True,
        place_name=place_name,
        details=details,
    )
