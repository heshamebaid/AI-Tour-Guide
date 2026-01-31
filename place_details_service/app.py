"""
Place Details Service — standalone API for location → visitor-friendly info via OpenRouter LLM.
Accepts lat/lng (reverse-geocoded via Nominatim) or place_name; returns LLM-generated sections as JSON.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Single .env at project root (used when run locally; Docker injects env)
_root = Path(__file__).resolve().parent.parent
_load = _root / ".env"
if _load.exists():
    load_dotenv(_load)

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
_raw_model = os.environ.get("OPEN_ROUTER_MODEL") or os.environ.get("LLM_MODEL") or "liquid/lfm-2.5-1.2b-thinking:free"
OPEN_ROUTER_MODEL = "liquid/lfm-2.5-1.2b-thinking:free" if _raw_model and "qwen" in _raw_model.lower() else _raw_model
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Section keys returned by the LLM (same order as frontend)
SECTION_KEYS = ["intro", "significance", "what_to_see", "practical_tips", "ancient_connection"]

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
    details: str | None = None  # legacy fallback
    sections: dict[str, str] | None = None  # intro, significance, what_to_see, practical_tips, ancient_connection
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


# Generic closing/invitation phrases to strip from section content (case-insensitive)
_PHRASES_TO_STRIP = [
    r"If you're looking for specific sites or landmarks,?\s*please (?:check if you can )?provide more details!?\.?\s*",
    r"If you're interested in (?:this area|specific (?:sites|landmarks)),?\s*consider (?:finding|providing) (?:specific )?(?:sites or )?landmarks[^.]*\.?\s*",
    r"Please (?:check|provide) if you can provide more details!?\.?\s*",
    r"For more (?:details|information),?\s*please (?:check|provide|ask)[^.]*\.?\s*",
    r"Feel free to (?:ask|provide) (?:for )?more (?:details|information)!?\.?\s*",
    r"Let me know if you (?:need|want) (?:more )?(?:details|information)!?\.?\s*",
]


def _sanitize_section_text(text: str) -> str:
    """Remove markdown **, emoji, and generic closing phrases from section content."""
    if not text:
        return text
    # Remove ** (and single * used for bold/italic)
    out = re.sub(r"\*+", "", text)
    # Remove emoji (common ranges)
    out = re.sub(
        r"[\U0001F300-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]",
        "",
        out,
    )
    # Strip generic closing/invitation phrases (case-insensitive)
    for pat in _PHRASES_TO_STRIP:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    # Collapse multiple spaces/newlines at end and trim
    out = re.sub(r"  +", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return out.strip()


# Optional key aliases if LLM returns different key names
_KEY_ALIASES: dict[str, str] = {
    "overview": "intro",
    "what to see": "what_to_see",
    "what_to_see": "what_to_see",
    "practical tips": "practical_tips",
    "practical_tips": "practical_tips",
    "ancient connection": "ancient_connection",
    "ancient_connection": "ancient_connection",
    "significance": "significance",
    "intro": "intro",
}


def _parse_sections_json(raw: str) -> dict[str, str] | None:
    """Extract JSON object from LLM response (strip markdown code fence if present)."""
    text = (raw or "").strip()
    if not text:
        return None
    # Strip optional ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        out = {}
        for k, v in obj.items():
            if not isinstance(v, str) or not v.strip():
                continue
            key_lower = (k or "").strip().lower()
            canonical = _KEY_ALIASES.get(key_lower) or key_lower.replace(" ", "_")
            if canonical in SECTION_KEYS:
                out[canonical] = _sanitize_section_text(v.strip())
        return out if out else None
    except (json.JSONDecodeError, TypeError):
        return None


def _place_details_llm(place_name_or_address: str) -> tuple[dict[str, str] | None, str | None]:
    """
    Call OpenRouter to generate visitor-friendly place details as JSON sections.
    Returns (sections_dict, error_message). If sections_dict is not None, error_message is None.
    """
    if not OPEN_ROUTER_API_KEY:
        return None, "Place details are unavailable: OPEN_ROUTER_API_KEY is not set."

    system = (
        "You are an AI tour guide for visitors. Given a place name or address, respond with "
        "ONLY a valid JSON object (no markdown, no code fence, no other text) with exactly these keys, "
        "each with a string value. Use clear short paragraphs.\n\n"
        "Rules: Do NOT use emojis. Do NOT use markdown (no ** or other formatting). Use plain text only.\n\n"
        "Required keys and content:\n"
        "- intro: (optional) One short introductory paragraph about the place. Omit or empty string if not needed.\n"
        "- significance: Why the place matters; brief history or context.\n"
        "- what_to_see: Main attractions, sights, experiences, local food if relevant.\n"
        "- practical_tips: Transport, language, negotiation, best times, etc.\n"
        "- ancient_connection: Link to ancient Egyptian or regional history; or one sentence like 'N/A' if not relevant.\n\n"
        "If the place is unknown, set significance to a polite message and suggest checking the name or a nearby landmark. "
        "Do not add closing invitations like 'If you're looking for specific sites...' or 'please provide more details'. "
        "End with factual content only."
    )
    user_content = f"Tell me about this place for a visitor. Return only a JSON object with keys: intro, significance, what_to_see, practical_tips, ancient_connection.\n\nPlace: {place_name_or_address[:2000]}"

    # Never send qwen (no free endpoints on OpenRouter)
    model_to_use = OPEN_ROUTER_MODEL
    if model_to_use and "qwen" in (model_to_use or "").lower():
        model_to_use = "liquid/lfm-2.5-1.2b-thinking:free"

    payload: dict[str, Any] = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }
    # Prefer JSON mode when supported (OpenRouter / OpenAI-compatible)
    try:
        payload["response_format"] = {"type": "json_object"}
    except Exception:
        pass

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
        # Retry with Liquid free model if "no endpoints found" (e.g. qwen)
        if not r.ok and model_to_use != "liquid/lfm-2.5-1.2b-thinking:free":
            err_text = (r.text or "").lower()
            try:
                err_body = r.json()
                err_text = (err_body.get("error") or {}).get("message", r.text or "") or err_text
            except Exception:  # noqa: S110
                pass
            if "no endpoints found" in err_text:
                payload["model"] = "liquid/lfm-2.5-1.2b-thinking:free"
                r = requests.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=45,
                )
        # Retry with another free model when daily free limit is reached
        if not r.ok and payload.get("model") != "meta-llama/llama-3.2-3b-instruct:free":
            err_text = (r.text or "").lower()
            try:
                err_body = r.json()
                err_text = (err_body.get("error") or {}).get("message", r.text or "") or err_text
            except Exception:  # pylint: disable=broad-except
                pass
            if r.status_code == 429 or "free-models-per-day" in err_text:
                logger.warning(
                    "OpenRouter daily free limit for %s; retrying with meta-llama/llama-3.2-3b-instruct:free",
                    payload.get("model"),
                )
                payload["model"] = "meta-llama/llama-3.2-3b-instruct:free"
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
            err_msg = r.text or f"HTTP {r.status_code}"
            try:
                err_body = r.json()
                err_msg = (err_body.get("error") or {}).get("message", r.text or "") or err_msg
            except Exception:  # pylint: disable=broad-except
                pass
            if r.status_code == 429 or "free-models-per-day" in (err_msg or "").lower():
                return None, (
                    "The daily free limit for the AI model has been reached. "
                    "Add credits at https://openrouter.ai/credits or try again tomorrow."
                )
            return None, f"Sorry, the guide service returned an error (HTTP {r.status_code})."
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        sections = _parse_sections_json(content)
        if sections:
            return sections, None
        return None, "The guide could not return structured sections. Please try again."
    except requests.exceptions.Timeout:
        return None, "The request timed out. Please try again."
    except Exception as e:
        logger.exception("OpenRouter request failed")
        return None, f"Error: {str(e)}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "place-details"}


@app.post("/details", response_model=PlaceDetailsResponse)
def get_place_details(body: PlaceDetailsRequest) -> PlaceDetailsResponse:
    """
    POST JSON: { "lat", "lng" } or { "place_name" }.
    Returns { "success", "place_name", "sections" } or { "success": false, "error" }.
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
    sections, err = _place_details_llm(place_name)
    if err:
        return PlaceDetailsResponse(success=False, error=err)
    return PlaceDetailsResponse(
        success=True,
        place_name=place_name,
        sections=sections,
    )
