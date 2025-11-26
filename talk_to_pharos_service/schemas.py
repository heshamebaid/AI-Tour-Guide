from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ConversationTurn(BaseModel):
    speaker: str = Field(..., description="Either 'user' or 'pharaoh'")
    content: str = Field(..., min_length=1, description="Natural language message")

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"user", "pharaoh"}:
            raise ValueError("speaker must be 'user' or 'pharaoh'")
        return normalized


class ConverseRequest(BaseModel):
    pharaoh_id: str = Field(..., description="Persona identifier, e.g., 'ramses-ii'")
    user_query: str = Field(..., min_length=1, description="Latest traveler utterance")
    history: List[ConversationTurn] = Field(default_factory=list)
    top_k: Optional[int] = Field(
        default=2, ge=1, le=5, description="Number of context chunks for retrieval"
    )


class PersonaResponse(BaseModel):
    id: str
    display_name: str
    throne_name: Optional[str]
    era: str
    short_bio: str
    personality: str
    speech_style: str
    virtues: List[str]
    guardrails: List[str]
    sample_questions: List[str]
    avatar_asset: Optional[str]
    voice_hint: Optional[str]


class ConverseResponse(BaseModel):
    pharaoh_id: str
    answer: str
    persona: PersonaResponse
    used_history_turns: int

