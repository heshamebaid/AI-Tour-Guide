from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from .persona_registry import PharaohPersona


def _format_history(persona: PharaohPersona, history: Sequence[Mapping[str, str]]) -> str:
    """
    Convert conversation history objects into a compact textual transcript.

    Each history item must contain `speaker` ("user" or "pharaoh") and `content`.
    """

    if not history:
        return "No previous conversation has been recorded."

    rendered_turns: List[str] = []
    for turn in history:
        if isinstance(turn, Mapping):
            speaker_val = turn.get("speaker", "")
            content_val = turn.get("content", "")
        else:
            speaker_val = getattr(turn, "speaker", "")
            content_val = getattr(turn, "content", "")

        speaker = (speaker_val or "").strip().lower()
        raw_text = (content_val or "").strip()
        if not raw_text:
            continue

        label = "Traveler"
        if speaker in {"pharaoh", "assistant"}:
            label = persona.display_name
        rendered_turns.append(f"{label}: {raw_text}")

    if not rendered_turns:
        return "No previous conversation has been recorded."
    return "\n".join(rendered_turns)


def build_persona_prompt(
    persona: PharaohPersona,
    user_query: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """
    Create a persona-aware user prompt for the underlying RAG query.

    Since `rag_query` only accepts a plain string question, we embed persona
    directives, tone, and recent history into a single textual payload.
    """

    safe_query = (user_query or "").strip()
    if not safe_query:
        raise ValueError("User query must be provided to build a persona prompt.")

    persona_intro = (
        f"You are {persona.display_name} (throne name: {persona.throne_name or 'unknown'}), "
        f"ruling during {persona.era}. {persona.short_bio} "
        f"Personality traits: {persona.personality}. Speech style: {persona.speech_style}."
    )

    virtues = ", ".join(persona.virtues) if persona.virtues else "Honor Ma'at and truth."
    guardrails = "; ".join(persona.guardrails) if persona.guardrails else ""

    history_text = _format_history(persona, history or [])

    prompt_sections: Iterable[str] = [
        persona_intro,
        f"Guiding virtues: {virtues}.",
        f"Behavior guardrails: {guardrails or 'Remain historically grounded and respectful.'}",
        "Previous conversation:",
        history_text,
        "Answer as the Pharaoh in first-person singular, referencing authentic historical details from the provided context when possible.",
        f"Traveler's latest question: {safe_query}",
    ]

    return "\n\n".join(section for section in prompt_sections if section.strip())

