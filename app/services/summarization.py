import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. "
    "Extract structured, actionable information from meeting transcripts with precision. "
    "Always respond with valid JSON only."
)

_ANALYSIS_PROMPT = """\
Analyze the meeting transcript below and return a single JSON object.

Required fields:
{
  "title": "<concise meeting title, max 10 words>",
  "key_points": ["<takeaway 1>", "...", "<5 to 8 total>"],
  "summary": "<2-3 paragraph narrative: what was discussed, key context, why it matters>",
  "decisions": ["<explicit decision made>"],
  "action_items": [
    {
      "item": "<specific, concrete task — not a vague observation>",
      "priority": "High | Medium | Low",
      "assignee": "<person responsible, or null if not mentioned>"
    }
  ]
}

Priority guide:
- High: urgent deadlines, blockers, critical path items
- Medium: important but not time-sensitive
- Low: nice-to-have, longer-term

Rules:
- Return ONLY the JSON object — no markdown, no explanation, no code fences
- decisions: use empty array [] if no explicit decisions were made
- action_items: only include tasks someone actually needs to do
- If the transcript is in a non-English language, still return the JSON in English

Transcript:
{transcript}"""

_MAX_TRANSCRIPT_CHARS = 80_000


async def analyze_meeting(transcript: str, ollama_url: str, model: str) -> dict[str, Any]:
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        logger.warning("Transcript truncated to %d chars", _MAX_TRANSCRIPT_CHARS)
        transcript = transcript[:_MAX_TRANSCRIPT_CHARS] + "\n[Transcript truncated due to length]"

    # Use replace instead of .format() so curly braces in the transcript don't crash
    user_prompt = _ANALYSIS_PROMPT.replace("{transcript}", transcript)

    client = AsyncOpenAI(api_key="ollama", base_url=ollama_url, timeout=120.0)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as exc:
        if "connection" in str(exc).lower() or "refused" in str(exc).lower():
            raise ValueError(
                f"Cannot reach Ollama at {ollama_url}. "
                "Make sure Ollama is running: open the Ollama app or run 'ollama serve'."
            ) from exc
        raise

    raw = (response.choices[0].message.content or "{}").strip()

    # Strip markdown code fences if the model wrapped the JSON (common with local LLMs)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Model returned invalid JSON: %s", raw[:500])
        raise ValueError(f"AI returned malformed JSON: {exc}") from exc
