import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. "
    "Extract structured, actionable information from meeting transcripts with precision."
)

_ANALYSIS_PROMPT = """\
Analyze the following meeting transcript and return a JSON object with these exact fields:

- title: string — concise, descriptive meeting title (≤10 words)
- key_points: array of 5–8 strings — the most important takeaways
- summary: string — a 2–3 paragraph narrative summary covering what was discussed and why it matters
- decisions: array of strings — explicit decisions made during the meeting (empty array if none)
- action_items: array of objects, each with:
    - item: string — a specific, concrete actionable task
    - priority: "High" | "Medium" | "Low"
    - assignee: string | null — person responsible if mentioned, otherwise null

Rules:
- Return ONLY valid JSON, no markdown code blocks or extra text
- action_items must be genuinely actionable (not vague observations)
- If the transcript is short or unclear, still do your best

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
