import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.config import get_settings
from app.models import (
    ActionItem,
    MeetingAnalysis,
    ProcessingResult,
    Segment,
    SummarizeRequest,
    TranscriptionResult,
)
from app.services.summarization import analyze_meeting
from app.services.transcription import transcribe_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["meetings"])

_ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus",
    ".mp4", ".webm", ".mov", ".avi", ".mkv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_file(file: UploadFile) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )


async def _save_upload(file: UploadFile, max_bytes: int) -> str:
    """Stream upload directly to a temp file — never loads the whole file into RAM."""
    ext = os.path.splitext(file.filename or "")[1].lower() or ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    total = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, f"File exceeds {max_bytes // (1024 * 1024)} MB limit.")
                f.write(chunk)
    except HTTPException:
        os.unlink(tmp_path)
        raise
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    if total == 0:
        os.unlink(tmp_path)
        raise HTTPException(400, "Uploaded file is empty.")

    return tmp_path


def _build_segments(raw: list) -> list[Segment]:
    return [
        Segment(
            start=float(s.get("start", 0)),
            end=float(s.get("end", 0)),
            text=str(s.get("text", "")),
        )
        for s in (raw or [])
    ]


def _parse_action_items(raw: list) -> list[ActionItem]:
    items = []
    for a in raw or []:
        priority = a.get("priority", "Medium")
        if priority not in ("High", "Medium", "Low"):
            priority = "Medium"
        items.append(
            ActionItem(
                item=str(a.get("item", "")).strip(),
                priority=priority,
                assignee=a.get("assignee") or None,
            )
        )
    return items


def _parse_analysis(data: dict) -> dict:
    return {
        "title": str(data.get("title") or "Meeting Summary"),
        "key_points": [str(p) for p in (data.get("key_points") or [])],
        "summary": str(data.get("summary") or ""),
        "decisions": [str(d) for d in (data.get("decisions") or [])],
        "action_items": _parse_action_items(data.get("action_items") or []),
    }


def _render_markdown(analysis: dict, transcript: str, language: str, duration: float) -> str:
    mins, secs = divmod(int(duration), 60)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    lines = [
        f"# {analysis['title']}",
        "",
        f"**Language:** {language.upper()}  |  **Duration:** {duration_str}",
        "",
        "---",
        "",
        "## Key Points",
        "",
    ]
    for point in analysis.get("key_points", []):
        lines.append(f"- {point}")

    lines.extend(["", "## Summary", "", analysis.get("summary", ""), ""])

    if analysis.get("decisions"):
        lines.extend(["## Decisions", ""])
        for decision in analysis["decisions"]:
            lines.append(f"- {decision}")
        lines.append("")

    if analysis.get("action_items"):
        lines.extend(["## Action Items", ""])
        for item in analysis["action_items"]:
            assignee = f" *(Owner: {item.assignee})*" if item.assignee else ""
            lines.append(f"- `[{item.priority}]` {item.item}{assignee}")
        lines.append("")

    lines.extend(["---", "", "## Full Transcript", "", transcript])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/transcribe",
    response_model=TranscriptionResult,
    summary="Transcribe audio or video to text",
)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResult:
    _validate_file(file)
    settings = get_settings()
    tmp_path = await _save_upload(file, settings.max_file_size_mb * 1024 * 1024)

    logger.info("Transcribing: %s", file.filename)
    try:
        result = await transcribe_file(tmp_path, settings.whisper_model_size)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return TranscriptionResult(
        transcript=result["text"],
        segments=_build_segments(result["segments"]),
        language=result["language"],
        duration=result["duration"],
    )


@router.post(
    "/analyze",
    response_model=MeetingAnalysis,
    summary="Analyze a transcript with AI (title, summary, actions, decisions)",
)
async def analyze(request: SummarizeRequest) -> MeetingAnalysis:
    settings = get_settings()
    logger.info("Analyzing transcript (%d chars)", len(request.transcript))

    try:
        data = await analyze_meeting(request.transcript, settings.ollama_url, settings.ollama_model)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("AI analysis failed")
        raise HTTPException(500, f"AI analysis failed: {exc}") from exc

    return MeetingAnalysis(**_parse_analysis(data))


@router.post(
    "/process",
    response_model=ProcessingResult,
    summary="Full pipeline: transcribe audio/video then analyze with AI",
)
async def process(file: UploadFile = File(...)) -> ProcessingResult:
    _validate_file(file)
    settings = get_settings()
    tmp_path = await _save_upload(file, settings.max_file_size_mb * 1024 * 1024)

    logger.info("Processing: %s", file.filename)
    try:
        tr_result = await transcribe_file(tmp_path, settings.whisper_model_size)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    transcript = tr_result["text"]

    try:
        analysis_data = await analyze_meeting(transcript, settings.ollama_url, settings.ollama_model)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("AI analysis failed")
        raise HTTPException(500, f"AI analysis failed: {exc}") from exc

    return ProcessingResult(
        transcript=transcript,
        segments=_build_segments(tr_result["segments"]),
        language=tr_result["language"],
        duration=tr_result["duration"],
        **_parse_analysis(analysis_data),
    )


@router.post(
    "/export/markdown",
    response_class=PlainTextResponse,
    summary="Export full meeting notes as a Markdown file",
)
async def export_markdown(file: UploadFile = File(...)) -> PlainTextResponse:
    _validate_file(file)
    settings = get_settings()
    tmp_path = await _save_upload(file, settings.max_file_size_mb * 1024 * 1024)

    try:
        tr_result = await transcribe_file(tmp_path, settings.whisper_model_size)
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    try:
        analysis_data = await analyze_meeting(
            tr_result["text"], settings.ollama_url, settings.ollama_model
        )
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"AI analysis failed: {exc}") from exc

    analysis = _parse_analysis(analysis_data)
    md = _render_markdown(analysis, tr_result["text"], tr_result["language"], tr_result["duration"])

    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in analysis["title"])
    filename = safe_title.strip().replace(" ", "-").lower() or "meeting-notes"

    return PlainTextResponse(
        md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}.md"'},
    )
