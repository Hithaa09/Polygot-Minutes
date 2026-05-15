import asyncio
import logging
from typing import Any

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_model_cache: dict[str, WhisperModel] = {}


def load_model(size: str) -> WhisperModel:
    if size in _model_cache:
        return _model_cache[size]
    logger.info("Loading faster-whisper '%s' (device: auto) ...", size)
    model = WhisperModel(size, device="auto", compute_type="auto")
    _model_cache[size] = model
    logger.info("faster-whisper '%s' ready.", size)
    return model


def _transcribe_sync(model: WhisperModel, path: str) -> dict[str, Any]:
    segments_gen, info = model.transcribe(path, beam_size=5)
    segments: list[dict] = []
    texts: list[str] = []
    for seg in segments_gen:
        text = seg.text.strip()
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        texts.append(text)
    return {
        "text": " ".join(texts),
        "segments": segments,
        "language": info.language,
        "duration": round(info.duration, 2),
    }


async def transcribe_file(path: str, model_size: str) -> dict[str, Any]:
    model = load_model(model_size)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_sync, model, path)
