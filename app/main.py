import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import HealthResponse
from app.routers.meetings import router
from app.services.transcription import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


async def _check_ollama(ollama_url: str, model: str) -> None:
    parsed = urlparse(ollama_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{base}/api/tags")
        logger.info("Ollama reachable at %s (model: %s)", base, model)
    except Exception:
        logger.warning(
            "Ollama not reachable at %s — open the Ollama app before processing files.", base
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Polyglot Minutes v2 ...")
    load_model(settings.whisper_model_size)
    await _check_ollama(settings.ollama_url, settings.ollama_model)
    logger.info("Ready — listening on %s:%d", settings.host, settings.port)
    yield
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Polyglot Minutes",
        version="2.0.0",
        description=(
            "AI-powered meeting transcription and summarization. "
            "Upload audio or video, get a transcript, key points, decisions, and action items."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok",
            whisper_model=settings.whisper_model_size,
            ollama_model=settings.ollama_model,
            version="2.0.0",
        )

    try:
        app.mount("/", StaticFiles(directory="static", html=True), name="static")
    except RuntimeError:
        logger.warning("No 'static/' directory — frontend not served. Run from project root.")

    return app


app = create_app()
