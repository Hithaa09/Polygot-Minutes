# Entry point kept for uvicorn CLI compatibility: `uvicorn main:app`
# Prefer running via: python run.py
from app.main import app

__all__ = ["app"]
