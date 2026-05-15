from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large"] = "small"
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"
    max_file_size_mb: int = 200
    host: str = "0.0.0.0"
    port: int = 8001

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
