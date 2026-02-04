"""Application configuration with .env support."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for ClinSync AI backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_env: str = "production"
    version: str = "1.0.0"
    cors_allow_origins: List[str] = ["*"]

    medgemma_model_id: str = "google/medgemma-4b-it"
    medasr_model_id: str = "google/medasr"
    hear_model_id: str = "google/hear"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"
    ollama_timeout_seconds: int = 30

    device: str = "cuda"
    max_workers: int = 4
    enable_quantization: bool = True

    redis_url: str = "redis://localhost:6379/0"
    redis_queue_name: str = "clinsync:queue"

    webrtc_room_ttl_seconds: int = 3600
    webrtc_max_rooms: int = 20
    webrtc_frame_interval_seconds: int = 5

    audio_sample_rate: int = 16000
    asr_chunk_seconds: float = 5.0

    anonymization_salt: str = "clinsync-ai"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
