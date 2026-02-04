"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
def health_check() -> dict:
    """Return basic health information."""
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.app_env,
        "version": settings.version,
    }
