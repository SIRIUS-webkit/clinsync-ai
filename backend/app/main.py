"""FastAPI entrypoint for ClinSync AI."""

from __future__ import annotations

import logging

# Configure logging to show all levels in the console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import audio, chat, health, vision
from app.core.config import get_settings
from app.services.ai_orchestrator import AIOrchestrator
from app.services.redis_queue import RedisQueue
from app.services.webrtc_manager import WebRTCManager
from app.services.firebase_service import initialize_firebase

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="ClinSync AI",
    version=settings.version,
    description="ClinSync AI backend for multimodal medical assistance.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(vision.router)
app.include_router(audio.router)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize lightweight services on startup."""
    logger.info("Starting ClinSync AI backend.")
    app.state.ai_orchestrator = AIOrchestrator(settings=settings)
    app.state.webrtc_manager = WebRTCManager(settings=settings)

    app.state.redis_queue = await RedisQueue.create(settings=settings)

    # Initialize Firebase
    initialize_firebase()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Gracefully shutdown services."""
    queue: RedisQueue | None = getattr(app.state, "redis_queue", None)
    if queue is not None:
        await queue.close()
    logger.info("ClinSync AI backend shutdown complete.")
