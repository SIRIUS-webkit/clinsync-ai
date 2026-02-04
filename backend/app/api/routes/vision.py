"""Vision endpoints for MedGemma."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.services.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/analyze")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
) -> dict:
    """Analyze a medical image and return findings."""
    orchestrator: AIOrchestrator = request.app.state.ai_orchestrator
    try:
        image_bytes = await image.read()
        return await orchestrator.process_request(
            text=prompt,
            image_bytes=image_bytes,
            audio_bytes=None,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Vision analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to analyze image.") from exc
