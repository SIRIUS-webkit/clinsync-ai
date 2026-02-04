"""Audio endpoints for MedASR and HeAR."""
from __future__ import annotations

import base64
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from app.services.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/analyze")
async def analyze_audio(
    request: Request,
    audio: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
) -> dict:
    """Analyze audio input for transcription and respiratory signals."""
    orchestrator: AIOrchestrator = request.app.state.ai_orchestrator
    try:
        audio_bytes = await audio.read()
        return await orchestrator.process_request(
            text=prompt,
            image_bytes=None,
            audio_bytes=audio_bytes,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Audio analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to analyze audio.") from exc


@router.websocket("/ws")
async def audio_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming audio."""
    await websocket.accept()
    orchestrator: AIOrchestrator = websocket.app.state.ai_orchestrator
    buffer = bytearray()
    prompt: Optional[str] = None
    try:
        while True:
            payload = await websocket.receive_json()
            chunk = payload.get("chunk")
            prompt = payload.get("prompt", prompt)
            is_final = bool(payload.get("final", False))
            if chunk:
                buffer.extend(base64.b64decode(chunk))
            if is_final and buffer:
                response = await orchestrator.process_request(
                    text=prompt,
                    image_bytes=None,
                    audio_bytes=bytes(buffer),
                )
                await websocket.send_json(response)
                buffer.clear()
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Audio WebSocket error: %s", exc)
        await websocket.close(code=1011)
