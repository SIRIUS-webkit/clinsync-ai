"""Chat and multimodal endpoints."""
from __future__ import annotations

import base64
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.services.ai_orchestrator import AIOrchestrator
from app.services.webrtc_manager import WebRTCManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class WebRTCOffer(BaseModel):
    """Payload for WebRTC offer/answer."""

    room_id: str
    peer_id: str
    sdp: dict


class WebRTCIceCandidate(BaseModel):
    """Payload for ICE candidates."""

    room_id: str
    peer_id: str
    candidate: dict


@router.post("/")
async def chat(
    request: Request,
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
) -> dict:
    """Handle chat requests with optional image and audio."""
    orchestrator: AIOrchestrator = request.app.state.ai_orchestrator
    image_bytes = await image.read() if image else None
    audio_bytes = await audio.read() if audio else None
    try:
        return await orchestrator.process_request(text=text, image_bytes=image_bytes, audio_bytes=audio_bytes)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Chat processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to process request.") from exc


@router.websocket("/ws")
async def chat_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    orchestrator: AIOrchestrator = websocket.app.state.ai_orchestrator
    try:
        while True:
            payload = await websocket.receive_json()
            text = payload.get("text")
            image_b64 = payload.get("image")
            audio_b64 = payload.get("audio")
            image_bytes = base64.b64decode(image_b64) if image_b64 else None
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None
            response = await orchestrator.process_request(
                text=text,
                image_bytes=image_bytes,
                audio_bytes=audio_bytes,
            )
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Chat WebSocket error: %s", exc)
        await websocket.close(code=1011)


@router.post("/webrtc/offer")
async def webrtc_offer(request: Request, payload: WebRTCOffer) -> dict:
    """Handle WebRTC offer signaling."""
    manager: WebRTCManager = request.app.state.webrtc_manager
    await manager.handle_offer(payload.room_id, payload.peer_id, payload.sdp)
    return {"status": "ok"}


@router.post("/webrtc/answer")
async def webrtc_answer(request: Request, payload: WebRTCOffer) -> dict:
    """Handle WebRTC answer signaling."""
    manager: WebRTCManager = request.app.state.webrtc_manager
    await manager.handle_answer(payload.room_id, payload.peer_id, payload.sdp)
    return {"status": "ok"}


@router.post("/webrtc/ice")
async def webrtc_ice(request: Request, payload: WebRTCIceCandidate) -> dict:
    """Handle ICE candidate signaling."""
    manager: WebRTCManager = request.app.state.webrtc_manager
    await manager.handle_ice_candidate(payload.room_id, payload.peer_id, payload.candidate)
    return {"status": "ok"}


@router.get("/webrtc/rooms/{room_id}")
async def get_room_state(request: Request, room_id: str) -> dict:
    """Fetch current room state for debugging."""
    manager: WebRTCManager = request.app.state.webrtc_manager
    room = await manager.get_room_state(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found.")
    return {
        "room_id": room.room_id,
        "peers": list(room.peers),
        "offers": room.offers,
        "answers": room.answers,
        "ice_candidates": room.ice_candidates,
    }
