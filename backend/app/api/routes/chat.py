"""Chat and multimodal endpoints with streaming support."""

from __future__ import annotations

import base64
import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.ai_orchestrator import AIOrchestrator, get_orchestrator
from app.services.webrtc_manager import WebRTCManager
from app.core.config import get_settings
from app.services.firebase_service import save_chat_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """Chat message structure."""

    text: Optional[str] = None
    image_url: Optional[str] = None  # base64 data URL or external URL
    audio_url: Optional[str] = None  # base64 data URL
    patient_id: Optional[str] = None
    consultation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Structured response for chat UI."""

    message_id: str
    timestamp: str
    response_text: str
    triage_level: str  # LOW | MODERATE | HIGH | CRITICAL
    triage_color: str  # green | yellow | orange | red
    confidence: float  # 0.0 - 1.0
    findings: list[dict]  # For findings list with percentages
    differential_diagnosis: list[dict]  # For differential list
    recommended_actions: list[dict]  # With priority badges
    is_streaming: bool = False
    raw_analysis: Optional[dict] = None  # Full data for debugging


class WebRTCOffer(BaseModel):
    room_id: str
    peer_id: str
    sdp: dict


class WebRTCIceCandidate(BaseModel):
    room_id: str
    peer_id: str
    candidate: dict


def generate_message_id() -> str:
    """Generate unique message ID."""
    import uuid

    return str(uuid.uuid4())[:8]


def format_findings_for_ui(raw_findings: list) -> list[dict]:
    """Format findings for frontend display with confidence bars."""
    formatted = []
    for i, finding in enumerate(raw_findings[:5]):  # Max 5 findings
        if isinstance(finding, str):
            # Extract confidence if embedded in text
            import re

            confidence = 0.7  # default
            match = re.search(r"(\d+)%", finding)
            if match:
                confidence = int(match.group(1)) / 100

            formatted.append(
                {
                    "id": f"finding_{i}",
                    "label": finding.replace(f" ({int(confidence*100)}%)", "").strip(),
                    "confidence": int(confidence * 100),
                    "confidence_label": f"{int(confidence * 100)}%",
                }
            )
        elif isinstance(finding, dict):
            formatted.append(
                {
                    "id": finding.get("id", f"finding_{i}"),
                    "label": finding.get("label", finding.get("text", "Unknown")),
                    "confidence": int(finding.get("confidence", 0.7) * 100),
                    "confidence_label": f"{int(finding.get('confidence', 0.7) * 100)}%",
                }
            )
    return formatted


def format_differential_for_ui(raw_data: dict) -> list[dict]:
    """Format differential diagnosis for frontend."""
    differential = []

    # Extract from various possible formats
    diagnoses = []
    if "differential_diagnoses" in raw_data:
        diagnoses = raw_data["differential_diagnoses"]
    elif "differential" in raw_data:
        diagnoses = raw_data["differential"]
    elif "diagnoses" in raw_data:
        diagnoses = raw_data["diagnoses"]

    for i, dx in enumerate(diagnoses[:5]):
        if isinstance(dx, dict):
            differential.append(
                {
                    "id": f"dx_{i}",
                    "condition": dx.get("condition", dx.get("name", "Unknown")),
                    "probability": int(dx.get("probability", 0.5) * 100),
                    "probability_label": f"{int(dx.get('probability', 0.5) * 100)}%",
                }
            )
        elif isinstance(dx, str):
            # Parse "Condition (48%)" format
            import re

            match = re.match(r"(.+?)\s*\((\d+)%\)", dx)
            if match:
                condition, prob = match.groups()
                differential.append(
                    {
                        "id": f"dx_{i}",
                        "condition": condition.strip(),
                        "probability": int(prob),
                        "probability_label": f"{prob}%",
                    }
                )
            else:
                differential.append(
                    {
                        "id": f"dx_{i}",
                        "condition": dx,
                        "probability": 50,
                        "probability_label": "50%",
                    }
                )

    # Fallback if no differential found
    if not differential and "findings" in raw_data:
        # Create generic differential from findings
        differential = [
            {
                "id": "dx_0",
                "condition": "Further evaluation needed",
                "probability": 100,
                "probability_label": "100%",
            }
        ]

    return differential


def format_actions_for_ui(raw_data: dict, response_text: str) -> list[dict]:
    """Format recommended actions with priority badges."""
    actions = []

    # Extract from image analysis recommendations
    if isinstance(raw_data.get("image"), dict):
        img_recs = raw_data["image"].get("recommendations", [])
        for i, rec in enumerate(img_recs[:3]):
            priority = "medium"
            if "urgent" in rec.lower() or "immediate" in rec.lower():
                priority = "high"
            elif "routine" in rec.lower() or "follow-up" in rec.lower():
                priority = "low"

            actions.append(
                {
                    "id": f"action_{i}",
                    "text": rec,
                    "priority": priority,  # high | medium | low
                    "priority_label": priority.capitalize(),
                    "priority_color": {
                        "high": "red",
                        "medium": "yellow",
                        "low": "green",
                    }[priority],
                }
            )

    # Add from synthesized response if no explicit recommendations
    if not actions and response_text:
        # Parse common recommendation patterns
        import re

        rec_patterns = re.findall(
            r"(?:Recommend|Suggested|Should|Consider)[:\s]+([^.\n]+)",
            response_text,
            re.IGNORECASE,
        )
        for i, rec in enumerate(rec_patterns[:3]):
            actions.append(
                {
                    "id": f"action_{i}",
                    "text": rec.strip(),
                    "priority": "medium",
                    "priority_label": "Medium",
                    "priority_color": "yellow",
                }
            )

    return (
        actions
        if actions
        else [
            {
                "id": "action_0",
                "text": "Consult healthcare provider for definitive diagnosis",
                "priority": "medium",
                "priority_label": "Medium",
                "priority_color": "yellow",
            }
        ]
    )


def determine_triage_color(triage_level: str) -> str:
    """Map triage level to color."""
    colors = {
        "LOW": "green",
        "MODERATE": "yellow",
        "HIGH": "orange",
        "CRITICAL": "red",
        "URGENT": "red",
    }
    return colors.get(triage_level.upper(), "gray")


@router.post("/", response_model=ChatResponse)
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    patient_id: Optional[str] = Form(default=None),
    consultation_id: Optional[str] = Form(default=None),
    mode: Optional[str] = Form(default="chat"),  # "chat" or "voice"
) -> ChatResponse:
    """
    Handle chat requests with optional image and audio.
    Returns structured response for UI display.

    Args:
        mode: "chat" for structured clinical response, "voice" for conversational TTS-friendly response
    """
    settings = get_settings()
    orchestrator = await get_orchestrator(settings)

    image_bytes = await image.read() if image else None
    audio_bytes = await audio.read() if audio else None

    try:
        result = await orchestrator.process_request(
            text=text,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            mode=mode or "chat",
        )

        # Format for UI
        triage_level = result.get("triage_level", "LOW")

        # Determine patient_id if not key
        actual_patient_id = patient_id or f"anon_{generate_message_id()}"

        # Prepare data for saving
        message_data = {
            "text": text,
            "has_image": image is not None,
            "has_audio": audio is not None,
            "consultation_id": consultation_id,
        }

        response_model = ChatResponse(
            message_id=generate_message_id(),
            timestamp=__import__("datetime").datetime.now().isoformat(),
            response_text=result.get("response", "Analysis complete."),
            triage_level=triage_level,
            triage_color=determine_triage_color(triage_level),
            confidence=result.get("confidence", 0.5),
            findings=format_findings_for_ui(result.get("findings", [])),
            differential_diagnosis=format_differential_for_ui(
                result.get("raw_results", {})
            ),
            recommended_actions=format_actions_for_ui(
                result.get("raw_results", {}), result.get("response", "")
            ),
            is_streaming=False,
            raw_analysis=result if getattr(settings, "debug", False) else None,
        )

        # Save to Firebase in background
        background_tasks.add_task(
            save_chat_message,
            patient_id=actual_patient_id,
            message_data=message_data,
            response_data=response_model.model_dump(),
        )

        return response_model

    except Exception as exc:
        logger.exception("Chat processing failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Unable to process request."
        ) from exc


@router.post("/stream")
async def chat_stream(
    request: Request,
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
) -> StreamingResponse:
    """
    Streaming response for real-time chat.
    Sends partial updates as AI processes.
    """
    settings = get_settings()
    orchestrator = await get_orchestrator(settings)

    image_bytes = await image.read() if image else None
    audio_bytes = await audio.read() if audio else None

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        message_id = generate_message_id()

        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing...'})}\n\n"

        try:
            # Simulate streaming by sending progress updates
            # In production, this would hook into actual model tokens

            result = await orchestrator.process_request(
                text=text, image_bytes=image_bytes, audio_bytes=audio_bytes
            )

            # Send final structured response
            triage_level = result.get("triage_level", "LOW")
            response = ChatResponse(
                message_id=message_id,
                timestamp=__import__("datetime").datetime.now().isoformat(),
                response_text=result.get("response", ""),
                triage_level=triage_level,
                triage_color=determine_triage_color(triage_level),
                confidence=result.get("confidence", 0.5),
                findings=format_findings_for_ui(result.get("findings", [])),
                differential_diagnosis=format_differential_for_ui(
                    result.get("raw_results", {})
                ),
                recommended_actions=format_actions_for_ui(
                    result.get("raw_results", {}), result.get("response", "")
                ),
                is_streaming=False,
            )

            yield f"data: {json.dumps({'type': 'complete', 'data': response.model_dump()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/ws")
async def chat_ws(websocket: WebSocket) -> None:
    """
    WebSocket for real-time bidirectional chat.
    Supports: text, image (base64), audio (base64)
    """
    await websocket.accept()

    settings = get_settings()
    orchestrator = await get_orchestrator(settings)

    try:
        while True:
            # Receive message from frontend
            payload = await websocket.receive_json()

            # Handle non-chat control messages (ping / pong / heartbeat)
            msg_type = payload.get("type")
            if msg_type in ("ping", "pong", "heartbeat"):
                await websocket.send_json({"type": "pong"})
                continue

            text = payload.get("text")
            image_b64 = payload.get("image")
            audio_b64 = payload.get("audio")
            patient_id = payload.get("patient_id")
            consultation_id = payload.get("consultation_id")

            # Skip empty messages (no text, image, or audio)
            if not text and not image_b64 and not audio_b64:
                logger.debug("Skipping empty chat payload: %s", payload)
                continue

            # Decode base64 media (support data URL format: data:image/jpeg;base64,... or raw b64)
            def decode_b64(value: Optional[str]):
                if not value:
                    return None
                if "," in value:
                    value = value.split(",", 1)[1]
                return base64.b64decode(value)

            image_bytes = decode_b64(image_b64)
            audio_bytes = decode_b64(audio_b64)

            # Send acknowledgment
            await websocket.send_json(
                {
                    "type": "status",
                    "message": "Processing...",
                    "message_id": generate_message_id(),
                }
            )

            # Process request
            result = await orchestrator.process_request(
                text=text, image_bytes=image_bytes, audio_bytes=audio_bytes
            )

            # Format and send structured response
            triage_level = result.get("triage_level", "LOW")
            response = ChatResponse(
                message_id=generate_message_id(),
                timestamp=__import__("datetime").datetime.now().isoformat(),
                response_text=result.get("response", ""),
                triage_level=triage_level,
                triage_color=determine_triage_color(triage_level),
                confidence=result.get("confidence", 0.5),
                findings=format_findings_for_ui(result.get("findings", [])),
                differential_diagnosis=format_differential_for_ui(
                    result.get("raw_results", {})
                ),
                recommended_actions=format_actions_for_ui(
                    result.get("raw_results", {}), result.get("response", "")
                ),
                is_streaming=False,
                raw_analysis=result if getattr(settings, "debug", False) else None,
            )

            await websocket.send_json(
                {"type": "response", "data": response.model_dump()}
            )

            # Save to Firebase (fire and forget via threadpool)
            actual_patient_id = patient_id or f"anon_{generate_message_id()}"
            message_data = {
                "text": text,
                "has_image": image_b64 is not None,
                "has_audio": audio_b64 is not None,
                "consultation_id": consultation_id,
            }
            from fastapi.concurrency import run_in_threadpool

            await run_in_threadpool(
                save_chat_message,
                patient_id=actual_patient_id,
                message_data=message_data,
                response_data=response.model_dump(),
            )

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception as exc:
        logger.exception("Chat WebSocket error: %s", exc)
        await websocket.send_json(
            {"type": "error", "message": "Processing failed. Please try again."}
        )
        await websocket.close(code=1011)


@router.websocket("/ws/stream")
async def chat_ws_stream(websocket: WebSocket) -> None:
    """
    WebSocket with streaming tokens for typing effect.
    Sends partial response as it's generated.
    """
    await websocket.accept()

    settings = get_settings()
    orchestrator = await get_orchestrator(settings)

    try:
        while True:
            payload = await websocket.receive_json()

            text = payload.get("text")
            image_b64 = payload.get("image")

            image_bytes = (
                base64.b64decode(image_b64.split(",")[1])
                if image_b64 and "," in image_b64
                else None
            )

            # Start processing
            await websocket.send_json(
                {"type": "start", "message_id": generate_message_id()}
            )

            # Simulate streaming (in production, hook into model.generate with callbacks)
            result = await orchestrator.process_request(
                text=text, image_bytes=image_bytes, audio_bytes=None
            )

            # Stream response word by word
            response_text = result.get("response", "")
            words = response_text.split()

            for i, word in enumerate(words):
                await websocket.send_json(
                    {
                        "type": "token",
                        "token": word + " ",
                        "index": i,
                        "is_complete": i == len(words) - 1,
                    }
                )
                await asyncio.sleep(0.05)  # Typing effect

            # Send final structured data
            await websocket.send_json(
                {
                    "type": "complete",
                    "data": {
                        "triage_level": result.get("triage_level", "LOW"),
                        "confidence": result.get("confidence", 0.5),
                        "findings": format_findings_for_ui(result.get("findings", [])),
                    },
                }
            )

    except WebSocketDisconnect:
        logger.info("Streaming WebSocket disconnected")
    except Exception as exc:
        logger.exception("Streaming WebSocket error: %s", exc)
        await websocket.close(code=1011)


# WebRTC endpoints remain unchanged...
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
    await manager.handle_ice_candidate(
        payload.room_id, payload.peer_id, payload.candidate
    )
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
    }


# ============================================================================
# SOAP NOTE GENERATION ENDPOINTS
# ============================================================================


class SOAPNoteRequest(BaseModel):
    """Request model for SOAP note generation."""

    patient_id: str
    consultation_type: str = "chat"  # "chat" | "video" | "general"
    consultation_data: dict  # Full consultation data


class SOAPNoteResponse(BaseModel):
    """Response model for SOAP note."""

    success: bool
    note_id: str
    soap_note: dict
    text_format: str  # Human-readable text version


@router.post("/soap/generate", response_model=SOAPNoteResponse)
async def generate_soap_note(request: SOAPNoteRequest) -> SOAPNoteResponse:
    """
    Generate a SOAP note from consultation data.

    This endpoint creates a structured clinical note following the SOAP format:
    - Subjective: Patient-reported symptoms
    - Objective: Clinical observations and test results
    - Assessment: Diagnosis and interpretation
    - Plan: Recommendations and follow-up
    """
    from app.services.soap_generator import get_soap_generator

    try:
        generator = get_soap_generator()
        soap_note = generator.generate_soap_note(
            patient_id=request.patient_id,
            consultation_data=request.consultation_data,
            consultation_type=request.consultation_type,
        )
        text_format = generator.to_text(soap_note)

        return SOAPNoteResponse(
            success=True,
            note_id=soap_note["note_id"],
            soap_note=soap_note,
            text_format=text_format,
        )
    except Exception as e:
        logger.error("SOAP note generation failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate SOAP note: {str(e)}"
        )


@router.get("/soap/download/{note_id}")
async def download_soap_note(
    note_id: str,
    format: str = "text",  # "text" | "json"
    patient_id: str = "anonymous",
    consultation_data: Optional[str] = None,  # JSON string
) -> StreamingResponse:
    """
    Download a SOAP note in the specified format.

    For demo purposes, this regenerates the note from provided data.
    In production, you would store and retrieve notes from a database.
    """
    from app.services.soap_generator import get_soap_generator
    import io

    try:
        generator = get_soap_generator()

        # Parse consultation data if provided
        data = {}
        if consultation_data:
            data = json.loads(consultation_data)

        # Generate SOAP note
        soap_note = generator.generate_soap_note(
            patient_id=patient_id,
            consultation_data=data,
            consultation_type="general",
        )

        if format == "json":
            content = generator.to_json(soap_note, pretty=True)
            media_type = "application/json"
            filename = f"{note_id}.json"
        else:
            content = generator.to_text(soap_note)
            media_type = "text/plain"
            filename = f"{note_id}.txt"

        # Create streaming response
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Note-ID": note_id,
            },
        )
    except Exception as e:
        logger.error("SOAP note download failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to download SOAP note: {str(e)}"
        )


@router.post("/soap/download")
async def download_soap_note_post(
    request: Request,
    patient_id: str = Form("anonymous"),
    consultation_type: str = Form("chat"),
    response_text: str = Form(""),
    findings: str = Form("[]"),  # JSON array
    recommendations: str = Form("[]"),  # JSON array
    triage_level: str = Form("LOW"),
    transcript: str = Form(""),
    format: str = Form("text"),
) -> StreamingResponse:
    """
    Generate and download a SOAP note from form data.

    This is the main endpoint for downloading consultation summaries.
    """
    from app.services.soap_generator import get_soap_generator
    import io

    try:
        generator = get_soap_generator()

        # Parse JSON fields
        try:
            findings_list = json.loads(findings) if findings else []
        except json.JSONDecodeError:
            findings_list = []

        try:
            recommendations_list = (
                json.loads(recommendations) if recommendations else []
            )
        except json.JSONDecodeError:
            recommendations_list = []

        # Build consultation data
        consultation_data = {
            "response_text": response_text,
            "response": response_text,
            "findings": findings_list,
            "recommendations": recommendations_list,
            "triage_level": triage_level,
            "transcript": transcript,
            "user_input": transcript,
            "models_used": ["medgemma"],
        }

        # Generate SOAP note
        soap_note = generator.generate_soap_note(
            patient_id=patient_id,
            consultation_data=consultation_data,
            consultation_type=consultation_type,
        )

        note_id = soap_note["note_id"]

        if format == "json":
            content = generator.to_json(soap_note, pretty=True)
            media_type = "application/json"
            filename = f"{note_id}.json"
        else:
            content = generator.to_text(soap_note)
            media_type = "text/plain"
            filename = f"{note_id}.txt"

        logger.info("Generated downloadable SOAP note: %s", note_id)

        # Create streaming response
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Note-ID": note_id,
            },
        )
    except Exception as e:
        logger.error("SOAP note download failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate SOAP note: {str(e)}"
        )
