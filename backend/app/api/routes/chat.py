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


@router.websocket("/ws/realtime")
async def realtime_voice_ws(websocket: WebSocket) -> None:
    """
    WebSocket for real-time voice conversation like Zoom/Gemini video calls.

    Supports continuous audio streaming with voice activity detection.
    Messages:
        - Client sends: {"type": "audio", "data": base64_audio_chunk}
        - Client sends: {"type": "audio_end"} when speech ends
        - Client sends: {"type": "image", "data": base64_image} for context
        - Server sends: {"type": "listening"} when ready
        - Server sends: {"type": "processing"} when processing speech
        - Server sends: {"type": "speaking", "text": response_text}
        - Server sends: {"type": "complete", "data": full_response}
    """
    await websocket.accept()

    settings = get_settings()
    orchestrator = await get_orchestrator(settings)

    # Session state
    audio_chunks: list[bytes] = []
    current_image_bytes: bytes | None = None
    patient_context: dict | None = None  # Store patient intake data
    context_images: list[bytes] = []  # Images from intake form

    try:
        # Notify client we're ready
        await websocket.send_json({"type": "listening"})

        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            # Handle ping/pong
            if msg_type in ("ping", "pong", "heartbeat"):
                await websocket.send_json({"type": "pong"})
                continue

            # Handle patient context (from intake form)
            if msg_type == "context":
                patient_context = payload.get("data", {})
                logger.info(
                    "Received patient context: %s, %sy, %s - %s",
                    patient_context.get("fullName", "Unknown"),
                    patient_context.get("age", "?"),
                    patient_context.get("gender", "?"),
                    patient_context.get("chiefComplaint", "No complaint")[:50],
                )
                continue

            # Handle image context (captured from video feed or intake form)
            if msg_type == "image":
                image_data = payload.get("data")
                image_type = payload.get("imageType", "live")  # "live" or from intake
                if image_data:
                    if "," in image_data:
                        image_data = image_data.split(",", 1)[1]
                    image_bytes = base64.b64decode(image_data)

                    if image_type == "live":
                        current_image_bytes = image_bytes
                    else:
                        # Store intake images for context
                        context_images.append(image_bytes)

                    logger.debug(
                        "Received %s image, %d bytes", image_type, len(image_bytes)
                    )
                continue

            # Handle audio chunk
            if msg_type == "audio":
                audio_data = payload.get("data")
                if audio_data:
                    if "," in audio_data:
                        audio_data = audio_data.split(",", 1)[1]
                    chunk = base64.b64decode(audio_data)
                    audio_chunks.append(chunk)
                continue

            # Handle end of speech - process accumulated audio
            if msg_type == "audio_end":
                if not audio_chunks:
                    await websocket.send_json({"type": "listening"})
                    continue

                # Concatenate all audio chunks
                full_audio = b"".join(audio_chunks)
                audio_chunks.clear()

                if len(full_audio) < 1000:  # Too short to be meaningful speech
                    await websocket.send_json({"type": "listening"})
                    continue

                # Notify client we're processing
                await websocket.send_json({"type": "processing"})

                try:
                    # Build context text from patient intake
                    context_text = None
                    if patient_context:
                        context_parts = []
                        if patient_context.get("contextPrompt"):
                            context_parts.append(patient_context["contextPrompt"])
                        elif patient_context.get("chiefComplaint"):
                            context_parts.append(
                                f"Chief complaint: {patient_context['chiefComplaint']}"
                            )
                            if patient_context.get("symptoms"):
                                context_parts.append(
                                    f"Symptoms: {', '.join(patient_context['symptoms'])}"
                                )
                            if patient_context.get("symptomDuration"):
                                context_parts.append(
                                    f"Duration: {patient_context['symptomDuration']}"
                                )
                            if patient_context.get("painLevel"):
                                context_parts.append(
                                    f"Pain level: {patient_context['painLevel']}/10"
                                )
                        context_text = (
                            "\n".join(context_parts) if context_parts else None
                        )

                    # Combine context image with live image if available
                    image_to_analyze = current_image_bytes
                    if not image_to_analyze and context_images:
                        image_to_analyze = context_images[0]  # Use first intake image

                    # Get consultation type from patient context
                    consultation_type = (
                        patient_context.get("consultationType", "general")
                        if patient_context
                        else "general"
                    )

                    # Process with orchestrator in voice mode with full patient context
                    result = await orchestrator.process_request(
                        text=context_text,
                        image_bytes=image_to_analyze,
                        audio_bytes=full_audio,
                        mode="voice",
                        patient_context=patient_context,  # Full intake data
                        consultation_type=consultation_type,
                    )

                    response_text = result.get("response", "")
                    transcript = result.get("transcript", "")

                    # Send response to client
                    await websocket.send_json(
                        {
                            "type": "speaking",
                            "text": response_text,
                            "transcript": transcript,
                        }
                    )

                    # Send full structured response
                    triage_level = result.get("triage_level", "LOW")
                    response = ChatResponse(
                        message_id=generate_message_id(),
                        timestamp=__import__("datetime").datetime.now().isoformat(),
                        response_text=response_text,
                        triage_level=triage_level,
                        triage_color=determine_triage_color(triage_level),
                        confidence=result.get("confidence", 0.5),
                        findings=format_findings_for_ui(result.get("findings", [])),
                        differential_diagnosis=format_differential_for_ui(
                            result.get("raw_results", {})
                        ),
                        recommended_actions=format_actions_for_ui(
                            result.get("raw_results", {}), response_text
                        ),
                        is_streaming=False,
                    )

                    await websocket.send_json(
                        {"type": "complete", "data": response.model_dump()}
                    )

                except Exception as e:
                    logger.error("Realtime processing error: %s", e)
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Processing failed. Please try again.",
                        }
                    )

                # Ready for next speech
                await websocket.send_json({"type": "listening"})
                continue

            # Handle text input (typed messages during call)
            if msg_type == "text":
                text = payload.get("text", "")
                if text:
                    await websocket.send_json({"type": "processing"})

                    result = await orchestrator.process_request(
                        text=text,
                        image_bytes=current_image_bytes,
                        audio_bytes=None,
                        mode="voice",
                    )

                    response_text = result.get("response", "")
                    await websocket.send_json(
                        {
                            "type": "speaking",
                            "text": response_text,
                        }
                    )

                    await websocket.send_json({"type": "listening"})
                continue

    except WebSocketDisconnect:
        logger.info("Realtime voice WebSocket disconnected")
    except Exception as exc:
        logger.exception("Realtime voice WebSocket error: %s", exc)
        try:
            await websocket.send_json(
                {"type": "error", "message": "Connection error occurred."}
            )
        except Exception:
            pass
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
# WEBRTC AI REAL-TIME VIDEO CALLING ENDPOINTS
# ============================================================================


class WebRTCAIOfferRequest(BaseModel):
    """Request for WebRTC AI session offer."""

    session_id: str
    sdp: str
    sdp_type: str = "offer"


class WebRTCAIAnswerResponse(BaseModel):
    """Response containing WebRTC answer."""

    session_id: str
    sdp: str
    sdp_type: str


class WebRTCAIIceCandidate(BaseModel):
    """ICE candidate for WebRTC AI session."""

    session_id: str
    candidate: str
    sdp_mid: Optional[str] = None
    sdp_m_line_index: Optional[int] = None


@router.post("/webrtc-ai/offer", response_model=WebRTCAIAnswerResponse)
async def webrtc_ai_offer(
    request: Request, payload: WebRTCAIOfferRequest
) -> WebRTCAIAnswerResponse:
    """
    Handle WebRTC offer for AI video calling.

    This creates a real-time WebRTC connection for video consultation with AI.
    The AI will process audio in real-time and respond via voice.
    """
    from app.services.webrtc_ai_handler import get_webrtc_ai_manager

    settings = get_settings()
    orchestrator = await get_orchestrator(settings)
    manager = get_webrtc_ai_manager()

    # Create or get session
    session = manager.get_session(payload.session_id)
    if session:
        await manager.close_session(payload.session_id)

    session = manager.create_session(
        session_id=payload.session_id,
        orchestrator=orchestrator,
    )

    # Handle the offer and get answer
    answer = await session.handle_offer(payload.sdp, payload.sdp_type)

    return WebRTCAIAnswerResponse(
        session_id=payload.session_id,
        sdp=answer.sdp,
        sdp_type=answer.type,
    )


@router.post("/webrtc-ai/ice")
async def webrtc_ai_ice(payload: WebRTCAIIceCandidate) -> dict:
    """Handle ICE candidate for WebRTC AI session."""
    from app.services.webrtc_ai_handler import get_webrtc_ai_manager

    manager = get_webrtc_ai_manager()
    session = manager.get_session(payload.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await session.add_ice_candidate(
        {
            "candidate": payload.candidate,
            "sdpMid": payload.sdp_mid,
            "sdpMLineIndex": payload.sdp_m_line_index,
        }
    )

    return {"status": "ok"}


@router.post("/webrtc-ai/close/{session_id}")
async def webrtc_ai_close(session_id: str) -> dict:
    """Close a WebRTC AI session."""
    from app.services.webrtc_ai_handler import get_webrtc_ai_manager

    manager = get_webrtc_ai_manager()
    await manager.close_session(session_id)

    return {"status": "closed"}


@router.websocket("/ws/webrtc-ai/{session_id}")
async def webrtc_ai_signaling_ws(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket for WebRTC AI signaling and real-time updates.

    This provides a bidirectional channel for:
    - WebRTC signaling (offer/answer/ice)
    - Real-time transcript updates
    - AI response notifications
    - State change notifications
    """
    await websocket.accept()

    from app.services.webrtc_ai_handler import get_webrtc_ai_manager

    settings = get_settings()
    orchestrator = await get_orchestrator(settings)
    manager = get_webrtc_ai_manager()

    # Callbacks to send updates to client
    async def on_transcript(text: str):
        try:
            await websocket.send_json(
                {
                    "type": "transcript",
                    "text": text,
                }
            )
        except Exception:
            pass

    async def on_response(text: str):
        try:
            await websocket.send_json(
                {
                    "type": "response",
                    "text": text,
                }
            )
        except Exception:
            pass

    async def on_state_change(state: str):
        try:
            await websocket.send_json(
                {
                    "type": "state",
                    "state": state,
                }
            )
        except Exception:
            pass

    # Create session with callbacks
    session = manager.create_session(
        session_id=session_id,
        orchestrator=orchestrator,
        on_transcript=lambda t: asyncio.create_task(on_transcript(t)),
        on_response=lambda r: asyncio.create_task(on_response(r)),
        on_state_change=lambda s: asyncio.create_task(on_state_change(s)),
    )

    try:
        await websocket.send_json({"type": "ready", "session_id": session_id})

        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "offer":
                # Handle WebRTC offer
                answer = await session.handle_offer(
                    payload.get("sdp"), payload.get("sdp_type", "offer")
                )
                await websocket.send_json(
                    {
                        "type": "answer",
                        "sdp": answer.sdp,
                        "sdp_type": answer.type,
                    }
                )

            elif msg_type == "ice":
                # Handle ICE candidate
                await session.add_ice_candidate(
                    {
                        "candidate": payload.get("candidate"),
                        "sdpMid": payload.get("sdp_mid"),
                        "sdpMLineIndex": payload.get("sdp_m_line_index"),
                    }
                )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebRTC AI signaling disconnected: %s", session_id)
    except Exception as exc:
        logger.exception("WebRTC AI signaling error: %s", exc)
    finally:
        await manager.close_session(session_id)


import asyncio


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
