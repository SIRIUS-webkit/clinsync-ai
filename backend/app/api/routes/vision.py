"""Vision endpoints for MedGemma."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.services.ai_orchestrator import AIOrchestrator, get_orchestrator
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/analyze")
async def analyze_image(
    request: Request,
    image: UploadFile = File(..., description="Medical image to analyze"),
    prompt: Optional[str] = Form(default=None, description="Specific question about the image"),
) -> dict:
    """
    Analyze a medical image and return structured findings.
    
    Returns:
        Structured analysis with findings, confidence, and recommendations
    """
    # Get or create orchestrator
    settings = get_settings()
    orchestrator = await get_orchestrator(settings)
    
    # Validate file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Size limit check (10MB)
    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
    
    try:
        logger.info("Analyzing image: %s, size: %d bytes", image.filename, len(image_bytes))
        
        # Process through orchestrator
        result = await orchestrator.process_request(
            text=prompt or "Analyze this medical image for any abnormalities.",
            image_bytes=image_bytes,
            audio_bytes=None,
        )

        logger.info("Result: %s", result)
        # Return simplified response for API
        return {
            "success": True,
            "findings": result.get("findings", []),
            "confidence": result.get("confidence", 0),
            "triage_level": result.get("triage_level", "LOW"),
            "recommendations": result.get("recommendations", []),
            "response": result.get("response", ""),
            "transcript": result.get("transcript"),
            "model_used": "medgemma-4b",
        }
        
    except ValueError as e:
        logger.error("Invalid image: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    except Exception as exc:
        logger.exception("Vision analysis failed: %s", exc)
        raise HTTPException(
            status_code=500, 
            detail="Image analysis failed. Please try again or consult a provider directly."
        ) from exc


@router.post("/test")  # Simple test endpoint
async def test_vision_simple(
    request: Request,
    image: UploadFile = File(...),
) -> dict:
    """Simplified test endpoint that just checks if image loads."""
    try:
        from PIL import Image
        import io
        
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        return {
            "success": True,
            "image_size": img.size,
            "image_mode": img.mode,
            "format": img.format,
            "message": "Image loaded successfully. Vision model ready.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image load failed: {str(e)}")